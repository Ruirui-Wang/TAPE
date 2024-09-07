import os, sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
from torch_sparse import SparseTensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from utils import init_random_state

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphgps.score.custom_score import LinkPredictor
# from functools import partial
from graphgps.network.ncn import predictor_dict, convdict, GCN, CNLinkPredictor, IncompleteCN1Predictor


class BertClassifier(PreTrainedModel):
    def __init__(self, model, cfg, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = LinkPredictor(hidden_dim, cfg.model.hidden_channels, 1, cfg.model.num_layers,
                                        cfg.model.dropout, 'dot')
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]

        outputs_1 = self.bert_encoder(input_ids=input_1,
                                      attention_mask=attention_mask_1,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        outputs_2 = self.bert_encoder(input_ids=input_2,
                                      attention_mask=attention_mask_2,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        emb_1 = self.dropout(outputs_1['hidden_states'][-1])
        emb_2 = self.dropout(outputs_2['hidden_states'][-1])
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb_1 = self.feat_shrink_layer(cls_token_emb_1)
            cls_token_emb_2 = self.feat_shrink_layer(cls_token_emb_2)

        logits = self.classifier(cls_token_emb_1, cls_token_emb_2)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        pos_out = logits[pos_mask]
        neg_out = logits[neg_mask]

        pos_loss = -torch.log(pos_out + 1e-15).mean() if pos_out.numel() > 0 else torch.tensor(0.0)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean() if neg_out.numel() > 0 else torch.tensor(0.0)

        loss = pos_loss + neg_loss
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]

        outputs_1 = self.bert_classifier.bert_encoder(input_ids=input_1,
                                                      attention_mask=attention_mask_1,
                                                      return_dict=return_dict,
                                                      output_hidden_states=True)
        outputs_2 = self.bert_classifier.bert_encoder(input_ids=input_2,
                                                      attention_mask=attention_mask_2,
                                                      return_dict=return_dict,
                                                      output_hidden_states=True)
        # outputs[0]=last hidden state
        emb_1 = outputs_1['hidden_states'][-1]
        emb_2 = outputs_2['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb_1 = self.feat_shrink_layer(cls_token_emb_1)
            cls_token_emb_2 = self.feat_shrink_layer(cls_token_emb_2)

        logits = self.bert_classifier.classifier(cls_token_emb_1, cls_token_emb_2).squeeze(dim=1)

        self.emb = torch.stack((cls_token_emb_1, cls_token_emb_2), dim=1).cpu().numpy().astype(np.float16)
        self.pred = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        pos_out = logits[pos_mask]
        neg_out = logits[neg_mask]

        pos_loss = -torch.log(pos_out + 1e-15).mean() if pos_out.numel() > 0 else torch.tensor(0.0)

        neg_loss = -torch.log(1 - neg_out + 1e-15).mean() if neg_out.numel() > 0 else torch.tensor(0.0)

        loss = pos_loss + neg_loss
        return TokenClassifierOutput(loss=loss, logits=logits)


class NCNClassifier(PreTrainedModel):
    def __init__(self, model, cfg, data, inf_edges, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.edges = inf_edges
        self.data = data
        hidden_dim = model.config.hidden_size
        # predfn = predictor_dict[cfg.decoder.model.type]
        if cfg.decoder.model.type == 'NCN':
            predfn = CNLinkPredictor
        elif cfg.decoder.model.type == 'NCNC':
            predfn = IncompleteCN1Predictor(predfn, scale=cfg.decoder.model.probscale,
                                            offset=cfg.decoder.model.proboffset, pt=cfg.decoder.model.pt)
        self.model = GCN(hidden_dim, cfg.decoder.model.hiddim, cfg.decoder.model.hiddim, cfg.decoder.model.mplayers,
                         cfg.decoder.model.gnndp, cfg.decoder.model.ln, cfg.decoder.model.res, cfg.decoder.data.max_x,
                         cfg.decoder.model.model, cfg.decoder.model.jk, cfg.decoder.model.gnnedp,
                         xdropout=cfg.decoder.model.xdp, taildropout=cfg.decoder.model.tdp,
                         noinputlin=False)

        self.predictor = predfn(cfg.decoder.model.hiddim, cfg.decoder.model.hiddim, 1, cfg.decoder.model.nnlayers,
                                cfg.decoder.model.predp, cfg.decoder.model.preedp, cfg.decoder.model.lnnn)

        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                token_type_ids=None,
                node_id=None,
                return_dict=None,
                preds=None):
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        edge_pos = node_id[pos_mask].T
        edge_neg = node_id[neg_mask].T

        edge_index = torch.tensor(self.edges, dtype=torch.long)
        adjmask = torch.ones_like(edge_index[0], dtype=torch.bool)  # mask for adj
        adjmask[edge_pos] = 0
        tei = edge_index[:, adjmask]  # get the target edge index
        # get the adj matrix
        adj = SparseTensor.from_edge_index(tei, sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
        adjmask[edge_pos] = 1
        adj = adj.to_symmetric()
        outputs_1 = self.bert_encoder(input_ids=input_1,
                                        attention_mask=attention_mask_1,
                                        return_dict=return_dict,
                                        output_hidden_states=True)
        outputs_2 = self.bert_encoder(input_ids=input_2,
                                        attention_mask=attention_mask_2,
                                        return_dict=return_dict,
                                        output_hidden_states=True)
        emb_1 = self.dropout(outputs_1['hidden_states'][-1])
        emb_2 = self.dropout(outputs_2['hidden_states'][-1])
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        self.data.x = self.data.x.to(cls_token_emb_1.device)
        adj = adj.to(cls_token_emb_1.device)
        node_id = node_id.to(cls_token_emb_1.device)
        self.data.x[node_id[:, 0]] = cls_token_emb_1
        self.data.x[node_id[:, 1]] = cls_token_emb_2
        h = self.model(self.data.x, adj).detach()  # get the node embeddings
        pos_outs = self.predictor.multidomainforward(h, adj, edge_pos)  # get the prediction
        pos_loss = -F.logsigmoid(pos_outs).mean()
        neg_outs = self.predictor.multidomainforward(h, adj, edge_neg)
        neg_loss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_loss + pos_loss
        return TokenClassifierOutput(loss=loss, logits=torch.cat((pos_outs, neg_outs)))


class NCNClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, data, inf_edges, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.edges = inf_edges
        self.data = data
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                token_type_ids=None,
                node_id=None,
                return_dict=None,
                preds=None):
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        edge_pos = node_id[pos_mask].T
        edge_neg = node_id[neg_mask].T

        edge_index = torch.tensor(self.edges, dtype=torch.long)
        adjmask = torch.ones_like(edge_index[0], dtype=torch.bool)  # mask for adj
        adjmask[edge_pos] = 0
        tei = edge_index[:, adjmask]  # get the target edge index
        # get the adj matrix
        adj = SparseTensor.from_edge_index(tei, sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
        adjmask[edge_pos] = 1
        adj = adj.to_symmetric()
        outputs_1 = self.bert_classifier.bert_encoder(input_ids=input_1,
                                                      attention_mask=attention_mask_1,
                                                      return_dict=return_dict,
                                                      output_hidden_states=True)
        outputs_2 = self.bert_classifier.bert_encoder(input_ids=input_2,
                                                      attention_mask=attention_mask_2,
                                                      return_dict=return_dict,
                                                      output_hidden_states=True)
        emb_1 = outputs_1['hidden_states'][-1]
        emb_2 = outputs_2['hidden_states'][-1]
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        self.data.x = self.data.x.to(cls_token_emb_1.device)
        adj = adj.to(cls_token_emb_1.device)
        node_id = node_id.to(cls_token_emb_1.device)
        self.data.x[node_id[:, 0]] = cls_token_emb_1
        self.data.x[node_id[:, 1]] = cls_token_emb_2
        h = self.bert_classifier.model(self.data.x, adj)  # get the node embeddings
        output = self.bert_classifier.predictor.multidomainforward(h, adj, node_id.permute(1, 0))
        loss = -F.logsigmoid(output).mean()
        return TokenClassifierOutput(loss=loss, logits=output)


class GCNClassifier(PreTrainedModel):
    def __init__(self, model, cfg, data, inf_edges, GCN_Model, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.edges = inf_edges
        self.data = data
        self.model = GCN_Model
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                token_type_ids=None,
                node_id=None,
                return_dict=None,
                preds=None):
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        edge_pos = node_id[pos_mask].T
        edge_neg = node_id[neg_mask].T

        edge_index = torch.tensor(self.edges, dtype=torch.long)
        adjmask = torch.ones_like(edge_index[0], dtype=torch.bool)  # mask for adj
        adjmask[edge_pos] = 0
        tei = edge_index[:, adjmask]  # get the target edge index
        # get the adj matrix
        adj = SparseTensor.from_edge_index(tei, sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
        adjmask[edge_pos] = 1
        adj = adj.to_symmetric()
        row, col, _ = adj.coo()
        batch_edge_index = torch.stack([col, row], dim=0)
        torch.cuda.empty_cache()

        outputs_1 = self.bert_encoder(input_ids=input_1,
                                      attention_mask=attention_mask_1,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        outputs_2 = self.bert_encoder(input_ids=input_2,
                                      attention_mask=attention_mask_2,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        emb_1 = self.dropout(outputs_1['hidden_states'][-1])
        emb_2 = self.dropout(outputs_2['hidden_states'][-1])
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        batch_edge_index = batch_edge_index.to(cls_token_emb_1.device)
        self.data.x = self.data.x.to(cls_token_emb_1.device)
        node_id = node_id.to(cls_token_emb_1.device)
        self.data.x[node_id[:, 0]] = cls_token_emb_1
        self.data.x[node_id[:, 1]] = cls_token_emb_2
        h = self.model.encoder(self.data.x, batch_edge_index).detach()
        loss = self.model.recon_loss(h, edge_pos, edge_neg)
        return TokenClassifierOutput(loss=loss, logits=h)


class GCNClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, data, inf_edges, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.edges = inf_edges
        self.data = data
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                token_type_ids=None,
                node_id=None,
                return_dict=None,
                preds=None):
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        edge_pos = node_id[pos_mask].T
        edge_neg = node_id[neg_mask].T

        edge_index = torch.tensor(self.edges, dtype=torch.long)
        adjmask = torch.ones_like(edge_index[0], dtype=torch.bool)  # mask for adj
        adjmask[edge_pos] = 0
        tei = edge_index[:, adjmask]  # get the target edge index
        # get the adj matrix
        adj = SparseTensor.from_edge_index(tei, sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
        adjmask[edge_pos] = 1
        adj = adj.to_symmetric()
        row, col, _ = adj.coo()
        batch_edge_index = torch.stack([col, row], dim=0)

        outputs_1 = self.bert_classifier.bert_encoder(input_ids=input_1,
                                                      attention_mask=attention_mask_1,
                                                      return_dict=return_dict,
                                                      output_hidden_states=True)
        outputs_2 = self.bert_classifier.bert_encoder(input_ids=input_2,
                                                      attention_mask=attention_mask_2,
                                                      return_dict=return_dict,
                                                      output_hidden_states=True)
        emb_1 = outputs_1['hidden_states'][-1]
        emb_2 = outputs_2['hidden_states'][-1]
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        self.data.x = self.data.x.to(cls_token_emb_1.device)
        adj = adj.to(cls_token_emb_1.device)
        batch_edge_index = batch_edge_index.to(cls_token_emb_1.device)
        node_id = node_id.to(cls_token_emb_1.device)
        self.data.x[node_id[:, 0]] = cls_token_emb_1
        self.data.x[node_id[:, 1]] = cls_token_emb_2
        h = self.bert_classifier.model.encoder(self.data.x, batch_edge_index).detach()
        loss = self.bert_classifier.model.recon_loss(h, edge_pos, edge_neg)
        pred = self.bert_classifier.model.decoder(h[node_id.T[0]], h[node_id.T[1]])
        return TokenClassifierOutput(loss=loss, logits=pred)
