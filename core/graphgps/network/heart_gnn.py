import os
import sys
# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch 
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
import torch.nn.functional as F
from torch.nn import Embedding
import math
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding)
from torch.nn import BatchNorm1d as BN
from torch_geometric.nn import global_sort_pool
from torch_geometric.utils import negative_sampling
from torch.nn import Module
from yacs.config import CfgNode as CN



class GCN_Variant(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 dropout, 
                 data_name=None):
        super(GCN_Variant, self).__init__()

        self.convs = torch.nn.ModuleList()

        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels,normalize=False ))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        
        else:
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT_Variant(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 dropout, 
                 head=None, 
                 data_name=None):
        super(GAT_Variant, self).__init__()

        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.head = head 
        self.in_channels = in_channels
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        if self.num_layers == 1:
            out_channels = int(self.out_channels/self.head)
            self.convs.append(GATConv(self.in_channels, out_channels, heads=self.head))

        else:
            hidden_channels= int(self.hidden_channels/self.head)
            self.convs.append(GATConv(self.in_channels, hidden_channels, heads=self.head))
            
            for _ in range(self.num_layers - 2):
                self.convs.append(
                    GATConv(self.hidden_channels, hidden_channels, heads=self.head))
                
            out_channels = int(self.out_channels/head)
            self.convs.append(GATConv(self.hidden_channels, out_channels, heads=self.head))
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gat: ', len(self.convs))
            self.invest = 0
        
        # x.shape = 2708, 1433
        # conv1 1433, 64, heads 4
        # x1 2708 256
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        
        return x


class SAGE_Variant(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False,  data_name=None):
        super(SAGE_Variant, self).__init__()

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))

        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in sage: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class mlp_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  data_name=None):
        super(mlp_model, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1
        self.num_layers = num_layers

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t=None):
        if self.invest == 1:
            print('layers in mlp: ', len(self.lins))
            self.invest = 0
       
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x.squeeze()


class GIN_Variant(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 dropout,  
                 mlp_layer=None,  data_name=None):
        super(GIN_Variant, self).__init__()
        self.convs = torch.nn.ModuleList()
        gin_mlp_layer = mlp_layer
        
        if num_layers == 1:
            self.mlp= mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
            self.convs.append(GINConv(self.mlp))
        # else:
        #     # self.mlp_layers = torch.nn.ModuleList()
        #     self.mlp1 = mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
            
        #     self.convs.append(GINConv(self.mlp1))
        #     for _ in range(num_layers - 2):
        #         self.mlp = mlp_model( hidden_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
        #         self.convs.append(GINConv(self.mlp))

        #     self.mlp2 = mlp_model( hidden_channels, hidden_channels, out_channels, gin_mlp_layer, dropout)
        #     self.convs.append(GINConv(self.mlp2))

        self.dropout = dropout
        self.invest = 1
          
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # self.mlp1.reset_parameters()
        # self.mlp2.reset_parameters()



    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in gin: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 dynamic_train=False, GNN=GCNConv, use_feature=False, 
                 node_embedding=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                sampled_train = train_dataset[:1000] if dynamic_train else train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)
        
        # embedding for DRNL?
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for _ in range(0, num_layers-1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1                 , conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        # batch is the batch idx for each data samples
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
   

class GAE_forall(torch.nn.Module):
    """graph auto encoder。
    """
    def __init__(self, 
                 encoder: Module, 
                 decoder: Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """In this script we use the binary cross entropy loss function.

        params
        ----
        z: output of encoder
        pos_edge_index: positive edge index
        neg_edge_index: negative edge index
        """
        EPS = 1e-15 

        pos_logits = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        pos_loss = -torch.log(
            pos_logits + EPS).mean() # loss for positive samples

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) # negative sampling
        neg_logits = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])
        neg_loss = -torch.log(
            1 - neg_logits + EPS).mean() # loss for negative samples

        return pos_loss + neg_loss

