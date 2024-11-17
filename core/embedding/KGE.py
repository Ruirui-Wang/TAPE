#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from collections import defaultdict

from ogb.linkproppred import Evaluator


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, feature_dim, gamma, args,
                 double_entity_embedding=True, double_relation_embedding=True):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = feature_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / feature_dim]),
            requires_grad=False
        )

        self.entity_dim = feature_dim * 2 if double_entity_embedding else feature_dim
        self.relation_dim = feature_dim * 2 if double_relation_embedding else feature_dim
        self.feature_dim = feature_dim

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE':
            self.relation_embedding = torch.nn.Embedding(1, feature_dim)
        else:
            self.relation_embedding = torch.nn.Embedding(1, feature_dim * 2)

        if model_name == 'ConvE':
            self.inp_drop = torch.nn.Dropout(args.input_drop)
            self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
            self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)

            self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 1, bias=args.use_bias)
            self.bn0 = torch.nn.BatchNorm2d(1)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.bn2 = torch.nn.BatchNorm1d(feature_dim * 2)
            self.fc = torch.nn.Linear(32 * 4 * feature_dim, feature_dim * 2)
            self.register_parameter('b', torch.nn.Parameter(torch.zeros(nentity)))

        self.model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'ConvE': self.ConvE,
        }

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            head = sample[:, 0].repeat(1, 2)
            edge_features = F.cosine_similarity(sample[:, 0],sample[:, 2], dim=-1).long()

            relation = self.relation_embedding(edge_features)


            tail = sample[:, 2].repeat(1, 2)

        else:
            raise ValueError('mode %s not supported' % mode)



        if self.model_name in self.model_func:
            score = self.model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def ConvE(self, head, relation, tail, mode):
        """ConvE model implementation."""
        # Embedding lookup
        head_emb = head.view(-1, 1, 2, self.feature_dim)
        relation_emb = relation.view(-1, 1, 2, self.feature_dim)
        tail_emb = tail.view(-1, 1, 2, self.feature_dim)
        # Concatenate the head and relation embeddings along the feature dimension
        stacked_inputs = torch.cat([head_emb, relation_emb], 2)
        stacked_inputs = self.bn0(stacked_inputs)


        # Apply convolutions, activations, and dropout
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)


        # Flatten and pass through fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Calculate the score
        score = torch.mm(x, self.relation_embedding.weight.transpose(1, 0))
        score += self.b.expand_as(score)

        # score = torch.sum(x * tail_emb, dim=1)
        score = torch.sigmoid(score)
        print(score)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=1)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=1)
        return score

    def ComplEx(self, head, relation, tail, mode):
        if head.dim() == 2:
            head = head.unsqueeze(1)
        if relation.dim() == 2:
            relation = relation.unsqueeze(1)
        if tail.dim() == 2:
            tail = tail.unsqueeze(1)
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=1)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
