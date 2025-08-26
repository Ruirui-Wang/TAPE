import os
import sys
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add # check
from torch_sparse import SparseTensor
import torch_sparse
from torch_scatter import scatter_add
from typing import Iterable, Final
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphgps.utility.ncn import adjoverlap


# a vanilla message passing layer
class PureConv(nn.Module):
    aggr: Final[str]

    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1 + adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x


convdict = {
    "gcn":
        GCNConv,
    "gcn_cached":
        lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage":
        lambda indim, outdim: GCNConv(
            indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin":
        lambda indim, outdim: GCNConv(
            indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max":
        lambda indim, outdim: GCNConv(
            indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax":
        lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum":
        lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean":
        lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn":
        lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none":
        None
}

predictor_dict = {}


# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool]  # whether to rescale edge weight

    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1 / (1 - dp))) # rescale ratio
        self.doscale = doscale # whether to rescale edge weight

    def forward(self, adj: SparseTensor) -> SparseTensor:
        # visualize 
        # adj_sparse = coo_tensor_to_coo_matrix(adj)
        #  coo = adj.coo()
        # row_indices = coo[0].numpy()
        # col_indices =  coo[1].numpy()
        # values = torch.ones(row_indices.shape[0])
        # sparse_original = coo_matrix((values, (row_indices, col_indices)), shape=shape)
        # shape = adj.sizes()
        # plot_coo_matrix(adj_sparse, 'test')
        # TODO visualize the adjacency matrix before and after the DropAdj
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value() * self.ratio, layout="coo")
            else:
                adj.fill_value_(1 / (1 - self.dp), dtype=torch.float)
        return adj


# Vanilla MPNN composed of several layers.
class GCN(nn.Module):
    def __init__(self,
                 in_channels, # input feature dimension
                 hidden_channels, # hidden feature dimension
                 out_channels,  # output feature dimension
                 num_layers,
                 dropout,
                 ln=False, # whether to use layer normalization
                 res=False, # whether to use residual connection
                 max_x=-1, # maximum feature index
                 conv_fn="gcn", # convolution function
                 jk=False, # whether to use JumpingKnowledge
                 edrop=0.0, # edge dropout rate
                 xdropout=0.0, # input feature dropout rate
                 taildropout=0.0, # dropout rate for the last layer
                 noinputlin=False): # whether to use linear transformation for input features
        super().__init__()
        self.adjdrop = DropAdj(edrop)
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout))  # nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers == 0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))

        if num_layers == 0 or conv_fn == "none":
            self.jk = False
            return

        convfn = convdict[conv_fn]
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                              nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                                   2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))
            if self.res and x1.shape[-1] == x.shape[-1]:  # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk:  # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx * sftmax, dim=0)
        return x


# NCN predictor
class CNLinkPredictor(nn.Module):
    cndeg: Final[int]

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout, # dropout rate
                 edrop=0.0, # edge dropout rate
                 ln=False, # whether to use layer normalization
                 cndeg=-1, # degree of common neighbors
                 use_xlin=True, # whether to use linear transformation for input features
                 tailact=True, # whether to use linear transformation for the last layer
                 twolayerlin=False, # whether to use two-layer linear transformation
                 beta=1): # weight for common neighbors
        super().__init__()
        self.register_parameter("beta", nn.Parameter(beta * torch.ones((1)))) # weight for common neighbors
        self.dropadj = DropAdj(edrop) # edge dropout
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity() # layer normalization

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                  nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_channels, hidden_channels),
                                  lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
                                  nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x, # input features
                           adj, # adjacency matrix
                           tar_ei, # target edge index
                           filled1: bool = False, # whether to fill the target edge
                           cndropprobs: Iterable[float] = []): # common neighbor dropout rate
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]] # input features for the source node
        xj = x[tar_ei[1]] # input features for the target node
        x = x + self.xlin(x)
        
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg) # common neighbors
        xcns = [spmm_add(cn, x)] # common neighbor features #TODO 
        xij = self.xijlin(xi * xj)

        return torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns], dim=-1
        )



    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# NCNC predictor
class IncompleteCN1Predictor(CNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=-1,
                 testresdeg=-1,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin,
                         tailact, twolayerlin, beta)
        self.learnablept = learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
                                   nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale * (prob - self.offset))
        return self.alpha * pt * p0 / (pt * p0 + 1 - p0)

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int = None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = xi * xj
        x = x + self.xlin(x)
        
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                adj,
                adj,
                tar_ei,
                filled1,
                calresadj=True,
                cnsampledeg=self.cndeg,
                ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        else:
            cn = adjoverlap(
                adj,
                adj,
                tar_ei,
                filled1,
                calresadj=False,
                cnsampledeg=self.cndeg,
                ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        xcns = [spmm_add(cn, x)]

        # completion oder configuration
        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth - 1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth - 1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i + self.splitsize] = self.forward(x, adj, ei1[:, i: i + self.splitsize], filled1,
                                                                     depth - 1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i + self.splitsize] = self.forward(x, adj, ei2[:, i: i + self.splitsize], filled1,
                                                                     depth - 1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]])
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            xcn1 = spmm_add(cnres1, x)
            xcn2 = spmm_add(cnres2, x)
            xcns[0] = xcns[0] + xcn2 + xcn1

        xij = self.xijlin(xij)

        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def setalpha(self, alpha: float):
        self.alpha.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [],
                                       depth)


predictor_dict = {
    "NCN": CNLinkPredictor,
    "NCNC": IncompleteCN1Predictor
}