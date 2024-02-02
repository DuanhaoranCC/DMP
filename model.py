import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, sort_edge_index, degree, dropout_edge, scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import SparseTensor
import copy
from functools import partial
import math
import time


def mask(x, mask_rate=0.5):
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    mask_nodes = perm[: num_mask_nodes]
    # perm = torch.randperm(x)
    # num_mask_nodes = int(mask_rate * x)
    #
    # mask_nodes = perm[: num_mask_nodes]

    return mask_nodes


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    # loss = -(x * y).sum(dim=-1)
    loss = loss.mean()
    return loss


class GAT(MessagePassing):
    def __init__(self, in_channels, dropout, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.dropout = dropout

        self.att_src = nn.Parameter(torch.Tensor(1, in_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x_src, x_dst = x

        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class Cross_View(nn.Module):
    def __init__(self, data, hidden_dim, feat_drop, att_drop1, att_drop2, r1, r2
                 , r3,
                 # r5, r6
                 ):
        super(Cross_View, self).__init__()
        self.data = data
        self.hidden_dim = hidden_dim
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        # self.r4 = r4
        # self.r5 = r5
        # self.r6 = r6
        self.fc = nn.ModuleDict({
            n_type: nn.Linear(
                data[n_type].x.shape[1],
                hidden_dim,
                bias=True
            )
            for n_type in data.use_nodes
        })

        self.feat_drop = nn.Dropout(feat_drop)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        # mask gnn
        self.intra = nn.ModuleList([
            GAT(hidden_dim, att_drop1)
            for _ in range(len(self.data.schema_dict))
        ])
        self.action = nn.ModuleList([
            nn.PReLU() for _ in range(len(self.data.schema_dict))
        ])
        self.act = nn.ModuleDict({
            s: nn.PReLU() for s in self.data.use_nodes
        })

        self.bn = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(len(self.data.schema_dict))
        ])
        self.schema_dict = {s: i for i, s in enumerate(self.data.schema_dict)}
        self.reset_parameter()
        # message passing gnn
        self.intra_mp = nn.ModuleList([
            GAT(hidden_dim, att_drop2)
            for _ in range(len(self.data.mp))
        ])
        self.action_mp = nn.ModuleList([
            nn.PReLU() for _ in range(len(self.data.mp))
        ])
        self.bn_mp = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(len(self.data.mp))
        ])
        self.mp = {s: i for i, s in enumerate(self.data.mp)}

    def reset_parameter(self):
        for fc in self.fc.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, data):

        h = {}
        for n_type in data.use_nodes:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )

        # Message Passing
        for n_type in data.mp:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra_mp[self.mp[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn_mp[self.mp[n_type]](embed1)
            h[dst] = self.action_mp[self.mp[n_type]](embed1)

        #########################################################################
        # Reconstruct
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # src, dst = sc
        # x = h[src], main_h
        # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # embed1 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # sc = ('s', 'p')
        # # sc = ('director', 'movie')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss1 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # ###########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('s', 'p')
        # # sc = ('director', 'movie')
        # src, dst = sc
        # x = h[src], main_h
        # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.05)
        # embed1 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss2 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # return loss1 + loss2
        # # #########################################################################
        # # # Aminer1
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss3 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # #####################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r4)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss4 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # #########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r5)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('R', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss5 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # ###################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r6)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('R', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss6 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # return loss1 + loss2 + loss4 + loss5 + loss6 + loss3
        # Aminer
        ########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict1:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # # embed2 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss1 = sce_loss(embed2[mask_node], h1[mask_node].detach())
        # ##########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict2:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('R', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # # embed2 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss2 = sce_loss(embed2[mask_node], h1[mask_node].detach())
        # ##########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict3:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # # embed2 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss3 = sce_loss(embed2[mask_node], h1[mask_node].detach())
        #
        # return loss1 + loss2 + loss3

        # Freebase
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict1:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('w', 'm')
        # src, dst = sc
        # x = h[src], main_h
        # # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # # embed1 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        # loss1 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # ###########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict2:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('d', 'm')
        # src, dst = sc
        # x = h[src], main_h
        # # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # # embed1 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss2 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # #####################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict3:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('a', 'm')
        # src, dst = sc
        # x = h[src], main_h
        # # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # # embed1 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss3 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        #
        # return loss1 + loss2 + loss3

        # CS
        ########################################################################
        mask_node = mask(h[data.main_node], mask_rate=self.r1)
        main_h = h[data.main_node].clone()
        main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        h1 = 0
        for n_type in data.schema_dict1:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        sc = ('F', 'P')
        src, dst = sc
        x = h[src], main_h
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # embed2 = self.intra[self.schema_dict[sc]](x, edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        loss1 = sce_loss(embed2[mask_node], h1[mask_node].detach())
        ##########################################################################
        mask_node = mask(h[data.main_node], mask_rate=self.r2)
        main_h = h[data.main_node].clone()
        main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        h1 = 0
        for n_type in data.schema_dict2:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        sc = ('A', 'P')
        src, dst = sc
        x = h[src], main_h
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # embed2 = self.intra[self.schema_dict[sc]](x, edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        loss2 = sce_loss(embed2[mask_node], h1[mask_node].detach())
        ##########################################################################
        mask_node = mask(h[data.main_node], mask_rate=self.r3)
        main_h = h[data.main_node].clone()
        main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        h1 = 0
        for n_type in data.schema_dict3:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        sc = ('V', 'P')
        src, dst = sc
        x = h[src], main_h
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # embed2 = self.intra[self.schema_dict[sc]](x, edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        loss3 = sce_loss(embed2[mask_node], h1[mask_node].detach())

        return loss1 + loss2 + loss3

    def get_embed(self, data):
        h = {}
        for n_type in data.use_nodes:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )

        h2 = 0.0
        for n_type in data.mp:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra_mp[self.mp[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn_mp[self.mp[n_type]](embed1)
            h[dst] = self.action_mp[self.mp[n_type]](embed1)

        for n_type in data.schema_dict:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h2 += self.action[self.schema_dict[n_type]](embed1)
        return h2.detach()


class SAGE(MessagePassing):
    def __init__(self, in_channels, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(node_dim=0, **kwargs)
        self.lin = nn.Linear(in_channels, in_channels)
        self.lin2 = nn.Linear(in_channels, in_channels)
        self.reset_parameters()
        self.act = nn.PReLU()
        self.bn = BatchNorm(in_channels)
        self.act1 = nn.PReLU()
        self.bn1 = BatchNorm(in_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x_src, x_dst = x
        x_src = self.act(self.bn(self.lin(x_src)))
        x = (x_src, x_dst)
        out = self.propagate(edge_index, x=x)
        out = self.lin2(out)
        out = self.act1(self.bn1(out))
        return out

    def message(self, x_j):
        return x_j


class Aggregator(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Aggregator, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att = nn.Parameter(torch.Tensor(1, hidden_dim))

        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

    def forward(self, embeds):
        sp = torch.tanh(self.fc(embeds)).mean(1)

        beta = self.attn_drop(self.att) @ sp.T
        beta = torch.softmax(beta, dim=1).view(-1, 1, 1)

        z = (beta * embeds).sum(0)
        return z


class CG(nn.Module):
    def __init__(self, data, hidden_dim, feat_drop, att_drop1, att_drop2, r1, r2
                 , r3, b):
        super(CG, self).__init__()
        self.data = data
        self.hidden_dim = hidden_dim
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.b = b
        self.fc = nn.ModuleDict({
            n_type: nn.Linear(
                data[n_type].x.shape[1],
                hidden_dim,
                bias=True
            )
            for n_type in data.use_nodes
        })

        self.feat_drop = nn.Dropout(feat_drop)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        # mask gnn
        self.intra = nn.ModuleList([
            GAT(hidden_dim, att_drop1)
            for _ in range(len(self.data.schema_dict))
        ])
        self.action = nn.ModuleList([
            nn.PReLU() for _ in range(len(self.data.schema_dict))
        ])
        self.act = nn.ModuleDict({
            s: nn.PReLU() for s in self.data.use_nodes
        })

        self.bn = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(len(self.data.schema_dict))
        ])
        self.schema_dict = {s: i for i, s in enumerate(self.data.schema_dict)}
        self.reset_parameter()
        # message passing gnn
        self.intra_mp = nn.ModuleList([
            SAGE(hidden_dim)
            for _ in range(len(self.data.mp))
        ])
        self.intra_mp1 = nn.ModuleList([
            SAGE(hidden_dim)
            for _ in range(len(self.data.mp))
        ])
        self.mp = {s: i for i, s in enumerate(self.data.mp)}
        self.inter = Aggregator(hidden_dim, att_drop2)
        self.inter1 = Aggregator(hidden_dim, att_drop2)
        self.proj_head1 = nn.Sequential(nn.Linear(hidden_dim, 256), BatchNorm(256), nn.ReLU(inplace=True),
                                        nn.Linear(256, hidden_dim))
        ########################################################################################################
        self.proj_head2 = nn.Sequential(nn.Linear(hidden_dim, 256), BatchNorm(256), nn.ReLU(inplace=True),
                                        nn.Linear(256, hidden_dim))
        self.proj_head3 = nn.Sequential(nn.Linear(hidden_dim, 256), BatchNorm(256), nn.ReLU(inplace=True),
                                        nn.Linear(256, hidden_dim))
        self.proj_head4 = nn.Sequential(nn.Linear(hidden_dim, 256), BatchNorm(256), nn.ReLU(inplace=True),
                                        nn.Linear(256, hidden_dim))
        #########################################################################################################
        # self.proj_head2 = nn.Sequential(nn.Linear(hidden_dim, 256), BatchNorm(256), nn.ReLU(inplace=True),
        #                                 nn.Linear(256, data[data.main_node]['degree']['A'].shape[1]))
        # self.proj_head3 = nn.Sequential(nn.Linear(hidden_dim, 256), BatchNorm(256), nn.ReLU(inplace=True),
        #                                 nn.Linear(256, data[data.main_node]['degree']['R'].shape[1]))
        # self.proj_head4 = nn.Sequential(nn.Linear(hidden_dim, 256), BatchNorm(256), nn.ReLU(inplace=True),
        #                                 nn.Linear(256, data[data.main_node]['degree']['C'].shape[1]))

    def reset_parameter(self):
        for fc in self.fc.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, data):

        h = {}
        for n_type in data.use_nodes:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )

        # Message Passing
        # embeds1 = 0
        h2 = []
        for n_type in data.mp:
            src, dst = n_type
            x = h[src], torch.zeros_like(h[dst])
            embed1 = self.intra_mp[self.mp[n_type]](x, data[n_type].edge_index)
            x = embed1, torch.zeros_like(h[src])
            embed1 = self.intra_mp1[self.mp[n_type]](x, data[n_type].edge_index.flip([0]))
            # embeds1 += embed1
            h2.append(embed1)
        embeds1 = torch.stack(h2)
        embeds1 = self.inter1(embeds1)
        #########################################################################
        h1 = []
        for n_type in data.schema_dict:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1.append(self.action[self.schema_dict[n_type]](embed1))
        embeds2 = torch.stack(h1)
        embeds2 = self.inter(embeds2)
        loss1 = sce_loss(self.proj_head1(embeds2), embeds1.detach())
        loss5 = sce_loss(self.proj_head1(embeds1), embeds2.detach())
        #########################################################################
        # # sc = ('a', 'p')
        # # sc = ('a', 'm')
        # # sc = ('actor', 'movie')
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        # re_embed = self.proj_head2(embed1)
        # # loss2 = sce_loss(re_embed[mask_node], data[data.main_node]['degree']['A'][mask_node].detach())
        # loss2 = sce_loss(re_embed[mask_node], data[data.main_node].x[mask_node].detach())
        # #########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # # sc = ('s', 'p')
        # # sc = ('d', 'm')
        # # sc = ('director', 'movie')
        # # sc = ('R', 'P')
        # sc = ('F', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        # re_embed = self.proj_head3(embed1)
        # # loss3 = sce_loss(re_embed[mask_node], data[data.main_node]['degree']['R'][mask_node].detach())
        # loss3 = sce_loss(re_embed[mask_node], data[data.main_node].x[mask_node].detach())
        # # print(loss1.item(), loss2.item(), loss3.item())
        # # return loss1 + loss2 + loss3
        # #########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # # sc = ('w', 'm')
        # # sc = ('C', 'P')
        # sc = ('P', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        # re_embed = self.proj_head4(embed1)
        # # loss4 = sce_loss(re_embed[mask_node], data[data.main_node]['degree']['C'][mask_node].detach())
        # loss4 = sce_loss(re_embed[mask_node], data[data.main_node].x[mask_node].detach())
        # # loss4 = sce_loss(re_embed[mask_node], h[data.main_node][mask_node].detach())
        # # print(loss1.item(), loss2.item(), loss3.item(), loss4.item())
        # return loss1 + self.b*(loss2 + loss3 + loss4)
        #####################################################################################
        h1 = 0
        for n_type in data.schema_dict1:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('a', 'p')
        # sc = ('C', 'P')
        # MAG, CS
        sc = ('F', 'P')
        src, dst = sc
        x = h[src], h[dst]
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        embed2 = self.proj_head2(embed2)
        loss2 = sce_loss(embed2, h1.detach())
        ##########################################################################
        h1 = 0
        for n_type in data.schema_dict2:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('s', 'p')
        # sc = ('R', 'P')
        # MAG, CS
        sc = ('A', 'P')
        src, dst = sc
        x = h[src], h[dst]
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        embed2 = self.proj_head3(embed2)
        loss3 = sce_loss(embed2, h1.detach())
        # return loss1 + self.b*(loss2 + loss3)
        ##########################################################################
        h1 = 0
        for n_type in data.schema_dict3:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('A', 'P')
        # CS
        sc = ('V', 'P')
        src, dst = sc
        x = h[src], h[dst]
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        embed2 = self.proj_head4(embed2)
        loss4 = sce_loss(embed2, h1.detach())
        ##########################################################################
        return loss1 + self.b * (loss2 + loss3 + loss4) + loss5
        # return loss1 + loss5
        # return self.b * (loss2 + loss3 + loss4)

    def get_embed(self, data):
        h = {}
        for n_type in data.use_nodes:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )

        h1 = []
        for n_type in data.schema_dict:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1.append(self.action[self.schema_dict[n_type]](embed1))
        embeds = torch.stack(h1)
        embeds = self.inter(embeds)

        h2 = 0.0
        for n_type in data.schema_dict:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h2 += self.action[self.schema_dict[n_type]](embed1)

        # h1 = []
        # for n_type in data.schema_dict:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1.append(self.action[self.schema_dict[n_type]](embed1))
        # embeds2 = torch.stack(h1)
        # embeds2 = self.inter(embeds2)

        return (embeds + h2).detach()
