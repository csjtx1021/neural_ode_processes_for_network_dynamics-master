"""Script that utilizes an ANP to regress points to a sine curve."""

import logging
import os
import time
import warnings
import sys
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from functools import partial
from sklearn.metrics import roc_auc_score

import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

import torchdiffeq as ode

from torchviz import make_dot

from tools import *
from plots import *
from load_dynamics_solution2and3 import dynamics_dataset, display_3D, display_diff

import argparse

parser = argparse.ArgumentParser(description='GraphNDP_OneForAll')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--topo_type', help='topo_type, default:"power_law"', default="power_law")
parser.add_argument('--dynamics_name', help='dynamics_name, default:"heat_diffusion_dynamics"',
                    default="heat_diffusion_dynamics")
parser.add_argument('--num_graphs', type=int, default=1000, metavar='S',
                    help='num_graphs (default: 1000)')
parser.add_argument('--time_steps', type=int, default=100, metavar='S',
                    help='time_steps (default: 100)')
parser.add_argument('--x_dim', type=int, default=1, metavar='S',
                    help='x_dim (default: 1)')
parser.add_argument('--latent_dim', type=int, default=20, metavar='S',
                    help='latent_dim (default: 20)')
parser.add_argument('--hidden_dim', type=int, default=20, metavar='S',
                    help='hidden_dim (default: 20)')
parser.add_argument('--gnn_type', help='gnn_typ, default:"gat"', default="gat")
parser.add_argument('--num_gnn_blocks', type=int, default=2, metavar='S',
                    help='num_gnn_blocks (default: 2)')
parser.add_argument('--use_ML_loss', action='store_true', default=False,
                    help='set use_ML_loss')
parser.add_argument('--is_determinate', action='store_true', default=False,
                    help='set is_determinate')
parser.add_argument('--is_uncertainty', action='store_true', default=False,
                    help='set is_determinate')
parser.add_argument('--train', action='store_true', default=False,
                    help='flag for training')
parser.add_argument('--test', action='store_true', default=False,
                    help='flag for testing')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--tune', action='store_true', default=False,
                    help='output the result for tuning the hyperparams')
parser.add_argument('--bound_t_context', type=float, default=0., metavar='S',
                    help='bound_t_context (default: 0.)')
parser.add_argument('--test_topo_type', help='topo_type, default:"none"', default="none")
parser.add_argument('--start_epoch', type=int, default=0, metavar='S',
                    help='start_epoch (default: 0)')
parser.add_argument('--num_epochs', type=int, default=20, metavar='S',
                    help='num_epochs (default: 20)')
parser.add_argument('--constraint_state', action='store_true', default=False,
                    help='flag for the state constraint')
parser.add_argument('--is_fine_tune', action='store_true', default=False,
                    help='flag for is_fine_tune')
parser.add_argument('--train_2nd_phase', action='store_true', default=False,
                    help='flag for train_2nd_phase')
parser.add_argument('--test_with_2nd_phase', action='store_true', default=False,
                    help='flag for test_with_2nd_phase')
parser.add_argument('--is_sparsity', action='store_true', default=False,
                    help='flag for sparsity')
parser.add_argument('--sparsity', type=float, default=0.01, metavar='S',
                    help='sparsity (default: 0.01)')
parser.add_argument('--task_features_dim', type=int, default=4, metavar='S',
                    help='task_features_dim (default: 4)')
                    
                    

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  #
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
device = torch.device("cuda:2" if args.cuda else "cpu")
print("using device : ", device)

os.chdir(".")
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)


def get_mask_dim(dim, max_dim):
    assert dim <= max_dim
    res = []
    for i in range(max_dim):
        if i < dim:
            res.append(1)
        else:
            res.append(0)
    return np.array(res)


##=====================================================================
##
##                  models
##
##====================================================================

class ScoreNet(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.4):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                    num_heads=num_heads,
                                                    batch_first=True,
                                                    dropout=dropout)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
        )
        
        self.read_out = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        

        # save_last_rep
        self.hidden_rep = None

    def forward(self, R):
        # R: [batch_size, #models, hidden_dim]
        R_res, weights = self.self_attention(R, R, R)
        # [batch_size, #models, hidden_dim]
        hidden_rep = self.scorer(R_res)
        self.hidden_rep = hidden_rep.detach()
        # [batch_size, #models, 1]
        scores = self.read_out(hidden_rep)
        # [batch_size, #models, 1] -> # [batch_size, #models,]
        # scores = torch.softmax(scores.sum(-1), dim=-1)

        return scores.sum(-1)


class GNN(torch.nn.Module):
    def __init__(self, name, num_layers, in_channels, hidden_channels, out_channels, use_edge_attr=False, num_heads=8):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.block_list = nn.ModuleList()
        # self.LN_list = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = in_channels
                out_dim = hidden_channels
            elif i > 0 and i < num_layers - 1:
                in_dim = hidden_channels
                out_dim = hidden_channels
            else:
                in_dim = hidden_channels
                out_dim = out_channels
            if name == 'gcn':
                block = GCNConv(in_dim, out_dim)
            elif name == 'gat':
                if i == 0:
                    if use_edge_attr:
                        block = GATConv(in_dim, out_dim, heads=num_heads, concat=True, edge_dim=1)
                    else:
                        block = GATConv(in_dim, out_dim, heads=num_heads, concat=True)
                elif i > 0 and i < num_layers - 1:
                    if use_edge_attr:
                        block = GATConv(in_dim * num_heads, out_dim, heads=num_heads, concat=True, edge_dim=1)
                    else:
                        block = GATConv(in_dim * num_heads, out_dim, heads=num_heads, concat=True)
                else:
                    if use_edge_attr:
                        block = GATConv(in_dim * num_heads, out_dim, heads=num_heads, concat=False, edge_dim=1)
                    else:
                        block = GATConv(in_dim * num_heads, out_dim, heads=num_heads, concat=False)
            elif name == 'sage':
                block = SAGEConv(in_dim, out_dim)
            self.block_list.append(block)

            self.act_func = nn.LeakyReLU(inplace=True)

            # self.LN_list.append(nn.LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        H = []
        for i in range(len(self.block_list)):
            if self.use_edge_attr:
                x = self.act_func(self.block_list[i](x, edge_index.long(), edge_attr))
            else:
                x = self.act_func(self.block_list[i](x, edge_index.long()))
            # x = self.LN_list[i](x)
            H.append(x)
        return H


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class ODEFunc(nn.Module):
    def __init__(self, decode_F, decode_G):
        super(ODEFunc, self).__init__()
        self.decode_F = decode_F
        self.decode_G = decode_G
        #
        self.Z = None
        self.adj = None
        self.task_info = None

    def update(self, Z, adj, task_info):
        self.Z = Z
        self.adj = adj
        self.task_info = task_info

    def forward(self, t, x):  # How to use t?
        # print(x.size())
        # [num_sampling, # points,d] -> [num_sampling, # points,d]
        # x_encoded = self.encode_x(x)
        # [num_sampling, # points,d], [num_sampling, # points,d] -> [num_sampling, # points,d + d]
        x_encoded_augment = torch.cat([x, self.Z], dim=-1)
        # [num_sampling, # points,d + d] -> [num_sampling, # points,d]
        out_F = self.decode_F(x_encoded_augment)
        # print('x has nan =', torch.isnan(x).sum())
        # print('self.Z has nan =', torch.isnan(self.Z).sum())
        # print('x_encoded_augment has nan =', torch.isnan(x_encoded_augment).sum())
        # print('out_F has nan =', torch.isnan(out_F).sum())
        if len(self.adj) > 2:
            row, col, edge_weights = self.adj
            # [num_sampling, # points's neighbors,d], [num_sampling, # points's neighbors,d+d] -> [num_sampling, # points's neighbors,d]
            # print(x_encoded[:, col, :].size(), x_encoded_augment[:, row, :].size())
            x_i_j_in = torch.cat([x[:, col.long(), :], x_encoded_augment[:, row.long(), :]], dim=-1)
            out_G = self.decode_G(x_i_j_in) * edge_weights.view(1, -1, 1).float()

            # print('edge_weights=',edge_weights)
            # print('out_G=', out_G)

            # print('edge_weights has nan =', torch.isnan(edge_weights).sum())

            # print('out_G has nan =', torch.isnan(out_G).sum())

        else:
            row, col = self.adj
            # [num_sampling, # points's neighbors,d], [num_sampling, # points's neighbors,d+d] -> [num_sampling, # points's neighbors,d]
            # print(x_encoded[:, col, :].size(), x_encoded_augment[:, row, :].size())
            x_i_j_in = torch.cat([x[:, col.long(), :], x_encoded_augment[:, row.long(), :]], dim=-1)
            out_G = self.decode_G(x_i_j_in)

        # [num_sampling, # points,out_dim], [num_sampling, # points's neighbors,out_dim] -> [num_sampling, # points,out_dim]
        out_dynamics = out_F + scatter_sum(out_G, col.long(), dim=1, dim_size=x.size(1))

        # print('out_dynamics has nan =', torch.isnan(out_dynamics).sum())

        return out_dynamics


class GNDP(nn.Module):
    def __init__(
            self,
            state_dim,
            latent_dim,
            hidden_dim,
            gnn_type,
            num_gnn_blocks=2,
            is_determinate=True,
            is_uncertainty=True,
            use_ML_loss=False,
            use_edge_attr=False,
    ):
        super().__init__()

        self.name = 'GNDP_OneForAll'
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.gnn_type = gnn_type

        self.use_ML_loss = use_ML_loss
        self.use_edge_attr = use_edge_attr

        # encoders
        # self.encode_t = Time2Vec('sin', hidden_dim).to(device)

        self.encode_t = nn.Sequential(
            nn.Linear(1, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # # nn.LayerNorm(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )

        self.encode_x = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  ## add one dim parameters for "null" node type
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # # nn.LayerNorm(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )

        ##
        # gnn_type = 'sage'
        self.encode_structure = GNN(gnn_type, num_gnn_blocks, hidden_dim, 16, 16, use_edge_attr)
        ##
        if gnn_type == 'gat':
            in_dim = 16 * 8 * (num_gnn_blocks - 1) + 16
        else:
            in_dim = 16 * num_gnn_blocks
        # in_dim = hidden_dim
        self.encode_self_phi = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.LayerNorm(hidden_dim),
        )
        self.encode_self_rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(inplace=True),
        )
        self.encode_z_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encode_z_logsigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )

        # cross attention layers for determinate_path
        # self.encode_cross_attention_enc_x = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        # )
        # self.encode_attention_enc_gnn = GNN(gnn_type, num_gnn_blocks, hidden_dim, 16, 16)
        # ##
        # if gnn_type == 'gat':
        #     in_dim = 16 * 8 * (num_gnn_blocks - 1) + 16
        # else:
        #     in_dim = 16 * num_gnn_blocks
        # self.encode_attention_enc_gnn_t = nn.Sequential(
        #     nn.Linear(in_dim + hidden_dim, hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        # )
        # self.encode_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=False)

        # parmas for path
        self.is_determinate = is_determinate
        self.is_uncertainty = is_uncertainty

        assert self.is_determinate + self.is_uncertainty > 0

        ## decoders
        self.decoder_encode_x = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.LayerNorm(latent_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )
        self.decoder_encode_x_mean = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder_encode_x_logsigma = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        in_dim_for_decoder_F = latent_dim
        if self.is_determinate:
            in_dim_for_decoder_F += hidden_dim
        if self.is_uncertainty:
            in_dim_for_decoder_F += hidden_dim
        # decoders
        self.decode_F = nn.Sequential(
            nn.Linear(in_dim_for_decoder_F, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.Tanh(),
            # nn.LayerNorm(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Tanh(),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.Tanh(),
        )
        in_dim_for_decoder_G = latent_dim + latent_dim
        if self.is_determinate:
            in_dim_for_decoder_G += hidden_dim
        if self.is_uncertainty:
            in_dim_for_decoder_G += hidden_dim
        self.decode_G = nn.Sequential(
            nn.Linear(in_dim_for_decoder_G, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.Tanh(),
            # nn.LayerNorm(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Tanh(),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.Tanh(),
        )

        in_dim_for_decode_state_hidden = latent_dim + hidden_dim
        if self.is_uncertainty:
            in_dim_for_decode_state_hidden += hidden_dim
        if self.is_determinate:
            in_dim_for_decode_state_hidden += hidden_dim
        self.decode_state_hidden = nn.Sequential(
            nn.Linear(in_dim_for_decode_state_hidden, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
        )
        self.decode_state_mean = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, state_dim),
        )
        self.decode_state_logsigma = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, state_dim),
        )

        # self.learned_logsigma = nn.Parameter(torch.zeros(state_dim))
        # self.learned_logsigma = nn.Parameter(torch.ones(state_dim))

        #
        self.ode_func = ODEFunc(self.decode_F, self.decode_G)

        # params for odeint
        self.adjoint = True
        # self.rtol = .01
        # self.atol = .001
        self.rtol = 1e-3
        self.atol = 1e-4
        self.method = 'dopri5'  # 'dopri5'

        # For intermediate storage
        self.Z = None
        self.x_0 = None
        self.adj = None
        self.task_info = None

        self.Z_determinate = None

        #  for Cyclical Annealing Schedule of KL's weight beta
        self.beta = torch.ones(1).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        modules = [self.encode_t,
                   self.encode_x,
                   self.encode_self_phi,
                   self.encode_self_rho,
                   self.encode_z_mean,
                   self.encode_z_logsigma,
                   self.decoder_encode_x,
                   self.decoder_encode_x_mean,
                   self.decoder_encode_x_logsigma,
                   self.decode_F,
                   self.decode_G,
                   self.decode_state_hidden,
                   self.decode_state_mean,
                   self.decode_state_logsigma,
                   ]
        for m in modules:
            if isinstance(m, nn.Linear):
                # print('** is linear')
                linear_init(m, activation=nn.LeakyReLU())
            elif isinstance(m, nn.Sequential):
                # print('** is not linear')
                # print(m)
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        linear_init(mm, activation=nn.LeakyReLU())
            else:
                # print('** is not linear')
                # print([mm for mm in m.modules()])
                weights_init(m)
        # exit(1)

        # weights_init(self)

    def update_beta(self, epoch, max_epoch):
        #  for Cyclical Annealing Schedule of KL's weight beta
        M = torch.tensor([4.])
        R = torch.tensor([.5])
        tau = (torch.tensor([epoch - 1]) % (torch.tensor([max_epoch]) / M).ceil()) / (torch.tensor([max_epoch]) / M)
        self.beta = tau / R if tau <= R else torch.ones(1)
        self.beta = self.beta.to(device)
        print('update beta %s' % str(self.beta))

    def encode_context_graph(self, points):
        #   points: (t, x_self, mask, point_info, adj)
        #       where,
        #       t [# points, ]
        #       x_self [# points * N, d],
        #       mask [# points * N,],
        #       point_info [# points * N, ],
        #       adj [2, # edges,],
        #       task_info [# points, ]

        t, x_self, mask, point_info, adj, task_info = points

        ## [#points, d]
        t_encoded_all = self.encode_t(torch.cat(t, dim=0).view(-1, 1))

        ## make all batch
        point_info_all = []
        adj_all = []
        num_nodes_last = 0
        for idx in range(len(t)):
            point_info_all.append(point_info[idx] + len(point_info_all))
            adj_all.append(torch.cat([adj[idx][:2, :] + num_nodes_last, adj[idx][2:, :]], dim=0))
            num_nodes_last += point_info_all[-1].size(0)
        ##
        x_self_all = torch.cat(x_self, dim=0)
        mask_all = torch.cat(mask, dim=0).long()
        point_info_all = torch.cat(point_info_all, dim=0).view(-1).long()
        adj_all = torch.cat(adj_all, dim=-1)
        task_info_all = torch.cat(task_info, dim=0).view(-1).long()

        ## make init embedding on nodes for each point
        # x_self_all_filter = torch.zeros_like(x_self_all)
        # x_self_all_filter[mask_all == 1] = x_self_all[mask_all == 1]
        # x_self_all_filter[mask_all < 1] = self.null_node_embedding
        # x_self_all_augment = torch.zeros_like(x_self_all)
        # x_self_all_augment[mask_all[:,0] == 1] = x_self_all[mask_all[:,0] == 1]
        x_self_all_augment = x_self_all * mask_all.float()
        ## add 0 at first dim of node states and add 1 at first dim of null node
        x_self_all_augment = torch.cat([1. - (mask_all.sum(-1) >= 1).float().view(-1, 1), x_self_all_augment], dim=-1)

        x_self_all_augment_embedding = self.encode_x(x_self_all_augment)

        if self.use_edge_attr:
            x_self_all_augment_embedding_new_list = self.encode_structure(x_self_all_augment_embedding,
                                                                          adj_all[:2, :].long(), adj_all[-1, :].float().view(-1, 1))
        else:
            x_self_all_augment_embedding_new_list = self.encode_structure(x_self_all_augment_embedding,
                                                                          adj_all[:2, :].long())
        x_self_graph_glo_embedding = []
        for x_self_all_augment_embedding_new_one in x_self_all_augment_embedding_new_list:
            x_self_graph_glo_embedding.append(scatter_sum(x_self_all_augment_embedding_new_one, point_info_all, dim=0))
        x_self_graph_glo_embedding = torch.cat(x_self_graph_glo_embedding, dim=-1)

        r_i_all = self.encode_self_phi(torch.cat([t_encoded_all, x_self_graph_glo_embedding], dim=-1))

        # self attention
        # make query key and value
        # graph_glo_embedding_list_1 = self.encode_attention_enc_gnn(self.encode_cross_attention_enc_x(mask_all.float()), adj_all)
        # graph_glo_embedding_1 = []
        # for x_self_all_augment_embedding_new_one in graph_glo_embedding_list_1:
        #     graph_glo_embedding_1.append(
        #             scatter_sum(x_self_all_augment_embedding_new_one, point_info_all, dim=0))
        # graph_glo_embedding_1 = torch.cat(graph_glo_embedding_1, dim=-1)
        # ##
        # query = self.encode_attention_enc_gnn_t(torch.cat([t_encoded_all, graph_glo_embedding_1], dim=-1))
        # key = self.encode_attention_enc_gnn_t(torch.cat([t_encoded_all, graph_glo_embedding_1], dim=-1))
        # value = r_i_all
        #
        # r_i_all_self_atten = []
        # for batch_idx in range(max(task_info_all)+1):
        #     # print(query[task_info_all==batch_idx].size(), key[task_info_all==batch_idx].size(), value[task_info_all==batch_idx].size())
        #     r_i_all_self_atten_batch_idx, _ = self.encode_attention(query[task_info_all==batch_idx], key[task_info_all==batch_idx], value[task_info_all==batch_idx])
        #     r_i_all_self_atten.append(r_i_all_self_atten_batch_idx)
        # r_i_all_self_atten = torch.cat(r_i_all_self_atten, dim=0)

        #  [batch_size, d]
        Z_determinate = scatter_mean(r_i_all, task_info_all, dim=0)

        z_hidden = self.encode_self_rho(Z_determinate)
        z_mean = self.encode_z_mean(z_hidden)
        z_logsigma = self.encode_z_logsigma(z_hidden)
        # Bound the variance
        # z_sigma = 0.01 + 0.99 * F.softplus(z_logsigma)
        # z_sigma = 0.1 + 0.9 * torch.sigmoid(z_logsigma)
        z_sigma = 0.1 + 0.9 * torch.sigmoid(z_logsigma)

        ## [batch_size, d]
        dist_z = torch.distributions.Normal(z_mean, z_sigma)

        return Z_determinate, dist_z

    def ode_integration(self, vt, x):

        integration_time_vector = vt.type_as(x)

        self.ode_func.update(self.Z, self.adj, self.task_info)

        if self.adjoint:
            out = ode.odeint_adjoint(self.ode_func,
                                     x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.ode_func,
                             x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # the size of out should be confirmed later
        return out  ## [#steps, num_sampling, #nodes, d]

    def forward(self, adj, x0, task_info, x_context, x_target, num_sampling=1, use_NPLoss=True):
        # x0 [# nodes, d],
        # adj [2, #edges],
        # task_info [# nodes,]
        #   x_context: (t, x_self, mask, point_info, adj)
        #       where,
        #       t [# points, ]
        #       x_self [# points, d],
        #       mask [# points,],
        #       point_info [# points, ],
        #       adj [2, # edges,],
        #       task_info [# points, ]
        # x_target: similar to x_context
        #   if x_self in points in x_target is None then, just testing; otherwise training

        t_target, x_self_target, mask_target, point_info_target, adj_target, task_info_target = x_target

        if x_self_target is not None:
            _train_flag = True
        else:
            _train_flag = False

        if not self.is_uncertainty or (_train_flag and use_NPLoss):
            num_sampling = 1

        # encode init_x in decoding process
        # [#batch_size*N, d]
        decoder_encode_x_hidden = self.decoder_encode_x(x0)
        decoder_encode_x_mean = self.decoder_encode_x_mean(decoder_encode_x_hidden)
        decoder_encode_x_logsigma = self.decoder_encode_x_logsigma(decoder_encode_x_hidden)
        # Bound the variance
        # decoder_encode_x_sigma = 0.01 + 0.99 * F.softplus(decoder_encode_x_logsigma)
        decoder_encode_x_sigma = 0.1 + 0.9 * torch.sigmoid(decoder_encode_x_logsigma)
        # [batch_size, d]
        dist_l_0 = torch.distributions.Normal(decoder_encode_x_mean, decoder_encode_x_sigma)
        # [num_sampling, #batch_size*N, d]
        self.x_0 = dist_l_0.rsample([num_sampling])
        # self.x_0 = x0.unsqueeze(0).repeat(num_sampling, 1, 1)  # [#batch_size*N, d]->[num_sampling, #batch_size*N, d]
        self.adj = adj
        self.task_info = task_info  # [#batch_size*N,]

        ## encode context
        Z_determinate, dist_prior_z = self.encode_context_graph(x_context)

        self.Z_determinate = Z_determinate

        if self.is_determinate:
            Z_determinate = Z_determinate.unsqueeze(0).repeat(num_sampling, 1,
                                                              1)  # [batch_size, d] -> [num_sampling, batch_size, d]
        if self.is_uncertainty:
            if _train_flag and use_NPLoss:
                _, dist_poster_z = self.encode_context_graph(x_target)
            else:
                dist_poster_z = dist_prior_z
            Z_sampling = dist_poster_z.rsample([num_sampling])  # [num_sampling, batch_size, d]
        if self.is_determinate and self.is_uncertainty:
            Z_augment = torch.cat([Z_determinate, Z_sampling], dim=-1)
        elif not self.is_determinate and self.is_uncertainty:
            Z_augment = Z_sampling
        elif self.is_determinate and not self.is_uncertainty:
            Z_augment = Z_determinate
        else:
            print("ERROR setting on is_determinate|is_uncertainty, %s,%s" % (
                self.is_determinate, self.is_uncertainty))
            exit(1)
        # print(Z_sampling.size(),dist_poster_z.loc, dist_poster_z.scale)
        # print(Z_sampling[0].view(-1))
        # print(Z_sampling[1].view(-1))
        # print(Z_sampling[2].view(-1))
        # print(Z_sampling[3].view(-1))
        # print(Z_sampling[4].view(-1))
        # Z_augment = torch.zeros_like(Z_augment).detach()

        ## decode
        # [num_sampling, batch_size, d] -> [num_sampling, batch_size*N, d]
        self.Z = Z_augment[:, task_info, :]

        # handle t
        t_target = torch.cat(t_target, dim=0).view(-1)
        t_target = t_target.clone().detach()
        t_target_remove_duplicates_and_sort_increasing = torch.unique(t_target)
        indices_t_target = t_target.detach()
        for ttt_idx in range(len(t_target_remove_duplicates_and_sort_increasing)):
            indices_t_target[indices_t_target == t_target_remove_duplicates_and_sort_increasing[ttt_idx]] = ttt_idx
        indices_t_target = indices_t_target.long()

        # print(t_target_remove_duplicates_and_sort_increasing, indices_t_target)
        # exit(1)

        # integration hidden
        # [#steps, num_sampling, batch_size*N, d]
        # print(t_target_remove_duplicates_and_sort_increasing, indices_t_target)
        # [#steps, num_sampling, #nodes, d]
        pre_state_out_hidden = self.ode_integration(t_target_remove_duplicates_and_sort_increasing, self.x_0)
        # [#steps, num_sampling, batch_size*N, d] -> [num_sampling, #steps, batch_size*N, d]
        pre_state_out_hidden = pre_state_out_hidden.transpose(0, 1)
        #  [num_sampling, #steps, batch_size*N, d] -> [num_sampling, #points, batch_size*N, d]
        pre_state_out_hidden = pre_state_out_hidden[:, indices_t_target, :]

        # decode hidden to state
        # [num_sampling, batch_size*N, d] -> [num_sampling, #points, batch_size*N, d]
        Z_ = self.Z.unsqueeze(1).repeat(1, pre_state_out_hidden.size(1), 1, 1)
        #  [#points, d]
        t_target_encoded = self.encode_t(t_target.detach().view(-1, 1))
        #  [#points, d] -> [num_sampling, #points, batch_size*N, d]
        t_target_encoded = t_target_encoded.view(1, -1, 1, self.hidden_dim).repeat(num_sampling, 1,
                                                                                   pre_state_out_hidden.size(2), 1)
        # [num_sampling, #points, batch_size*N, d + d + d]
        pre_state_out_hidden_ = torch.cat([pre_state_out_hidden, Z_, t_target_encoded], dim=-1)
        # pre_state_out_hidden_ = torch.cat([pre_state_out_hidden, t_target_encoded], dim=-1)
        # [num_sampling, #points, batch_size*N, d + d + d] -> [num_sampling, #points, batch_size*N, d]
        # print(pre_state_out_hidden_.size())
        pre_state_out_hidden_ = self.decode_state_hidden(pre_state_out_hidden_)
        # [num_sampling, #points, batch_size*N, d + d] -> [num_sampling, #points, batch_size*N, d]
        pre_state_out_hidden_ = torch.cat([pre_state_out_hidden, pre_state_out_hidden_], dim=-1)
        pre_state = self.decode_state_mean(pre_state_out_hidden_)
        pre_state_logsigma = self.decode_state_logsigma(pre_state_out_hidden_)
        # Bound the variance
        pre_state_sigma = 0.01 + 0.99 * F.softplus(pre_state_logsigma)
        # pre_state_sigma = torch.sigmoid(pre_state_logsigma)

        #  [num_sampling, #points, batch_size*N, d] -> [num_sampling, #points * batch_size*N, d]
        pre_state = pre_state.view(num_sampling, -1, self.state_dim)
        ## add state constraint when args.constraint_state is True
        if args.constraint_state:
            # pre_state = F.softplus(pre_state) / torch.sum(F.softplus(pre_state), dim=-1, keepdim=True)
            pre_state = F.softmax(pre_state, dim=-1)
        pre_state_sigma = pre_state_sigma.view(num_sampling, -1, self.state_dim)

        # make mask and x_self
        mask_all = torch.cat(mask_target, dim=0).long()

        mask_target_extend = []
        for idx in range(len(t_target)):
            mask_target_ = torch.zeros_like(x0)
            mask_target_[task_info == task_info_target[idx], :] = mask_target[idx].float()
            mask_target_extend.append(mask_target_)
        ## [#points * batch_size*N, d]
        mask_target_extend = torch.cat(mask_target_extend, dim=0).view(-1, self.state_dim)

        ## [num_sampling,  # observations]
        pre_state_target_mean = pre_state[:, mask_target_extend == 1]
        pre_state_target_sigma = pre_state_sigma[:, mask_target_extend == 1]
        poster_dist = torch.distributions.Normal(pre_state_target_mean, pre_state_target_sigma)

        loss = None
        loss_detail = None
        if _train_flag:
            # training
            # make x_self
            x_self_target_extend = []
            for idx in range(len(t_target)):
                x_self_target_ = torch.zeros_like(x0)
                x_self_target_[task_info == task_info_target[idx]] = x_self_target[idx]
                x_self_target_extend.append(x_self_target_)
            ## [#points * batch_size*N, d] -> [num_sampling, #points * batch_size*N, d]
            x_self_target_extend = torch.cat(x_self_target_extend, dim=0).unsqueeze(0).repeat(num_sampling, 1, 1)

            if self.is_determinate and not self.is_uncertainty:
                # [1, #observations] -> [#observations]
                log_p_ = poster_dist.log_prob(
                    x_self_target_extend[:, mask_target_extend == 1]).view(-1)
                # [batch_size*N, ] -> [#points, batch_size*N, ] -> [#points*batch_size*N,] -> [# observations,]

                # [batch_size*N, ] -> [batch_size*N, d] -> [#points, batch_size*N, d] -> [#points*batch_size*N, d] -> [# observations,]
                task_info_observations = \
                    task_info.unsqueeze(1).repeat(1, self.state_dim).unsqueeze(0).repeat(len(t_target), 1, 1).view(-1,
                                                                                                                   self.state_dim)[
                        mask_target_extend == 1]

                # [batch_size,]
                log_p = scatter_sum(log_p_, task_info_observations, dim=0)
                # [batch_size] -> 1
                loss = -log_p.mean()

                loss_detail = {'neg_log_p': 0., 'kl': 0.}
            else:
                if use_NPLoss:  ##use NP loss
                    # get log probability
                    # Get KL between prior and posterior
                    # [batch_size, d] -> [batch_size,]
                    loss_kl = torch.distributions.kl_divergence(dist_poster_z, dist_prior_z).sum(-1)

                    # [num_sampling,  #observations]
                    log_p_ = poster_dist.log_prob(
                        x_self_target_extend[:, mask_target_extend == 1])

                    # [batch_size*N, ] -> [batch_size*N, d] -> [#points, batch_size*N, d] -> [#points*batch_size*N, d] -> [# observations,]
                    task_info_observations = \
                        task_info.unsqueeze(1).repeat(1, self.state_dim).unsqueeze(0).repeat(len(t_target), 1, 1).view(
                            -1,
                            self.state_dim)[
                            mask_target_extend == 1]
                    # [num_sampling,  #observations] -> [num_sampling,  batch_size]
                    log_p = scatter_sum(log_p_, task_info_observations, dim=1)
                    # [num_sampling,  batch_size] -> [batch_size]
                    log_p = log_p.mean(0)

                    # print('torch.mean(loss_kl) =', torch.mean(loss_kl))
                    loss = - (log_p - self.beta * loss_kl).mean()

                    loss_detail = {'neg_log_p': (-log_p.mean()).item(), 'kl': loss_kl.mean().item()}
                    # loss = -log_p
                else:  # use ML loss
                    # get log probability
                    # [num_sampling,  #observations, d]
                    log_p_ = poster_dist.log_prob(
                        x_self_target_extend[:, mask_target_extend == 1, :])
                    # [batch_size*N, ] -> [#points, batch_size*N, ] -> [#points*batch_size*N,] -> [# observations,]
                    task_info_observations = task_info.unsqueeze(0).repeat(len(t_target), 1).view(-1)[
                        mask_target_extend == 1]
                    # [num_sampling,  #observations, d] -> [num_sampling,  batch_size, d]
                    log_p = scatter_sum(log_p_, task_info_observations, dim=1)
                    # [num_sampling,  batch_size, d] -> [num_sampling,  batch_size]
                    log_p = log_p.sum(-1)
                    # [num_sampling,  batch_size] -> [batch_size]
                    log_p = torch.logsumexp(log_p, 0) - torch.log(torch.tensor([num_sampling])).to(device)

                    # print('torch.mean(loss_kl) =', torch.mean(loss_kl))
                    loss = -log_p.mean()

                    loss_detail = {'neg_log_p': 0., 'kl': 0.}

        return {'pre_dist': poster_dist, 'loss': loss, "loss_detail": loss_detail}


def make_model(x_dim, latent_dim, hidden_dim, gnn_type, num_gnn_blocks, is_determinate, is_uncertainty, use_ML_loss, use_edge_attr=False):
    # make model
    model = GNDP(state_dim=x_dim,
                 latent_dim=latent_dim,
                 hidden_dim=hidden_dim,
                 gnn_type=gnn_type,
                 num_gnn_blocks=num_gnn_blocks,
                 is_determinate=is_determinate,
                 is_uncertainty=is_uncertainty,
                 use_ML_loss=use_ML_loss,
                 use_edge_attr=use_edge_attr)

    print(f"# parameters of np_model: {count_parameters(model):,d}")
    # exit(1)
    return model


##=====================================================================
##
##                  make batch
##
##====================================================================

def make_batch(data, batch_size, is_shuffle=True, bound_t_context=None, is_test=False, is_shuffle_target=True,
               max_x_dim=None):
    if max_x_dim is None:
        max_x_dim = x_dim
    tasks_data = data['tasks']
    if is_shuffle:
        points_data_index_shuffle = []
        for idx in range(len(data['tasks'])):
            points_data = tasks_data[idx]['points']
            index_shuffle = torch.randperm(len(points_data) - 1)
            index_shuffle = torch.cat([torch.tensor([0]).long(),
                                       index_shuffle + torch.tensor([1]).long()], dim=-1)
            points_data_index_shuffle.append(index_shuffle)
        tasks_data_index_shuffle = torch.randperm(len(tasks_data))
    else:
        points_data_index_shuffle = []
        for idx in range(len(data['tasks'])):
            points_data = tasks_data[idx]['points']
            index_shuffle = torch.linspace(0, len(points_data) - 1, len(points_data)).long()
            points_data_index_shuffle.append(index_shuffle)
        tasks_data_index_shuffle = torch.linspace(0, len(tasks_data) - 1, len(tasks_data)).long()

    # print(points_data_index_shuffle)
    # we got tasks_data_shuffle,
    # which is [(adj,
    #            x0,
    #            task_info,
    #            [{"t":, "x_self":, "mask":},...,{}]),
    #           (adj,
    #            x0,
    #            task_info,
    #            [{},...,{}]),
    #          ...]

    start_idx = 0
    while start_idx < len(tasks_data_index_shuffle):
        start_time = time.time()

        batch_adj = []  #
        batch_x0 = []  #
        batch_task_info = []  #

        batch_name = []

        contexts_batch_t = []
        contexts_batch_x_self = []
        contexts_batch_mask = []
        contexts_batch_point_info = []
        contexts_batch_adj = []
        contexts_batch_task_info = []

        targets_batch_t = []
        targets_batch_x_self = []
        targets_batch_mask = []
        targets_batch_point_info = []
        targets_batch_adj = []
        targets_batch_task_info = []

        for task_i_idx in tasks_data_index_shuffle[start_idx:start_idx + batch_size]:

            # make batch for adj, x0 and task_info
            task_i_adj_in_batch = tasks_data[task_i_idx]['adj']
            task_i_x0_in_batch = tasks_data[task_i_idx]['X0']
            task_i_task_info_in_batch = tasks_data[task_i_idx]['task_info']
            task_i_points_in_batch = tasks_data[task_i_idx]['points']

            batch_name.append(tasks_data[task_i_idx]['name'])

            num_nodes_last = 0
            for x0_ in batch_x0:
                num_nodes_last += x0_.shape[0]
            if len(task_i_adj_in_batch) > 2:
                batch_adj.append(
                    np.concatenate([task_i_adj_in_batch[:2, :] + num_nodes_last, task_i_adj_in_batch[2:, :]], axis=0))
            else:
                batch_adj.append(task_i_adj_in_batch + num_nodes_last)
            # padding zeros to x0
            task_i_x0_in_batch = np.concatenate(
                [task_i_x0_in_batch, np.zeros((task_i_x0_in_batch.shape[0], max_x_dim - task_i_x0_in_batch.shape[-1]))],
                axis=-1)
            batch_x0.append(task_i_x0_in_batch)
            batch_task_info.append(task_i_task_info_in_batch + len(batch_task_info))

            # make context points and target points
            num_targets = len(task_i_points_in_batch)  ## number of targets = 50
            ## number of contexts, at least #nodes (i.e., the number of points with t=0)
            # num_contexts = np.random.randint(2, max(num_targets // 5, 4))  # [2, num_targets // 5)
            # num_contexts = np.random.randint(1, int(num_targets * 0.6))  # [2, num_targets // 5)
            if 'RealEpidemicData' in args.dynamics_name:
                num_contexts = np.random.randint(1, max(int(num_targets/2)+1, 2))
            else:
                num_contexts = np.random.randint(2, 31)

            # num_contexts = num_targets - 1
            contexts_index = points_data_index_shuffle[task_i_idx][:num_contexts]
            # print("context num = %s, target num = %s" % (num_contexts, num_targets))
            if is_shuffle_target:
                targets_index = points_data_index_shuffle[task_i_idx][:num_targets]
            else:
                targets_index = torch.linspace(0,
                                               len(task_i_points_in_batch) - 1,
                                               len(task_i_points_in_batch)).long()[:num_targets]
            #print("num_contexts=%s, num_targets=%s" % (num_contexts, num_targets))
            assert num_contexts < num_targets

            contexts_point_mask = []
            for contexts_point_idx in contexts_index:
                contexts_point_i = task_i_points_in_batch[contexts_point_idx]
                if is_test and contexts_point_i['t'] > 0.:
                    num_nodes = contexts_point_i['mask'].shape[0]
                    num_sampling_points_per_time = np.random.randint(1, int(num_nodes / 2) + 1)  ## [1, N/2]
                    sampled_idxs = np.random.choice(a=list(range(num_nodes)), size=num_sampling_points_per_time,
                                                    replace=False)
                    new_mask = np.zeros(num_nodes)
                    new_mask[sampled_idxs] = 1.
                    contexts_point_mask.append(new_mask)
                else:
                    contexts_point_mask.append(contexts_point_i['mask'])
                # contexts_point_mask.append(contexts_point_i['mask'])

            # make contexts
            for idx in range(len(contexts_index)):
                contexts_point_idx = contexts_index[idx]
                contexts_point_i = task_i_points_in_batch[contexts_point_idx]
                if bound_t_context is not None:
                    if contexts_point_i['t'] > bound_t_context:
                        continue

                # print(contexts_point_i['t'])

                contexts_batch_t.append(torch.from_numpy(contexts_point_i['t']).float())
                # contexts_batch_x_self.append(torch.from_numpy(contexts_point_i['x_self']).float())
                # padding zeros to x_self
                contexts_batch_x_self.append(torch.from_numpy(np.concatenate(
                    [contexts_point_i['x_self'],
                     np.zeros((contexts_point_i['x_self'].shape[0], max_x_dim - contexts_point_i['x_self'].shape[-1]))],
                    axis=-1)).float())
                contexts_batch_mask.append(torch.from_numpy(
                    contexts_point_mask[idx].reshape(-1, 1) * get_mask_dim(contexts_point_i['x_self'].shape[-1],
                                                                           max_x_dim).reshape(1, -1)).long())
                contexts_batch_adj.append(torch.from_numpy(contexts_point_i['adj']))
                contexts_batch_point_info.append(torch.from_numpy(contexts_point_i['point_info']).long())
                contexts_batch_task_info.append(torch.tensor([len(batch_task_info) - 1]).long())

            if is_test:
                print("number of context observations = ",
                      sum([contexts_batch_mask[i].sum() for i in range(len(contexts_batch_mask))]).item())

            # make targets
            for targets_point_idx in targets_index:
                targets_point_i = task_i_points_in_batch[targets_point_idx]

                targets_batch_t.append(torch.from_numpy(targets_point_i['t']).float())
                # targets_batch_x_self.append(torch.from_numpy(targets_point_i['x_self']).float())
                # padding zeros to x_self
                targets_batch_x_self.append(torch.from_numpy(np.concatenate(
                    [targets_point_i['x_self'],
                     np.zeros((targets_point_i['x_self'].shape[0], max_x_dim - targets_point_i['x_self'].shape[-1]))],
                    axis=-1)).float())
                # targets_batch_mask.append(torch.from_numpy(targets_point_i['mask']).long())
                targets_batch_mask.append(torch.from_numpy(
                    targets_point_i['mask'].reshape(-1, 1) * get_mask_dim(targets_point_i['x_self'].shape[-1],
                                                                          max_x_dim).reshape(1, -1)).long())
                targets_batch_adj.append(torch.from_numpy(targets_point_i['adj']))
                targets_batch_point_info.append(torch.from_numpy(targets_point_i['point_info']).long())
                targets_batch_task_info.append(torch.tensor([len(batch_task_info) - 1]).long())

        batch_adj = np.concatenate(batch_adj, axis=-1)
        batch_x0 = np.concatenate(batch_x0, axis=0)
        batch_task_info = np.concatenate(batch_task_info, axis=0)

        start_idx = start_idx + batch_size
        batch_data = {"adj": torch.from_numpy(batch_adj),
                      "x0": torch.from_numpy(batch_x0).float(),
                      "task_info": torch.from_numpy(batch_task_info).long(),
                      "contexts": {"t": contexts_batch_t,
                                   "x_self": contexts_batch_x_self,
                                   "mask": contexts_batch_mask,
                                   "adj": contexts_batch_adj,
                                   "point_info": contexts_batch_point_info,
                                   "task_info": contexts_batch_task_info
                                   },
                      "targets": {"t": targets_batch_t,
                                  "x_self": targets_batch_x_self,
                                  "mask": targets_batch_mask,
                                  "adj": targets_batch_adj,
                                  "point_info": targets_batch_point_info,
                                  "task_info": targets_batch_task_info
                                  },
                      'name': batch_name,
                      }
        print("make batch cost = %.2f" % (time.time() - start_time))
        yield batch_data


def move_list_to_device(a_list, device):
    return [item.to(device) for item in a_list]


##=====================================================================
##
##                  train
##
##====================================================================

def fit_net_dynamic(model, dataset, start_epoch, num_epochs):
    start_time = time.time()
    end_epoch = start_epoch + num_epochs
    # train
    # lr = 1e-2
    lr = 5e-3
    # weight_decay = 5e-4
    weight_decay = 0.
    decay_lr = 10.
    batch_size = 8
    # batch_size = 16
    # batch_size = 32
    # batch_size = 32
    optim = torch.optim.Adam(
        model.parameters(),
        lr,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=float((1. / decay_lr) ** (1. / num_epochs)))

    loss_restore = []
    for epoch in range(start_epoch, end_epoch):

        lr = scheduler.get_last_lr()[0]

        print("epoch = " + str(epoch), 'lr = ', str(lr), end='\r\n')

        model.train()

        num_processed_total = 0
        loss_total = 0
        neg_log_p_total = 0
        kl_total = 0
        step = -1
        gen_batch = make_batch(dataset.data, batch_size, is_shuffle=True)
        for batch_data in gen_batch:
            start_forward_time = time.time()

            step += 1
            optim.zero_grad()

            adj = batch_data['adj'].to(device)
            x0 = batch_data['x0'].to(device)
            task_info = batch_data['task_info'].to(device)

            x_context = (move_list_to_device(batch_data['contexts']['t'], device),
                         move_list_to_device(batch_data['contexts']['x_self'], device),
                         move_list_to_device(batch_data['contexts']['mask'], device),
                         move_list_to_device(batch_data['contexts']['point_info'], device),
                         move_list_to_device(batch_data['contexts']['adj'], device),
                         move_list_to_device(batch_data['contexts']['task_info'], device),
                         )
            x_target = (move_list_to_device(batch_data['targets']['t'], device),
                        move_list_to_device(batch_data['targets']['x_self'], device),
                        move_list_to_device(batch_data['targets']['mask'], device),
                        move_list_to_device(batch_data['targets']['point_info'], device),
                        move_list_to_device(batch_data['targets']['adj'], device),
                        move_list_to_device(batch_data['targets']['task_info'], device),
                        )

            if model.use_ML_loss:
                res = model(adj, x0, task_info, x_context, x_target, num_sampling=16, use_NPLoss=False)
            else:
                if model.is_uncertainty:
                    model.update_beta(epoch + 1, start_epoch + num_epochs)
                res = model(adj, x0, task_info, x_context, x_target, num_sampling=1, use_NPLoss=True)

            # if epoch < num_epochs // 2:
            #     res = model(adj, x0, task_info, x_context, x_target, num_sampling=16, use_NPLoss=False)
            # else:
            #     res = model(adj, x0, task_info, x_context, x_target, num_sampling=1, use_NPLoss=True)

            print("forward cost = %.2f" % (time.time() - start_forward_time))

            # {'pred_dist': poster_dist,
            #  'pred_dist_self': torch.distributions.Normal(out_F_mean, torch.sqrt(out_F_logvar.exp())),
            #  'pred_dist_interact': torch.distributions.Normal(out_G_mean, torch.sqrt(out_G_logvar.exp())),
            #  'loss': loss}

            loss = res['loss']

            # comput_graph = make_dot(loss)
            # comput_graph.render(filename="comput_graph", view=False)
            # exit(1)

            start_backward_time = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()

            print("backward cost = %.2f" % (time.time() - start_backward_time))

            num_processed = max(task_info).item() + 1
            num_processed_total += num_processed

            loss_total += loss.item() * num_processed
            neg_log_p_total += res['loss_detail']['neg_log_p'] * num_processed
            kl_total += res['loss_detail']['kl'] * num_processed

            print('epoch %s, step %s (has %s), loss = %s (neg_log_p %s, kl %s), '
                  'total loss = %s (neg_log_p %s, kl %s), '
                  'cost_time = %.2f' \
                  % (epoch + 1, step, max(task_info).item() + 1,
                     loss.item(), res['loss_detail']['neg_log_p'], res['loss_detail']['kl'],
                     loss_total / num_processed_total, neg_log_p_total / num_processed_total,
                     kl_total / num_processed_total,
                     time.time() - start_time), end='\r\n')

            # if step == 2:
            #     break
        scheduler.step()

        loss_restore.append(
            (loss_total / num_processed_total, neg_log_p_total / num_processed_total, kl_total / num_processed_total))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放显存

        if (epoch + 1) % 10 == 0 or (epoch + 1) == end_epoch:
            if args.constraint_state:
                torch.save(model.state_dict(),
                           "./saved_models/saved_model_%s_MLloss%s_deter%s_uncer%s_gnn%s_%s_%s_x%s_numgraph%s_timestep%s_epoch%s_%s.pkl"
                           % (
                               model.name,
                               model.use_ML_loss, model.is_determinate, model.is_uncertainty, model.gnn_type,
                               dynamics_name,
                               topo_type,
                               x_dim,
                               num_graphs,
                               time_steps,
                               epoch + 1,
                               args.constraint_state))
            else:
                torch.save(model.state_dict(),
                           "./saved_models/saved_model_%s_MLloss%s_deter%s_uncer%s_gnn%s_%s_%s_x%s_numgraph%s_timestep%s_epoch%s.pkl"
                           % (
                               model.name,
                               model.use_ML_loss, model.is_determinate, model.is_uncertainty, model.gnn_type,
                               dynamics_name,
                               topo_type,
                               x_dim,
                               num_graphs,
                               time_steps,
                               epoch + 1))
            if args.tune:
                res_list = []
                for _ in range(5):
                    res = test_net_dynamic(model, dataset, path=None, add_str="_%s" % epoch)
                    res_list.append(res)
                np.savetxt('./tune_hyperparams/temp_black_func_eval.txt', np.array([np.mean(res_list)]))
            # else:
            # test_net_dynamic(model, dataset, path=None, add_str="_%s" % epoch)

        for jj in range(len(loss_restore[0])):
            plt.plot([ii[jj] for ii in loss_restore])

        title_str = 'training statistics ' + ' %s ' % model.name + ' MLloss%s_deter%s_uncer%s' % (
            model.use_ML_loss, model.is_determinate, model.is_uncertainty) + str(dynamics_name) + ' ' + str(
            topo_type) + ' on #graph' + str(
            num_graphs) + ' #timestep' + str(time_steps)
        plt.savefig('./results/' + title_str)
        plt.close()
        # plt.show()


def get_score_label(task_names):
    labels = []
    for name in task_names:
        if 'heat' in name:
            label = 0
        elif 'mutu' in name:
            label = 1
        elif 'gene' in name:
            label = 2
        elif 'opinion' in name:
            label = 3
        elif 'SI_' in name:
            label = 4
        elif 'SIS_' in name:
            label = 5
        elif 'SIR_Individual' in name:
            label = 6
        elif 'SEIS_' in name:
            label = 7
        elif 'SEIR_' in name:
            label = 8
        elif 'Coupled' in name:
            label = 9
        elif 'SIR_meta' in name:
            label = 10
        elif 'RealEpidemicData' in name:
            label = 11
        else:
            print('error [%s]' % name)
        labels.append(label)
    return labels

def get_task_features(task_names):
    features = []
    for name in task_names:
        if 'heat' in name:
            dim = 1
            max_val = 25.
            avg_val = 12.5
            min_val = 0.
        elif 'mutu' in name:
            dim = 1
            max_val = 25.
            avg_val = 12.5
            min_val = 0.
        elif 'gene' in name:
            dim = 1
            max_val = 25.
            avg_val = 12.5
            min_val = 0.
        elif 'opinion' in name:
            dim = 2
            max_val = 30.
            avg_val = 0.
            min_val = -30.
        elif 'SI_' in name:
            dim = 2
            max_val = 1.
            avg_val = 0.5
            min_val = 0.
        elif 'SIS_' in name:
            dim = 2
            max_val = 1.
            avg_val = 0.5
            min_val = 0.
        elif 'SIR_Individual' in name:
            dim = 3
            max_val = 1.
            avg_val = 0.5
            min_val = 0.
        elif 'SEIS_' in name:
            dim = 3
            max_val = 1.
            avg_val = 0.5
            min_val = 0.
        elif 'SEIR_' in name:
            dim = 4
            max_val = 1.
            avg_val = 0.5
            min_val = 0.
        elif 'Coupled' in name:
            dim = 4
            max_val = 1.
            avg_val = 0.5
            min_val = 0.
        elif 'SIR_meta' in name:
            dim = 3
            max_val = 1.
            avg_val = 0.5
            min_val = 0.
        elif 'RealEpidemicData' in name:
            dim = 1
            max_val = 1000.
            avg_val = (1000.+1.)/2.
            min_val = 1.
        else:
            print('error [%s]' % name)
        features.append([dim, max_val, avg_val, min_val])
    return features


def fit_score_net(score_net_path, paths, dataset):
    start_time = time.time()

    print('loading pre models set ...')
    pre_models = []
    for path_i in paths:
        if 'SIR_meta_pop' in path_i or 'real_data' in path_i or 'RealEpidemicData' in path_i:
            use_edge_attr = True
        else:
            use_edge_attr = False
        
        if 'RealEpidemicData' in path_i:
            model_i = make_model(x_dim, 50, 50, gnn_type, num_gnn_blocks, is_determinate, is_uncertainty,
                                 use_ML_loss, use_edge_attr).to(device)
        else:
            model_i = make_model(x_dim, latent_dim, hidden_dim, gnn_type, num_gnn_blocks, is_determinate, is_uncertainty,
                             use_ML_loss, use_edge_attr).to(device)

        # model_i.load_state_dict(torch.load(path_i, map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'}))
        model_i.load_state_dict(torch.load(path_i))

        pre_models.append(model_i)

    print('making train set ...')
    gen_batch = make_batch(dataset.data, 10, is_shuffle=False)
    # make train datas
    Rs = []
    ys = []
    count_ = 0
    for batch_data in gen_batch:
        adj = batch_data['adj'].to(device)
        x0 = batch_data['x0'].to(device)
        task_info = batch_data['task_info'].to(device)

        task_names = batch_data['name']

        labels = get_score_label(task_names)
        task_features = get_task_features(task_names)

        x_context = (move_list_to_device(batch_data['contexts']['t'], device),
                     move_list_to_device(batch_data['contexts']['x_self'], device),
                     move_list_to_device(batch_data['contexts']['mask'], device),
                     move_list_to_device(batch_data['contexts']['point_info'], device),
                     move_list_to_device(batch_data['contexts']['adj'], device),
                     move_list_to_device(batch_data['contexts']['task_info'], device),
                     )
        Rs_ = []
        for idx in range(len(pre_models)):
            res = pre_models[idx](adj, x0, task_info, x_context,
                                  (x_context[0], None, x_context[2], x_context[3], x_context[4], x_context[5]),
                                  num_sampling=1)

            R_i = pre_models[idx].Z_determinate.detach()  # [batch_size, d]

            
            # [batch_size, 1, d] + task features [batch_size, 1, d] -> [batch_size, 1, d+d]
            Rs_.append(torch.cat([R_i.unsqueeze(1).cpu(),torch.tensor(task_features).float().unsqueeze(1).cpu()],dim=-1))
            

        # [batch_size, len(pre_models), d]
        Rs_ = torch.cat(Rs_, dim=1)
        
        # [batch_size, ]
        ys_ = torch.tensor(labels).long()

        count_ += Rs_.size(0)
        print('%s ...' % (count_), Rs_.size())

        # print(Rs_.size(), ys_)
        # exit(1)

        Rs.append(Rs_)
        ys.append(ys_)

    # [batch_size, len(pre_models), d]
    Rs = torch.cat(Rs, dim=0)
    # [batch_size, ]
    ys = torch.cat(ys, dim=0).long()
    print(Rs.size(), ys.size())

    print('make model')
    score_net = ScoreNet(hidden_dim+task_features_dim, num_heads=4)
    if score_net_path is not None:
        score_net.load_state_dict(torch.load(score_net_path))

    score_net = score_net.to(device)

    num_epochs = 5000
    b_size = 200
    lr = 1e-3
    # weight_decay = 1e-4
    weight_decay = 0.
    optim = torch.optim.Adam(
        score_net.parameters(),
        lr,
        weight_decay=weight_decay)

    func_loss = torch.nn.CrossEntropyLoss()
    min_loss_val = 100000000

    # shuffle
    index_shuffled = torch.randperm(len(ys))
    data_R_train = Rs[index_shuffled[:int(len(ys) * 0.8)]]
    data_ys_train = ys[index_shuffled[:int(len(ys) * 0.8)]]
    data_R_val = Rs[index_shuffled[int(len(ys) * 0.8):]]
    data_ys_val = ys[index_shuffled[int(len(ys) * 0.8):]]

    # print(data_R_train.size(), data_ys_train.size())

    for epoch in range(num_epochs):
        score_net.train()
        # shuffle
        index_shuffled = torch.randperm(len(data_ys_train))
        for step in range(data_R_train.size(0) // b_size):
            optim.zero_grad()

            selected_idx = index_shuffled[step * b_size:step * b_size + b_size]
            data_R_batch = data_R_train[selected_idx].to(device)
            data_ys_batch = data_ys_train[selected_idx].to(device)

            pre_ = score_net(data_R_batch)
            # print(pre_.size(), data_ys_batch.size())
            loss = func_loss(pre_, data_ys_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.)
            optim.step()

        # val
        score_net.eval()
        pre_val_ = score_net(data_R_val.to(device))
        loss_val = func_loss(pre_val_, data_ys_val.to(device))

        if epoch % 1 == 0:
            print('Train 2rd phase: epoch %s, loss = %s , loss_val = %s, cost_time = %.2f' \
                  % (epoch + 1, loss.item(), loss_val.item(), time.time() - start_time), end='\r\n')

        if loss_val.item() < min_loss_val:
            print('saving model ... at epoch %s' % (epoch + 1))
            torch.save(score_net.state_dict(),
                       "./saved_models/saved_model_score_net_num_scenario%s.pkl" % (len(paths)))
            min_loss_val = loss_val.item()

    # val
    score_net.load_state_dict(torch.load("./saved_models/saved_model_score_net_num_scenario%s.pkl" % (len(paths))))

    score_net.eval()
    pre_on_trainset = score_net(Rs.to(device))
    # hidden_rep = score_net.hidden_rep

    np.savetxt('score_net_pre_on_trainset_%s.txt' % len(paths),
               torch.softmax(pre_on_trainset, dim=-1).detach().cpu().numpy())
    np.savetxt('score_net_label_on_trainset_%s.txt' % len(paths), ys.detach().cpu().numpy())

    roc_auc = roc_auc_score(ys, torch.softmax(pre_on_trainset, dim=-1).detach().cpu(), multi_class='ovr')
    print('roc_auc: ', roc_auc, )
    roc_auc = roc_auc_score(ys, torch.softmax(pre_on_trainset, dim=-1).detach().cpu(), multi_class='ovr', average=None)
    print('roc_auc: ', roc_auc, )

    return score_net


def test_score_net(score_net, pre_models, adj, x0, task_info, x_context, task_names):

    task_features = get_task_features(task_names)
        
    Rs_ = []
    for idx in range(len(pre_models)):
        res = pre_models[idx](adj, x0, task_info, x_context,
                              (x_context[0], None, x_context[2], x_context[3], x_context[4], x_context[5]),
                              num_sampling=1)

        R_i = pre_models[idx].Z_determinate.detach()  # [batch_size, d]
        
        # [batch_size, 1, d] + task features [batch_size, 1, d] -> [batch_size, 1, d+d]
        Rs_.append(torch.cat([R_i.unsqueeze(1),torch.tensor(task_features).float().unsqueeze(1)],dim=-1))


    # [batch_size, len(pre_models), d]
    Rs_ = torch.cat(Rs_, dim=1)

    score_net.eval()
    pre_score = score_net(Rs_.to(device))

    return torch.softmax(pre_score, dim=-1).detach()  # [batch_size, len(pre_models)]


##=====================================================================
##
##                  test
##
##====================================================================

def fine_tune(model, batch_data):
    start_time = time.time()
    lam = 0.1
    num_epochs_fine_tune = 20
    lr = 1e-4
    weight_decay = 1e-5
    optim = torch.optim.Adam(
        model.parameters(),
        lr,
        weight_decay=weight_decay)

    model.train()
    l1_loss = torch.nn.L1Loss(reduction='sum')
    min_loss_val = 100000000
    for epoch in range(num_epochs_fine_tune):

        optim.zero_grad()

        adj = batch_data['adj'].to(device)
        x0 = batch_data['x0'].to(device)
        task_info = batch_data['task_info'].to(device)

        x_context = (move_list_to_device(batch_data['contexts']['t'], device),
                     move_list_to_device(batch_data['contexts']['x_self'], device),
                     move_list_to_device(batch_data['contexts']['mask'], device),
                     move_list_to_device(batch_data['contexts']['point_info'], device),
                     move_list_to_device(batch_data['contexts']['adj'], device),
                     move_list_to_device(batch_data['contexts']['task_info'], device),
                     )

        num_sampling = 20

        res = model(adj, x0, task_info, x_context,
                    (x_context[0], None, x_context[2], x_context[3], x_context[4], x_context[5]),
                    num_sampling=num_sampling)

        ## [num_sampling,  #observations, ]
        mean_ys = res['pre_dist'].loc
        std_ys = res['pre_dist'].scale

        # [#observations, ]
        mean_ys_ = torch.mean(mean_ys, dim=0)
        std_ys_ = torch.sqrt(torch.mean(mean_ys ** 2 + std_ys ** 2, dim=0) - mean_ys_ ** 2)

        pre_dist_ = torch.distributions.Normal(mean_ys_, std_ys_)

        # [# t steps, #nodes]
        mask = torch.cat([item.view(1, -1, x_dim) for item in x_context[2]], dim=0).long()
        # [# t steps, #nodes, dim]
        gt_x_self = torch.cat([item.view(1, -1, x_dim) for item in x_context[1]], dim=0)
        # MAE
        split_radio = 0.8
        loss = l1_loss(mean_ys_[:int(mean_ys_.size(0) * split_radio)],
                       gt_x_self[mask == 1][:int(mean_ys_.size(0) * split_radio)]) + lam * std_ys_[:int(
            mean_ys_.size(0) * split_radio)].sum()

        loss_val = l1_loss(mean_ys_[int(mean_ys_.size(0) * split_radio):],
                           gt_x_self[mask == 1][int(mean_ys_.size(0) * split_radio):])

        # loss = pre_dist_.log_prob(gt_x_self[mask==1]).sum()

        # print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()

        print('fine-tune: epoch %s, loss = %s , loss_val = %s, cost_time = %.2f' \
              % (epoch + 1, loss.item(), loss_val.item(), time.time() - start_time), end='\r\n')

        if loss_val.item() < min_loss_val:
            torch.save(model.state_dict(),
                       "./saved_models/saved_model_fine_tune_temp.pkl")
            min_loss_val = loss_val.item()

    model.load_state_dict(torch.load("./saved_models/saved_model_fine_tune_temp.pkl"))

    return model


def test_net_dynamic(model, path=None, batch_data=None, saved_results=None, add_str="", is_plot=True):
    if path is not None:
        model.load_state_dict(torch.load(path))

    test_time_steps = time_steps

    if batch_data is None:
        # test_time_steps = 200
        test_data = dataset.query(seed=seed, num_graph=1, max_time_points=test_time_steps,
                                  query_all_node=True, query_all_t=True,
                                  N=test_N
                                  )

        # save test data
        import pickle
        fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s.pickle' % (
            dataset.dynamics_name, dataset.topo_type, dataset.x_dim, 1, time_steps, seed,
            test_data['tasks'][0]['X0'].shape[0])
        f = open(fname, 'wb')
        pickle.dump(test_data, f)
        f.close()

        gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context, is_test=True,
                               is_shuffle_target=False)

        batch_data = next(gen_batch)

    if args.is_fine_tune and len(batch_data['contexts']['t']) > 1:
        model = fine_tune(model, batch_data)
    model.eval()

    if True:

        adj = batch_data['adj'].to(device)
        x0 = batch_data['x0'].to(device)
        task_info = batch_data['task_info'].to(device)

        x_context = (move_list_to_device(batch_data['contexts']['t'], device),
                     move_list_to_device(batch_data['contexts']['x_self'], device),
                     move_list_to_device(batch_data['contexts']['mask'], device),
                     move_list_to_device(batch_data['contexts']['point_info'], device),
                     move_list_to_device(batch_data['contexts']['adj'], device),
                     move_list_to_device(batch_data['contexts']['task_info'], device),
                     )
        x_target = (move_list_to_device(batch_data['targets']['t'], device),
                    move_list_to_device(batch_data['targets']['x_self'], device),
                    move_list_to_device(batch_data['targets']['mask'], device),
                    move_list_to_device(batch_data['targets']['point_info'], device),
                    move_list_to_device(batch_data['targets']['adj'], device),
                    move_list_to_device(batch_data['targets']['task_info'], device),
                    )

        num_sampling = 20
        # res = model(adj, x0, task_info, x_context,
        #             (x_target[0], x_target[1], None, x_target[3], x_target[4], x_target[5], x_target[6]),
        #             num_sampling=num_sampling)

        res = model(adj, x0, task_info, x_context,
                    (x_target[0], None, x_target[2], x_target[3], x_target[4], x_target[5]),
                    num_sampling=num_sampling)

        if not model.is_uncertainty:
            num_sampling = 1
        ## [num_sampling,  #observations, d] -> [num_sampling,  #steps, #nodes , d]
        ## -> [num_sampling,  #nodes, 1, #steps, d]
        mean_ys = res['pre_dist'].loc.detach().view(num_sampling, test_time_steps, -1, x_dim_).transpose(1,
                                                                                                         2).contiguous().view(
            num_sampling, -1, 1, test_time_steps, x_dim_)
        std_ys = res['pre_dist'].scale.detach().view(num_sampling, test_time_steps, -1, x_dim_).transpose(1,
                                                                                                          2).contiguous().view(
            num_sampling, -1, 1, test_time_steps, x_dim_)
        # mean_ys = torch.zeros_like(mean_ys)
        # std_ys = torch.zeros_like(std_ys)
        x_time = torch.cat(x_target[0], dim=0).view(1, test_time_steps).repeat(x0.size(0), 1).view(-1, 1,
                                                                                                   test_time_steps, 1)

        ground_truth_ys = torch.cat(x_target[1], dim=0)[torch.cat(x_target[2], dim=0) == 1].view(test_time_steps, -1,
                                                                                                 x_dim_).transpose(0,
                                                                                                                   1).reshape(
            -1, 1,
            test_time_steps,
            x_dim_)

        # plt.ion()
        # fig = plt.figure()

        t_start = 0.
        t_end = 1.
        t_inc = (t_end-t_start) / (time_steps)

        if is_plot:
            display_3D(ground_truth_ys.cpu().view(-1, test_time_steps, x_dim_),
                       mean_ys.cpu().view(num_sampling, -1, test_time_steps, x_dim_),
                       std_ys.cpu().view(num_sampling, -1, test_time_steps, x_dim_),
                       adj.cpu(),
                       t_start, t_end, t_inc,
                       x_context,
                       available_time_list=None, animation=False,
                       name="test_%s_MLloss%s_deter%s_uncer%s_%s_%s_x%s_numgraph%s_timestep%s_%s.pkl" %
                            (model.name, model.use_ML_loss, model.is_determinate, model.is_uncertainty, dynamics_name,
                             topo_type, x_dim_,
                             num_graphs,
                             time_steps, add_str + '_num_nodes%s' % (test_N)),
                       pos=None, max_value=None, min_value=None)

        GT_sum = torch.sum(ground_truth_ys.cpu().reshape(-1, test_time_steps), dim=0).view(-1)
        predictions_sum_diff = []
        prediction_states = mean_ys.view(num_sampling, -1, test_time_steps, x_dim_)
        for one_step in range(test_time_steps):
            one_state = prediction_states[:, :, one_step, :]
            predictions_sum_diff.append(
                torch.sum(
                    one_state.view(num_sampling, -1, x_dim_),
                    dim=1
                ).view(num_sampling, 1, x_dim_).detach()
            )
        predictions_sum_diff = torch.cat(predictions_sum_diff, dim=1)
        if is_plot:
            display_diff(torch.cat(x_target[0], dim=0).view(-1).cpu(), GT_sum.cpu(), predictions_sum_diff.cpu(),
                         name="test_%s_CONSTANT_MLloss%s_deter%s_uncer%s_%s_%s_x%s_numgraph%s_timestep%s_%s.pkl" %
                              (model.name, model.use_ML_loss, model.is_determinate, model.is_uncertainty, dynamics_name,
                               topo_type,
                               x_dim_, num_graphs,
                               time_steps, add_str + '_num_nodes%s' % (test_N)))
        
        print('batch_data[\'contexts\'][\'adj\'][-1]=',batch_data['contexts']['adj'][-1])

        saved_results['observations'].append({'t': batch_data['contexts']['t'],
                                              'x_self': batch_data['contexts']['x_self'],
                                              'mask': batch_data['contexts']['mask'],
                                              'adj': batch_data['contexts']['adj']})

        saved_results['groundtruth'].append(ground_truth_ys.cpu().view(-1, test_time_steps, x_dim_))
        saved_results['predictions'].append({'mean': mean_ys.cpu().view(num_sampling, -1, test_time_steps, x_dim_),
                                             'std': std_ys.cpu().view(num_sampling, -1, test_time_steps, x_dim_)})
        saved_results['groundtruth_sum'].append(GT_sum.cpu())
        saved_results['predictions_sum'].append(predictions_sum_diff.cpu())

        l1_store = []
        l2_store = []
        for ii in range(mean_ys.size(0)):
            if is_plot:
                fig, ax = plt.subplots(1, 1, figsize=(18, 6))
                plot_functions(ax,
                               x_time.view(-1, 1, test_time_steps, 1).cpu(),
                               ground_truth_ys.cpu(),
                               x_time.view(-1, 1, test_time_steps, 1).cpu(),
                               ground_truth_ys.cpu(),
                               mean_ys[ii].cpu(),
                               std_ys[ii].cpu())

            l1_loss_value = torch.sum(
                torch.abs(mean_ys[ii].cpu() - ground_truth_ys.cpu()).sum(-1) / torch.abs(ground_truth_ys.cpu()).sum(
                    -1).reshape(-1).mean()
            ).item() / torch.cat(x_target[1], dim=0).cpu().size(0)

            l2_loss_value = torch.sum(
                ((mean_ys[ii].cpu() - ground_truth_ys.cpu()) ** 2).sum(-1)
            ).item() / torch.cat(x_target[1], dim=0).cpu().size(0)

            l1_store.append(l1_loss_value)
            l2_store.append(l2_loss_value)

            print("L1 loss = ", l1_loss_value)
            print("L2 loss (MSE) =", l2_loss_value)

        print('mean of l1 loss = ', np.mean(l1_store), 'std of l1 loss = ', np.std(l1_store))
        print('mean of l2 loss = ', np.mean(l2_store), 'std of l2 loss = ', np.std(l2_store))

        saved_results['nl1_error'].append([np.mean(l1_store), np.std(l1_store)])
        saved_results['l2_error'].append([np.mean(l2_store), np.std(l2_store)])

        if is_plot:
            plt.scatter(torch.cat(x_context[0], dim=0).view(-1, 1).repeat(1, x0.size(0)).view(-1)[
                            torch.cat(x_context[2], dim=0) == 1].cpu(),
                        torch.cat(x_context[1], dim=0).view(-1).detach()[torch.cat(x_context[2], dim=0) == 1].cpu(),
                        color='r')

            title_str = 'test' + ' %s ' % model.name + str(dynamics_name) + ' ' + str(
                topo_type) + ' training on #graph' + str(
                num_graphs) + ' #timestep' + str(time_steps) + add_str + \
                        '_num_nodes' + str(test_N) + '_' + \
                        'L1=%.4f' % (np.mean(l1_store)) + '+-%.4f' % (np.std(l1_store)) + '_' + \
                        'L2=%.4f' % (np.mean(l2_store)) + '+-%.4f' % (np.std(l2_store)) + '.png'
            plt.suptitle(title_str)

            # plt.show()

            plt.savefig('./results/' + title_str)
            # plt.show()
            plt.close()

        return np.mean(l1_store)


def test_net_dynamic_with_2nd_phase(paths=None, score_net_path=None, batch_data=None, saved_results=None, add_str="",
                                    is_plot=True):
    print('loading pretrained models ...')
    pre_models = []
    for path_i in paths:

        if 'SIR_meta_pop' in path_i or 'real_data' in path_i or 'RealEpidemicData' in path_i:
            use_edge_attr = True
        else:
            use_edge_attr = False
        if 'RealEpidemicData' in path_i:
            model_i = make_model(x_dim, 50, 50, gnn_type, num_gnn_blocks, is_determinate, is_uncertainty,
                             use_ML_loss, use_edge_attr).to(device)
        else:
            model_i = make_model(x_dim, latent_dim, hidden_dim, gnn_type, num_gnn_blocks, is_determinate, is_uncertainty,
                             use_ML_loss, use_edge_attr).to(device)

        # model_i.load_state_dict(torch.load(path_i, map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'}))
        model_i.load_state_dict(torch.load(path_i))

        model_i.eval()

        pre_models.append(model_i)

    print('loading scorer model ...')
    score_net = ScoreNet(hidden_dim+task_features_dim, num_heads=4).to(device)
    score_net.load_state_dict(torch.load(score_net_path))
    score_net.eval()

    test_time_steps = time_steps

    if batch_data is None:
        # test_time_steps = 200
        test_data = dataset.query(seed=seed, num_graph=1, max_time_points=test_time_steps,
                                  query_all_node=True, query_all_t=True,
                                  N=test_N
                                  )

        # save test data
        import pickle
        fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s.pickle' % (
            dataset.dynamics_name, dataset.topo_type, dataset.x_dim, 1, time_steps, seed,
            test_data['tasks'][0]['X0'].shape[0])
        f = open(fname, 'wb')
        pickle.dump(test_data, f)
        f.close()

        gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context, is_test=True,
                               is_shuffle_target=False)

        batch_data = next(gen_batch)
    # data
    adj = batch_data['adj'].to(device)
    x0 = batch_data['x0'].to(device)
    task_info = batch_data['task_info'].to(device)

    x_context = (move_list_to_device(batch_data['contexts']['t'], device),
                 move_list_to_device(batch_data['contexts']['x_self'], device),
                 move_list_to_device(batch_data['contexts']['mask'], device),
                 move_list_to_device(batch_data['contexts']['point_info'], device),
                 move_list_to_device(batch_data['contexts']['adj'], device),
                 move_list_to_device(batch_data['contexts']['task_info'], device),
                 )
    x_target = (move_list_to_device(batch_data['targets']['t'], device),
                move_list_to_device(batch_data['targets']['x_self'], device),
                move_list_to_device(batch_data['targets']['mask'], device),
                move_list_to_device(batch_data['targets']['point_info'], device),
                move_list_to_device(batch_data['targets']['adj'], device),
                move_list_to_device(batch_data['targets']['task_info'], device),
                )
                
    task_names = batch_data['name']

    ## [batch_size=1, len(pre_models)]
    weights = test_score_net(score_net, pre_models, adj, x0, task_info, x_context, task_names)

    saved_results['weights'].append(weights)

    print('weights=', weights.view(-1))

    mean_ys_ = []
    std_ys_ = []
    for model_i in pre_models:

        num_sampling = 20

        res = model_i(adj, x0, task_info, x_context,
                      (x_target[0], None, x_target[2], x_target[3], x_target[4], x_target[5]),
                      num_sampling=num_sampling)

        if not model_i.is_uncertainty:
            num_sampling = 1
        ## [num_sampling,  #observations, d] -> [num_sampling,  #steps, #nodes , d]
        ## -> [num_sampling,  #nodes, 1, #steps, d] -> [1, num_sampling,  #nodes, 1, #steps, d]
        mean_ys_i = res['pre_dist'].loc.detach().view(num_sampling, test_time_steps, -1, x_dim_).transpose(1,
                                                                                                           2).contiguous().view(
            1, num_sampling, -1, 1, test_time_steps, x_dim_)
        std_ys_i = res['pre_dist'].scale.detach().view(num_sampling, test_time_steps, -1, x_dim_).transpose(1,
                                                                                                            2).contiguous().view(
            1, num_sampling, -1, 1, test_time_steps, x_dim_)

        mean_ys_.append(mean_ys_i)
        std_ys_.append(std_ys_i)

    mean_ys_ = torch.cat(mean_ys_, dim=0)  ## [len(pre_models), num_sampling,  #nodes, 1, #steps, d]
    std_ys_ = torch.cat(std_ys_, dim=0)  ## [len(pre_models), num_sampling,  #nodes, 1, #steps, d]

    # weights.view(-1,1,1,1,1,1) ## [len(pre_models), 1,  1, 1, 1, 1]
    mean_ys = torch.sum(mean_ys_ * weights.view(-1, 1, 1, 1, 1, 1), dim=0)  ## [num_sampling,  #nodes, 1, #steps, d]
    std_ys = torch.sqrt(torch.sum((mean_ys_ ** 2 + std_ys_ ** 2) * weights.view(-1, 1, 1, 1, 1, 1),
                                  dim=0) - mean_ys ** 2)  ## [num_sampling,  #nodes, 1, #steps, d]

    x_time = torch.cat(x_target[0], dim=0).view(1, test_time_steps).repeat(x0.size(0), 1).view(-1, 1,
                                                                                               test_time_steps, 1)

    ground_truth_ys = torch.cat(x_target[1], dim=0)[torch.cat(x_target[2], dim=0) == 1].view(test_time_steps, -1,
                                                                                             x_dim_).transpose(0,
                                                                                                               1).reshape(
        -1, 1,
        test_time_steps,
        x_dim_)

    t_start = 0.
    t_end = 1.
    t_inc = (t_end - t_start) / time_steps

    saved_results['observations'].append({'t': batch_data['contexts']['t'],
                                          'x_self': batch_data['contexts']['x_self'],
                                          'mask': batch_data['contexts']['mask'],
                                          'adj': batch_data['contexts']['adj']})

    saved_results['groundtruth'].append(ground_truth_ys.cpu().view(-1, test_time_steps, x_dim_))
    saved_results['predictions'].append({'mean': mean_ys.cpu().view(num_sampling, -1, test_time_steps, x_dim_),
                                         'std': std_ys.cpu().view(num_sampling, -1, test_time_steps, x_dim_)})

    l1_store = []
    l2_store = []
    for ii in range(mean_ys.size(0)):
        l1_loss_value = torch.sum(
            torch.abs(mean_ys[ii].cpu() - ground_truth_ys.cpu()).sum(-1) / torch.abs(ground_truth_ys.cpu()).sum(
                -1).reshape(-1).mean()
        ).item() / torch.cat(x_target[1], dim=0).cpu().size(0)

        l2_loss_value = torch.sum(
            ((mean_ys[ii].cpu() - ground_truth_ys.cpu()) ** 2).sum(-1)
        ).item() / torch.cat(x_target[1], dim=0).cpu().size(0)

        l1_store.append(l1_loss_value)
        l2_store.append(l2_loss_value)

        print("L1 loss = ", l1_loss_value)
        print("L2 loss (MSE) =", l2_loss_value)

    print('mean of l1 loss = ', np.mean(l1_store), 'std of l1 loss = ', np.std(l1_store))
    print('mean of l2 loss = ', np.mean(l2_store), 'std of l2 loss = ', np.std(l2_store))

    saved_results['nl1_error'].append([np.mean(l1_store), np.std(l1_store)])
    saved_results['l2_error'].append([np.mean(l2_store), np.std(l2_store)])

    return np.mean(l1_store)


def set_rand_seed(rseed):
    torch.manual_seed(rseed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(rseed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rseed)  # 为所有GPU设置随机种子
    random.seed(rseed)
    np.random.seed(rseed)


if __name__ == '__main__':

    seed = args.seed

    num_graphs = args.num_graphs
    time_steps = args.time_steps
    dynamics_name = args.dynamics_name
    topo_type = args.topo_type

    x_dim = args.x_dim
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    task_features_dim = args.task_features_dim
    gnn_type = args.gnn_type
    num_gnn_blocks = args.num_gnn_blocks

    use_ML_loss = args.use_ML_loss
    is_determinate = args.is_determinate
    is_uncertainty = args.is_uncertainty

    start_epoch = args.start_epoch
    num_epochs = args.num_epochs

    bound_t_context = args.bound_t_context

    print(args)

    if 'SIR_meta_pop' in dynamics_name or 'real_data' in dynamics_name or 'RealEpidemicData' in dynamics_name or 'sim_epidemic' in dynamics_name:
        use_edge_attr = True
    else:
        use_edge_attr = False
    model = make_model(x_dim, latent_dim, hidden_dim, gnn_type, num_gnn_blocks, is_determinate, is_uncertainty,
                       use_ML_loss, use_edge_attr).to(device)

    set_rand_seed(seed)

    # train
    # num_epochs = 70
    # num_epochs = 50
    # num_epochs = 20
    # num_epochs = 30
    # num_epochs = 50
    if args.train:
        # make train data
        dataset = dynamics_dataset(dynamics_name, topo_type,
                                   num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim)

        if start_epoch > 0:
            path = "./saved_models/saved_model_%s_MLloss%s_deter%s_uncer%s_gnn%s_%s_%s_x%s_numgraph%s_timestep%s_epoch%s.pkl" % \
                   (model.name, model.use_ML_loss, model.is_determinate,
                    model.is_uncertainty, model.gnn_type,
                    dynamics_name,
                    topo_type, x_dim, num_graphs,
                    time_steps, start_epoch)

            model.load_state_dict(torch.load(path))

        fit_net_dynamic(model, dataset, start_epoch, num_epochs)

    if args.train_2nd_phase:
        # make train data
        dataset = dynamics_dataset(dynamics_name, topo_type,
                                   num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim)

        # train_2nd_phase
        paths = [
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_heat_diffusion_dynamics_all_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_mutualistic_interaction_dynamics_all_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_gene_regulatory_dynamics_all_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_opinion_dynamics_Baumann2021_2topic_small_world_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SI_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SIS_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SIR_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SEIS_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SEIR_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_Coupled_Epidemic_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
            "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SIR_meta_pop_dynamics_directed_full_connected_x4_numgraph1000_timestep100_epoch30.pkl",
            #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_123_power_law_x4_numgraph1000_timestep100_epoch1000.pkl",  
            #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_124_power_law_x4_numgraph1000_timestep100_epoch1000.pkl", 
            #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_134_power_law_x4_numgraph1000_timestep100_epoch1000.pkl", 
            #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_234_power_law_x4_numgraph1000_timestep100_epoch1000.pkl", 
            # "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_mix_power_law_x4_numgraph1000_timestep100_epoch60.pkl",
             
        ]
        fit_score_net(None, paths, dataset)

    # test
    if args.test:

        dynamics_topo_names = None

        if topo_type == 'all':
            topo_types = ['grid', 'power_law', 'random', 'small_world', 'community']
            # topo_types = ['power_law', 'random', 'small_world', 'community']
        else:
            if args.test_topo_type == 'none':
                topo_types = [topo_type]
            else:
                topo_types = [args.test_topo_type]

        if dynamics_name == 'all':
            # dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics', 'gene_regulatory_dynamics']
            # dynamics_names = ['combination_dynamics']
            # dynamics_names = ['combination_dynamics_vary_coeff']
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics', 'gene_regulatory_dynamics']
            # dynamics_names = ['vary_dynamics_with_vary_type_and_coeff']
        elif dynamics_name == 'all_with_combination_dynamics':
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics', 'gene_regulatory_dynamics',
                              'combination_dynamics']
            # dynamics_names = ['gene_regulatory_dynamics']
        elif dynamics_name == 'all_with_combination_dynamics_vary_coeff':
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics', 'gene_regulatory_dynamics',
                              'combination_dynamics_vary_coeff']
        elif dynamics_name == 'all_with_combination_dynamics_vary_coeff_vary_dynamics':
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics', 'gene_regulatory_dynamics',
                              'combination_dynamics_vary_coeff', 'vary_dynamics_with_vary_type_and_coeff']
        elif dynamics_name == 'all_epidemic':
            dynamics_names = ['SI_Individual_dynamics', 'SIS_Individual_dynamics', 'SIR_Individual_dynamics',
                              'SEIS_Individual_dynamics', 'SEIR_Individual_dynamics']
        elif dynamics_name == 'One_For_All':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   ('combination_dynamics_vary_coeff', 'grid'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   ('combination_dynamics_vary_coeff', 'power_law'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   ('combination_dynamics_vary_coeff', 'random'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   ('combination_dynamics_vary_coeff', 'small_world'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   ('combination_dynamics_vary_coeff', 'community'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   ('SEIS_Individual_dynamics', 'power_law'),
                                   ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        else:
            dynamics_names = [dynamics_name]

        if dynamics_topo_names is None:
            dynamics_topo_names = []
            for i in dynamics_names:
                for j in topo_types:
                    dynamics_topo_names.append((i, j))

        for dynamics_name_i, topo_type_i in dynamics_topo_names:
            print(dynamics_name_i, topo_type_i)
            test_N = 200
            test_num_trials = 100
            add_str = '_is_shuffleTrue'
            if 'RealEpidemicData' in dynamics_name_i:
                x_dim_ = 1
                #test_num_trials = 18
                test_num_trials = 150
                if '12' in dynamics_name_i:
                    test_N = 17
                elif '13' in dynamics_name_i:
                    test_N = 5
                elif '23' in dynamics_name_i:
                    test_N = 42
                
                test_N = 234
                """
                if '123' in dynamics_name_i:
                    test_N = 52
                elif '124' in dynamics_name_i:
                    test_N = 17
                elif '134' in dynamics_name_i:
                    test_N = 5
                elif '234' in dynamics_name_i:
                    test_N = 42
                """
            elif 'sim_epidemic' in dynamics_name_i:
                x_dim_ = 1
                test_N = 130
                test_num_trials = 50
            elif 'real_data_spain_covid19' in dynamics_name_i:
                x_dim_ = 3
                test_N = 52
                test_num_trials = 4
            elif 'SIR_meta_pop' in dynamics_name_i:
                x_dim_ = 3
                test_N = 52
            elif 'SEIR' in dynamics_name_i:
                x_dim_ = 4
            elif 'SIR_Individual' in dynamics_name_i or 'SEIS' in dynamics_name_i:
                x_dim_ = 3
            elif 'SI' in dynamics_name_i or 'SIS' in dynamics_name_i:
                x_dim_ = 2
            elif 'opinion' in dynamics_name_i:
                x_dim_ = 2
                add_str = '_is_shuffleTrue_024'
            elif 'brain' in dynamics_name_i:
                x_dim_ = 2
                add_str = ''
                test_num_trials=3
            elif 'phototaxis' in dynamics_name_i:
                x_dim_ = 5
                add_str = ''
                test_num_trials=3
                test_N = 40
            else:
                x_dim_ = 1
                test_N = 225
                add_str = ''
                test_num_trials = 20

            saved_results = {'test_id': [],
                             'observations': [],
                             'groundtruth': [], 'groundtruth_sum': [],
                             'predictions': [], 'predictions_sum': [],
                             'nl1_error': [], 'l2_error': [],
                             'weights': []}

            import pickle
                        
            if args.is_sparsity:
                test_data_set_path = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph1_timestep%s_seed666_num_nodes=%s_sparsity=%s_split_train_and_test%s.pickle' % \
                                     (dynamics_name_i, topo_type_i, x_dim_, time_steps, test_N, args.sparsity, add_str)
            else:
                test_data_set_path = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph1_timestep100_seed666_num_nodes=%s_split_train_and_test%s.pickle' % \
                                     (dynamics_name_i, topo_type_i, x_dim_, test_N, add_str)
                                     
            with open(test_data_set_path, 'rb') as f:
                batch_all_test = pickle.load(f)
            for i in range(test_num_trials):
                if 'RealEpidemicData' in dynamics_name_i:
                    time_steps = 100
                saved_results['test_id'].append(i)
                if args.constraint_state:
                    model_path = "./saved_models/saved_model_%s_MLloss%s_deter%s_uncer%s_gnn%s_%s_%s_x%s_numgraph%s_timestep%s_epoch%s_%s.pkl" % \
                                 (model.name, model.use_ML_loss, model.is_determinate,
                                  model.is_uncertainty, model.gnn_type,
                                  dynamics_name,
                                  topo_type, x_dim, num_graphs,
                                  time_steps, num_epochs,
                                  args.constraint_state
                                  )
                else:
                    model_path = "./saved_models/saved_model_%s_MLloss%s_deter%s_uncer%s_gnn%s_%s_%s_x%s_numgraph%s_timestep%s_epoch%s.pkl" % \
                                 (model.name, model.use_ML_loss, model.is_determinate,
                                  model.is_uncertainty, model.gnn_type,
                                  dynamics_name,
                                  topo_type, x_dim, num_graphs,
                                  time_steps, num_epochs
                                  )
                if 'RealEpidemicData' in dynamics_name_i:
                    time_steps = args.time_steps
                
                res = test_net_dynamic(model,
                                       path=model_path,
                                       batch_data=batch_all_test[
                                           (dynamics_name_i, topo_type_i, bound_t_context, i)],
                                       saved_results=saved_results,
                                       add_str="test_on_%s_%s_epoch%s_bound_t_context%s_seed%s_%s" % (
                                           dynamics_name_i, topo_type_i, num_epochs, bound_t_context, seed, i),
                                       is_plot=False)

            if topo_type == 'all' or args.test_topo_type != 'none':
                topo_type_i_add = topo_type + topo_type_i
            else:
                topo_type_i_add = topo_type_i

            if 'all' in dynamics_name:
                dynamics_name_i_add = dynamics_name + dynamics_name_i
            else:
                dynamics_name_i_add = dynamics_name_i

            if args.constraint_state:
                saved_fname = "./results/saved_test_results_%s_MLloss%s_deter%s_uncer%s_%s_%s_x%s_numgraph%s_timestep%s" % \
                              (model.name, model.use_ML_loss, model.is_determinate, model.is_uncertainty,
                               dynamics_name_i_add,
                               topo_type_i_add, x_dim_, num_graphs,
                               time_steps) + "_epoch%s_bound_t_context%s_seed%s_num_nodes%s_%s" % \
                              (num_epochs, bound_t_context, seed, test_N, args.constraint_state)
            else:
                saved_fname = "./results/saved_test_results_%s_MLloss%s_deter%s_uncer%s_%s_%s_x%s_numgraph%s_timestep%s" % \
                              (model.name, model.use_ML_loss, model.is_determinate, model.is_uncertainty,
                               dynamics_name_i_add,
                               topo_type_i_add, x_dim_, num_graphs,
                               time_steps) + "_epoch%s_bound_t_context%s_seed%s_num_nodes%s" % \
                              (num_epochs, bound_t_context, seed, test_N)
            
            if args.is_sparsity:
                saved_fname += '_sparsity%s' % args.sparsity
                
            if args.is_fine_tune:
                saved_fname += 'is_fine_tune' + str(args.is_fine_tune)

            print('saving results to ', saved_fname)
            with open(saved_fname + '.pkl', 'wb') as f:
                pickle.dump(saved_results, f)
    # test
    if args.test_with_2nd_phase:

        dynamics_topo_names = None

        if topo_type == 'all':
            topo_types = ['grid', 'power_law', 'random', 'small_world', 'community']
            # topo_types = ['power_law', 'random', 'small_world', 'community']
        else:
            if args.test_topo_type == 'none':
                topo_types = [topo_type]
            else:
                topo_types = [args.test_topo_type]

        if dynamics_name == 'all':
            # dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics', 'gene_regulatory_dynamics']
            # dynamics_names = ['combination_dynamics']
            # dynamics_names = ['combination_dynamics_vary_coeff']
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics',
                              'gene_regulatory_dynamics']
            # dynamics_names = ['vary_dynamics_with_vary_type_and_coeff']
        elif dynamics_name == 'all_with_combination_dynamics':
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics',
                              'gene_regulatory_dynamics',
                              'combination_dynamics']
            # dynamics_names = ['gene_regulatory_dynamics']
        elif dynamics_name == 'all_with_combination_dynamics_vary_coeff':
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics',
                              'gene_regulatory_dynamics',
                              'combination_dynamics_vary_coeff']
        elif dynamics_name == 'all_with_combination_dynamics_vary_coeff_vary_dynamics':
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics',
                              'gene_regulatory_dynamics',
                              'combination_dynamics_vary_coeff', 'vary_dynamics_with_vary_type_and_coeff']
        elif dynamics_name == 'all_epidemic':
            dynamics_names = ['SI_Individual_dynamics', 'SIS_Individual_dynamics', 'SIR_Individual_dynamics',
                              'SEIS_Individual_dynamics', 'SEIR_Individual_dynamics']
        elif dynamics_name == 'One_For_All':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   ('combination_dynamics_vary_coeff', 'grid'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   ('combination_dynamics_vary_coeff', 'power_law'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   ('combination_dynamics_vary_coeff', 'random'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   ('combination_dynamics_vary_coeff', 'small_world'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   ('combination_dynamics_vary_coeff', 'community'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   ('SEIS_Individual_dynamics', 'power_law'),
                                   ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        else:
            dynamics_names = [dynamics_name]

        if dynamics_topo_names is None:
            dynamics_topo_names = []
            for i in dynamics_names:
                for j in topo_types:
                    dynamics_topo_names.append((i, j))

        for dynamics_name_i, topo_type_i in dynamics_topo_names:
            test_N = 200
            test_num_trials = 100
            add_str = '_is_shuffleTrue'
            if 'RealEpidemicData_123' in dynamics_name_i:
                x_dim_ = 1
                test_N = 52
                test_num_trials = 1
            elif 'real_data_spain_covid19' in dynamics_name_i:
                x_dim_ = 3
                test_N = 52
                test_num_trials = 4
            elif 'SIR_meta_pop' in dynamics_name_i:
                x_dim_ = 3
                test_N = 52
            elif 'SEIR' in dynamics_name_i or 'Coupled' in dynamics_name_i:
                x_dim_ = 4
            elif 'SIR' in dynamics_name_i or 'SEIS' in dynamics_name_i:
                x_dim_ = 3
            elif 'SI' in dynamics_name_i or 'SIS' in dynamics_name_i:
                x_dim_ = 2
            elif 'opinion' in dynamics_name_i:
                x_dim_ = 2
                add_str = '_is_shuffleTrue_024'
            elif 'brain' in dynamics_name_i:
                x_dim_ = 2
                add_str = ''
                test_num_trials=3
            else:
                x_dim_ = 1
                test_N = 225
                add_str = ''
                test_num_trials = 20

            saved_results = {'test_id': [],
                             'observations': [],
                             'groundtruth': [], 'groundtruth_sum': [],
                             'predictions': [], 'predictions_sum': [],
                             'nl1_error': [], 'l2_error': [],
                             'weights': []}

            import pickle

            if args.is_sparsity:
                test_data_set_path = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph1_timestep100_seed666_num_nodes=%s_sparsity=%s_split_train_and_test%s.pickle' % \
                                     (dynamics_name_i, topo_type_i, x_dim_, test_N, args.sparsity, add_str)
            else:
                test_data_set_path = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph1_timestep100_seed666_num_nodes=%s_split_train_and_test%s.pickle' % \
                                     (dynamics_name_i, topo_type_i, x_dim_, test_N, add_str)
            with open(test_data_set_path, 'rb') as f:
                batch_all_test = pickle.load(f)
            for i in range(test_num_trials):
                saved_results['test_id'].append(i)
                if args.constraint_state:
                    model_path = "./saved_models/saved_model_%s_MLloss%s_deter%s_uncer%s_gnn%s_%s_%s_x%s_numgraph%s_timestep%s_epoch%s_%s.pkl" % \
                                 (model.name, model.use_ML_loss, model.is_determinate,
                                  model.is_uncertainty, model.gnn_type,
                                  dynamics_name,
                                  topo_type, x_dim, num_graphs,
                                  time_steps, num_epochs,
                                  args.constraint_state
                                  )
                else:
                    model_path = "./saved_models/saved_model_%s_MLloss%s_deter%s_uncer%s_gnn%s_%s_%s_x%s_numgraph%s_timestep%s_epoch%s.pkl" % \
                                 (model.name, model.use_ML_loss, model.is_determinate,
                                  model.is_uncertainty, model.gnn_type,
                                  dynamics_name,
                                  topo_type, x_dim, num_graphs,
                                  time_steps, num_epochs
                                  )

                # pretrained models' paths
                paths = [
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_heat_diffusion_dynamics_all_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_mutualistic_interaction_dynamics_all_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_gene_regulatory_dynamics_all_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_opinion_dynamics_Baumann2021_2topic_small_world_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SI_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SIS_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SIR_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SEIS_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SEIR_Individual_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
                    "saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_Coupled_Epidemic_dynamics_power_law_x4_numgraph1000_timestep100_epoch30.pkl",
                    #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_SIR_meta_pop_dynamics_directed_full_connected_x4_numgraph1000_timestep100_epoch30.pkl",
                    #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_123_power_law_x4_numgraph1000_timestep100_epoch1000.pkl",  
                    #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_124_power_law_x4_numgraph1000_timestep100_epoch1000.pkl", 
                    #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_134_power_law_x4_numgraph1000_timestep100_epoch1000.pkl", 
                    #"saved_models/saved_model_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_gnngat_RealEpidemicData_234_power_law_x4_numgraph1000_timestep100_epoch1000.pkl", 
                ]
                # trained 2nd model's path
                score_net_path = "./saved_models/saved_model_score_net_num_scenario%s.pkl" % (len(paths))
                res = test_net_dynamic_with_2nd_phase(paths=paths,
                                                      score_net_path=score_net_path,
                                                      batch_data=batch_all_test[
                                                          (dynamics_name_i, topo_type_i, bound_t_context, i)],
                                                      saved_results=saved_results,
                                                      add_str="test_on_%s_%s_epoch%s_bound_t_context%s_seed%s_%s" % (
                                                          dynamics_name_i, topo_type_i, num_epochs, bound_t_context,
                                                          seed, i),
                                                      is_plot=False)

            if topo_type == 'all' or args.test_topo_type != 'none':
                topo_type_i_add = topo_type + topo_type_i
            else:
                topo_type_i_add = topo_type_i

            if 'all' in dynamics_name:
                dynamics_name_i_add = dynamics_name + dynamics_name_i
            else:
                dynamics_name_i_add = dynamics_name_i

            if args.constraint_state:
                saved_fname = "./results/saved_test_results_%s_MLloss%s_deter%s_uncer%s_%s_%s_x%s_numgraph%s_timestep%s" % \
                              (model.name, model.use_ML_loss, model.is_determinate, model.is_uncertainty,
                               dynamics_name_i_add,
                               topo_type_i_add, x_dim_, num_graphs,
                               time_steps) + "_epoch%s_bound_t_context%s_seed%s_num_nodes%s_%s" % \
                              (num_epochs, bound_t_context, seed, test_N, args.constraint_state)
            else:
                saved_fname = "./results/saved_test_results_%s_MLloss%s_deter%s_uncer%s_%s_%s_x%s_numgraph%s_timestep%s" % \
                              (model.name, model.use_ML_loss, model.is_determinate, model.is_uncertainty,
                               dynamics_name_i_add,
                               topo_type_i_add, x_dim_, num_graphs,
                               time_steps) + "_epoch%s_bound_t_context%s_seed%s_num_nodes%s" % \
                              (num_epochs, bound_t_context, seed, test_N)

            if args.is_sparsity:
                saved_fname += '_sparsity%s' % args.sparsity

            if args.is_fine_tune:
                saved_fname += 'is_fine_tune' + str(args.is_fine_tune)

            saved_fname += '_with_2nd_phase'

            print('saving results to ', saved_fname)
            with open(saved_fname + '.pkl', 'wb') as f:
                pickle.dump(saved_results, f)
