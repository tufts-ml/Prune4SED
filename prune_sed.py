import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GINConv, GATv2Conv

from models import NormSEDModel
from topk import TopKPooling


class MutualForward(nn.Module):
    def __init__(self, args):
        super(MutualForward, self).__init__()

        self.n_layers = args.head_layer
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim

        self.pre = nn.Linear(self.input_dim, self.hidden_dim)

        self.convs = nn.ModuleList()
        self.mhas = nn.ModuleList()
        self.mha_linears = nn.ModuleList()
        self.batchnorm1d_s = nn.ModuleList()
        self.batchnorm1d_t = nn.ModuleList()

        # input layer
        self.mhas.append(nn.MultiheadAttention(args.hidden_dim, 4, add_zero_attn=True, batch_first=True))
        self.mha_linears.append(nn.Sequential(
            nn.Linear(args.hidden_dim, 1),
            nn.Sigmoid(),
        ))
        self.batchnorm1d_s.append(nn.BatchNorm1d(args.hidden_dim))
        self.batchnorm1d_t.append(nn.BatchNorm1d(args.hidden_dim))

        for l in range(self.n_layers + 1):
            if l < self.n_layers:
                self.convs.append(GATv2Conv(self.hidden_dim, self.hidden_dim, heads=1, concat=False))

            self.mhas.append(nn.MultiheadAttention(args.hidden_dim, 4, add_zero_attn=True, batch_first=True))
            self.mha_linears.append(nn.Sequential(nn.Linear(args.hidden_dim, 1), nn.Sigmoid()))

            self.batchnorm1d_s.append(nn.BatchNorm1d(args.hidden_dim))
            self.batchnorm1d_t.append(nn.BatchNorm1d(args.hidden_dim))

    def forward_mha(self, h_s, h_batch, t_s, t_batch, layer_index):
        h_s, s_mask = to_dense_batch(h_s, h_batch, fill_value=0)
        h_t, t_mask = to_dense_batch(t_s, t_batch, fill_value=0)

        h_cross, _ = self.mhas[layer_index](h_s, h_t, h_t, key_padding_mask=~t_mask)
        h_cross = h_cross[s_mask]
        h_score = self.mha_linears[layer_index](h_cross)

        h_s = h_s[s_mask]
        h_t = h_t[t_mask]
        h_s = h_s * h_score

        h_s = self.batchnorm1d_s[layer_index](h_s)
        h_t = self.batchnorm1d_t[layer_index](h_t)

        return h_s, h_t

    def forward(self, data_s, data_t):
        x_s = self.pre(data_s.x)
        x_t = self.pre(data_t.x)

        x_s, x_t = self.forward_mha(x_s, data_s.batch, x_t, data_t.batch, layer_index=0)

        emb_s = x_s
        emb_t = x_t

        for i in range(self.n_layers):
            x_s = self.convs[i](x_s, data_s.edge_index)
            x_t = self.convs[i](x_t, data_t.edge_index)
            x_s = F.relu(x_s)
            x_t = F.relu(x_t)

            x_s, x_t = self.forward_mha(x_s, data_s.batch, x_t, data_t.batch, layer_index=i + 1)

            emb_s = torch.cat((emb_s, x_s), dim=1)
            emb_t = torch.cat((emb_t, x_t), dim=1)

        return emb_s, emb_t


class PruneSED(nn.Module):
    def __init__(self, args):
        super(PruneSED, self).__init__()

        self.heads = args.heads
        self.hop = args.hop

        self.model_step_1 = MutualForward(args)
        self.predictor_s2 = NormSEDModel(args.predictor_layer, args.input_dim, args.hidden_dim, output_dim=args.hidden_dim, conv='gin')

        self.linear_step_2 = nn.Sequential(
            nn.Linear(args.input_dim + args.hidden_dim * (1 + args.head_layer), args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
        )
        self.cross = nn.MultiheadAttention(args.hidden_dim * (1 + args.head_layer), 4, add_zero_attn=True,
                                           batch_first=True)

        self.selector_cross_linear = nn.ModuleList()
        self.selector = nn.ModuleList()
        for _ in range(args.heads):
            self.selector_cross_linear.append(nn.Sequential(
                nn.Linear(args.hidden_dim * (1 + args.head_layer), args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
            )
            self.selector.append(TopKPooling(args.hidden_dim, ratio=args.top_k, nonlinearity=torch.sigmoid))

    def forward(self, data_t, data_s, lb=None, ub=None):
        h_s_init, h_t_init = self.model_step_1(data_s, data_t)

        h_s_ = h_s_init

        h_s_step_2 = data_s.x
        h_t_step_2 = data_t.x

        g_t_inter = Data(x=h_t_step_2, edge_index=data_t.edge_index, batch=data_t.batch)
        pred_list = []
        for i in range(self.heads):
            score_logits = self.selector_cross_linear[i](h_s_)
            x, edge_index, batch, _perm, _score, _score_raw = self.selector[i](h_s_step_2, data_s.edge_index, self.hop,
                                                              batch=data_s.batch, attn=score_logits)
            g_s_inter = Data(x=x, edge_index=edge_index, batch=batch)

            pred_i = self.predictor_s2(g_t_inter, g_s_inter)
            pred_list.append(pred_i[None, :])

        agg_pred = torch.vstack(pred_list)
        agg_pred = agg_pred.mean(0)

        if lb is not None and ub is not None:
            loss = self.predictor_s2.criterion(lb, ub, agg_pred)
        else:
            loss = None

        return agg_pred, loss
