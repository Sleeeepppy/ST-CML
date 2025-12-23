import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor
from copy import deepcopy
from torch_geometric.nn import GATConv
from datasets import BBDefinedError

# Meta linear:
class MetaLinear(nn.Module):
    def __init__(self, meta_dim, in_feature_dim, out_feature_dim, bias=True):
        super(MetaLinear, self).__init__()
        self.meta_dim = meta_dim
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.bias = bias
        self.build()

    def build(self):
        self.w_linear = nn.Linear(self.meta_dim, self.in_feature_dim * self.out_feature_dim)
        if self.bias:
            self.b_linear = nn.Linear(self.meta_dim, self.out_feature_dim)

    def forward(self, meta_knowledge, input, dim=3, nonlinear='None'):
        # meta_knowledge shape is [batch_size, node_num, meta_dim]
        # input shape is [batch_size, node_num, in_feature_dim]
        # output shape is [batch_size, node_num, out_feature_dim]

        if dim == 3:
            batch_size, node_num, _ = input.shape
            meta_knowledge = torch.reshape(meta_knowledge, (-1, self.meta_dim))
            x = torch.reshape(input, (-1, self.in_feature_dim)).unsqueeze(1)
        elif dim == 2:
            x = input.unsqueeze(1)
            # print("[Meta-Linear] meta knowledge shape is {}, x shape is {}".format(meta_knowledge.shape, x.shape))
        else:
            raise BBDefinedError("dim error.")

        w = self.w_linear(meta_knowledge)
        w = torch.reshape(w, (-1, self.in_feature_dim, self.out_feature_dim))

        if self.bias:
            b = self.b_linear(meta_knowledge)
            b = torch.reshape(b, (-1, 1, self.out_feature_dim))
            # print("[Meta Linear] w shape is {}, b shape is {}".format(w.shape, b.shape))
            y = torch.bmm(x, w) + b
        else:
            y = torch.bmm(x, w)

        if dim == 3:
            y = torch.reshape(y.squeeze(1), (batch_size, node_num, self.out_feature_dim))

        if nonlinear == 'None':
            output = y
        elif nonlinear == 'relu':
            output = nn.ReLU(y)
        elif nonlinear == 'leaky':
            output = F.LeakyReLU(y)
        elif nonlinear == 'tanh':
            output = nn.Tanh(y)
        else:
            print("[Warning] Unsupported nonlinear function")
            output = y
        return output

# MetaGRU Cell:
class MetaGRUCell(nn.Module):
    def __init__(self, meta_dim, input_dim, hidden_dim):
        super(MetaGRUCell, self).__init__()
        self.meta_dim = meta_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.build()

    def build(self):
        self.r_x_linear = MetaLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.r_h_linear = MetaLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)
        self.z_x_linear = MetaLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.z_h_linear = MetaLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)
        self.c_x_linear = MetaLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.c_h_linear = MetaLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)

    def forward(self, meta_knowledge, x, hidden):
        """
        : params meta_knowledge shape is [batch_size, node_num, meta_dim]
        : params x shape is [batch_size, node_num, input_dim]
        : params hidden shape is [batch_size, node_num, hidden_dim]
        : return next_hidden shape is [batch_size, node_num, hidden_dim]
        """
        r = torch.sigmoid(
            self.r_x_linear(meta_knowledge, x) + self.r_h_linear(meta_knowledge, hidden)
        )
        z = torch.sigmoid(
            self.z_x_linear(meta_knowledge, x) + self.z_h_linear(meta_knowledge, hidden)
        )
        c = torch.tanh(
            self.c_x_linear(meta_knowledge, x) + r * self.c_h_linear(meta_knowledge, hidden)
        )
        next_hidden = (1 - z) * c + z * hidden
        return next_hidden

# Meta-Learner(GRU+GAT):
class STMetaLearner_add(nn.Module):
    def __init__(self, model_args, task_args):
        super(STMetaLearner_add, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.tp = model_args['tp']
        self.sp = model_args['sp']
        self.node_feature_dim = model_args['node_feature_dim']
        self.edge_feature_dim = model_args['edge_feature_dim']
        self.message_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.meta_in = self.node_feature_dim + self.message_dim * self.his_num
        self.meta_out = model_args['meta_dim']
        self.build()

    def build(self):
        if self.tp:
            print("tp is True.")
            self.tp_learner = nn.GRU(self.message_dim, 1, batch_first=True)
        if self.sp:
            print("sp is True.")
            self.sp_learner = GATConv(self.message_dim * self.his_num, self.his_num, 3, False, dropout=0.1)

        if self.tp and self.sp:
            self.alpha = nn.Parameter(torch.FloatTensor(self.his_num))
            stdv = 1. / math.sqrt(self.alpha.shape[0])
            self.alpha.data.uniform_(-stdv, stdv)

        if self.tp == False and self.sp == False:
            print("sp and tp are all False.")
            self.meta_knowledge = nn.Parameter(torch.FloatTensor(self.his_num))

        self.mk_learner = nn.Linear(self.his_num, self.meta_out)

    def forward(self, data):
        batch_size, node_num, his_len, message_dim = data.x.shape
        # print("node_feature: {}, message_feature: {}, edge_attr: {}".format(node_feature.shape, message_feature.shape, edge_attr.shape))

        if self.tp:
            # tp_learner -> [batch_size * node_num, his_len]
            self.tp_learner.flatten_parameters()
            tp_input = torch.reshape(data.x, (batch_size * node_num, his_len, message_dim))
            tp_output, _ = self.tp_learner(tp_input)
            tp_output = tp_output.squeeze(-1)

        if self.sp:
            # sp_learner -> [batch_size * node_num, his_len]
            sp_input = torch.reshape(data.x, (batch_size * node_num, his_len, message_dim))
            sp_input = torch.reshape(sp_input, (batch_size * node_num, his_len * message_dim))
            sp_output = self.sp_learner(sp_input, data.edge_index)

        if self.tp and self.sp:
            mk_input = torch.sigmoid(self.alpha) * sp_output + (1 - torch.sigmoid(self.alpha)) * tp_output
        elif self.tp:
            mk_input = tp_output
        elif self.sp:
            mk_input = sp_output
        else:
            mk_input = self.meta_knowledge
            # print("sp and tp are all False.")

        meta_knowledge = self.mk_learner(mk_input)
        meta_knowledge = torch.reshape(meta_knowledge, (batch_size, node_num, self.meta_out))
        return meta_knowledge

# Parameter Generation GRU:
class MetaGRU(nn.Module):
    def __init__(self, model_args, task_args, input_dim=None, hidden_dim=None, output_dim=None):
        super(MetaGRU, self).__init__()
        self.meta_dim = model_args['meta_dim']
        self.model_args = model_args
        self.task_args = task_args
        self.input_dim = model_args['message_dim'] if input_dim == None else input_dim
        self.hidden_dim = model_args['hidden_dim'] if hidden_dim == None else hidden_dim
        self.output_dim = model_args['output_dim'] if output_dim == None else output_dim

        self.build()

    def build(self):
        self.metagru_layer = MetaGRUCell(self.meta_dim, self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, meta_knowledge, data, hidden=None, input=None):
        """
        : params meta_knowledge shape is [batch_size, node_num, meta_dim]
        : params input shape is [batch_size, node_num, seq_len, input_dim]
        : return output shape is [batch_size, node_num, seq_len, output_dim]
        """
        input = data.x if input == None else input
        batch_size, node_num, seq_len, _ = input.shape
        if hidden is None:
            hidden = torch.zeros(batch_size, node_num, self.hidden_dim).cuda()
        h_outputs = []
        for i in range(seq_len):
            input_i = input[:, :, i, :]
            hidden = self.metagru_layer(meta_knowledge, input_i, hidden)
            output = self.output_layer(hidden)
            h_outputs.append(output.unsqueeze(0))
        h_outputs = torch.cat(h_outputs, 0)
        return output, h_outputs

# Meta-Learner(GRU+GAT) + Parameter Generation GRU:
class MetaSTNN(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaSTNN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.gen_graph_dimension = model_args['gen_graph_dimension']
        self.build()

    def build(self):
        self.mk_learner = STMetaLearner_add(self.model_args, self.task_args)
        self.meta_gru = MetaGRU(self.model_args, self.task_args)
        self.predictor = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])

    def forward(self, data, A_wave):
        """
        : return output shape is [batch_size, node_num, output_seq_len]
        """
        meta_knowledge = self.mk_learner(data) # bs,num_node, frame-->bs,num_node, meta_out

        # softmax_func = nn.Softmax(dim=-1)

        _, gru_h_outputs = self.meta_gru(meta_knowledge, data)
        input = gru_h_outputs.squeeze(-1).permute(1, 2, 0)
        input = F.relu(input)
        output = self.predictor(input)

        return output, meta_knowledge

# Meta-GWN:
class MetaGWN(nn.Module):
    pass