import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from torch_geometric.profile import count_parameters

import utils
from model import *
from utils import *
from copy import deepcopy
from tqdm import tqdm
import scipy.sparse as sp
import time

from model import MetaSTNN


def asym_adj(adj):
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def filter_negative(input_, thres):
    times = input_[:, 0, 0, 1]

    m = []
    cnt = 0
    c = thres / 288
    for t in times:
        if t < c:
            st = times < 0
            gt = torch.logical_and(times <= (1 + t - c), times >= (t + c))
        elif t > (1 - c):
            st = torch.logical_and(times <= (t - c), times >= (c + t - 1))
            gt = times > 1
        else:
            st = times <= (t - c)
            gt = times >= (t + c)

        res = torch.logical_or(st, gt).view(1, -1)
        res[0, cnt] = True
        cnt += 1
        m.append(res)
    m = torch.cat(m)
    return m

class STMAML(nn.Module):

    def __init__(self, data_args, task_args, con_args, model_args, model='GRU'):
        super(STMAML, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.con_args = con_args
        self.update_lr = model_args['update_lr']
        self.meta_lr = model_args['meta_lr']
        self.update_step = model_args['update_step']
        self.update_step_test = model_args['update_step_test']
        self.task_num = task_args['task_num']
        self.model_name = model

        self.loss_lambda = model_args['loss_lambda']
        # print("loss_lambda = ", self.loss_lambda)

        if model == 'GRU':
            # Meta-GRU
            self.model = MetaSTNN(model_args, task_args)
            print("MAML Model: GRU")
        elif model == 'GWN':
            self.model = MetaGWN()
            print("MAML Model: GraphWave Net")
        else:
            self.model = MetaSTNN(model_args, task_args)
            print("MAML Model: GRU (default)")

        # print(self.model)
        print("model params: ", count_parameters(self.model))
        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        # self.meta_optim = torch.optim.SGD(self.model.parameters(), lr=self.update_lr, momentum=0.9)
        self.loss_criterion = utils.masked_mae

    # 损失函数和负过滤函数参照paper，感觉要改一下
    def contrastive_loss(self, matrix,input_data, meta_knowledge, aug_meta_knowledge):
        # temporal contrast:
        # mk(bs,node,meta_dim)
        tempo_mk = meta_knowledge.transpose(0,1) # tp_mk(node,bs,meta_dim)
        tempo_mk_norm = tempo_mk.norm(dim=2).unsqueeze(dim=2)
        aug_tempo_mk = aug_meta_knowledge.transpose(0,1)
        aug_tempo_mk_norm = aug_tempo_mk.norm(dim=2).unsqueeze(dim=2)

        tempo_matrix = torch.matmul(tempo_mk, aug_tempo_mk.transpose(1,2)/
                                    torch.matmul(tempo_mk_norm,aug_tempo_mk_norm.transpose(1,2)))

        tempo_matrix = torch.exp(tempo_matrix / self.tempe)
        # temporal filter:
        if self.fn_t:
            m = filter_negative(input_data, self.fn_t)
            tempo_matrix = tempo_matrix * m
        tempo_neg = torch.sum(tempo_matrix, dim=2) # (node,bs)
        # spatio contrast:
        spatio_norm = meta_knowledge.norm(dim=2).unsqueeze(dim=2)
        aug_spatio_norm = aug_meta_knowledge.norm(dim=2).unsqueeze(dim=2)
        spatio_matrix = (torch.matmul(meta_knowledge,aug_meta_knowledge.transpose(1,2)) /
                         torch.matmul(spatio_norm,aug_spatio_norm.transpose(1,2)))
        spatio_matrix = torch.exp(spatio_matrix / self.tempe)
        diag = torch.eye(meta_knowledge.shape[1],dtype=torch.bool).cuda()
        pos_sum = torch.sum(spatio_matrix * diag,dim=2) # (bs,node)
        # spatio filter
        if self.fn_t:
            adj = (matrix == 0)
            adj = adj.clone().detach().cuda()
            adj = adj + diag
            spatio_matrix = spatio_matrix * adj
        spatio_neg = torch.sum(spatio_matrix,dim=2) # (bs,node)
        eps = 1e-8
        ratio = pos_sum / (spatio_neg + tempo_neg.transpose(0,1) - pos_sum + eps)
        u_loss = torch.mean(-torch.log(ratio))

        return u_loss

    def con_calculate_loss(self, out, y, meta_knowledge, matrix, batch_size, input_data,
                           aug_meta_knowledge,stage='target', graph_loss=True, loss_lambda=1.5,):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y,0.0)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y,0.0)
                loss_contrastive = self.contrastive_loss(matrix, input_data,meta_knowledge,aug_meta_knowledge)
            else:
                loss_predict = self.loss_criterion(out, y,0.0)
                loss_contrastive = self.loss_criterion(meta_knowledge, aug_meta_knowledge, 0.0)
            loss = loss_predict + loss_lambda * loss_contrastive
        else:
            loss = self.loss_criterion(out, y,0.0)
        return loss

    def calculate_loss(self, out, y, stage='target', graph_loss=True, loss_lambda=1):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y,0.0)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y,0.0)
            else:
                loss_predict = self.loss_criterion(out, y,0.0)
            loss = loss_predict
        else:
            loss = self.loss_criterion(out, y,0.0)

        return loss

    def contrastive_meta_train(self, data_spt, matrix_spt, data_qry, matrix_qry, aug_data_spt, aug_matrix_spt, aug_data_qry, aug_matrix_qry):

        self.tempe = self.con_args['tempe']
        self.fn_t = self.con_args['fn_t']
        model_loss = 0

        init_model = deepcopy(self.model)

        for i in range(self.task_num):
            maml_model = deepcopy(init_model)
            optimizer = optim.Adam(maml_model.parameters(), lr=self.update_lr, weight_decay=1e-2)

            for k in range(self.update_step):
                batch_size, node_num, seq_len, _ = data_spt[i].x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [matrix_spt[i], (matrix_spt[i]).t()]
                    aug_adj_mx = [aug_matrix_spt[i], (aug_matrix_spt[i]).t()]
                    out, meta_knowledge = maml_model(data_spt[i], adj_mx)
                    _, aug_meta_knowledge = maml_model(aug_data_spt[i], aug_adj_mx)
                else:
                    out, meta_knowledge = maml_model(data_spt[i], matrix_spt[i])
                    # con
                    _, aug_meta_knowledge = maml_model(aug_data_spt[i],aug_matrix_spt[i])

                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss = self.loss_criterion(out, data_spt[i].y,0.0)
                else:
                    # loss = self.calculate_loss(out, data_spt[i].y, meta_graph, matrix_spt[i], 'source', graph_loss=False)
                    loss = self.con_calculate_loss(out, data_spt[i].y, meta_knowledge, matrix_spt[i],batch_size,data_spt[i].x,
                                                   aug_meta_knowledge,'source', loss_lambda=self.loss_lambda)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.model_name == 'GWN':
                adj_mx = [matrix_qry[i], (matrix_qry[i]).t()]
                aug_adj_mx = [aug_matrix_qry[i], (aug_matrix_qry[i]).t()]
                out, meta_knowledge = self.model(data_qry[i], adj_mx)
                _, aug_meta_knowledge = maml_model(aug_data_qry[i], aug_adj_mx)
            else:
                out, meta_knowledge = self.model(data_qry[i], matrix_qry[i])
                _, aug_meta_knowledge = maml_model(aug_data_qry[i], aug_matrix_qry[i])

            if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                loss_q = self.loss_criterion(out, data_qry[i].y,0.0)
            else:
                # loss_q = self.calculate_loss(out, data_qry[i].y, meta_graph, matrix_qry[i], 'target_maml', graph_loss=False)
                loss_q = self.con_calculate_loss(out, data_qry[i].y, meta_knowledge, matrix_qry[i], batch_size ,data_spt[i].x,
                                                 aug_meta_knowledge,'target_maml', loss_lambda=self.loss_lambda)
            model_loss += loss_q

        model_loss = model_loss / self.task_num
        self.meta_optim.zero_grad()
        model_loss.backward()
        self.meta_optim.step()

        return model_loss.detach().cpu().numpy()

    def forward(self, data, matrix):
        out, meta_knowledge = self.model(data, matrix)
        return out, meta_knowledge

    def finetuning(self, target_dataloader, test_dataloader, target_epochs, test_dataset):

        maml_model = deepcopy(self.model)

        optimizer = optim.Adam(maml_model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        best_loss = float('inf')
        best_result = ''
        best_meta_graph = -1
        # print(test_dataset)
        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            maml_model.train()
            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                # data.node_num = data.node_num[0]
                # data augment
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, _ = maml_model(data, adj_mx)
                else:
                    out, _ = maml_model(data, A_wave[0].float())

                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss = self.loss_criterion(out, data.y,0.0)
                else:
                    # loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', graph_loss=False)
                    loss = self.calculate_loss(out, data.y, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses) / len(train_losses)
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_model_state = deepcopy(maml_model.state_dict())
            end_time = time.time()
            if epoch % 2 == 0:
                print("[Target Fine-tune] epoch #{}: loss is {}, "
                      "fine-tuning time is {}".format(epoch + 1,avg_train_loss,end_time - start_time))

        maml_model.load_state_dict(best_model_state)
        maml_model.eval()
        with torch.no_grad():
            test_start = time.time()

            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                # data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, _ = maml_model(data, adj_mx)
                else:
                    out, _ = maml_model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu()
            y_label = y_label.permute(0, 2, 1).detach().cpu()
            result = mask_metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            # result = mask_metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, info_name='Evaluate', test_dataset=test_dataset)
            print("[Target Test] testing time is {}".format(test_end - test_start))
