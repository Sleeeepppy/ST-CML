import numpy as np
import torch
from utils import *
import random
from torch_geometric.data import Data, Dataset, DataLoader
from scipy.fftpack import dct, idct

class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self)
        self.errorinfo=ErrorInfo

    def __str__(self):
        return self.errorinfo


class traffic_dataset(Dataset):
    def __init__(self, data_args, task_args, con_args, stage, args, add_target=True, test_data='metr-la', target_days=3):
        super(traffic_dataset, self).__init__()
        self.device = args.device
        self.data_args = data_args
        self.task_args = task_args
        self.con_args = con_args
        self.stage = stage
        self.test_data = test_data
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.add_target = add_target
        self.test_dataset = test_data
        self.target_days = target_days
        self.load_data(stage, test_data)
        if self.add_target:
            self.data_list = np.append(self.data_list, self.test_data)
        print("[INFO] Dataset init finished!")

    def load_data(self, stage, test_data):

        self.A_list, self.edge_index_list = {}, {}  # A_list : 邻接矩阵；edge_index_list : 边索引
        self.edge_attr_list, self.node_feature_list = {}, {}  # edge_attr_list : 边特征；node_feature_list : 节点特征
        self.x_list, self.y_list = {}, {}  # x_list : 输入数据； y_list : 标签数据
        self.means_list, self.stds_list = {}, {}
        # 增强数据
        self.A_aug_list = {}
        self.x_aug_list, self.y_aug_list = {}, {}
        self.means_aug_list, self.stds_aug_list = {}, {}

        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
            #print("proceed data_list:",self.data_list)
            # 处理训练数据，从 data_keys 中去掉测试数据集 'metr-la'，将它保留用于测试，其余三个数据集将被用于训练模型。
        elif stage == 'target' or stage == 'target_maml':
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError('Error: Unsupported Stage')
        print("[INFO] {} dataset: {}".format(stage, self.data_list))

        for dataset_name in self.data_list:
            A = np.load(self.data_args[dataset_name]['adjacency_matrix_path'])
            edge_index = self.get_attr_func(self.data_args[dataset_name]['adjacency_matrix_path'])

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[dataset_name] = edge_index

            X = np.load(self.data_args[dataset_name]['dataset_path'])
            X, means, stds = get_normalized_data(X)
            if stage == 'source':
                X = X
            elif stage == 'target' or stage == 'target_maml':
                X = X[:, :, :288 * self.target_days]
            elif stage == 'test':
                X = X[:, :, int(X.shape[2]*0.8):]
            else:
                raise BBDefinedError('Error: Unsupported Stage')
            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'],
                                                   means, stds)

            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs

        # 处理训练数据集
        if stage == 'source' and self.add_target:
            #print("data_list --check:",self.data_list)
            A = np.load(self.data_args[test_data]['adjacency_matrix_path'])
            edge_index = self.get_attr_func(self.data_args[test_data]['adjacency_matrix_path'])

            self.A_list[test_data] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[test_data] = edge_index

            X_test = np.load(self.data_args[test_data]['dataset_path'])
            X_test, means_test, stds_test = get_normalized_data(X_test)

            X_test = X_test[:, :, :288 * self.target_days]
            x_test_input ,y_test_output = generate_dataset(X_test, self.task_args['his_num'],
                                                          self.task_args['pred_num'], means_test, stds_test)
            self.x_list[test_data] = x_test_input
            self.y_list[test_data] = y_test_output

    def get_attr_func(self, matrix_path):
        a, b = [], []  # 分别存储边的起始节点和终止节点的索引。
        # edge_attr = []  # None
        # node_feature = None
        matrix = np.load(matrix_path)  # 邻接矩阵
        # edge_feature_matrix = np.load(edge_feature_matrix_path)
        # node_feature = np.load(node_feature_path)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if (matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a, b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index

    def get_maml_task_batch(self, task_num):

        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        for i in range(task_num * 2):
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = self.A_list[select_dataset].shape[0]
            # augment
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            data_i.data_name = select_dataset
            A_wave = self.A_list[select_dataset].float()

            if i % 2 == 0:
                spt_task_data.append(data_i)
                spt_task_A_wave.append(A_wave)
            else:
                qry_task_data.append(data_i)
                qry_task_A_wave.append(A_wave)
        # spt_task_data = [data.pin_memory().to('cuda:0', non_blocking=True) for data in spt_task_data]
        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave

    def get_edge_feature(self, edge_index, x_data):
        pass

    def get_maml_task_batch_aug(self, task_num):

        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        aug_spt_task_data, aug_qry_task_data = [], []
        aug_spt_task_A_wave, aug_qry_task_A_wave = [], []

        #print("data_list:",self.data_list)
        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']
        # print("batch_size:",batch_size)
        for i in range(task_num * 2):

            frame = self.x_list[select_dataset].shape[2]
            num_node = self.x_list[select_dataset].shape[1]
            bs = self.x_list[select_dataset].shape[0]

            permutation = torch.randperm(bs)
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = num_node

            # add adj augment
            A_wave = self.A_list[select_dataset].float()
            aug_A_wave = self.adj_aug(con_args=self.con_args, A_wave=A_wave)
            # add augment
            aug_x_data = self.data_augment(x_data, y_data, A_wave,con_args=self.con_args)
            #print("x_data :",x_data.size())
            #print("aug:",aug_x_data.size())
            # storage data
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            data_i.data_name = select_dataset
            # storage augment data
            aug_data_i = Data(node_num=node_num, x=aug_x_data, y=y_data)
            aug_data_i.edge_index = self.edge_index_list[select_dataset]
            aug_data_i.data_name = select_dataset


            if i % 2 == 0:
                spt_task_data.append(data_i.cuda())
                spt_task_A_wave.append(A_wave.cuda())

                aug_spt_task_data.append(aug_data_i.cuda())
                aug_spt_task_A_wave.append(aug_A_wave.cuda())

            else:
                qry_task_data.append(data_i.cuda())
                qry_task_A_wave.append(A_wave.cuda())

                aug_qry_task_data.append(aug_data_i.cuda())
                aug_qry_task_A_wave.append(aug_A_wave.cuda())
        # spt_task_data = [data.pin_memory().to('cuda:0', non_blocking=True) for data in spt_task_data]
        # return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave
        return (spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave,
                aug_spt_task_data, aug_spt_task_A_wave, aug_qry_task_data, aug_qry_task_A_wave)

    def adj_aug(self, con_args, A_wave):
        self.em_t = con_args['em_t']
        if self.em_t is not None:
            rand = torch.rand(A_wave.shape[0], A_wave.shape[1])
            supports = A_wave * (rand >= self.em_t)
            return supports
        else:
            return A_wave

    def data_augment(self, x_data, y_data, A_wave, con_args):

        self.im_t = con_args['im_t']
        self.ts_t = con_args['ts_t']
        self.ism_t = con_args['ism_t']
        self.em_t = con_args['em_t']
        self.ism_e = con_args['ism_e']

        # print("node_num:", node_num)
        shape = x_data.size()
        frame = shape[2]
        num_node = shape[1]
        # print("num_node", num_node)
        bs = shape[0]

        if self.im_t or self.ts_t or self.ism_t:

            input_ = x_data.detach().clone()

            if self.im_t:
                rand = torch.rand(bs, num_node, frame)
                input_[:, :, :, 0] = input_[:, :, :, 0] * (rand >= self.im_t)

            if self.ts_t:
                # print("y size:", y_data.size())
                # print("ipt size:", input_.size())
                s = torch.cat((input_[:, :, :, 0], y_data), dim=2)[:, :, :frame + 1]
                rand = (1 - self.ts_t) * torch.rand(bs, 1, 1) + self.ts_t
                rand = rand.expand(bs, num_node, frame + 1)
                input_[:, :, :, 0] = (s * rand + torch.roll(s, -1, 2) * (1 - rand))[:, :, :frame]

            if self.ism_t:
                s = torch.cat((input_[:, :, :, 0], y_data), dim=2)
                o = []
                for i in range(bs):
                    t = np.array(s[i])
                    m1 = np.ones((num_node, self.ism_e))
                    # print("t:", t.shape)

                    m2 = np.random.uniform(low=self.ism_t, high=1.0, size=(num_node, t.shape[1] - self.ism_e))
                    m2 = np.matmul(A_wave, m2)
                    m2 = np.matmul(A_wave, m2)
                    # print("m1", m1.shape)
                    # print("m2", m2.shape)
                    mall = np.concatenate((m1, m2), axis=1)
                    t = dct(t, norm='ortho')
                    t = np.multiply(t, mall)
                    t = idct(t, norm='ortho')
                    o.append(t)
                o = np.stack(o)
                input_[:, :, :, 0] = torch.tensor(o[:, :, :frame])
            # diff = torch.mean(torch.abs(input_[:, :, :, 0] - x_data[:, :, :, 0])).item()
            # print("diff:", diff)

        else:
            input_ = x_data.detach().clone()

        return input_

    def __getitem__(self, index):
        """
        : data.node_num record the node number of each batch
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        : data.node_feature shape is [batch_size, node_num, node_dim]
        """

        if self.stage == 'source':
            select_dataset = random.choice(self.data_list)
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        elif self.stage == 'target_maml':
            select_dataset = self.data_list[0]
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        else:
            select_dataset = self.data_list[0]
            x_data = self.x_list[select_dataset][index: index+1]
            y_data = self.y_list[select_dataset][index: index+1]

        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]
        return data_i, A_wave


    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length







