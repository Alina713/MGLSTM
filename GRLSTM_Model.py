import torch
import torch.nn as nn
import logging
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from pars_args import args
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # 输入层到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # 隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 通过第一个全连接层和ReLU激活函数
        x = self.fc1(x)
        x = self.relu(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x
    
class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first)

    def forward(self, x):
        # LSTM layer
        output, _ = self.lstm(x)  # output shape: (batch_size, seq_length, hidden_dim)
        output, length = pad_packed_sequence(output, batch_first=True)

        # attention_3d_block 
        # https://github.com/ningshixian/LSTM_Attention/blob/master/attModel2/attention_lstm.py
        a = output.permute(0, 2, 1)
        a = F.softmax(a, dim=-1)
        a = torch.mean(a, dim=1)
        a = a.unsqueeze(1).repeat(1, output.shape[2], 1)
        a_probs = a.permute(0, 2, 1)
        attention_mul = output * a_probs

        attention_output = torch.add(output, attention_mul)

        packed_attention_output = pack_padded_sequence(attention_output, length, batch_first=True)
        
        return packed_attention_output

class FeatureViewModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureViewModule, self).__init__()
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        feature_emb = F.relu(self.feature_embedding(x))
        return feature_emb

class GRLSTM(nn.Module):
    def __init__(self, args, device, batch_first=True):
        super(GRLSTM, self).__init__()
        self.nodes = args.nodes
        self.latent_dim = args.latent_dim
        self.device = device
        self.batch_first = batch_first
        self.num_heads = args.num_heads
        self.fea_size = args.fea_size

        logging.info('Initializing model: latent_dim=%d' % self.latent_dim)

        self.poi_neighbors = np.load(
            args.poi_file, allow_pickle=True)['neighbors']
        
        self.poi_str_neighbors = np.load(
            args.poi_str_file, allow_pickle=True)['neighbors']
        
        self.feature_list = np.load(
            args.poi_fea_file, allow_pickle=True)['features']

        self.poi_features = torch.randn(
            self.nodes, self.latent_dim).to(self.device)

        self.lstm_list = nn.ModuleList([
            # nn.LSTM(input_size=self.latent_dim, hidden_size=self.latent_dim,
            #         num_layers=1, batch_first=True)
            LSTMAttention(input_size=self.latent_dim, hidden_size=self.latent_dim,
                          num_layers=1, batch_first=True)
            for _ in range(args.lstm_layers)
        ])

        self.gat1 = GATConv(in_channels=self.latent_dim, out_channels=16,
                           heads=8, dropout=0.1, concat=True)
        
        self.gat2 = GATConv(in_channels=self.latent_dim, out_channels=16,
                    heads=8, dropout=0.1, concat=True)
        
        self.latent_dim2 = self.latent_dim * 2
        self.mlp1 = MLP(self.latent_dim2, self.latent_dim, self.latent_dim)
        self.mlp2 = MLP(self.latent_dim2, self.latent_dim, self.latent_dim)

        self.fea_module = FeatureViewModule(self.fea_size, self.latent_dim)

    def _construct_edge_index(self, batch_x_flatten):
        batch_x_flatten = batch_x_flatten.cpu().numpy()
        neighbors = self.poi_neighbors[batch_x_flatten]

        batch_x_flatten = batch_x_flatten.repeat(neighbors.shape[1])

        neighbors = neighbors.reshape(-1)

        edge_index = np.vstack((neighbors, batch_x_flatten))
        batch_x_flatten = np.vstack((batch_x_flatten, batch_x_flatten))
        edge_index = np.concatenate((batch_x_flatten, edge_index), axis=1)
        edge_index = np.unique(edge_index, axis=1)

        return torch.tensor(edge_index).to(self.device)
    
    def _construct_str_edge_index(self, batch_x_flatten):
        batch_x_flatten = batch_x_flatten.cpu().numpy()
        neighbors = self.poi_str_neighbors[batch_x_flatten]

        edge_index = np.empty((2, 0))

        ind = len(neighbors)
        for i in range(ind):
            x = batch_x_flatten[i]
            y = neighbors[i]

            if len(y) > 0:
                y = np.array(y)
                # trick去掉y中大于57254的值
                y = y[y < args.nodes]
                x = torch.tensor([x])
                x = x.repeat(len(y))
            else:
                x = []
                y = []
            
            tmp_edge_index = np.vstack((x, y))
            edge_index = np.concatenate((edge_index, tmp_edge_index), axis=1)

        edge_index = np.unique(edge_index, axis=1)
        edge_index = edge_index.astype(int)

        return torch.tensor(edge_index).to(self.device)
    
    def _construction_feature_list(self):
        fea = torch.tensor(self.feature_list)
        fea = fea.unsqueeze(1)
        fea = fea.to(torch.float32)

        return fea.to(self.device)

    def forward(self, fps, pos=True):
        fea_x = self._construction_feature_list()
        # trick 将fea_x进行归一化操作 （如果不归一化loss为nan）
        fea_x = F.normalize(fea_x, p=2, dim=-1)
        if pos:
            batch_x, batch_x_len, _, _ = fps
            batch_x_flatten = batch_x.reshape(-1)
            batch_x_flatten = torch.unique(batch_x_flatten)
            batch_edge_index = self._construct_edge_index(batch_x_flatten)
            batch_str_edge_index = self._construct_str_edge_index(batch_x_flatten)
            embedding_weight1 = self.gat1(self.poi_features, batch_edge_index)
            # embedding_weight1 = self.poi_features + embedding_weight1
            embedding_weight2 = self.gat2(self.poi_features, batch_str_edge_index)
            # embedding_weight2 = self.poi_features + embedding_weight2
            embedding_weight3 = self.fea_module(fea_x)

            embedding_weight = torch.cat((embedding_weight1, embedding_weight2), dim=1)
            embedding_weight = self.mlp1(embedding_weight)
            embedding_weight = torch.cat((embedding_weight, embedding_weight3), dim=1)
            embedding_weight = self.mlp2(embedding_weight)
            
            batch_emb = embedding_weight[batch_x]

            batch_emb_pack = rnn_utils.pack_padded_sequence(
                batch_emb, batch_x_len, batch_first=self.batch_first)

            for lstm in self.lstm_list[:-1]:
                out_emb  = lstm(batch_emb_pack)
                out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(
                    out_emb, batch_first=self.batch_first)
                out_emb_pad = batch_emb + F.relu(out_emb_pad)
                batch_emb_pack = rnn_utils.pack_padded_sequence(
                    out_emb_pad, out_emb_len, batch_first=self.batch_first)
            out_emb  = self.lstm_list[-1](batch_emb_pack)
            out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(
                out_emb, batch_first=self.batch_first)

            idx = (torch.LongTensor(batch_x_len) - 1).view(-1, 1).expand(
                len(batch_x_len), out_emb_pad.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if out_emb_pad.is_cuda:
                idx = idx.cuda(out_emb_pad.data.get_device())
            last_output_emb = out_emb_pad.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

            traj_poi_emb = batch_emb
            poi_emb = batch_emb
        else:
            batch_n, batch_n_len, poi, batch_traj_poi = fps
            batch_n_flatten = batch_n.reshape(-1)
            batch_n_flatten = torch.unique(batch_n_flatten)
            batch_n_edge_index = self._construct_edge_index(batch_n_flatten)
            batch_str_edge_index = self._construct_str_edge_index(batch_n_flatten)
            embedding_weight1 = self.gat1(self.poi_features, batch_n_edge_index)
            embedding_weight2 = self.gat2(self.poi_features, batch_str_edge_index)
            # 将embedding_weight1和embedding_weight2进行concat
            embedding_weight = torch.cat((embedding_weight1, embedding_weight2), dim=1)
            embedding_weight = self.mlp1(embedding_weight)
            embedding_weight3 = self.fea_module(fea_x)

            embedding_weight = torch.cat((embedding_weight, embedding_weight3), dim=1)
            embedding_weight = self.mlp2(embedding_weight)
            
            batch_n_emb = embedding_weight[batch_n]

            sorted_seq_lengths, indices = torch.sort(
                torch.IntTensor(batch_n_len), descending=True)
            batch_n_emb = batch_n_emb[indices]
            _, desorted_indices = torch.sort(indices, descending=False)

            batch_emb_n_pack = rnn_utils.pack_padded_sequence(batch_n_emb,
                                                              sorted_seq_lengths,
                                                              batch_first=self.batch_first)

            for lstm in self.lstm_list[:-1]:
                out_n_emb = lstm(batch_emb_n_pack)
                out_n_emb_pad, out_n_emb_len = rnn_utils.pad_packed_sequence(
                    out_n_emb, batch_first=self.batch_first)
                out_n_emb_pad = batch_n_emb + F.relu(out_n_emb_pad)
                batch_emb_n_pack = rnn_utils.pack_padded_sequence(out_n_emb_pad,
                                                                  out_n_emb_len,
                                                                  batch_first=self.batch_first)
            out_n_emb = self.lstm_list[-1](batch_emb_n_pack)
            out_n_emb_pad, out_n_emb_len = rnn_utils.pad_packed_sequence(
                out_n_emb, batch_first=self.batch_first)

            out_n_emb_pad = out_n_emb_pad[desorted_indices]
            idx = (torch.LongTensor(batch_n_len) - 1).view(-1, 1).expand(
                len(batch_n_len), out_n_emb_pad.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if out_n_emb_pad.is_cuda:
                idx = idx.cuda(out_n_emb_pad.data.get_device())
            last_output_emb = out_n_emb_pad.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

            if batch_traj_poi is not None:
                batch_traj_poi_flatten = batch_traj_poi.reshape(-1)
                batch_traj_poi_flatten = torch.unique(batch_traj_poi_flatten)
                batch_traj_poi_edge_index = self._construct_edge_index(
                    batch_traj_poi_flatten)
                batch_traj_poi_str_edge_index = self._construct_str_edge_index(
                    batch_traj_poi_flatten)
                embedding_weight1 = self.gat1(
                    self.poi_features, batch_traj_poi_edge_index)
                embedding_weight2 = self.gat2(
                    self.poi_features, batch_traj_poi_str_edge_index)
                embedding_weight = torch.cat(
                    (embedding_weight1, embedding_weight2), dim=1)
                embedding_weight = self.mlp1(embedding_weight)
                embedding_weight3 = self.fea_module(fea_x)

                embedding_weight = torch.cat((embedding_weight, embedding_weight3), dim=1)
                embedding_weight = self.mlp2(embedding_weight)

                traj_poi_emb = embedding_weight[batch_traj_poi]
            else:
                traj_poi_emb = None

            if poi is not None:
                poi_flatten = poi.reshape(-1)
                poi_flatten = torch.unique(poi_flatten)
                poi_edge_index = self._construct_edge_index(poi_flatten)
                poi_str_edge_index = self._construct_str_edge_index(poi_flatten)
                embedding_weight1 = self.gat1(self.poi_features, poi_edge_index)
                embedding_weight2 = self.gat2(self.poi_features, poi_str_edge_index)
                embedding_weight = torch.cat((embedding_weight1, embedding_weight2), dim=1)
                embedding_weight = self.mlp1(embedding_weight)
                embedding_weight3 = self.fea_module(fea_x)

                embedding_weight = torch.cat((embedding_weight, embedding_weight3), dim=1)
                embedding_weight = self.mlp2(embedding_weight)

                poi_emb = embedding_weight[poi]
            else:
                poi_emb = None

        return last_output_emb, poi_emb, traj_poi_emb