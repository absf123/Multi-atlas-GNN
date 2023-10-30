import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from graph_utils import laplacian_norm_adj, add_self_loop_cheb


import torch
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, args, numROI, init_ch, channel, K=2, bias=True):
        super(GNN, self).__init__()
        self.args = args
        self.numROI = numROI
        self.channel = channel
        self.gcn1 = myChebConv(in_features=init_ch, out_features=channel[1], K=K, bias=bias)
        self.gcn2 = myChebConv(in_features=channel[1], out_features=channel[2], K=K, bias=bias)
        self.mish = nn.Mish()
        self.fc1 = nn.Linear(self.numROI * channel[2], 1000)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.fc2 = nn.Linear(1000, 2)
        self.dropout = nn.Dropout(p=args.dropout_ratio)

    def forward(self, x, A):
        out = self.mish(self.gcn1(x=x, adj=A))
        x1 = out.clone()

        out = self.mish(self.gcn2(x=out, adj=A))
        x2 = out.clone()

        x = out.reshape(-1, self.numROI * self.channel[-1])
        out_features = F.dropout(F.mish(self.bn1(self.fc1(x))), p=self.args.dropout_ratio)
        logits = self.fc2(out_features)

        return x1, x, out_features, logits

class myChebConv(torch.nn.Module):
    def __init__(self, in_features, out_features, K=4, bias=True):
        # input
        super(myChebConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(K, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    def forward(self, x, adj):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        adj = adj.clone()
        adj = laplacian_norm_adj(adj)  
        adj = add_self_loop_cheb(-adj)
        Tx_0 = x
        Tx_1 = x
        out = torch.matmul(Tx_0, self.weight[0])
        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.shape[0] > 1:
            Tx_1 = torch.matmul(adj, x)
            out = out + torch.matmul(Tx_1, self.weight[1])
        for k in range(2, self.weight.shape[0]):
            Tx_2 = torch.matmul(adj, Tx_1)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + torch.matmul(Tx_1, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out += self.bias
        return out
    def __repr__(self):
        # print layer's structure
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

