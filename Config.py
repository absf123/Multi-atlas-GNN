import argparse
# from argparse import ArgumentParser
import torch
from datetime import datetime
import re

class Config():
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='Argparse single GCN')
        timestamp = datetime.today().strftime("%Y%m%d%H%M%S")

        # pytorch base
        parser.add_argument("--cuda_num", default=1, type=str, help="0~5")
        parser.add_argument("--device")

        # training hyperparams
        parser.add_argument('--seed', type=int, default=100, help='random seed')
        parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
        parser.add_argument("--batch_size", default=16, type=int, help="batch size")
        parser.add_argument('--num_epoch', default=200, type=int, help='num_epoch')
        parser.add_argument('--test_epoch_checkpoint', default=10, type=int, help='step of test function')
        parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay')
        parser.add_argument('--optim', default="Adam", type=str, help='optimizer')
        parser.add_argument('--betas', default=(0.5, 0.9), type=tuple, help='adam betas')
        parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum - SGD, MADGRAD")
        parser.add_argument("--gamma", default=0.995, type=float, help="gamma for lr learning")
        parser.add_argument("--info_gain_weight", default=[0.522, 0.478], type=list, help="info gain weight")
        parser.add_argument("--p_value", default=0.05, type=float, help="AAL Harvard AALHarvard")
        parser.add_argument("--edge_binary", default=True, type=bool, help="0, 1")
        parser.add_argument("--edge_abs", default=True, type=bool, help="-,+ -> +")
        parser.add_argument("--Multi_p_value", default=[0.05, 0.05, 0.05], type=list, help="AAL Harvard AALHarvard")
        parser.add_argument("--cheb_k", default=2, type=int, help="ChebGCN k for combine GCN")
        parser.add_argument("--Holistic_atlas", default="AH",
                            type=str, help="atlas AH 228 AC 316 AP 380 HC 312 HP 376 CP 464 ")
        parser.add_argument("--Multi_atlas", default=["AAL", "Harvard"], type=list, help="T1, T2, T3")
        parser.add_argument("--Single_atlas", default="AAL", type=str, help="T1, T2, T3")
        parser.add_argument("--num_atlas", default=2, type=int, help="2, 3")

        parser.add_argument("--dropout_ratio", default=0.0, type=float, help="fc layer dropout")

        parser.add_argument("--embCh", default=16, type=int, help="")

        parser.add_argument("--Hol_numROI", default=228, type=int,
                            help="AH : 228, AC : 316, HC : 312"
                                 "and num nodes")
        parser.add_argument("--Multi_numROI", default=[116, 112], type=list,
                            help="Multi atlas nodes")
        parser.add_argument("--Single_numROI", default=116, type=list,
                            help="single atlas nodes")
        # source file
        parser.add_argument('--fold_num', default=1, type=int, help='num of fold txt')

        # save & load
        parser.add_argument("--save", default=False, type=bool, help="saving model.pth")
        parser.add_argument("--model_save_overwrite", default=False, type=bool, help="")
        parser.add_argument("--model_save_timestamp", default=timestamp if not parser.parse_args().model_save_overwrite else "", type=str, help="")


        self.args = parser.parse_args()
        self.cuda_num = self.args.cuda_num
        self.device = torch.device("cuda:{}".format(self.cuda_num) if torch.cuda.is_available() else "cpu")

        self.seed = self.args.seed
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size
        self.num_epoch = self.args.num_epoch
        self.test_epoch_checkpoint = self.args.test_epoch_checkpoint
        self.weight_decay = self.args.weight_decay
        self.optim = self.args.optim
        self.betas = self.args.betas
        self.momentum = self.args.momentum
        self.gamma = self.args.gamma
        self.info_gain_weight = self.args.info_gain_weight
        self.p_value = self.args.p_value
        self.edge_binary = self.args.edge_binary
        self.edge_abs = self.args.edge_abs
        self.Multi_p_value = self.args.Multi_p_value
        self.cheb_k = self.args.cheb_k
        self.Holistic_atlas = self.args.Holistic_atlas
        self.Multi_atlas = self.args.Multi_atlas
        self.Single_atlas = self.args.Single_atlas
        self.num_atlas = self.args.num_atlas
        self.dropout_ratio = self.args.dropout_ratio
        self.embCh = f"[{self.args.embCh}, {self.args.embCh}, {self.args.embCh}]"
        self.Single_embCh = list(map(int, re.findall("\d+", str(self.embCh))))
        self.T1_embCh = list(map(int, re.findall("\d+", str(self.embCh))))
        self.T2_embCh = list(map(int, re.findall("\d+", str(self.embCh))))
        self.T3_embCh = list(map(int, re.findall("\d+", str(self.embCh))))
        self.Hol_embCh = list(map(int, re.findall("\d+", str(self.embCh))))
        self.Hol_numROI = self.args.Hol_numROI
        self.Multi_numROI = self.args.Multi_numROI
        self.Single_numROI = self.args.Single_numROI

        self.fold_num = self.args.fold_num
        self.save = self.args.save
        self.model_save_overwrite = self.args.model_save_overwrite
        self.model_save_timestamp = self.args.model_save_timestamp

