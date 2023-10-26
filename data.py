from scipy import io
from scipy import sparse as sp
import numpy as np
import sys
import os
import pandas as pd
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from graph_utils import define_node_edge
from seed import set_seed



def get_subject_id(file_name):
    mat_full_name = str(file_name)  # file_name : ROISignals_S1-1-0001.mat
    file_name_label = file_name.split('-')[1]

    if os.path.splitext(mat_full_name)[1] == ".npy":
        mat_full_name = mat_full_name.replace('.npy', '.mat')

    if file_name_label == "1":
        symptom = 'MDD_Data'
        subject_ID = str(file_name[11:-4])
    elif file_name_label == "2":
        symptom = 'NC'
        subject_ID = str(file_name[11:-4])  # S1-1-0001
    else:
        print('where is symptoms!')
        sys.exit()
    return symptom, subject_ID, mat_full_name  # MDD, S1-1-0001, ROISignals_S1-1-0001.mat


# 2021.8.18
def get_FC_map(txt=None, nan_fc_subject_list=None, atlas="Harvard", fold_num=1):
    data_load_path = f"Data/{atlas}/MDD_{atlas}_FC"

    data = []
    label = []
    train_weight_index = [0, 0]

    topology = f"Topology/{atlas}/ttest_pvalue/t_test_{atlas}_fold_{fold_num}.npy"

    new_sub_list = []
    if len(nan_fc_subject_list) != 0:
        for sub_name in txt["Subject ID"]:
            if sub_name not in nan_fc_subject_list:
                new_sub_list.append(sub_name)
    else:
        for sub_name in txt["Subject ID"]:
            if sub_name in nan_fc_subject_list:
                new_sub_list.append(sub_name)

    for file_name in new_sub_list:
        symptom, subject_ID, mat_full_name = get_subject_id(file_name)  # MDD_Data, S1-1-0001, ROISignals_S1-1-0001.mat
        mat = io.loadmat(
            data_load_path + "/" + mat_full_name)
        mat = mat['ROI_Functional_connectivity']
        mat[np.isinf(mat)] = 1  # make diagonal 1
        if symptom in ['MDD_Data']:  # MDD
            label.append(1)
            train_weight_index[1] += 1
        else:  # NC
            label.append(0)
            train_weight_index[0] += 1
        data.append(mat)
    [data, label] = [np.array(data), np.array(label)]

    return [data, label, train_weight_index, topology]


def single_atlas_DataLoader(args, single_atlas):
    txt_train_dir = f'Data_txt_list/MDD_train_data_list_fold_' + str(args.fold_num) + '.txt'
    txt_test_dir = f'Data_txt_list/MDD_test_data_list_fold_' + str(args.fold_num) + '.txt'
    nan_fc_subject_list = ['ROISignals_S20-1-0028.mat', 'ROISignals_S20-1-0038.mat', 'ROISignals_S20-1-0061.mat',
                           'ROISignals_S20-1-0094.mat', 'ROISignals_S20-1-0251.mat', 'ROISignals_S20-2-0038.mat',
                           'ROISignals_S20-2-0045.mat', 'ROISignals_S20-2-0063.mat', 'ROISignals_S20-2-0095.mat']

    txt_train = pd.read_csv(txt_train_dir, names=['Subject ID'])
    txt_test = pd.read_csv(txt_test_dir, names=['Subject ID'])
    [train_data, train_label, train_weight_index, topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=single_atlas, fold_num=args.fold_num)
    [test_data, test_label, _, _] = get_FC_map(txt=txt_test, atlas=single_atlas, fold_num=args.fold_num)

    train_static_edge, \
    test_static_edge = define_node_edge(train_data=train_data, test_data=test_data, t=topology, p_value=args.p_value, edge_binary=args.edge_binary, edge_abs=args.edge_abs)

    train_Node_list = torch.FloatTensor(train_data).to(args.device)
    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)
    train_label = torch.LongTensor(train_label).to(args.device)

    test_Node_list = torch.FloatTensor(test_data).to(args.device)
    test_A_list = torch.FloatTensor(test_static_edge).to(args.device)
    test_label = torch.LongTensor(test_label).to(args.device)

    # dataloader
    train_dataset = custom_single_dataset(train_Node_list, train_A_list, train_label)
    test_dataset = custom_single_dataset(test_Node_list, test_A_list, test_label)

    set_seed(args.seed)
    if args.batch_size > len(train_dataset):
        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True)
    set_seed(args.seed)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    weight = torch.tensor([train_weight_index[1] / (train_weight_index[0] + train_weight_index[1]),
                           train_weight_index[0] / (train_weight_index[0] + train_weight_index[1])]).to(args.device)

    return train_loader, test_loader, weight


class custom_single_dataset(torch.utils.data.Dataset):
    # for single atlas
    def __init__(self, node_tensor, edge_tensor, label_tensor):
        super(custom_single_dataset, self).__init__()
        self.node = node_tensor
        self.edge = edge_tensor
        self.label = label_tensor

    def __getitem__(self, index):
        return self.node[index], self.edge[index], self.label[index]

    def __len__(self):
        return len(self.node)


def multi_atlas_DataLoader(args, Multi_atlas, Holistic_atlas):
    txt_train_dir = f'Data_txt_list/MDD_train_data_list_fold_' + str(args.fold_num) + '.txt'
    txt_test_dir = f'Data_txt_list/MDD_test_data_list_fold_' + str(args.fold_num) + '.txt'

    txt_train = pd.read_csv(txt_train_dir, names=['Subject ID'])
    txt_test = pd.read_csv(txt_test_dir, names=['Subject ID'])
    nan_fc_subject_list = ['ROISignals_S20-1-0028.mat', 'ROISignals_S20-1-0038.mat', 'ROISignals_S20-1-0061.mat',
                           'ROISignals_S20-1-0094.mat', 'ROISignals_S20-1-0251.mat', 'ROISignals_S20-2-0038.mat',
                           'ROISignals_S20-2-0045.mat', 'ROISignals_S20-2-0063.mat', 'ROISignals_S20-2-0095.mat']
    [T1_train_data, T1_train_label, T1_train_weight_index, T1_topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[0], fold_num=args.fold_num)
    [T1_test_data, T1_test_label, _, _] = get_FC_map(txt=txt_test, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[0], fold_num=args.fold_num)

    [T2_train_data, T2_train_label, T2_train_weight_index, T2_topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[1], fold_num=args.fold_num)
    [T2_test_data, T2_test_label, _, _] = get_FC_map(txt=txt_test, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[1], fold_num=args.fold_num)

    [Hol_train_data, Hol_train_label, Hol_train_weight_index, Hol_topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=Holistic_atlas, fold_num=args.fold_num)
    [Hol_test_data, Hol_test_label, _, _] = get_FC_map(txt=txt_test, nan_fc_subject_list=nan_fc_subject_list, atlas=Holistic_atlas, fold_num=args.fold_num)


    T1_train_static_edge,\
    T1_test_static_edge = define_node_edge(train_data=T1_train_data,
                                            test_data=T1_test_data, t=T1_topology, p_value=args.Multi_p_value[0], edge_binary=args.edge_binary,edge_abs=args.edge_abs)
    T2_train_static_edge,\
    T2_test_static_edge = define_node_edge(train_data=T2_train_data,
                                            test_data=T2_test_data, t=T2_topology, p_value=args.Multi_p_value[1], edge_binary=args.edge_binary,edge_abs=args.edge_abs)
    Hol_train_static_edge,\
    Hol_test_static_edge = define_node_edge(train_data=Hol_train_data,
                                            test_data=Hol_test_data, t=Hol_topology, p_value=args.Multi_p_value[2], edge_binary=args.edge_binary,edge_abs=args.edge_abs)



    T1_train_Node_list = torch.FloatTensor(T1_train_data).to(args.device)
    T1_train_A_list = torch.FloatTensor(T1_train_static_edge).to(args.device)
    T1_train_label = torch.LongTensor(T1_train_label).to(args.device)

    T1_test_Node_list = torch.FloatTensor(T1_test_data).to(args.device)
    T1_test_A_list = torch.FloatTensor(T1_test_static_edge).to(args.device)
    T1_test_label = torch.LongTensor(T1_test_label).to(args.device)

    T2_train_Node_list = torch.FloatTensor(T2_train_data).to(args.device)
    T2_train_A_list = torch.FloatTensor(T2_train_static_edge).to(args.device)
    T2_train_label = torch.LongTensor(T2_train_label).to(args.device)

    T2_test_Node_list = torch.FloatTensor(T2_test_data).to(args.device)
    T2_test_A_list = torch.FloatTensor(T2_test_static_edge).to(args.device)
    T2_test_label = torch.LongTensor(T2_test_label).to(args.device)

    Hol_train_Node_list = torch.FloatTensor(Hol_train_data).to(args.device)
    Hol_train_A_list = torch.FloatTensor(Hol_train_static_edge).to(args.device)
    Hol_train_label = torch.LongTensor(Hol_train_label).to(args.device)

    Hol_test_Node_list = torch.FloatTensor(Hol_test_data).to(args.device)
    Hol_test_A_list = torch.FloatTensor(Hol_test_static_edge).to(args.device)
    Hol_test_label = torch.LongTensor(Hol_test_label).to(args.device)

    # dataloader
    train_dataset = custom_multi_dataset(T1_train_Node_list, T1_train_A_list, T2_train_Node_list, T2_train_A_list, Hol_train_Node_list, Hol_train_A_list, T1_train_label)
    test_dataset = custom_multi_dataset(T1_test_Node_list, T1_test_A_list, T2_test_Node_list, T2_test_A_list, Hol_test_Node_list, Hol_test_A_list, T1_test_label)

    set_seed(args.seed)
    if args.batch_size > len(train_dataset):  # full batch
        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    set_seed(args.seed)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    weight = torch.tensor([T1_train_weight_index[1] / (T1_train_weight_index[0] + T1_train_weight_index[1]),
                           T1_train_weight_index[0] / (T1_train_weight_index[0] + T1_train_weight_index[1])]).to(args.device)

    return train_dataset, test_dataset, train_loader, test_loader, weight


class custom_multi_dataset(torch.utils.data.Dataset):
    """
    Multi atlas
    T1 : AAL
    T2 : Harvard
    Hol : AAL&Harvard
    """
    def __init__(self, T1_node_tensor, T1_edge_tensor,  T2_node_tensor, T2_edge_tensor,
                 Hol_node_tensor, Hol_edge_tensor, label_tensor, train_mask=None, test_mask=None):
        super(custom_multi_dataset, self).__init__()
        self.T1_x = T1_node_tensor
        self.T2_x = T2_node_tensor
        self.Hol_x = Hol_node_tensor
        self.T1_edge = T1_edge_tensor
        self.T2_edge = T2_edge_tensor
        self.Hol_edge = Hol_edge_tensor
        self.y = label_tensor
        self.train_mask = train_mask
        self.test_mask = test_mask

    def __getitem__(self, index):
        if self.train_mask is None:
            return self.T1_x[index], self.T1_edge[index], self.T2_x[index], self.T2_edge[index],\
                   self.Hol_x[index], self.Hol_edge[index], self.y[index]
        else:
            return self.T1_x[index], self.T1_edge[index], self.T2_x[index], self.T2_edge[index],\
                   self.Hol_x[index], self.Hol_edge[index], self.y[index],\
                   self.train_mask[index], self.test_mask[index]

    def __len__(self):
        return len(self.T1_x)


# 수정필요 2021.11.22 ~ 2023.01.17
def multi_atlas_DataLoader_Three(args, Multi_atlas, Holistic_atlas):
    txt_train_dir = f'Data_txt_list/MDD_train_data_list_fold_' + str(args.fold_num) + '.txt'
    txt_test_dir = f'Data_txt_list/MDD_test_data_list_fold_' + str(args.fold_num) + '.txt'

    txt_train = pd.read_csv(txt_train_dir, names=['Subject ID'])
    txt_test = pd.read_csv(txt_test_dir, names=['Subject ID'])
    nan_fc_subject_list = ['ROISignals_S20-1-0028.mat', 'ROISignals_S20-1-0038.mat', 'ROISignals_S20-1-0061.mat',
                           'ROISignals_S20-1-0094.mat', 'ROISignals_S20-1-0251.mat', 'ROISignals_S20-2-0038.mat',
                           'ROISignals_S20-2-0045.mat', 'ROISignals_S20-2-0063.mat', 'ROISignals_S20-2-0095.mat']
    [T1_train_data, T1_train_label, T1_train_weight_index, T1_topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[0], fold_num=args.fold_num)
    [T1_test_data, T1_test_label, _, _] = get_FC_map(txt=txt_test, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[0], fold_num=args.fold_num)

    [T2_train_data, T2_train_label, T2_train_weight_index, T2_topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[1], fold_num=args.fold_num)
    [T2_test_data, T2_test_label, _, _] = get_FC_map(txt=txt_test, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[1], fold_num=args.fold_num)

    [T3_train_data, T3_train_label, T3_train_weight_index, T3_topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=Multi_atlas[2], fold_num=args.fold_num)
    [T3_test_data, T3_test_label, _, _] = get_FC_map(txt=txt_test, atlas=Multi_atlas[2], nan_fc_subject_list=nan_fc_subject_list, fold_num=args.fold_num)

    [Hol_train_data, Hol_train_label, Hol_train_weight_index, Hol_topology] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, atlas=Holistic_atlas, fold_num=args.fold_num)
    [Hol_test_data, Hol_test_label, _, _] = get_FC_map(txt=txt_test, nan_fc_subject_list=nan_fc_subject_list, atlas=Holistic_atlas, fold_num=args.fold_num)


    T1_train_static_edge,\
    T1_test_static_edge = define_node_edge(train_data=T1_train_data,
                                            test_data=T1_test_data, t=T1_topology, p_value=args.Multi_p_value[0], edge_binary=args.edge_binary,edge_abs=args.edge_abs)
    T2_train_static_edge,\
    T2_test_static_edge = define_node_edge(train_data=T2_train_data,
                                            test_data=T2_test_data, t=T2_topology, p_value=args.Multi_p_value[1], edge_binary=args.edge_binary,edge_abs=args.edge_abs)
    T3_train_static_edge,\
    T3_test_static_edge = define_node_edge(train_data=T3_train_data,
                                            test_data=T3_test_data, t=T3_topology, p_value=args.Multi_p_value[2], edge_binary=args.edge_binary,edge_abs=args.edge_abs,)
    Hol_train_static_edge,\
    Hol_test_static_edge = define_node_edge(train_data=Hol_train_data,
                                            test_data=Hol_test_data, t=Hol_topology, p_value=args.Multi_p_value[3], edge_binary=args.edge_binary,edge_abs=args.edge_abs)

    T1_train_Node_list = torch.FloatTensor(T1_train_data).to(args.device)
    T1_train_A_list = torch.FloatTensor(T1_train_static_edge).to(args.device)
    T1_train_label = torch.LongTensor(T1_train_label).to(args.device)

    T1_test_Node_list = torch.FloatTensor(T1_test_data).to(args.device)
    T1_test_A_list = torch.FloatTensor(T1_test_static_edge).to(args.device)
    T1_test_label = torch.LongTensor(T1_test_label).to(args.device)

    T2_train_Node_list = torch.FloatTensor(T2_train_data).to(args.device)
    T2_train_A_list = torch.FloatTensor(T2_train_static_edge).to(args.device)
    T2_train_label = torch.LongTensor(T2_train_label).to(args.device)

    T2_test_Node_list = torch.FloatTensor(T2_test_data).to(args.device)
    T2_test_A_list = torch.FloatTensor(T2_test_static_edge).to(args.device)
    T2_test_label = torch.LongTensor(T2_test_label).to(args.device)

    T3_train_Node_list = torch.FloatTensor(T3_train_data).to(args.device)
    T3_train_A_list = torch.FloatTensor(T3_train_static_edge).to(args.device)
    T3_train_label = torch.LongTensor(T3_train_label).to(args.device)

    T3_test_Node_list = torch.FloatTensor(T3_test_data).to(args.device)
    T3_test_A_list = torch.FloatTensor(T3_test_static_edge).to(args.device)
    T3_test_label = torch.LongTensor(T3_test_label).to(args.device)

    Hol_train_Node_list = torch.FloatTensor(Hol_train_data).to(args.device)
    Hol_train_A_list = torch.FloatTensor(Hol_train_static_edge).to(args.device)
    Hol_train_label = torch.LongTensor(Hol_train_label).to(args.device)

    Hol_test_Node_list = torch.FloatTensor(Hol_test_data).to(args.device)
    Hol_test_A_list = torch.FloatTensor(Hol_test_static_edge).to(args.device)
    Hol_test_label = torch.LongTensor(Hol_test_label).to(args.device)


    # dataloader
    train_dataset = custom_multi_dataset_Three(T1_train_Node_list, T1_train_A_list, T2_train_Node_list, T2_train_A_list, T3_train_Node_list, T3_train_A_list, Hol_train_Node_list, Hol_train_A_list, T1_train_label)
    test_dataset = custom_multi_dataset_Three(T1_test_Node_list, T1_test_A_list, T2_test_Node_list, T2_test_A_list, T3_test_Node_list, T3_test_A_list, Hol_test_Node_list, Hol_test_A_list, T1_test_label)

    set_seed(args.seed)
    if args.batch_size > len(train_dataset):  # full batch
        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    set_seed(args.seed)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    weight = torch.tensor([T1_train_weight_index[1] / (T1_train_weight_index[0] + T1_train_weight_index[1]),
                           T1_train_weight_index[0] / (T1_train_weight_index[0] + T1_train_weight_index[1])]).to(args.device)

    return train_dataset, test_dataset, train_loader, test_loader, weight


class custom_multi_dataset_Three(torch.utils.data.Dataset):
    """
    Multi atlas
    T1 : AAL
    T2 : Harvard
    Hol : AAL&Harvard
    """
    def __init__(self, T1_node_tensor, T1_edge_tensor,  T2_node_tensor, T2_edge_tensor, T3_node_tensor, T3_edge_tensor,
                 Hol_node_tensor, Hol_edge_tensor, label_tensor, train_mask=None, test_mask=None):
        super(custom_multi_dataset_Three, self).__init__()
        self.T1_x = T1_node_tensor
        self.T2_x = T2_node_tensor
        self.T3_x = T3_node_tensor
        self.Hol_x = Hol_node_tensor
        self.T1_edge = T1_edge_tensor
        self.T2_edge = T2_edge_tensor
        self.T3_edge = T3_edge_tensor
        self.Hol_edge = Hol_edge_tensor
        self.y = label_tensor
        self.train_mask = train_mask
        self.test_mask = test_mask

    def __getitem__(self, index):
        if self.train_mask is None:
            return self.T1_x[index], self.T1_edge[index], self.T2_x[index], self.T2_edge[index], self.T3_x[index], self.T3_edge[index],\
                   self.Hol_x[index], self.Hol_edge[index], self.y[index]
        else:
            return self.T1_x[index], self.T1_edge[index], self.T2_x[index], self.T2_edge[index], self.T3_x[index], self.T3_edge[index],\
                   self.Hol_x[index], self.Hol_edge[index], self.y[index],\
                   self.train_mask[index], self.test_mask[index]

    def __len__(self):
        return len(self.T1_x)

