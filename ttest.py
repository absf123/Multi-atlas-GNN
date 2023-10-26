import numpy as np
import pandas as pd
from scipy import stats
import sys
from data import get_subject_id
from scipy import io
import os, sys

def get_FC_map(txt=None, nan_fc_subject_list=None, atlas="Harvard", fold_num=1):
    data_load_path = f"Data/{atlas}/MDD_{atlas}_FC"

    data = []
    label = []
    train_weight_index = [0, 0]

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

    return [data, label, train_weight_index]


def flatten_fc(data):
    x, y = np.triu_indices(data.shape[1], k=1)
    FC_flatten = data[:, x, y]
    return FC_flatten

def flatten2dense(flatten, ROI):
    x, y = np.triu_indices(ROI, k=1)
    sym = np.zeros((ROI, ROI))
    sym[x, y] = flatten
    sym = sym + sym.T
    return sym

def t_test_topology(ROI=112, atlas="AAL"):
    nan_fc_subject_list = ['ROISignals_S20-1-0028.mat', 'ROISignals_S20-1-0038.mat', 'ROISignals_S20-1-0061.mat',
                           'ROISignals_S20-1-0094.mat', 'ROISignals_S20-1-0251.mat', 'ROISignals_S20-2-0038.mat',
                           'ROISignals_S20-2-0045.mat', 'ROISignals_S20-2-0063.mat', 'ROISignals_S20-2-0095.mat']
    for fold in range(1, 6):
        txt_train_dir = f'Data_txt_list/MDD_train_data_list_fold_' + str(fold) + '.txt'

        txt_train = pd.read_csv(txt_train_dir, names=['Subject ID'])
        [train_data, train_label, _] = get_FC_map(txt=txt_train, nan_fc_subject_list=nan_fc_subject_list, fold_num=fold)
        flatten = flatten_fc(train_data)
        MDDflatten = flatten[train_label == 1]
        NCflatten = flatten[train_label == 0]
        p_value = stats.ttest_ind(MDDflatten, NCflatten, equal_var=False).pvalue
        p_value = flatten2dense(p_value, ROI)

        np.save(f"Topology/{atlas}/ttest_pvalue/t_test_{atlas}_fold_{fold}.npy", p_value)
        print(f"atlas:{atlas} pvalue shape:{p_value.shape}")

if __name__ == "__main__":
    atlas = 'AAL'
    num_ROI = 116
    t_test_topology(ROI=num_ROI, atlas=atlas)
