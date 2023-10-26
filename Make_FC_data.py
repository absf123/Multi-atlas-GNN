import numpy as np
import pandas as pd
from scipy import io
import os, sys

import time

def read_txt():
    with open(f"/Data_txt_list/MDD_RoI_signal_list.txt", 'r') as f:
            f_names = f.read().splitlines()
    return f_names

def make_fc_map(f_names, atlas="AAL"):

    data_save_dir = 'Data/ROISignals_FunImgARCWF_static_FCmap/MDD_{}_FC_data'.format(atlas)

    if not (os.path.isdir(data_save_dir)):
        os.mkdir(data_save_dir)

    print("atlas: ", atlas)
    for f_name in f_names:
        mat_file = io.loadmat('Data/ROISignals_FunImgARCWF/{}'.format(f_name))
        mat = mat_file['ROISignals']  # signal, node
        # single
        if atlas == "AAL":
            mat = mat[:, :116]   # AAL 116 ROI
        elif atlas == "Harvard":
            mat = mat[:, 116:228]  # Harvard Oxford 112 ROI
        elif atlas == "Craddock":
            mat = mat[:, 228:428]   # 200 ROI

        # Holistic atlas: early fusion
        elif atlas == "AH":  # AAL, Harvard 228
            mat = mat[:, :228]
        elif atlas == "AC":  # AAL, Craddock 316
            mat1 = mat[:, :116]
            mat = np.concatenate([mat1, mat[:, 228:428]], axis=1)
        elif atlas == "HC":  # Harvard, Craddock 312
            mat = mat[:, 116:428]


        mat_transpose = np.array(mat).transpose()

        roi_dict = {}
        for i in range(len(mat_transpose)):
            single_roi = mat_transpose[i]
            roi_dict["{}".format(i)] = single_roi

        roi_df = pd.DataFrame(roi_dict)

        # correlation
        corr_df = roi_df.corr(method='pearson')

        corr_matrix = np.array(corr_df, dtype=[('ROI_Functional_connectivity', 'float64')])

        # numpy파일 mat변환
        corr_matrix_dict = {}

        # list로 변환하고 mat 저장
        for varname in corr_matrix.dtype.names:
            corr_matrix_dict[varname] = corr_matrix[varname]

        # functional connectivity matrix
        io.savemat('Data/ROISignals_FunImgARCWF_static_FCmap/MDD_{}_FC_data/{}'.format(atlas, f_name), corr_matrix_dict)

    return f"Data_{atlas}_FC_data"


def fisher_z_transformation(FC_data_fold="", atlas="AAL"):

    timestamp = FC_data_fold[:14]
    print("fisher_z, {}, {}".format(timestamp, atlas))
    data_save_dir = 'Data/ROISignals_FunImgARCWF_FCmap_fisher_z/MDD_{}_FC_fisher_ztrans_data_inf'.format(atlas)
    if not (os.path.isdir(data_save_dir)):
        os.mkdir(data_save_dir)
    path_dir = 'Data/ROISignals_FunImgARCWF_FCmap/{}'.format(FC_data_fold)

    f_names = os.listdir(path_dir)
    print(len(f_names))


    for f_name in f_names:
        corr_matrix = io.loadmat('Data/ROISignals_FunImgARCWF_FCmap/{}/{}'.format(FC_data_fold, f_name))

        corr_matrix = corr_matrix['ROI_Functional_connectivity']
        corr_matrix = np.array(corr_matrix)

        for row in range(len(corr_matrix)):
            for col in range(len(corr_matrix)):
                rho = corr_matrix[row][col]
                corr_matrix[row][col] = (1.0 / 2.0) * (np.log((1.0 + rho) / (1.0 - rho)))


        corr_matrix_dict = {}
        corr_matrix = np.array(corr_matrix, dtype=[('ROI_Functional_connectivity', 'float64')])

        for varname in corr_matrix.dtype.names:
            corr_matrix_dict[varname] = corr_matrix[varname]

        io.savemat(f"Data/{atlas}/MDD_{atlas}_FC/{f_name}",  corr_matrix_dict)

if __name__ == "__main__":

    f_names = read_txt()

    atlas = 'AAL'
    start_time = time.time()
    FC_data_fold = make_fc_map(f_names, atlas=atlas)
    print(FC_data_fold)

    fisher_z_transformation(FC_data_fold=FC_data_fold, atlas=atlas)

    print("time: ", time.time() - start_time)