import os, sys
import re
import torch
import torch.nn as nn
import pandas as pd
import time
import numpy as np
from utils import selecting_optim
from data import multi_atlas_DataLoader, multi_atlas_DataLoader_Three
from model import Single_GNN
from Config import Config
from train_Ours import train_two, test_two, train_three, test_three
from seed import set_seed


def main(args, i):
    args.timestamp = args.model_save_timestamp.strip('_')
    args.fold_num = i
    print("fold :{} device {}".format(args.fold_num, args.device))
    set_seed(args.seed)

    if args.num_atlas==2:
        train_dataset, test_dataset, train_loader, test_loader, weight = multi_atlas_DataLoader(args, Multi_atlas=args.Multi_atlas, Holistic_atlas=args.Holistic_atlas)
    elif args.num_atlas==3:
        train_dataset, test_dataset, train_loader, test_loader, weight = multi_atlas_DataLoader_Three(args, Multi_atlas=args.Multi_atlas, Holistic_atlas=args.Holistic_atlas)

    T1_model = Single_GNN(args=args, numROI=args.Multi_numROI[0], init_ch=args.Multi_numROI[0], channel=args.T1_embCh, K=args.cheb_k).to(args.device)
    T2_model = Single_GNN(args=args, numROI=args.Multi_numROI[1], init_ch=args.Multi_numROI[1], channel=args.T2_embCh, K=args.cheb_k).to(args.device)
    if args.num_atlas==3:
        T3_model = Single_GNN(args=args, numROI=args.Multi_numROI[2], init_ch=args.Multi_numROI[2], channel=args.T3_embCh, K=args.cheb_k).to(args.device)
    Hol_model = Single_GNN(args=args, numROI=args.Hol_numROI, init_ch=args.Hol_numROI, channel=args.Hol_embCh, K=args.cheb_k).to(args.device)

    # optimizer
    optimizer_T1_model = selecting_optim(args=args, model=T1_model, lr=args.lr)
    optimizer_T2_model = selecting_optim(args=args, model=T2_model, lr=args.lr)
    if args.num_atlas==3:
        optimizer_T3_model = selecting_optim(args=args, model=T3_model, lr=args.lr)
    optimizer_Hol_model = selecting_optim(args=args, model=Hol_model, lr=args.lr)

    # ExponentialLR
    scheduler_T1_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_T1_model, gamma=args.gamma)
    scheduler_T2_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_T2_model, gamma=args.gamma)
    if args.num_atlas==3:
        scheduler_T3_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_T3_model, gamma=args.gamma)
    scheduler_Hol_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_Hol_model, gamma=args.gamma)

    loss_ce = nn.CrossEntropyLoss(weight=weight)

    if args.num_atlas==2:
        model_list = [T1_model, T2_model, Hol_model]
        optimizer_list = [optimizer_T1_model, optimizer_T2_model, optimizer_Hol_model]
    elif args.num_atlas==3:
        model_list = [T1_model, T2_model, T3_model, Hol_model]
        optimizer_list = [optimizer_T1_model, optimizer_T2_model, optimizer_T3_model, optimizer_Hol_model]

    # save log
    if not (os.path.isdir(f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}')):
        os.makedirs(os.path.join(f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}'))
    path_save_info = f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}' + os.path.sep + f"train_info{args.timestamp}_{args.fold_num}.csv"

    # single results
    T1_path_save_info = f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}' + os.path.sep + "train_T1_info{}_{}.csv".format(args.timestamp,
                                                                                                args.fold_num)
    T2_path_save_info = f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}' + os.path.sep + "train_T2_info{}_{}.csv".format(args.timestamp,
                                                                                                args.fold_num)
    T3_path_save_info = f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}' + os.path.sep + "train_T3_info{}_{}.csv".format(args.timestamp,
                                                                                                args.fold_num)
    Hol_path_save_info = f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}' + os.path.sep + "train_Hol_info{}_{}.csv".format(args.timestamp,
                                                                                                args.fold_num)
    if args.num_atlas==2:
        path_save_info_list = [path_save_info, T1_path_save_info, T2_path_save_info, Hol_path_save_info]
    elif args.num_atlas==3:
        path_save_info_list = [path_save_info, T1_path_save_info, T2_path_save_info, T3_path_save_info, Hol_path_save_info]

    # total results
    with open(path_save_info, "w") as f:
        f.write("tot_loss,acc,sen,spec,f1\n")
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("tot_loss,acc,sen,spec,f1\n")

    with open(T1_path_save_info, "w") as f:
        f.write("loss,acc,sen,spec,f1\n")
    with open(T1_path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("loss,acc,sen,spec,f1\n")

    with open(T2_path_save_info, "w") as f:
        f.write("loss,acc,sen,spec,f1\n")
    with open(T2_path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("loss,acc,sen,spec,f1\n")

    if args.num_atlas==3:
        with open(T3_path_save_info, "w") as f:
            f.write("loss,acc,sen,spec,f1\n")
        with open(T3_path_save_info.replace(".csv", "_test.csv"), "w") as f:
            f.write("loss,acc,sen,spec,f1\n")

    with open(Hol_path_save_info, "w") as f:
        f.write("loss,acc,sen,spec,f1\n")
    with open(Hol_path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("loss,acc,sen,spec,f1\n")

    for epoch in range(1, args.num_epoch + 1):
        start_time = time.time()
        if args.num_atlas==2:
            # train
            train_two(model_list, train_loader, optimizer_list, loss_ce, epoch, path_save_info_list)

            # test
            if epoch % args.test_epoch_checkpoint == 0:
                test_two(model_list, test_loader, loss_ce, epoch, args, path_save_info_list)
        elif args.num_atlas == 3:
            # train
            train_three(model_list, train_loader, optimizer_list, loss_ce, epoch, path_save_info_list)

            # test
            if epoch % args.test_epoch_checkpoint == 0:
                test_three(model_list, test_loader, loss_ce, epoch, path_save_info_list)

        scheduler_T1_model.step()
        scheduler_T2_model.step()
        if args.num_atlas==3:
            scheduler_T3_model.step()
        scheduler_Hol_model.step()


def cross(args):
    # # Final
    total_result = [[0 for j in range(5)] for i in range(4)]
    T1_result = [[0 for j in range(5)] for i in range(4)]
    T2_result = [[0 for j in range(5)] for i in range(4)]
    Hol_result = [[0 for j in range(5)] for i in range(4)]

    # fold1-5
    for i in range(1, 6):
        print("cross validation start...")
        print("""========================START[!] [{}/5] validation...========================""".format(i))
        main(args, i)

        print()
        if i != 5:
            print("finish fold{}".format(i))
            print("===== NEXT fold =====")
        print()

    print("===============finish training===============")
    print("[!!] cross validation successfully complete..")
    print("validation id : [{}]".format(args.model_save_timestamp))


    print(f"model timestamp: {args.timestamp} | atlas: {args.Multi_atlas[0], args.Multi_atlas[1], args.Holistic_atlas}")
    print("="*10+"fold results"+"="*10)
    for fold in range(1, 6):
        tot_result_csv = pd.read_csv(
            f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/train_info{args.timestamp}_{fold}_test.csv')
        tot_test_result = tot_result_csv.iloc[-1]

        with open(f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_result{args.timestamp}.csv', 'a') as f:
            f.write('{},{},{},{},{}\n'.format(fold, round(tot_test_result[1], 5) * 100, round(tot_test_result[2], 5) * 100,
                                              round(tot_test_result[3], 5) * 100, round(tot_test_result[4], 5) * 100 ))
            total_result[0][fold - 1] = tot_test_result[1]
            total_result[1][fold - 1] = tot_test_result[2]
            total_result[2][fold - 1] = tot_test_result[3]
            total_result[3][fold - 1] = tot_test_result[4]

        print('fold{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(fold,
                                                          round(tot_test_result[1], 5),
                                                          round(tot_test_result[2], 5),
                                                          round(tot_test_result[3], 5),
                                                          round(tot_test_result[4], 5)))
    with open(f'results//Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_result{args.timestamp}.csv', 'a') as f:
        f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100,
                                                    np.mean(total_result[2]) * 100,np.mean(total_result[3]) * 100))
        f.write('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0], ddof=1) * 100, np.nanstd(total_result[1], ddof=1) * 100,
                                                  np.nanstd(total_result[2], ddof=1) * 100,np.nanstd(total_result[3], ddof=1) * 100))


    print("="*10+"average results"+"="*10)
    print('avg,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100,
                                                np.mean(total_result[2]) * 100, np.mean(total_result[3]) * 100))
    print('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0], ddof=1) * 100, np.nanstd(total_result[1], ddof=1) * 100,
                                                  np.nanstd(total_result[2], ddof=1) * 100,np.nanstd(total_result[3], ddof=1) * 100))

    for fold in range(1, 6):
        t1_result_csv = pd.read_csv(
            f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/train_T1_info{args.timestamp}_{fold}_test.csv')
        t1_test_result = t1_result_csv.iloc[-1]

        with open(
                f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_t1_result{args.timestamp}.csv',
                'a') as f:
            f.write('{},{},{},{},{}\n'.format(fold, round(t1_test_result[1], 5) * 100, round(t1_test_result[2], 5) * 100,
                                              round(t1_test_result[3], 5) * 100, round(t1_test_result[4], 5) * 100))
            T1_result[0][fold - 1] = t1_test_result[1]
            T1_result[1][fold - 1] = t1_test_result[2]
            T1_result[2][fold - 1] = t1_test_result[3]
            T1_result[3][fold - 1] = t1_test_result[4]

        print('fold{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(fold,
                                                          round(t1_test_result[1], 5),
                                                          round(t1_test_result[2], 5),
                                                          round(t1_test_result[3], 5),
                                                          round(t1_test_result[4], 5)))
    with open(
            f'results//Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_t1_result{args.timestamp}.csv',
            'a') as f:
        f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(T1_result[0]) * 100, np.mean(T1_result[1]) * 100,
                                                           np.mean(T1_result[2]) * 100, np.mean(T1_result[3]) * 100))
        f.write('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(T1_result[0], ddof=1) * 100, np.nanstd(T1_result[1], ddof=1) * 100,
                                                         np.nanstd(T1_result[2], ddof=1) * 100, np.nanstd(T1_result[3], ddof=1) * 100))

    print("=" * 10 + "average t1 results" + "=" * 10)
    print('avg,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(T1_result[0]) * 100, np.mean(T1_result[1]) * 100,
                                                   np.mean(T1_result[2]) * 100, np.mean(T1_result[3]) * 100))
    print('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(T1_result[0], ddof=1) * 100, np.nanstd(T1_result[1], ddof=1) * 100,
                                                   np.nanstd(T1_result[2], ddof=1) * 100, np.nanstd(T1_result[3], ddof=1) * 100))


    for fold in range(1, 6):
        t2_result_csv = pd.read_csv(
            f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/train_T2_info{args.timestamp}_{fold}_test.csv')
        t2_test_result = t2_result_csv.iloc[-1]

        with open(
                f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_t2_result{args.timestamp}.csv',
                'a') as f:
            f.write('{},{},{},{},{}\n'.format(fold, round(t2_test_result[1], 5) * 100, round(t2_test_result[2], 5) * 100,
                                              round(t2_test_result[3], 5) * 100, round(t2_test_result[4], 5) * 100))
            T2_result[0][fold - 1] = t2_test_result[1]
            T2_result[1][fold - 1] = t2_test_result[2]
            T2_result[2][fold - 1] = t2_test_result[3]
            T2_result[3][fold - 1] = t2_test_result[4]

        print('fold{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(fold,
                                                          round(t2_test_result[1], 5),
                                                          round(t2_test_result[2], 5),
                                                          round(t2_test_result[3], 5),
                                                          round(t2_test_result[4], 5)))
    with open(
            f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_t2_result{args.timestamp}.csv',
            'a') as f:
        f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(T2_result[0]) * 100, np.mean(T2_result[1]) * 100,
                                                           np.mean(T2_result[2]) * 100, np.mean(T2_result[3]) * 100))
        f.write('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(T2_result[0], ddof=1) * 100, np.nanstd(T2_result[1], ddof=1) * 100,
                                                         np.nanstd(T2_result[2], ddof=1) * 100, np.nanstd(T2_result[3], ddof=1) * 100))

    print("=" * 10 + "average t2 results" + "=" * 10)
    print('avg,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(T2_result[0]) * 100, np.mean(T2_result[1]) * 100,
                                                   np.mean(T2_result[2]) * 100, np.mean(T2_result[3]) * 100))
    print('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(T2_result[0], ddof=1) * 100, np.nanstd(T2_result[1], ddof=1) * 100,
                                                   np.nanstd(T2_result[2], ddof=1) * 100, np.nanstd(T2_result[3], ddof=1) * 100))

    for fold in range(1, 6):
        hol_result_csv = pd.read_csv(
            f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/train_Hol_info{args.timestamp}_{fold}_test.csv')
        hol_test_result = hol_result_csv.iloc[-1]

        with open(
                f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_hol_result{args.timestamp}.csv',
                'a') as f:
            f.write('{},{},{},{},{}\n'.format(fold, round(hol_test_result[1], 5) * 100, round(hol_test_result[2], 5) * 100,
                                              round(hol_test_result[3], 5) * 100, round(hol_test_result[4], 5) * 100))
            Hol_result[0][fold - 1] = hol_test_result[1]
            Hol_result[1][fold - 1] = hol_test_result[2]
            Hol_result[2][fold - 1] = hol_test_result[3]
            Hol_result[3][fold - 1] = hol_test_result[4]

        print('fold{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(fold,
                                                          round(hol_test_result[1], 5),
                                                          round(hol_test_result[2], 5),
                                                          round(hol_test_result[3], 5),
                                                          round(hol_test_result[4], 5)))
    with open(
            f'results/Multi_atlas/{args.Multi_atlas[0]}_{args.Multi_atlas[1]}_{args.Holistic_atlas}/model{args.timestamp}/cross_val_hol_result{args.timestamp}.csv',
            'a') as f:
        f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(Hol_result[0]) * 100, np.mean(Hol_result[1]) * 100,
                                                           np.mean(Hol_result[2]) * 100, np.mean(Hol_result[3]) * 100))
        f.write('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(Hol_result[0], ddof=1) * 100, np.nanstd(Hol_result[1], ddof=1) * 100,
                                                         np.nanstd(Hol_result[2], ddof=1) * 100, np.nanstd(Hol_result[3], ddof=1) * 100))

    print("=" * 10 + "average hol results" + "=" * 10)
    print('avg,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(Hol_result[0]) * 100, np.mean(Hol_result[1]) * 100,
                                                   np.mean(Hol_result[2]) * 100, np.mean(Hol_result[3]) * 100))
    print('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(Hol_result[0], ddof=1) * 100, np.nanstd(Hol_result[1], ddof=1) * 100,
                                                   np.nanstd(Hol_result[2], ddof=1) * 100, np.nanstd(Hol_result[3], ddof=1) * 100))



if __name__ == '__main__':
    start_time = time.time()
    args = Config()
    cross(args)
