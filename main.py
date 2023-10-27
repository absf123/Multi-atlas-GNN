import os, sys
import re
import torch
import torch.nn as nn
import pandas as pd
import time
import numpy as np
import utils
from utils import selecting_optim
from data import single_atlas_DataLoader
from model import Single_GNN
from Config import Config
from train import train, test
from seed import set_seed



def main(args, i):
    args.timestamp = args.model_save_timestamp.strip('_')
    args.fold_num = i
    print("fold :{} device {}".format(args.fold_num, args.device))
    set_seed(args.seed)

    train_loader, test_loader, weight = single_atlas_DataLoader(args=args, single_atlas=args.Single_atlas)

    set_seed(args.seed)
    ## ChebNet
    model = Single_GNN(args=args, numROI=args.numROI, init_ch=args.numROI, channel=args.embCh, K=args.cheb_k).to(args.device)

    # optimizer
    optimizer_model = selecting_optim(args=args, model=model, lr=args.lr)
    # ExponentialLR
    scheduler_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_model, gamma=args.gamma)

    loss_ce = nn.CrossEntropyLoss(weight=weight)

    # save log
    if not (os.path.isdir(f'results/single/{args.Single_atlas}/model{args.timestamp}')):
        os.makedirs(os.path.join(f'results/single/{args.Single_atlas}/model{args.timestamp}'))
    path_save_info = f'results/single/{args.Single_atlas}/model{args.timestamp}' + os.path.sep + f"train_info{args.timestamp}_{args.fold_num}.csv"

    path_save_info_list = [path_save_info]
    # total results
    with open(path_save_info, "w") as f:
        f.write("loss,acc,sen,spec,f1\n")
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("loss,acc,sen,spec,f1\n")

    for epoch in range(1, args.num_epoch + 1):
        start_time = time.time()
        train(model, train_loader, optimizer_model, loss_ce, epoch, path_save_info_list)

        # test
        if epoch % args.test_epoch_checkpoint == 0:
            test(model, test_loader, loss_ce, epoch, path_save_info_list)

        scheduler_model.step()


def cross(args):
    # Final
    total_result = [[0 for j in range(5)] for i in range(4)]
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


    print(f"model timestamp: {args.timestamp} | atlas: {args.Single_atlas}")
    print("="*10+"fold results"+"="*10)
    for fold in range(1, 6):
        result_csv = pd.read_csv(
            f'results/single/{args.Single_atlas}/model{args.timestamp}/train_info{args.timestamp}_{fold}_test.csv')
        test_result = result_csv.iloc[-1]
        with open(f'results/single/{args.Single_atlas}/model{args.timestamp}/cross_val_result{args.timestamp}.csv', 'a') as f:
            f.write('{},{},{},{},{}\n'.format(fold, round(test_result[1], 5) * 100, round(test_result[2], 5) * 100,
                                              round(test_result[3], 5) * 100, round(test_result[4], 5) * 100 ))
            total_result[0][fold - 1] = test_result[1]
            total_result[1][fold - 1] = test_result[2]
            total_result[2][fold - 1] = test_result[3]
            total_result[3][fold - 1] = test_result[4]

        print('fold{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(fold,
                                                          round(test_result[1], 5),
                                                          round(test_result[2], 5),
                                                          round(test_result[3], 5),
                                                          round(test_result[4], 5)))
    with open(f'results/single/{args.Single_atlas}/model{args.timestamp}/cross_val_result{args.timestamp}.csv', 'a') as f:
        f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100,
                                                    np.mean(total_result[2]) * 100,np.mean(total_result[3]) * 100))
        f.write('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0], ddof=1) * 100, np.nanstd(total_result[1], ddof=1) * 100,
                                                  np.nanstd(total_result[2], ddof=1) * 100,np.nanstd(total_result[3], ddof=1) * 100))


    print("="*10+"average results"+"="*10)
    print('avg,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100,
                                                np.mean(total_result[2]) * 100, np.mean(total_result[3]) * 100))
    print('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0], ddof=1) * 100, np.nanstd(total_result[1], ddof=1) * 100,
                                                  np.nanstd(total_result[2], ddof=1) * 100,np.nanstd(total_result[3], ddof=1) * 100))



if __name__ == '__main__':
    start_time = time.time()
    args = Config()
    cross(args)
