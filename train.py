import os, sys
import torch
import time
import numpy as np
from utils import accuracy, sensitivity, specificity
from sklearn.metrics import f1_score

def train(model, train_loader, optimizer, loss_ce, epoch, path_save_info_list):
    start_time = time.time()
    model.train()
    train_epoch_cost = 0
    train_out_stack = []; train_label_stack = []

    [path_save_info] = path_save_info_list

    for i, loader in enumerate(train_loader):
        train_Node_list, train_A_list, train_label = loader

        optimizer.zero_grad()

        _, _, _, logits = model(x=train_Node_list, A=train_A_list)

        loss = loss_ce(logits, train_label)

        train_epoch_cost += loss.item()
        loss.backward()
        optimizer.step()

        # final
        train_label_stack.extend(train_label.cpu().detach().tolist())
        prob = torch.softmax(logits, dim=1)
        train_out_stack.extend(np.argmax(prob.cpu().detach().tolist(), axis=1))

    # final
    f1 = f1_score(train_label_stack, train_out_stack)

    if epoch % 10 == 0:
        print("[Train]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_epoch_cost / len(train_loader),
                     accuracy(train_out_stack, train_label_stack),
                     sensitivity(train_out_stack, train_label_stack),
                     specificity(train_out_stack, train_label_stack),
                     f1))

    with open(path_save_info.replace(".csv", "_train.csv"), "a") as f:
        f.write('{},{},{},{},{},{}\n'.format(epoch, train_epoch_cost / len(train_loader),
                                             accuracy(train_out_stack, train_label_stack),
                                             sensitivity(train_out_stack, train_label_stack),
                                             specificity(train_out_stack, train_label_stack),
                                             f1))


def test(model, test_loader, loss_ce, epoch,  path_save_info_list):
    start_time = time.time()
    test_epoch_cost = 0
    test_out_stack = []; test_label_stack = []

    [path_save_info] = path_save_info_list
    with torch.no_grad():
        model.eval()
        for i, loader in enumerate(test_loader):
            test_Node_list, test_A_list, test_label = loader

            x1, x2, out_features, logits = model(x=test_Node_list, A=test_A_list)

            loss = loss_ce(logits, test_label)

            test_epoch_cost += loss.item()

            test_label_stack.extend(test_label.cpu().detach().tolist())
            prob = torch.softmax(logits, dim=1)
            test_out_stack.extend(np.argmax(prob.cpu().detach().tolist(), axis=1))

        # final
        f1 = f1_score(test_label_stack, test_out_stack)

        if epoch % 10 == 0:

            print("[Test]   [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_epoch_cost / len(test_loader),
                         accuracy(test_out_stack, test_label_stack),
                         sensitivity(test_out_stack, test_label_stack),
                         specificity(test_out_stack, test_label_stack),
                         f1))

        with open(path_save_info.replace(".csv", f"_test.csv"), "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_epoch_cost / len(test_loader),
                                               accuracy(test_out_stack, test_label_stack),
                                               sensitivity(test_out_stack, test_label_stack),
                                               specificity(test_out_stack, test_label_stack),
                                               f1))


