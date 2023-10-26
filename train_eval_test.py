import os, sys
import torch
import time
import numpy as np
from utils import accuracy, sensitivity, specificity
from sklearn.metrics import f1_score


def train_two(model_list, train_loader, optimizer_list, loss_ce, epoch, path_save_info_list):
    start_time = time.time()
    for i in range(len(model_list)):
        model_list[i].train()
    train_epoch_cost = 0; train_T1_ce_cost = 0; train_T2_ce_cost = 0
    train_out_stack = []; train_label_stack = []

    # branch network evaluation
    train_T1_out_stack = []; train_T1_label_stack = []
    train_T2_out_stack = []; train_T2_label_stack = []

    [path_save_info, T1_path_save_info, T2_path_save_info] = path_save_info_list

    for i, loader in enumerate(train_loader):
        T1_train_Node_list, T1_train_A_list, \
        T2_train_Node_list, T2_train_A_list, \
        _, _, train_label = loader

        for i in range(len(optimizer_list)):
            optimizer_list[i].zero_grad()

        _, _, _, T1_logits = model_list[0](x=T1_train_Node_list, A=T1_train_A_list)
        _, _, _, T2_logits = model_list[1](x=T2_train_Node_list, A=T2_train_A_list)

        T1_ce_loss = loss_ce(T1_logits, train_label)
        T2_ce_loss = loss_ce(T2_logits, train_label)

        loss = T1_ce_loss + T2_ce_loss

        train_T1_ce_cost += T1_ce_loss.item()
        train_T2_ce_cost += T2_ce_loss.item()
        train_epoch_cost += loss.item()
        loss.backward()

        for i in range(len(optimizer_list)):
            optimizer_list[i].step()

        train_label_stack.extend(train_label.cpu().detach().tolist())
        T1_prob = torch.softmax(T1_logits, dim=1)
        T2_prob = torch.softmax(T2_logits, dim=1)
        train_out_stack.extend(np.argmax(((T1_prob + T2_prob) / 2).cpu().detach().tolist(), axis=1))

        train_T1_label_stack.extend(train_label.cpu().detach().tolist())
        train_T1_out_stack.extend(np.argmax(T1_prob.cpu().detach().tolist(), axis=1))
        train_T2_label_stack.extend(train_label.cpu().detach().tolist())
        train_T2_out_stack.extend(np.argmax(T2_prob.cpu().detach().tolist(), axis=1))

    # final
    f1 = f1_score(train_label_stack, train_out_stack)
    T1_f1 = f1_score(train_T1_label_stack, train_T1_out_stack)
    T2_f1 = f1_score(train_T2_label_stack, train_T2_out_stack)

    if epoch % 10 == 0:
        print("[Train Total]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_epoch_cost / len(train_loader),
                     accuracy(train_out_stack, train_label_stack),
                     sensitivity(train_out_stack, train_label_stack),
                     specificity(train_out_stack, train_label_stack),
                     f1))
        print("[Train    T1]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_T1_ce_cost / len(train_loader),
                     accuracy(train_T1_out_stack, train_T1_label_stack),
                     sensitivity(train_T1_out_stack, train_T1_label_stack),
                     specificity(train_T1_out_stack, train_T1_label_stack),
                     T1_f1))
        print("[Train    T2]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_T2_ce_cost / len(train_loader),
                     accuracy(train_T2_out_stack, train_T2_label_stack),
                     sensitivity(train_T2_out_stack, train_T2_label_stack),
                     specificity(train_T2_out_stack, train_T2_label_stack),
                     T2_f1))


    # training checkpoint
    with open(path_save_info.replace(".csv", "_train.csv"), "a") as f:
        f.write(
            "{:},{:},{:},{:},{:},{:}\n".format(epoch, train_epoch_cost / len(train_loader),
                                               accuracy(train_out_stack, train_label_stack),
                                               sensitivity(train_out_stack, train_label_stack),
                                               specificity(train_out_stack, train_label_stack),
                                               f1))
    with open(T1_path_save_info.replace(".csv", "_train.csv"), "a") as f:
        f.write(
            "{:},{:},{:},{:},{:},{:}\n".format(epoch, train_T1_ce_cost / len(train_loader),
                                               accuracy(train_T1_out_stack, train_T1_label_stack),
                                               sensitivity(train_T1_out_stack, train_T1_label_stack),
                                               specificity(train_T1_out_stack, train_T1_label_stack),
                                               T1_f1))
    with open(T2_path_save_info.replace(".csv", "_train.csv"), "a") as f:
        f.write(
            "{:},{:},{:},{:},{:},{:}\n".format(epoch, train_T2_ce_cost / len(train_loader),
                                               accuracy(train_T2_out_stack, train_T2_label_stack),
                                               sensitivity(train_T2_out_stack, train_T2_label_stack),
                                               specificity(train_T2_out_stack, train_T2_label_stack),
                                               T2_f1))


def test_two(model_list, test_loader, loss_ce, epoch, path_save_info_list):
    start_time = time.time()
    test_epoch_cost = 0; test_T1_ce_cost = 0; test_T2_ce_cost = 0
    test_out_stack = []; test_label_stack = []

    # branch network evaluation
    test_T1_out_stack = []; test_T1_label_stack = []
    test_T2_out_stack = []; test_T2_label_stack = []

    [path_save_info, T1_path_save_info, T2_path_save_info] = path_save_info_list
    with torch.no_grad():
        for i in range(len(model_list)):
            model_list[i].eval()
        for i, loader in enumerate(test_loader):
            T1_test_Node_list, T1_test_A_list, \
            T2_test_Node_list, T2_test_A_list, \
            _, _, test_label = loader

            T1_x1, T1_x2, T1_out_features, T1_logits = model_list[0](x=T1_test_Node_list, A=T1_test_A_list)
            T2_x1, T2_x2, T2_out_features, T2_logits = model_list[1](x=T2_test_Node_list, A=T2_test_A_list)

            T1_ce_loss = loss_ce(T1_logits, test_label)
            T2_ce_loss = loss_ce(T2_logits, test_label)

            loss = T1_ce_loss + T2_ce_loss

            test_T1_ce_cost += T1_ce_loss.item()
            test_T2_ce_cost += T2_ce_loss.item()
            test_epoch_cost += loss.item()

            # final
            test_label_stack.extend(test_label.cpu().detach().tolist())
            T1_prob = torch.softmax(T1_logits, dim=1)
            T2_prob = torch.softmax(T2_logits, dim=1)
            test_out_stack.extend(np.argmax(((T1_prob + T2_prob) / 2).cpu().detach().tolist(), axis=1))
            test_T1_label_stack.extend(test_label.cpu().detach().tolist())
            test_T1_out_stack.extend(np.argmax(T1_prob.cpu().detach().tolist(), axis=1))
            test_T2_label_stack.extend(test_label.cpu().detach().tolist())
            test_T2_out_stack.extend(np.argmax(T2_prob.cpu().detach().tolist(), axis=1))

        # final
        f1 = f1_score(test_label_stack, test_out_stack)
        T1_f1 = f1_score(test_T1_label_stack, test_T1_out_stack)
        T2_f1 = f1_score(test_T2_label_stack, test_T2_out_stack)

        if epoch % 10 == 0:
            print("[Test  Total]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_epoch_cost / len(test_loader),
                         accuracy(test_out_stack, test_label_stack),
                         sensitivity(test_out_stack, test_label_stack),
                         specificity(test_out_stack, test_label_stack),
                         f1))
            print("[Test     T1]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_T1_ce_cost / len(test_loader),
                         accuracy(test_T1_out_stack, test_T1_label_stack),
                         sensitivity(test_T1_out_stack, test_T1_label_stack),
                         specificity(test_T1_out_stack, test_T1_label_stack),
                         T1_f1))
            print("[Test     T2]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_T2_ce_cost / len(test_loader),
                         accuracy(test_T2_out_stack, test_T2_label_stack),
                         sensitivity(test_T2_out_stack, test_T2_label_stack),
                         specificity(test_T2_out_stack, test_T2_label_stack),
                         T2_f1))
        # testation checkpoint : 아직 구현x
        with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_epoch_cost / len(test_loader),
                                                   accuracy(test_out_stack, test_label_stack),
                                                   sensitivity(test_out_stack, test_label_stack),
                                                   specificity(test_out_stack, test_label_stack),
                                                   f1))

        with open(T1_path_save_info.replace(".csv", "_test.csv"), "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_T1_ce_cost / len(test_loader),
                                                   accuracy(test_T1_out_stack, test_T1_label_stack),
                                                   sensitivity(test_T1_out_stack, test_T1_label_stack),
                                                   specificity(test_T1_out_stack, test_T1_label_stack),
                                                   T1_f1))

        with open(T2_path_save_info.replace(".csv", "_test.csv"), "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_T2_ce_cost / len(test_loader),
                                                   accuracy(test_T2_out_stack, test_T2_label_stack),
                                                   sensitivity(test_T2_out_stack, test_T2_label_stack),
                                                   specificity(test_T2_out_stack, test_T2_label_stack),
                                                   T2_f1))

def train_three(model_list, train_loader, optimizer_list, loss_ce, epoch, path_save_info_list):
    start_time = time.time()
    for i in range(len(model_list)):
        model_list[i].train()
    train_epoch_cost = 0; train_T1_ce_cost = 0; train_T2_ce_cost = 0; train_T3_ce_cost = 0
    train_out_stack = []; train_label_stack = []

    # branch network evaluation
    train_T1_out_stack = []; train_T1_label_stack = []
    train_T2_out_stack = []; train_T2_label_stack = []
    train_T3_out_stack = []; train_T3_label_stack = []

    [path_save_info, T1_path_save_info, T2_path_save_info, T3_path_save_info] = path_save_info_list

    for i, loader in enumerate(train_loader):
        T1_train_Node_list, T1_train_A_list, \
        T2_train_Node_list, T2_train_A_list, \
        T3_train_Node_list, T3_train_A_list, \
        _, _, train_label = loader

        for i in range(len(optimizer_list)):
            optimizer_list[i].zero_grad()

        _, _, _, T1_logits = model_list[0](x=T1_train_Node_list, A=T1_train_A_list)
        _, _, _, T2_logits = model_list[1](x=T2_train_Node_list, A=T2_train_A_list)
        _, _, _, T3_logits = model_list[2](x=T3_train_Node_list, A=T3_train_A_list)

        T1_ce_loss = loss_ce(T1_logits, train_label)
        T2_ce_loss = loss_ce(T2_logits, train_label)
        T3_ce_loss = loss_ce(T3_logits, train_label)

        loss = T1_ce_loss + T2_ce_loss + T3_ce_loss

        train_T1_ce_cost += T1_ce_loss.item()
        train_T2_ce_cost += T2_ce_loss.item()
        train_T3_ce_cost += T3_ce_loss.item()
        train_epoch_cost += loss.item()
        loss.backward()
        for i in range(len(optimizer_list)):
            optimizer_list[i].step()

        train_label_stack.extend(train_label.cpu().detach().tolist())
        T1_prob = torch.softmax(T1_logits, dim=1)
        T2_prob = torch.softmax(T2_logits, dim=1)
        T3_prob = torch.softmax(T3_logits, dim=1)
        train_out_stack.extend(np.argmax(((T1_prob + T2_prob + T3_prob) / 3).cpu().detach().tolist(), axis=1))

        train_T1_label_stack.extend(train_label.cpu().detach().tolist())
        train_T1_out_stack.extend(np.argmax(T1_prob.cpu().detach().tolist(), axis=1))
        train_T2_label_stack.extend(train_label.cpu().detach().tolist())
        train_T2_out_stack.extend(np.argmax(T2_prob.cpu().detach().tolist(), axis=1))
        train_T3_label_stack.extend(train_label.cpu().detach().tolist())
        train_T3_out_stack.extend(np.argmax(T3_prob.cpu().detach().tolist(), axis=1))

    # final
    f1 = f1_score(train_label_stack, train_out_stack)
    T1_f1 = f1_score(train_T1_label_stack, train_T1_out_stack)
    T2_f1 = f1_score(train_T2_label_stack, train_T2_out_stack)
    T3_f1 = f1_score(train_T3_label_stack, train_T3_out_stack)

    if epoch % 10 == 0:
        print("[Train Total]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_epoch_cost / len(train_loader),
                     accuracy(train_out_stack, train_label_stack),
                     sensitivity(train_out_stack, train_label_stack),
                     specificity(train_out_stack, train_label_stack),
                     f1))
        print("[Train    T1]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_T1_ce_cost / len(train_loader),
                     accuracy(train_T1_out_stack, train_T1_label_stack),
                     sensitivity(train_T1_out_stack, train_T1_label_stack),
                     specificity(train_T1_out_stack, train_T1_label_stack),
                     T1_f1))
        print("[Train    T2]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_T2_ce_cost / len(train_loader),
                     accuracy(train_T2_out_stack, train_T2_label_stack),
                     sensitivity(train_T2_out_stack, train_T2_label_stack),
                     specificity(train_T2_out_stack, train_T2_label_stack),
                     T2_f1))
        print("[Train    T3]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
              format(epoch, time.time() - start_time, train_T3_ce_cost / len(train_loader),
                     accuracy(train_T3_out_stack, train_T3_label_stack),
                     sensitivity(train_T3_out_stack, train_T3_label_stack),
                     specificity(train_T3_out_stack, train_T3_label_stack),
                     T3_f1))

        # training checkpoint
    with open(path_save_info.replace(".csv", "_train.csv"), "a") as f:
        f.write(
            "{:},{:},{:},{:},{:},{:}\n".format(epoch, train_epoch_cost / len(train_loader),
                                               accuracy(train_out_stack, train_label_stack),
                                               sensitivity(train_out_stack, train_label_stack),
                                               specificity(train_out_stack, train_label_stack),
                                               f1))
    with open(T1_path_save_info.replace(".csv", "_train.csv"), "a") as f:
        f.write(
            "{:},{:},{:},{:},{:},{:}\n".format(epoch, train_T1_ce_cost / len(train_loader),
                                               accuracy(train_T1_out_stack, train_T1_label_stack),
                                               sensitivity(train_T1_out_stack, train_T1_label_stack),
                                               specificity(train_T1_out_stack, train_T1_label_stack),
                                               T1_f1))
    with open(T2_path_save_info.replace(".csv", "_train.csv"), "a") as f:
        f.write(
            "{:},{:},{:},{:},{:},{:}\n".format(epoch, train_T2_ce_cost / len(train_loader),
                                               accuracy(train_T2_out_stack, train_T2_label_stack),
                                               sensitivity(train_T2_out_stack, train_T2_label_stack),
                                               specificity(train_T2_out_stack, train_T2_label_stack),
                                               T2_f1))

    with open(T3_path_save_info, "a") as f:
        f.write(
            "{:},{:},{:},{:},{:},{:}\n".format(epoch, train_T3_ce_cost / len(train_loader),
                                               accuracy(train_T3_out_stack, train_T3_label_stack),
                                               sensitivity(train_T3_out_stack, train_T3_label_stack),
                                               specificity(train_T3_out_stack, train_T3_label_stack),
                                               T3_f1))


def test_three(model_list, test_loader, loss_ce, epoch, path_save_info_list):
    start_time = time.time()
    test_epoch_cost = 0; test_T1_ce_cost = 0; test_T2_ce_cost = 0; test_T3_ce_cost = 0
    test_out_stack = []; test_label_stack = []

    # branch network evaluation
    test_T1_out_stack = []; test_T1_label_stack = []
    test_T2_out_stack = []; test_T2_label_stack = []
    test_T3_out_stack = []; test_T3_label_stack = []

    [path_save_info, T1_path_save_info, T2_path_save_info, T3_path_save_info] = path_save_info_list
    with torch.no_grad():
        for i in range(len(model_list)):
            model_list[i].eval()
        for i, loader in enumerate(test_loader):
            T1_test_Node_list, T1_test_A_list, \
            T2_test_Node_list, T2_test_A_list, \
            T3_test_Node_list, T3_test_A_list, \
            _, _, test_label = loader

            _, _, _, T1_logits = model_list[0](x=T1_test_Node_list, A=T1_test_A_list)
            _, _, _, T2_logits = model_list[1](x=T2_test_Node_list, A=T2_test_A_list)
            _, _, _, T3_logits = model_list[2](x=T3_test_Node_list, A=T3_test_A_list)

            T1_ce_loss = loss_ce(T1_logits, test_label)
            T2_ce_loss = loss_ce(T2_logits, test_label)
            T3_ce_loss = loss_ce(T3_logits, test_label)

            loss = T1_ce_loss + T2_ce_loss + T3_ce_loss

            test_T1_ce_cost += T1_ce_loss.item()
            test_T2_ce_cost += T2_ce_loss.item()
            test_T3_ce_cost += T3_ce_loss.item()
            test_epoch_cost += loss.item()

            test_label_stack.extend(test_label.cpu().detach().tolist())
            T1_prob = torch.softmax(T1_logits, dim=1)
            T2_prob = torch.softmax(T2_logits, dim=1)
            T3_prob = torch.softmax(T3_logits, dim=1)
            test_out_stack.extend(np.argmax(((T1_prob + T2_prob + T3_prob) / 3).cpu().detach().tolist(), axis=1))

            test_T1_label_stack.extend(test_label.cpu().detach().tolist())
            test_T1_out_stack.extend(np.argmax(T1_prob.cpu().detach().tolist(), axis=1))
            test_T2_label_stack.extend(test_label.cpu().detach().tolist())
            test_T2_out_stack.extend(np.argmax(T2_prob.cpu().detach().tolist(), axis=1))
            test_T3_label_stack.extend(test_label.cpu().detach().tolist())
            test_T3_out_stack.extend(np.argmax(T3_prob.cpu().detach().tolist(), axis=1))
        # final
        f1 = f1_score(test_label_stack, test_out_stack)
        T1_f1 = f1_score(test_T1_label_stack, test_T1_out_stack)
        T2_f1 = f1_score(test_T2_label_stack, test_T2_out_stack)
        T3_f1 = f1_score(test_T3_label_stack, test_T3_out_stack)
        if epoch % 10 == 0:
            print("[Test  Total]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_epoch_cost / len(test_loader),
                         accuracy(test_out_stack, test_label_stack),
                         sensitivity(test_out_stack, test_label_stack),
                         specificity(test_out_stack, test_label_stack),
                         f1))
            print("[Test     T1]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_T1_ce_cost / len(test_loader),
                         accuracy(test_T1_out_stack, test_T1_label_stack),
                         sensitivity(test_T1_out_stack, test_T1_label_stack),
                         specificity(test_T1_out_stack, test_T1_label_stack),
                         T1_f1))
            print("[Test     T2]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_T2_ce_cost / len(test_loader),
                         accuracy(test_T2_out_stack, test_T2_label_stack),
                         sensitivity(test_T2_out_stack, test_T2_label_stack),
                         specificity(test_T2_out_stack, test_T2_label_stack),
                         T2_f1))
            print("[Test     T3]  [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}] [F1 C:{:.4f}]".
                  format(epoch, time.time() - start_time, test_T3_ce_cost / len(test_loader),
                         accuracy(test_T3_out_stack, test_T3_label_stack),
                         sensitivity(test_T3_out_stack, test_T3_label_stack),
                         specificity(test_T3_out_stack, test_T3_label_stack),
                         T3_f1))

        with open(path_save_info.replace(".csv", f"_test.csv"), "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_epoch_cost / len(test_loader),
                                                   accuracy(test_out_stack, test_label_stack),
                                                   sensitivity(test_out_stack, test_label_stack),
                                                   specificity(test_out_stack, test_label_stack),
                                                   f1))

        with open(T1_path_save_info.replace(".csv", f"_test.csv"), "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_T1_ce_cost / len(test_loader),
                                                   accuracy(test_T1_out_stack, test_T1_label_stack),
                                                   sensitivity(test_T1_out_stack, test_T1_label_stack),
                                                   specificity(test_T1_out_stack, test_T1_label_stack),
                                                   T1_f1))

        with open(T2_path_save_info.replace(".csv", f"_test.csv"), "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_T2_ce_cost / len(test_loader),
                                                   accuracy(test_T2_out_stack, test_T2_label_stack),
                                                   sensitivity(test_T2_out_stack, test_T2_label_stack),
                                                   specificity(test_T2_out_stack, test_T2_label_stack),
                                                   T2_f1))
        with open(T3_path_save_info, "a") as f:
            f.write(
                "{:},{:},{:},{:},{:},{:}\n".format(epoch, test_T3_ce_cost / len(test_loader),
                                                   accuracy(test_T3_out_stack, test_T3_label_stack),
                                                   sensitivity(test_T3_out_stack, test_T3_label_stack),
                                                   specificity(test_T3_out_stack, test_T3_label_stack),
                                                   T3_f1))

