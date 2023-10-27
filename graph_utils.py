import numpy as np
import torch


def laplacian_norm_adj(A):
    """
    return norm adj matrix
    A` = D'^(-0.5) * A * D'^(-0.5)
    :param A: (N, V, V)
    :return: norm matrix
    """
    A = remove_self_loop(A)
    adj = graph_norm(A)
    return adj


def remove_self_loop(adj):
    """
    remove self loop
    :param adj: (N, V, V)
    :return: (N, V, V)
    """
    num_node = adj.shape[1]
    if isinstance(adj, torch.Tensor):
        x = torch.arange(0, num_node)
    else:
        x = np.arange(0, num_node)
    adj = adj.unsqueeze(0) if int(adj.ndim) == 2 else adj
    adj[:, x, x] = 0
    return adj

def graph_norm(A):
    D = A.sum(dim=-1).clamp(min=1).pow(-0.5)
    adj = D.unsqueeze(-1) * A * D.unsqueeze(-2)
    return adj

def add_self_loop_cheb(adj, fill_value: float = 1.):
    """
    add self loop to matrix A
    :param adj: (N,V,V) matrix
    :param fill_value: value of diagonal
    :return: self loop added matrix A
    """
    num_node = adj.shape[1]
    if isinstance(adj, torch.Tensor):
        x = torch.arange(0, num_node)
    else:
        x = np.arange(0, num_node)
    adj[:, x, x] = fill_value
    return adj

# p value masking
def pValueMasking(feature, t_test, p_value, binary=True, abs=True):
    data = feature.copy()  # numpy copy
    mask = t_test <= p_value
    if abs:
        data = np.abs(data)
    data = data * mask
    if binary:
        data[data != 0] = 1
    return data

def define_node_edge(train_data, test_data, t, p_value, edge_binary, edge_abs):
    train_static_edge = pValueMasking(feature=train_data, t_test=t, p_value=p_value, binary=edge_binary, abs=edge_abs)
    test_static_edge = pValueMasking(feature=test_data, t_test=t, p_value=p_value, binary=edge_binary, abs=edge_abs)

    return train_static_edge, test_static_edge
