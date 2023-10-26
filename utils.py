import matplotlib.pyplot as plt
import numpy as np
import torch



def accuracy(out, label):
    out = np.array(out)
    label = np.array(label)
    total = out.shape[0]
    correct = (out == label).sum().item() / total
    return correct

def sensitivity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label == 1.)
    sens = np.sum(out[mask]) / np.sum(mask)
    return sens

def specificity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label <= 1e-5)
    total = np.sum(mask)
    spec = (total - np.sum(out[mask])) / total
    return spec


def selecting_optim(args, model, lr, state=None):
    if args.optim == "Adam":
        print("optimizer: ", "Adam")
        optimizer_gcn_model = torch.optim.Adam(model.parameters(), lr=lr, betas=args.betas, weight_decay=args.weight_decay)
        if state is not None:
            optimizer_gcn_model.load_state_dict(state)
    elif args.optim == "AdamW":
        print("optimizer: ", "AdamW")
        optimizer_gcn_model = torch.optim.AdamW(model.parameters(), lr=lr, betas=args.betas,
                                                weight_decay=args.weight_decay)
        if state is not None:
            optimizer_gcn_model.load_state_dict(state)
    elif args.optim == "RAdam":
        print("optimizer: ", "RAdam")
        optimizer_gcn_model = torch.optim.RAdam(model.parameters(), lr=lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optim == "SGD":
        print("optimizer: ", "SGD")
        optimizer_gcn_model = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if state is not None:
            optimizer_gcn_model.load_state_dict(state)

    return optimizer_gcn_model
