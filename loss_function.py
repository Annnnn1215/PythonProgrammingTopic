import torch
import torch.nn as nn


def one_hot(x, num_classes):
    return torch.eye(num_classes, dtype=torch.float32)[x, :]


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))
