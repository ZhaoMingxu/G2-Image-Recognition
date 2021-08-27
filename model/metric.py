import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# out and target are tensor
def score(output,target):
    output=output.cpu().detach().numpy()
    target=target.cpu().detach().numpy()
    target = np.eye(output.shape[1])[target]
    try:
        s=roc_auc_score(target,output,average='macro',multi_class='ovo')
    except ValueError:
        s=0.
    return s 