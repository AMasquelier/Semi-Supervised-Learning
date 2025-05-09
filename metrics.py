import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score


def AUC(true, pred, verbose=False):
    num_classes = true.shape[1]
    scored_classes = (np.sum(true,axis=0) > 0)
    auc = roc_auc_score(true[:,scored_classes], pred[:,scored_classes], average='macro')
    return auc

def Error_Rate(true, pred, verbose=False):    
    return 1-np.sum(np.argmax(true, axis=1)==np.argmax(pred, axis=1))/len(true)