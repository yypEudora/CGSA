from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import precision_recall_fscore_support as prf, roc_auc_score, average_precision_score, \
    precision_recall_fscore_support
import numpy



def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    elif metric == 'precision':
        thresh = 0.20
        scores=scores.numpy()
        labels=labels.numpy()
        y_pred = (scores >= thresh).astype(int)
        y_true = labels.astype(int)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        return precision
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, save=False):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap
