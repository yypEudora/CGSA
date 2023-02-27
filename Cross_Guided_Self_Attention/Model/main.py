import os

import argparse
import importlib
import logging
from run import train
from evaluate import roc, evaluate, auprc
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,average_precision_score,recall_score

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # nohup python main.py --model cadgmm --KNN 5
    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomadaly Detector.')
    parser.add_argument('--dataset', nargs="?", default='pascal_sentence',#arrhythmia,cifar10,ECG,satellite,FreezerRegularTrain,grid
                        choices=['flicker8k','pascal_sentence'],
                        help='the name of the dataset you want to run the experiments on')
    parser.add_argument('--gpu', nargs="?", type=int, default=6, help='GPU device id')
    parser.add_argument('--epoch', nargs="?", type=int, default=100)
    parser.add_argument('--batch_size', nargs="?", type=int, default=5)
    parser.add_argument('--lr', nargs="?", type=float, default=0.000001)
    parser.add_argument('--seed', nargs="?", type=int, default=0)
    parser.add_argument('--weight_decay', nargs="?", type=float, default=0.1)

    train(parser.parse_args())

