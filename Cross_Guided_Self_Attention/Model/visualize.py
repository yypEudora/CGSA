# -*- coding: utf-8 -*-
# @Time    : 2019/3/7 23:46
# @Author  : Haoyi Fan
# @Email   : isfanhy@gmail.com
# @File    : visualize.py

# coding: utf-8
import matplotlib
import torch

matplotlib.use('Agg')

import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from itertools import product
from tqdm import tqdm
import numpy as np
import os

def create_logdir(model_name, dataset, K, KNN, l1, l2, l3):
    """ Directory to save training logs, weights, biases, etc."""
    return "../train_logs/{}/{}_KNN{}_K{}_v0.0_l1{}_l2{}_l3{}".format(model_name, dataset, KNN, K, l1, l2, l3)

def create_logdir_D(model_name, dataset, K, KNN, l1, l2, l3):
    """ Directory to save training logs, weights, biases, etc."""
    return "../train_logs/{}/{}_KNN{}_K{}_conn0.5_l1{}_l2{}_l3{}".format(model_name, dataset, KNN, K, l1, l2, l3)

def create_logdir_oc(model_name, dataset):
    """ Directory to save training logs, weights, biases, etc."""
    return "../train_logs/{}/nu{}".format(model_name, dataset)

def create_logdir_alad():
    """ Directory to save training logs, weights, biases, etc."""
    return "./draw/alad_False_dzz0_dzzenabled0_False_label0.0_rd"

def create_logdir_ano():
    """ Directory to save training logs, weights, biases, etc."""
    return "./draw/label0_weight0.1_rd"

# model_name='KNN-DyGraph'
model_name = 'CGSA'
# model_name='alad'

# ['kdd','arrhythmia', 'satellite']
# dataset='arrhythmia'
dataset = 'pascal_sentence'
# K=7#arr
# K=4
K=2
# KNNs=[-1]
KNNs=[13]#arr
# KNNs=[-1]
l1=0.1
l2=0.005
l3=0.005

# l1=1
# l2=1
# l3=1

PRs={}
RCs={}
F1s={}
Max_Epochs={}

# EPOCHS=np.arange(200,300,20)
EPOCHS=[20]
SEEDS=[0,1,6,9,10]

PLOT_2D = True

for KNN in KNNs:
    # save_dir = create_logdir(model_name, dataset, K, KNN, l1, l2, l3)
    # save_dir = create_logdir_oc(model_name, dataset)


    # save_dir = create_logdir_alad(model_name, dataset,nb_exp)
    save_dir = create_logdir_ano()
    logdir = os.path.sep.join([save_dir])

    # vis_dir = os.path.sep.join(['../figure', 'embed_vis2', model_name,
    #                             "{}_KNN{}_K{}_v0.0_l1{}_l2{}_l3{}/{}".format(dataset, KNN, K, l1, l2, l3, nb_exp)])
    vis_dir = './figure'

    # for t in [37000, 38000, 40000, 42000]:

    for epoch in EPOCHS:
        fig_name = '{}_{}_embeddings'.format(model_name,dataset)
        #[:t]
        embeddings1 = np.load('./results/pascal_sentence/embeddings.npy')
        labels1 = np.load('./results/pascal_sentence/labels.npy')
        labels1 = np.array(labels1, np.int_)
        labels2 = labels1
        labels3 = labels1

        labels = np.concatenate([labels1,labels2],axis=0)

        embeddings1 = torch.Tensor(embeddings1)

        num1 = embeddings1.shape[0]
        num2 = embeddings1.shape[2]
        embeddings1 = torch.reshape(embeddings1, (num1, num2))

        embeddings2 = embeddings1

        embeddings = torch.cat([embeddings1, embeddings2], dim=0)

        if len(embeddings.shape) != 2:
            continue

        pos = TSNE(n_components=20).fit_transform(embeddings)
        df = pd.DataFrame()
        df['x'] = pos[:, 0]
        df['y'] = pos[:, 1]
        # df['z'] = pos[:, 2]
        legends = list(range(10000))
        df['class'] = [legends[l] for l in labels]

        if PLOT_2D:
            print(1)
            sns.set_context("notebook", font_scale=1.5)
            sns.set_style("ticks")

            # Create scatterplot of dataframe
            sns.lmplot('x',  # Horizontal axis
                       'y',  # Vertical axis
                       data=df,  # Data source
                       fit_reg=False,  # Don't fix a regression line
                       hue="class",  # Set color,
                       legend=False,
                       scatter_kws={"s": 10, 'alpha': 0.8})  # S marker size#0.8

            sns.despine(top=True, left=True, right=True, bottom=True)
        else:

            ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
            ax.scatter(
                xs=df["x"],
                ys=df["y"],
                zs=df["z"],
                c=df["y"],
                cmap='tab10'
            )
            ax.set_xlabel('pca-one')
            ax.set_ylabel('pca-two')
            ax.set_zlabel('pca-three')
            plt.show()

        plt.xticks([])
        plt.yticks([])

        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()
        # plt.savefig(vis_dir+'/' +fig_name+'_{}_.png'.format(t), bbox_inches='tight')
        plt.savefig(vis_dir+'/' +fig_name+'.png', bbox_inches='tight')
        # plt.savefig(vis_dir+'/' +fig_name+'_{}_.pdf'.format(t), bbox_inches='tight')
        plt.savefig(vis_dir+'/' +fig_name+'.pdf', bbox_inches='tight')
        # print(vis_dir+'/' +fig_name+'_{}_.png'.format(t))



