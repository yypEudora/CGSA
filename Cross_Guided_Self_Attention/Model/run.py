# clas以前的
import datetime

import torch
import model_net
import random
# from faster_rcnn import predict
from average_pooling_usr import average_pooling_usr
import self_attention
import sys
sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# sys.path.append('Cross_Guided_Self_Attention/Text')
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,average_precision_score,recall_score
import Text.text_bert as text_bert
import cross_attention
import torch.nn as nn
import numpy as np
from transformers import logging
from dataloader import loader_train,loader_test
import os
from sklearn.utils import shuffle

logging.set_verbosity_warning()
logging.set_verbosity_error()

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

name = "bicycle"

def train(args):
    epoch = args.epoch
    batch_size=args.batch_size

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_net.Model_Net(args)
    # net.to(device)



    # label_all=label_all.float()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()
    list_k=['1000_01', '3-1', '1000']
    # list_k = ['l1', 'l2', 'l3']

    log_dir = './results/{}'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_time = str(datetime.datetime.now()).replace(':', '_')
    current_time = current_time.replace('.', '_')
    file2print = open("{}/test {}.log".format(log_dir, current_time), 'a+')

    print('batch size = {}\t\tlr = {}\n'.format(args.batch_size,args.lr), file=file2print)

    max_acc = 0

    for i in range(epoch):
        # img_all, img_all_shape, text_all, label_all = shuffle(img_all, img_all_shape, text_all, label_all,
        #                                                       random_state=i)
        print(datetime.datetime.now(), file=file2print)
        print("Epoch\tAccuracy\tRecall\tF1", file=file2print)
        print("Epoch: {}==============================================================================".format(i))
        file2print.flush()
        # for train_id,k in enumerate(list_k):

        # print('哈哈哈_{}'.format(k))
        # print(img_all.shape)
        # print(img_all_shape.shape)
        # print(text_all.shape)
        # print(label_all.shape)
        text_all, label_all = loader_train()
        n_batch = int(text_all.shape[0] / batch_size)

        random.seed(i)
        random.shuffle(text_all)
        random.seed(i)
        random.shuffle(label_all)

        #train
        for batch_idx in range(n_batch):
            batch_text = text_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_label = label_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_label=torch.Tensor(batch_label.tolist()).float()
            ypre1, _ = net(batch_text)
            loss = loss_func(ypre1, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            len1 = len(text_all)

            # print('batch_idx: {}\t batch_label: {}'.format(batch_idx,(torch.max(batch_label.data, 1)[1]).data.numpy()))
            # print('Epoch:{} [{}/{}({:.0f}%)]\t train_data{}\t Loss:{:.6f}\t batch_idx: {}\t batch_label: {}'.format(i, batch_idx * len(batch_img),
            #                                                                          len1,
            #                                                                          100. * batch_idx / n_batch,
            #                                                                          train_id,
            #                                                                          loss.data.item(),
            #                                                                          batch_idx,(torch.max(batch_label.data, 1)[1]).data.numpy()))

            print('Epoch:{} [{}/{}({:.0f}%)]\t Loss:{:.6f}\t batch_idx: {}\t batch_label: {}'.format(i, batch_idx * len(text_all),
                                                                                     len1,
                                                                                     100. * batch_idx / n_batch,
                                                                                     loss.data.item(),
                                                                                     batch_idx,(torch.max(batch_label.data, 1)[1]).data.numpy()))


        del text_all
        del label_all
        del n_batch
        del batch_text
        del batch_label
        del len1

        print('---------------------------------------------------------------------------------------------------------')
        # test--batch
        print('testing--------------------------------------------------------------------------------------------------')
        correct = 0
        recall = 0
        f1 = 0
        acc = 0
        # img_all_test, img_all_shape_test, text_all_test, label_all_test = loader_test()
        # print(img_all_test.shape)
        # print(img_all_shape_test.shape)
        # print(text_all_test.shape)
        # print(label_all_test.shape)
        # test_batch = int(img_all_test.shape[0] / batch_size)

        rr = []

        text_all_test, label_all_test = loader_test()
        test_batch = int(text_all_test.shape[0] / batch_size)

        # random.seed(i)
        # random.shuffle(img_all_test)
        # random.seed(i)
        # random.shuffle(img_all_shape_test)
        # random.seed(i)
        # random.shuffle(text_all_test)
        # random.seed(i)
        # random.shuffle(label_all_test)

        for test_batch_idx in range(test_batch):

            batch_text_test = text_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
            batch_label_test = label_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
            batch_label_test = torch.Tensor(batch_label_test.tolist()).float()
            batch_ypre_all_test, ypre_cat_all_test = net(batch_text_test)
            predicted = torch.max(batch_ypre_all_test.data, 1)[1]
            batch_label_ = torch.max(batch_label_test.data, 1)[1]
            correct += (predicted == batch_label_).sum()

            recall += recall_score(batch_label_.data.numpy(), predicted.data.numpy(), average='macro')
            f1 += f1_score(batch_label_.data.numpy(), predicted.data.numpy(), average='macro')
            acc += accuracy_score(batch_label_.data.numpy(), predicted.data.numpy())
            #ap = average_precision_score(y_true, predicted)
            # print('Accuracy:{:.3f}\t F1:{:.3f}\t Recall:{:.3f}'.format(float(correct) / (float(args.batch_size) * (test_batch))))
            # print('--epoch: {}\t test_all: {}\t train_data{}\t test_idx: {}\t Acc: {:.3f}\t pre: {}\t true: {}'.format(i,test_batch,train_id,test_batch_idx+1,float(correct) / (float(args.batch_size) * (test_batch_idx + 1)),predicted.data.numpy(),batch_label_.data.numpy()))
            print('--epoch: {}\t test_all: {}\t test_idx: {}\t Acc: {:.3f}\t pre: {}\t true: {}'.format(i,test_batch,test_batch_idx+1,float(correct) / (float(args.batch_size) * (test_batch_idx + 1)),predicted.data.numpy(),batch_label_.data.numpy()))
            print('Accuracy:{:.4f}\t F1:{:.4f}\t Recall:{:.4f}'.format(acc / (test_batch_idx + 1),
                                                                       f1 / (test_batch_idx + 1),
                                                                       recall / (test_batch_idx + 1)))

            rr.append(ypre_cat_all_test)
        print("\t{}\t{:.4f}\t\t{:.4f}\t{:.4f}".format(i, acc/test_batch, recall/test_batch, f1/test_batch), file=file2print)
        file2print.flush()


        del text_all_test
        del label_all_test
        del batch_text_test
        del batch_label_test
        del test_batch

        # if acc/test_batch >= max_acc:
        #     max_acc = acc/test_batch
        #     result_dir = './results/rr/test'
        #     if not os.path.exists(log_dir):
        #         os.makedirs(log_dir)
        #
        #     g = 0
        #     for re in rr:
        #         for j in range(args.batch_size):
        #             np.save(result_dir+'/{}.npy'.format(g), re[j].detach().numpy())
        #             g += 1
        #
        #     file2print1 = open(result_dir+'/log.log', 'a+')
        #
        #     print('epoch = {}'.format(i), file=file2print1)
        #     file2print1.flush()
        #     file2print1.close()





# # clas以前的
# import datetime
#
# import torch
# import model_net
# import random
# # from faster_rcnn import predict
# from average_pooling_usr import average_pooling_usr
# import self_attention
# import sys
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# # sys.path.append('Cross_Guided_Self_Attention/Text')
# from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,average_precision_score,recall_score
# import Text.text_bert as text_bert
# import cross_attention
# import torch.nn as nn
# import numpy as np
# from transformers import logging
# from dataloader import loader_train,loader_test
# import os
# from sklearn.utils import shuffle
#
# logging.set_verbosity_warning()
# logging.set_verbosity_error()
#
# # import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#
# name = "bicycle"
#
# def train(args):
#     epoch = args.epoch
#     batch_size=args.batch_size
#
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = model_net.Model_Net(args)
#     # net.to(device)
#
#
#
#     img_all_test, img_all_shape_test, text_all_test, label_all_test = loader_test()
#     test_batch = int(img_all_test.shape[0] / batch_size)
#     # test_batch = 5
#
#     # label_all=label_all.float()
#     optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
#     loss_func = torch.nn.CrossEntropyLoss()
#     list_k=['1000_01', '3-1', '1000']
#     # list_k = ['l1', 'l2', 'l3']
#
#     log_dir = './results/{}/{}'.format(args.dataset, name)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
#     current_time = str(datetime.datetime.now()).replace(':', '_')
#     current_time = current_time.replace('.', '_')
#     file2print = open("{}/test {}.log".format(log_dir, current_time), 'a+')
#
#     print('batch size = {}\t\tlr = {}\n'.format(args.batch_size,args.lr), file=file2print)
#
#     max_acc = 0
#
#     for i in range(epoch):
#         # img_all, img_all_shape, text_all, label_all = shuffle(img_all, img_all_shape, text_all, label_all,
#         #                                                       random_state=i)
#         print(datetime.datetime.now(), file=file2print)
#         print("Epoch\tAccuracy\tRecall\tF1", file=file2print)
#         print("Epoch: {}==============================================================================".format(i))
#         file2print.flush()
#         # for train_id,k in enumerate(list_k):
#
#         # print('哈哈哈_{}'.format(k))
#         # print(img_all.shape)
#         # print(img_all_shape.shape)
#         # print(text_all.shape)
#         # print(label_all.shape)
#         img_all, img_all_shape, text_all, label_all = loader_train('train')
#         n_batch = int(img_all.shape[0] / batch_size)
#
#         random.seed(i)
#         random.shuffle(img_all)
#         random.seed(i)
#         random.shuffle(img_all_shape)
#         random.seed(i)
#         random.shuffle(text_all)
#         random.seed(i)
#         random.shuffle(label_all)
#
#         #train
#         for batch_idx in range(n_batch):
#             batch_img = img_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
#             batch_text = text_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
#             batch_label = label_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
#             batch_img_shape = img_all_shape[batch_idx * batch_size:(batch_idx + 1) * batch_size]
#             batch_label=torch.Tensor(batch_label.tolist()).float()
#             ypre1, ypre2, ypre3, ypre_all, _ = net(batch_img, batch_img_shape, batch_text)
#             loss1 = loss_func(ypre1, batch_label)
#             loss2 = loss_func(ypre2, batch_label)
#             loss3 = loss_func(ypre3, batch_label)
#             loss4 = loss_func(ypre_all, batch_label)
#             loss = loss1+loss2+loss3+loss4
#             # loss = loss1
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             len1 = len(img_all)
#
#             # print('batch_idx: {}\t batch_label: {}'.format(batch_idx,(torch.max(batch_label.data, 1)[1]).data.numpy()))
#             # print('Epoch:{} [{}/{}({:.0f}%)]\t train_data{}\t Loss:{:.6f}\t batch_idx: {}\t batch_label: {}'.format(i, batch_idx * len(batch_img),
#             #                                                                          len1,
#             #                                                                          100. * batch_idx / n_batch,
#             #                                                                          train_id,
#             #                                                                          loss.data.item(),
#             #                                                                          batch_idx,(torch.max(batch_label.data, 1)[1]).data.numpy()))
#
#             print('Epoch:{} [{}/{}({:.0f}%)]\t Loss:{:.6f}\t batch_idx: {}\t batch_label: {}'.format(i, batch_idx * len(batch_img),
#                                                                                      len1,
#                                                                                      100. * batch_idx / n_batch,
#                                                                                      loss.data.item(),
#                                                                                      batch_idx,(torch.max(batch_label.data, 1)[1]).data.numpy()))
#
#         del img_all
#         del img_all_shape
#         del text_all
#         del label_all
#         del n_batch
#         del batch_img
#         del batch_text
#         del batch_label
#         del batch_img_shape
#         del len1
#
#         print('---------------------------------------------------------------------------------------------------------')
#         # test--batch
#         print('testing--------------------------------------------------------------------------------------------------')
#         correct = 0
#         recall = 0
#         f1 = 0
#         acc = 0
#         # img_all_test, img_all_shape_test, text_all_test, label_all_test = loader_test()
#         # print(img_all_test.shape)
#         # print(img_all_shape_test.shape)
#         # print(text_all_test.shape)
#         # print(label_all_test.shape)
#         # test_batch = int(img_all_test.shape[0] / batch_size)
#
#         rr = []
#
#         for test_batch_idx in range(test_batch):
#             batch_img_test = img_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
#             batch_text_test = text_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
#             batch_label_test = label_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
#             batch_img_shape_test = img_all_shape_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
#             batch_label_test = torch.Tensor(batch_label_test.tolist()).float()
#             _, _, _, batch_ypre_all_test, ypre_cat_all_test = net(batch_img_test, batch_img_shape_test, batch_text_test)
#             predicted = torch.max(batch_ypre_all_test.data, 1)[1]
#             batch_label_ = torch.max(batch_label_test.data, 1)[1]
#             correct += (predicted == batch_label_).sum()
#
#             recall += recall_score(batch_label_.data.numpy(), predicted.data.numpy())
#             f1 += f1_score(batch_label_.data.numpy(), predicted.data.numpy())
#             acc += accuracy_score(batch_label_.data.numpy(), predicted.data.numpy())
#             #ap = average_precision_score(y_true, predicted)
#             # print('Accuracy:{:.3f}\t F1:{:.3f}\t Recall:{:.3f}'.format(float(correct) / (float(args.batch_size) * (test_batch))))
#             # print('--epoch: {}\t test_all: {}\t train_data{}\t test_idx: {}\t Acc: {:.3f}\t pre: {}\t true: {}'.format(i,test_batch,train_id,test_batch_idx+1,float(correct) / (float(args.batch_size) * (test_batch_idx + 1)),predicted.data.numpy(),batch_label_.data.numpy()))
#             print('--epoch: {}\t test_all: {}\t test_idx: {}\t Acc: {:.3f}\t pre: {}\t true: {}'.format(i,test_batch,test_batch_idx+1,float(correct) / (float(args.batch_size) * (test_batch_idx + 1)),predicted.data.numpy(),batch_label_.data.numpy()))
#             print('Accuracy:{:.4f}\t F1:{:.4f}\t Recall:{:.4f}'.format(acc / (test_batch_idx + 1),
#                                                                        f1 / (test_batch_idx + 1),
#                                                                        recall / (test_batch_idx + 1)))
#
#             rr.append(ypre_cat_all_test)
#         print("\t{}\t{:.4f}\t\t{:.4f}\t{:.4f}".format(i, acc/test_batch, recall/test_batch, f1/test_batch), file=file2print)
#         file2print.flush()
#
#         if acc/test_batch >= max_acc:
#             max_acc = acc/test_batch
#             result_dir = './results/{}/{}/test'.format(args.dataset, name, i)
#             if not os.path.exists(log_dir):
#                 os.makedirs(log_dir)
#
#             g = 0
#             for re in rr:
#                 for j in range(args.batch_size):
#                     np.save(result_dir+'/{}.npy'.format(g), re[j].detach().numpy())
#                     g += 1
#
#             file2print1 = open(result_dir+'/log.log', 'a+')
#
#             print('epoch = {}'.format(i), file=file2print1)
#             file2print1.flush()
#             file2print1.close()




















# import datetime
#
# import torch
# import model_net
# import random
# import sys
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,average_precision_score,recall_score
# from transformers import logging
# from dataloader import loader_train,loader_test
# import os
#
# logging.set_verbosity_warning()
# logging.set_verbosity_error()
#
#
# def train(args):
#     epoch = args.epoch
#     batch_size=args.batch_size
#
#     net = model_net.Model_Net(args)
#
#     clases_all_test, text_all_test, label_all_test = loader_test()
#     test_batch = int(clases_all_test.shape[0] / batch_size)
#
#     # label_all=label_all.float()
#     optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
#     loss_func = torch.nn.CrossEntropyLoss()
#     list_k=['1000_01', '3-1', '1000']
#     # list_k = ['l1', 'l2', 'l3']
#
#     log_dir = './results/{}'.format(args.dataset)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
#     current_time = str(datetime.datetime.now()).replace(':', '_')
#     current_time = current_time.replace('.', '_')
#     file2print = open("{}/test {}.log".format(log_dir, current_time), 'a+')
#
#     print('batch size = {}\t\tlr = {}\n'.format(args.batch_size,args.lr), file=file2print)
#
#     for i in range(epoch):
#         print(datetime.datetime.now(), file=file2print)
#         print("Epoch\tAccuracy\tRecall\tF1", file=file2print)
#         print("Epoch: {}==============================================================================".format(i))
#         file2print.flush()
#
#         clases_all, text_all, label_all = loader_train('train')
#         n_batch = int(clases_all.shape[0] / batch_size)
#
#         random.seed(i)
#         random.shuffle(clases_all)
#         random.seed(i)
#         random.shuffle(text_all)
#         random.seed(i)
#         random.shuffle(label_all)
#
#         #train
#         for batch_idx in range(n_batch):
#             batch_clas = clases_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
#             batch_text = text_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
#             batch_label = label_all[batch_idx * batch_size:(batch_idx + 1) * batch_size]
#             batch_label=torch.Tensor(batch_label.tolist()).float()
#             # ypre1, ypre2, ypre3, ypre_all = net(batch_clas, batch_text)
#             ypre1 = net(batch_clas, batch_text).float()
#             loss1 = loss_func(ypre1, batch_label)
#             # loss2 = loss_func(ypre2, batch_label)
#             # loss3 = loss_func(ypre3, batch_label)
#             # loss4 = loss_func(ypre_all, batch_label)
#             # loss = loss1+loss2+loss3+loss4
#             loss = loss1
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             len1 = len(clases_all)
#
#             print('Epoch:{} [{}/{}({:.0f}%)]\t Loss:{:.6f}\t batch_idx: {}\t batch_label: {}'.format(i, batch_idx * len(batch_clas),
#                                                                                      len1,
#                                                                                      100. * batch_idx / n_batch,
#                                                                                      loss.data.item(),
#                                                                                      batch_idx,(torch.max(batch_label.data, 1)[1]).data.numpy()))
#
#         del clases_all
#         del text_all
#         del label_all
#         del n_batch
#         del batch_clas
#         del batch_text
#         del batch_label
#         del len1
#
#         print('---------------------------------------------------------------------------------------------------------')
#         # test--batch
#         print('testing--------------------------------------------------------------------------------------------------')
#         correct = 0
#         recall = 0
#         f1_macro = 0
#         f1_weighted = 0
#         acc = 0
#
#
#         for test_batch_idx in range(test_batch):
#             batch_clases_test = clases_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
#             batch_text_test = text_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
#             batch_label_test = label_all_test[test_batch_idx * batch_size:(test_batch_idx + 1) * batch_size]
#             batch_label_test = torch.Tensor(batch_label_test.tolist()).float()
#             # _, _, _, batch_ypre_all_test = net(batch_clases_test, batch_text_test)
#             batch_ypre_all_test = net(batch_clases_test, batch_text_test).float()
#             predicted = torch.max(batch_ypre_all_test.data, 1)[1]
#             batch_label_ = torch.max(batch_label_test.data, 1)[1]
#             correct += (predicted == batch_label_).sum()
#
#             recall += recall_score(batch_label_.data.numpy(), predicted.data.numpy(), average='macro')
#             f1_macro += f1_score(batch_label_.data.numpy(), predicted.data.numpy(), average='macro')
#             f1_weighted += f1_score(batch_label_.data.numpy(), predicted.data.numpy(), average='weighted')
#             acc += accuracy_score(batch_label_.data.numpy(), predicted.data.numpy())
#             #ap = average_precision_score(y_true, predicted)
#             # print('Accuracy:{:.3f}\t F1:{:.3f}\t Recall:{:.3f}'.format(float(correct) / (float(args.batch_size) * (test_batch))))
#             # print('--epoch: {}\t test_all: {}\t train_data{}\t test_idx: {}\t Acc: {:.3f}\t pre: {}\t true: {}'.format(i,test_batch,train_id,test_batch_idx+1,float(correct) / (float(args.batch_size) * (test_batch_idx + 1)),predicted.data.numpy(),batch_label_.data.numpy()))
#             print('--epoch: {}\t test_all: {}\t test_idx: {}\t Acc: {:.3f}\t pre: {}\t true: {}'.format(i,test_batch,test_batch_idx+1,float(correct) / (float(args.batch_size) * (test_batch_idx + 1)),predicted.data.numpy(),batch_label_.data.numpy()))
#             # print('--epoch: {}\t test_all: {}\t test_idx: {}\t pre: {}\t true: {}'.format(i,test_batch,test_batch_idx+1, predicted.data.numpy(),batch_label_.data.numpy()))
#
#             print('Accuracy:{:.4f}\t F1_Macro:{:.4f}\t F1_Weighted:{:.4f}\t Recall:{:.4f}'.format(acc / (test_batch_idx + 1),
#                                                                        f1_macro / (test_batch_idx + 1),
#                                                                        f1_weighted / (test_batch_idx + 1),
#                                                                        recall / (test_batch_idx + 1)))
#         print("\t{}\t{:.4f}\t\t{:.4f}\t{:.4f}\t{:.4f}".format(i, acc/test_batch, recall/test_batch, f1_macro/test_batch, f1_weighted/test_batch), file=file2print)
#         file2print.flush()














