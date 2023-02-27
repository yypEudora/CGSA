# # clas以前的
# import sys
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# import numpy as np
# from transformers import logging
#
# logging.set_verbosity_warning()
# logging.set_verbosity_error()
#
# name = 'train'
#
# image_test_dir = "./data/pascal_sentence_data/crop_imgs_test.npy"
# image_test_shape_dir = "./data/pascal_sentence_data/crop_imgs_shape_test.npy"
# text_test_dir = "./data/pascal_sentence_data/text_test.npy"
# label_test_dir1 = "./data/pascal_sentence_data/label_test.npy"
#
#
# def loader_train(k):
#     image_dir = "./data/pascal_sentence_data/crop_imgs_{}.npy".format(k)
#     image_shape_dir = "./data/pascal_sentence_data/crop_imgs_shape_{}.npy".format(k)
#     text_dir = "./data/pascal_sentence_data/text_{}.npy".format(k)
#     label_dir1 = "./data/pascal_sentence_data/label_{}.npy".format(k)
#
#
#     img=np.load(image_dir,allow_pickle=True)
#     text=np.load(text_dir,allow_pickle=True)
#     label=np.load(label_dir1,allow_pickle=True)
#     shape=np.load(image_shape_dir,allow_pickle=True)
#
#     return img,shape,text,label
#
# def loader_test():
#     img=np.load(image_test_dir,allow_pickle=True)
#     text=np.load(text_test_dir,allow_pickle=True)
#     label=np.load(label_test_dir1,allow_pickle=True)
#     shape = np.load(image_test_shape_dir,allow_pickle=True)
#     return img,shape,text,label



# clas以前的
import sys
sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
import numpy as np
from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

name = 'train'

text_test_dir = "./data/pascal_sentence_data/test/text_test.npy"
label_test_dir1 = "./data/pascal_sentence_data/test/label_test.npy"


def loader_train():
    text_dir = "./data/pascal_sentence_data/test/text_train.npy"
    label_dir1 = "./data/pascal_sentence_data/test/label_train.npy"


    text=np.load(text_dir,allow_pickle=True)
    label=np.load(label_dir1,allow_pickle=True)

    return text,label

def loader_test():
    text=np.load(text_test_dir,allow_pickle=True)
    label=np.load(label_test_dir1,allow_pickle=True)
    return text,label








# import torch
# import model_net
# # from faster_rcnn import predict
# # from average_pooling import average_pooling
# import self_attention
# import sys
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# # sys.path.append('Cross_Guided_Self_Attention/Text')
# import Text.text_bert as text_bert
# import cross_attention
# import torch.nn as nn
# import numpy as np
# from transformers import logging
#
# logging.set_verbosity_warning()
# logging.set_verbosity_error()
#
#
#
#
# # image_test_dir = "./data/crop_imgs_01_test1.npy"
# # image_test_shape_dir = "./data/crop_imgs_shape_01_test1.npy"
# # text_test_dir = "./data/text_01_test1.npy"
# # label_test_dir1 = "./data/label_01_test1.npy"
#
# # k='l2'
# # image_test_dir = "./data/crop_imgs_01_{}.npy".format(k)
# # image_test_shape_dir = "./data/crop_imgs_shape_01_{}.npy".format(k)
# # text_test_dir = "./data/text_01_{}.npy".format(k)
# # label_test_dir1 = "./data/label_01_{}.npy".format(k)
#
# image_test_dir = "./data/test/crop_imgs_test.npy"
# image_test_shape_dir = "./data/test/crop_imgs_shape_test.npy"
# text_test_dir = "./data/test/text_test.npy"
# label_test_dir1 = "./data/test/label_test.npy"
#
#
# def loader_train(k):
#     image_dir = "./data/test/crop_imgs_{}.npy".format(k)
#     image_shape_dir = "./data/test/crop_imgs_shape_{}.npy".format(k)
#     text_dir = "./data/test/text_{}.npy".format(k)
#     label_dir1 = "./data/test/label_{}.npy".format(k)
#
#     # image_dir = "./data/crop_imgs_01_{}.npy".format(k)
#     # image_shape_dir = "./data/crop_imgs_shape_01_{}.npy".format(k)
#     # text_dir = "./data/text_01_{}.npy".format(k)
#     # label_dir1 = "./data/label_01_{}.npy".format(k)
#
#     img=np.load(image_dir,allow_pickle=True)
#     text=np.load(text_dir,allow_pickle=True)
#     label=np.load(label_dir1,allow_pickle=True)
#     shape=np.load(image_shape_dir,allow_pickle=True)
#     # img = np.load(image_dir)
#     # text = np.load(text_dir)
#     # label = np.load(label_dir1)
#     # shape = np.load(image_shape_dir)
#     return img,shape,text,label
#
# def loader_test():
#     img=np.load(image_test_dir,allow_pickle=True)
#     text=np.load(text_test_dir,allow_pickle=True)
#     label=np.load(label_test_dir1,allow_pickle=True)
#     shape = np.load(image_test_shape_dir,allow_pickle=True)
#     # img = np.load(image_test_dir)
#     # text = np.load(text_test_dir)
#     # label = np.load(label_test_dir1)
#     # shape = np.load(image_test_shape_dir)
#     return img,shape,text,label

















# import sys
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# import numpy as np
# from transformers import logging
#
# logging.set_verbosity_warning()
# logging.set_verbosity_error()
#
#
#
# crop_clases_test_dir = "./data/pascal_sentence_data/test/crop_clases_test.npy"
# text_test_dir = "./data/pascal_sentence_data/test/text_test.npy"
# label_test_dir1 = "./data/pascal_sentence_data/test/label_test.npy"
#
#
# def loader_train(k):
#     crop_clases_dir = "./data/pascal_sentence_data/test/crop_clases_{}.npy".format(k)
#     text_dir = "./data/pascal_sentence_data/test/text_{}.npy".format(k)
#     label_dir1 = "./data/pascal_sentence_data/test/label_{}.npy".format(k)
#
#
#     clases=np.load(crop_clases_dir,allow_pickle=True)
#     text=np.load(text_dir,allow_pickle=True)
#     label=np.load(label_dir1,allow_pickle=True)
#
#     return clases,text,label
#
# def loader_test():
#     clases=np.load(crop_clases_test_dir,allow_pickle=True)
#     text=np.load(text_test_dir,allow_pickle=True)
#     label=np.load(label_test_dir1,allow_pickle=True)
#     return clases,text,label





