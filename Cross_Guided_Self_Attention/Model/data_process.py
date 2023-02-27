import re
import xlrd
import sys
sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
from faster_rcnn import predict
import numpy as np
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()


k = 'test_test'
data = xlrd.open_workbook(r'D:\Files\PycharmProjects\PascalSentence\{}.xlsx'.format(k))
table = data.sheets()[0]


tables = []

def import_excel(excel):
    for rown in range(excel.nrows):
        array = {'img_name':'', 'text':'', 'label':''}
        array['img_name'] = table.cell_value(rown, 0)
        array['text'] = table.cell_value(rown, 1)
        array['label'] = int(table.cell_value(rown, 2))
        tables.append(array)

def get_word(classes, text):
    result = ' '
    text_words = re.split(r"[^A-Za-z]", text.strip())
    text_classes = re.split(r"[^A-Za-z]", classes.strip())
    lent = len(text_words)
    lenc = len(text_classes)
    a = 1
    b = 1
    for word_t in text_words:
        if a > lent:
            break

        for word_c in text_classes:
            if word_t == word_c:
                result = word_t
                return result
        a = a+1
    return result


def get_name(num):
    if num == 1:
        return "airplane"
    elif num == 2:
        return "bicycle"
    elif num == 3:
        return "bird"
    elif num == 4:
        return "boat"
    elif num == 5:
        return "bottle"
    elif num == 6:
        return "bus"
    elif num == 7:
        return "car"
    elif num == 8:
        return "cat"
    elif num == 9:
        return "chair"
    elif num == 10:
        return "cow"
    elif num == 11:
        return "diningtable"
    elif num == 12:
        return "dog"
    elif num == 13:
        return "horse"
    elif num == 14:
        return "motorbike"
    elif num == 15:
        return "person"
    elif num == 16:
        return "pottedplant"
    elif num == 17:
        return "sheep"
    elif num == 18:
        return "sofa"
    elif num == 19:
        return "train"
    elif num == 20:
        return "tvmonitor"

def process():

    img_path = 'D:\\Files\\PycharmProjects\\PascalSentence\\images\\'

    import_excel(table)
    labels = []
    texts = []
    g = 1

    for i in tables:
        img_name = i['img_name']
        text = i['text']
        label = i['label']
        img = img_path + img_name
        flag, clas = predict.main(img)

        if flag == 0:
            clas = get_name(label+1)


        word = get_word(clas, text)

        if word == ' ':
            word = get_name(label + 1)

        texts.append(word)
        lab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        lab[label] = 1
        labels.append(lab)

        print(g)
        g = g + 1

    np.save('./data/pascal_sentence_data/test/label_{}.npy'.format(k), labels)
    np.save('./data/pascal_sentence_data/test/text_{}.npy'.format(k), texts)



if __name__ == '__main__':
    process()







## clas之前的
# import numpy as np
# import torch
# import xlrd
# import torch
# import model_net
#
# # from average_pooling import average_pooling
# # import self_attention
# import sys
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# # sys.path.append('Cross_Guided_Self_Attention/Text')
# # sys.path.append('faster_rcnn')
# # sys.path.append('fliker8k')
# from faster_rcnn import predict
# import Text.text_bert
# import torch.nn as nn
# import numpy as np
# from transformers import logging
# logging.set_verbosity_warning()
# logging.set_verbosity_error()
#
#
# k = 'train'
# name = 'bicycle'
# data = xlrd.open_workbook(r'D:\Files\PycharmProjects\PascalSentence\{}_{}.xlsx'.format(k, name))
# # data = xlrd.open_workbook('Data/Annotations{}.xlsx'.format(k))
# table = data.sheets()[0]
#
#
# tables = []
#
# def import_excel(excel):
#     for rown in range(excel.nrows):
#         array = {'img_name':'', 'text':'', 'label':''}
#         array['img_name'] = table.cell_value(rown, 0)
#         array['text'] = table.cell_value(rown, 1)
#         array['label'] = int(table.cell_value(rown, 2))
#         tables.append(array)
#
# # labels=np.load(image_dir1)
# #
# # all_lab=[]
# # for i in labels:
# #     lab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# #     lab[i]=1
# #     all_lab.append(lab)
# #
# # labels=np.array(all_lab)
# # labels=torch.from_numpy(labels)
# # labels=labels.float()
#
# def process():
#     # device = torch.device("cpu")
#
#     img_path = 'D:\\Files\\PycharmProjects\\PascalSentence\\images\\'
#
#     # name = '667626_18933d713e.jpg'
#     # img = img_path + name
#     # flag, crop_img_list, crop_img_list_shape = predict.main(img)
#     # text = 'I love Beijing'
#     # label = [0, 1, 0, 0]
#     # label = torch.tensor(label)
#     # label = torch.reshape(label, (1, -1)).float()
#
#     import_excel(table)
#     labels = []
#     crop_imgs = []
#     crop_imgs_shape = []
#     texts = []
#     g = 1
#
#     for i in tables:
#         img_name = i['img_name']
#         text = i['text']
#         label = i['label']
#         img = img_path + img_name
#         flag, crop_img_list, crop_img_list_shape = predict.main(img)
#
#         if flag == 1:
#             crop_imgs.append(crop_img_list)
#             crop_imgs_shape.append(crop_img_list_shape)
#             texts.append(text)
#             lab = [0, 0]
#             lab[label] = 1
#             labels.append(lab)
#             print(g)
#             g = g + 1
#     np.save('./data/pascal_sentence_data/{}/label_{}.npy'.format(name, k), labels)
#     np.save('./data/pascal_sentence_data/{}/text_{}.npy'.format(name, k), texts)
#     np.save('./data/pascal_sentence_data/{}/crop_imgs_{}.npy'.format(name, k), crop_imgs)
#     np.save('./data/pascal_sentence_data/{}/crop_imgs_shape_{}.npy'.format(name, k), crop_imgs_shape)
#
#
#
# if __name__ == '__main__':
#     process()


































# import numpy as np
# import torch
# import xlrd
# import torch
# import model_net
#
# # from average_pooling import average_pooling
# # import self_attention
# import sys
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# # sys.path.append('Cross_Guided_Self_Attention/Text')
# # sys.path.append('faster_rcnn')
# # sys.path.append('fliker8k')
# from faster_rcnn import predict
# import Text.text_bert
# import torch.nn as nn
# import numpy as np
# from transformers import logging
# logging.set_verbosity_warning()
# logging.set_verbosity_error()
#
# k = 'train'
# data = xlrd.open_workbook(r'D:\Files\PycharmProjects\Cross_Guided_Self_Attention\Data\test\Annotations_{}.xlsx'.format(k))
# # data = xlrd.open_workbook('Data/Annotations{}.xlsx'.format(k))
# table = data.sheets()[0]
#
#
# tables = []
#
# def import_excel(excel):
#     for rown in range(excel.nrows):
#         array = {'img_name':'', 'text':'', 'label':''}
#         array['img_name'] = table.cell_value(rown, 0)
#         array['text'] = table.cell_value(rown, 1)
#         array['label'] = int(table.cell_value(rown, 2))
#         tables.append(array)
#
# # labels=np.load(image_dir1)
# #
# # all_lab=[]
# # for i in labels:
# #     lab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# #     lab[i]=1
# #     all_lab.append(lab)
# #
# # labels=np.array(all_lab)
# # labels=torch.from_numpy(labels)
# # labels=labels.float()
#
# def process():
#     # device = torch.device("cpu")
#
#     img_path = 'D:\\Files\\PycharmProjects\\flicker8k\\Flicker8k_Dataset\\'
#
#     # name = '667626_18933d713e.jpg'
#     # img = img_path + name
#     # flag, crop_img_list, crop_img_list_shape = predict.main(img)
#     # text = 'I love Beijing'
#     # label = [0, 1, 0, 0]
#     # label = torch.tensor(label)
#     # label = torch.reshape(label, (1, -1)).float()
#
#     import_excel(table)
#     labels = []
#     crop_imgs = []
#     crop_imgs_shape = []
#     texts = []
#     g = 1
#
#     for i in tables:
#         img_name = i['img_name']
#         text = i['text']
#         label = i['label']
#         img = img_path + img_name
#         flag, crop_img_list, crop_img_list_shape = predict.main(img)
#
#         if flag == 1:
#             crop_imgs.append(crop_img_list)
#             crop_imgs_shape.append(crop_img_list_shape)
#             texts.append(text)
#             lab = [0, 0]
#             lab[label-1] = 1
#             labels.append(lab)
#             print(g)
#             g = g + 1
#     np.save('./data/test/label_{}.npy'.format(k), labels)
#     np.save('./data/test/text_{}.npy'.format(k), texts)
#     np.save('./data/test/crop_imgs_{}.npy'.format(k), crop_imgs)
#     np.save('./data/test/crop_imgs_shape_{}.npy'.format(k), crop_imgs_shape)
#
#
# def concat_data():
#     crop_img=np.load('./data/text{}.npy'.format(k),allow_pickle=True)
#     return crop_img
#
#
# if __name__ == '__main__':
#     process()
#     # concat_data()



