import torch

from faster_rcnn import predict
# from average_pooling_usr import average_pooling_usr
import self_attention
import sys
sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
import text_bert
import cross_attention
import torch.nn as nn
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import logging
import torch
from bidirectional_cross_attention import BidirectionalCrossAttention

logging.set_verbosity_warning()
logging.set_verbosity_error()

bert_path = 'D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text\\bert-base-uncased'


img_path = 'D:\\Files\\PycharmProjects\\flicker8k\\Flicker8k_Dataset\\'

def average_poolings(imglist, imglist_shape):
    global m2
    result_list = []
    for img, img_shape in zip(imglist, imglist_shape):
        num1 = img_shape[0]
        num2 = img_shape[1]
        re = 3 * num1 * num2

        m1 = torch.reshape(torch.Tensor(img), (1, 3, num2, num1))
        avg_pool = torch.nn.AdaptiveAvgPool2d((num2, num1))

        # self.re = re
        # self.MLP = torch.nn.Linear(re, 768)
        # self.avg_pool = torch.nn.AdaptiveAvgPool2d((num2, num1))

        output = avg_pool(m1)
        m2 = torch.reshape(output, (1, re))
        MLP=torch.nn.Linear(re, 768)
        x = MLP(m2)
        x1 = torch.reshape(x, (-1, 1))
        result_list.append(x1.tolist())
        result = torch.Tensor(result_list)
        #print(result)
        #print(result.shape)
        num3 = result.shape[0]
        num4 = result.shape[1]
        m2 = torch.reshape(torch.Tensor(result), (1, num4, num3))
    return m2

def get_text_embedding(text):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BertModel.from_pretrained(bert_path, return_dict=True, output_hidden_states=True)

    #inputs = tokenizer("I love Beijing", return_tensors="pt")

    inputs = tokenizer(text, return_tensors="pt")
    outputs = bert(**inputs)
    num = outputs[0][0].shape[0]
    out = torch.reshape(outputs[0][0], (1, 768, num))

    #print(outputs[0].shape)   # (1, n,768)
    #print(outputs[0][0])  # 与图像进行交叉引导   文本特征向量，维度是(n,768)
    # print(outputs[1][0]) # 文本全局嵌入
    return out

def cross_attention(img, text):
    # img = torch.randn(1, 768, 768)
    # txt = torch.randn(1, 768, n)

    img=torch.reshape(img,(1,768,768))
    img_mask = torch.ones((1, 768)).bool()
    text_mask = torch.ones((1, 768)).bool()

    num_text_dimension = text.shape[2]
    joint_cross_attn = BidirectionalCrossAttention(
        dim=768,
        heads=16,
        dim_head=48,
        context_dim=num_text_dimension
    )

    img_out, text_out = joint_cross_attn(  # img交叉引导注意力特征向量，维度是(1,768,768)
        img,  # txt交叉引导注意力特征向量，维度是(1, 768, n)
        text,
        mask=img_mask,
        context_mask=text_mask
    )
    return img_out, text_out

if __name__ == '__main__':

    name = '754852108_72f80d421f.jpg'
    img = img_path + name
    flag, crop_img_list, crop_img_list_shape = predict.main(img)
    # list=[]
    # list.append(crop_img_list)
    # list.append(crop_img_list[0])
    if flag == 0:
        print('未检测到！！！')
    else:
        print('检测到啦！！！')
        print(crop_img_list_shape)

        # 图像特征池化处理 (1,768,n)
        img_pooling = average_poolings(crop_img_list, crop_img_list_shape)
        # 图像自注意力特征, (1,768,768)
        img_self_attention = self_attention.main(img_pooling)

        # 文本词嵌入特征向量 (1,768,n)
        text = 'I love Beijing'
        text_word_embedding = get_text_embedding(text)

        # 图像-文本交叉注意特征向量
        img_out, text_out = cross_attention(img_self_attention, text_word_embedding)

        # 处理为2维
        img_att = torch.reshape(img_self_attention, (768, 768))
        img_corss_att = torch.reshape(img_out, (768, 768))
        text_att1 = torch.reshape(text_word_embedding, (768, text_word_embedding.shape[2]))
        text_cross_att1 = torch.reshape(text_out, (768, text_out.shape[2]))

        # -------------------------------------------------------------------------------------------
        # 采样层
        # 先把文本信息输出成(768, 768)
        fc = torch.nn.Linear(text_out.shape[2], 768)
        text_att = fc(text_att1)
        text_cross_att = fc(text_cross_att1)

        # --------------------------------------------------------------------------------------------
        # 跨模态融合层
        f_va_vc = np.dot(img_att.detach(), img_corss_att.detach())
        f_ta_tc = np.dot(text_att.detach(), text_cross_att.detach())

        f_va_ta = np.dot(text_att.detach(), img_att.detach())
        f_vc_tc = np.dot(text_cross_att.detach(), img_corss_att.detach())

        # --------------------------------------------------------------------------------------------
        # 全局融合层
        fglobal1 = np.dot(f_va_ta, f_vc_tc)
        fglobal2 = np.dot(fglobal1, f_ta_tc)
        fglobal = np.dot(fglobal2, f_va_vc)

        print(fglobal)







