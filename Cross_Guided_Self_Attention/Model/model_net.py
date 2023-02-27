# clas以前的
import torch

import sys

sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# sys.path.append('Cross_Guided_Self_Attention/Text')
# sys.path.append('Cross_Guided_Self_Attention/Model')
import Model.self_attention as self_attention
import Model.cross_attention as cross_attention
import Text.text_bert as text_bert
import numpy as np
import Model.average_pooling_usr as average_pooling_usr

import warnings
warnings.filterwarnings("ignore")

class Fc(torch.nn.Module):
    def __init__(self,input,output):
        super(Fc, self).__init__()
        self.out1=torch.nn.Linear(input,output)
    def forward(self,x):
        x=self.out1(x)
        return x

class Model_Net(torch.nn.Module):
    def __init__(self, args):
        super(Model_Net, self).__init__()
        self.args = args
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.out1 = torch.nn.Linear(768 * 768 * 4, 2)
        self.out2 = torch.nn.Linear(768 * 768, 20)
        self.out3 = torch.nn.Linear(6, 2)
        self.out4 = torch.nn.Linear(768, 20)

    def average_pool(self, crop_img_list, crop_img_list_shape):
        global m2
        list = []
        for img1, img_shape1 in zip(crop_img_list, crop_img_list_shape):
            result_list = []
            for img, img_shape in zip(img1, img_shape1):
                num1 = img_shape[0]
                num2 = img_shape[1]
                re = 3 * num1 * num2
                avg_pooling = average_pooling_usr.average_pooling_usr(re, num2, num1)
                m1 = torch.reshape(torch.Tensor(img), (1, 3, num2, num1))
                # output = avg_pooling.avg_pool(m1)
                # m2 = torch.reshape(output, (1, re))
                # x = avg_pooling.MLP(m2)
                x = avg_pooling(m1)
                x1 = torch.reshape(x, (-1, 1))
                result_list.append(x1.tolist())
                result = torch.Tensor(result_list)
                # print(result)
                # print(result.shape)
                num3 = result.shape[0]
                num4 = result.shape[1]
                m2 = torch.reshape(torch.Tensor(result), (1, num4, num3))
            img_self_attention = self_attention.main(m2)
            list.append(img_self_attention)
        all_tensor = torch.reshape(torch.stack(list), (-1, 768, 768))
        return all_tensor

    def get_attension(self, crop_img_list, crop_img_list_shape, text):
        img_self_attention = self.average_pool(crop_img_list, crop_img_list_shape)
        text_word_embedding = text_bert.get_text_embedding(text)
        return img_self_attention, text_word_embedding

    def cross_attension(self, img_self_attention, text_word_embedding):
        img_out, text_out = cross_attention.cross_attention(img_self_attention, text_word_embedding)
        return img_out, text_out

    def CAT(self, text_word_embedding, text_out):
        list_text_att = []
        list_text_cross_att = []
        for text_em, text_ot in zip(text_word_embedding, text_out):
            text_att1 = torch.reshape(text_em, (768, text_em.shape[2]))
            text_cross_att1 = torch.reshape(text_ot, (768, text_ot.shape[2]))
            # fc = torch.nn.Linear()
            fc=Fc(text_ot.shape[2], 768)
            text_att = fc(text_att1)
            text_cross_att = fc(text_cross_att1)
            list_text_att.append(text_att)
            list_text_cross_att.append(text_cross_att)
        list_text_att = torch.reshape(torch.stack(list_text_att), (-1, 768, 768))
        list_text_cross_att = torch.reshape(torch.stack(list_text_cross_att), (-1, 768, 768))
        return list_text_att, list_text_cross_att


    def CAT_1(self, text_word_embedding, text_out):
        list_text_att = []
        for text_em, text_ot in zip(text_word_embedding, text_out):
            text_att1 = torch.reshape(text_em, (768, text_em.shape[2]))
            fc=Fc(text_ot.shape[2], 768)
            text_att = fc(text_att1)
            list_text_att.append(text_att)
        list_text_att = torch.reshape(torch.stack(list_text_att), (-1, 768, 768))
        return list_text_att

    def forward(self, text):
        text_word_embedding = text_bert.get_text_embedding(text)
        # text_att = self.CAT_1(text_word_embedding, text_word_embedding)

        text_word_embedding = torch.Tensor(text_word_embedding)
        text_att = torch.reshape(text_word_embedding, (-1, text_word_embedding.shape[1] * text_word_embedding.shape[2]))
        ypre = self.softmax(self.relu(self.out4(text_att)))

        return ypre, text_att






# # clas以前的
# import torch
#
# import sys
#
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# # sys.path.append('Cross_Guided_Self_Attention/Text')
# # sys.path.append('Cross_Guided_Self_Attention/Model')
# import Model.self_attention as self_attention
# import Model.cross_attention as cross_attention
# import Text.text_bert as text_bert
# import numpy as np
# import Model.average_pooling_usr as average_pooling_usr
#
# import warnings
# warnings.filterwarnings("ignore")
#
# class Fc(torch.nn.Module):
#     def __init__(self,input,output):
#         super(Fc, self).__init__()
#         self.out1=torch.nn.Linear(input,output)
#     def forward(self,x):
#         x=self.out1(x)
#         return x
#
# class Model_Net(torch.nn.Module):
#     def __init__(self, args):
#         super(Model_Net, self).__init__()
#         self.args = args
#         self.relu = torch.nn.ReLU()
#         self.softmax = torch.nn.Softmax()
#         self.out1 = torch.nn.Linear(768 * 768 * 4, 2)
#         self.out2 = torch.nn.Linear(768 * 768, 20)
#         self.out3 = torch.nn.Linear(6, 2)
#         self.out4 = torch.nn.Linear(768*768, 20)
#
#     def average_pool(self, crop_img_list, crop_img_list_shape):
#         global m2
#         list = []
#         for img1, img_shape1 in zip(crop_img_list, crop_img_list_shape):
#             result_list = []
#             for img, img_shape in zip(img1, img_shape1):
#                 num1 = img_shape[0]
#                 num2 = img_shape[1]
#                 re = 3 * num1 * num2
#                 avg_pooling = average_pooling_usr.average_pooling_usr(re, num2, num1)
#                 m1 = torch.reshape(torch.Tensor(img), (1, 3, num2, num1))
#                 # output = avg_pooling.avg_pool(m1)
#                 # m2 = torch.reshape(output, (1, re))
#                 # x = avg_pooling.MLP(m2)
#                 x = avg_pooling(m1)
#                 x1 = torch.reshape(x, (-1, 1))
#                 result_list.append(x1.tolist())
#                 result = torch.Tensor(result_list)
#                 # print(result)
#                 # print(result.shape)
#                 num3 = result.shape[0]
#                 num4 = result.shape[1]
#                 m2 = torch.reshape(torch.Tensor(result), (1, num4, num3))
#             img_self_attention = self_attention.main(m2)
#             list.append(img_self_attention)
#         all_tensor = torch.reshape(torch.stack(list), (-1, 768, 768))
#         return all_tensor
#
#     def get_attension(self, crop_img_list, crop_img_list_shape, text):
#         img_self_attention = self.average_pool(crop_img_list, crop_img_list_shape)
#         text_word_embedding = text_bert.get_text_embedding(text)
#         return img_self_attention, text_word_embedding
#
#     def cross_attension(self, img_self_attention, text_word_embedding):
#         img_out, text_out = cross_attention.cross_attention(img_self_attention, text_word_embedding)
#         return img_out, text_out
#
#     def CAT(self, text_word_embedding, text_out):
#         list_text_att = []
#         list_text_cross_att = []
#         for text_em, text_ot in zip(text_word_embedding, text_out):
#             text_att1 = torch.reshape(text_em, (768, text_em.shape[2]))
#             text_cross_att1 = torch.reshape(text_ot, (768, text_ot.shape[2]))
#             # fc = torch.nn.Linear()
#             fc=Fc(text_ot.shape[2], 768)
#             text_att = fc(text_att1)
#             text_cross_att = fc(text_cross_att1)
#             list_text_att.append(text_att)
#             list_text_cross_att.append(text_cross_att)
#         list_text_att = torch.reshape(torch.stack(list_text_att), (-1, 768, 768))
#         list_text_cross_att = torch.reshape(torch.stack(list_text_cross_att), (-1, 768, 768))
#         return list_text_att, list_text_cross_att
#
#     def forward(self, crop_img_list, crop_img_list_shape, text):
#         img_self_attention, text_word_embedding = self.get_attension(crop_img_list, crop_img_list_shape, text)
#         img_out, text_out = self.cross_attension(img_self_attention, text_word_embedding)
#         # img_att = img_self_attention
#         # img_corss_att = img_out
#         text_att, text_cross_att = self.CAT(text_word_embedding, text_out)
#         #
#         # # f_va_vc = np.dot(img_att.detach(), img_corss_att.detach())
#         # f_va_vc = img_att * img_corss_att
#         # # f_va_vc = torch.matmul(img_att.detach(), img_corss_att.detach())
#         # f_ta_tc = text_att * text_cross_att
#         # f_va_ta = text_att * img_att
#         # f_vc_tc = text_cross_att * img_corss_att
#         # fglobal1 = f_va_ta * f_vc_tc
#         # fglobal2 = fglobal1 * f_ta_tc
#         # fglobal = fglobal2 * f_va_vc
#         #
#         # f_va_vc = torch.tensor(f_va_vc)
#         # f_ta_tc = torch.tensor(f_ta_tc)
#         # f_va_ta = torch.tensor(f_va_ta)
#         # f_vc_tc = torch.tensor(f_vc_tc)
#         # fglobal1 = torch.tensor(fglobal1)
#         # fglobal2 = torch.tensor(fglobal2)
#         # fglobal = torch.tensor(fglobal)
#         #
#         # att_cat = torch.cat([img_att, img_corss_att, text_att, text_cross_att], dim=1)
#         # att_cat = torch.reshape(att_cat, (-1, att_cat.shape[1] * att_cat.shape[2]))
#         # ypre1 = self.softmax(self.relu(self.out1(att_cat)))
#         #
#         # vatt_cat = torch.cat([f_va_vc, f_ta_tc, f_va_ta, f_vc_tc], dim=1)
#         # vatt_cat = torch.reshape(vatt_cat, (-1, vatt_cat.shape[1] * vatt_cat.shape[2]))
#         # ypre2 = self.softmax(self.relu(self.out1(vatt_cat)))
#         #
#         # fglobal = torch.reshape(fglobal, (-1, fglobal.shape[1] * fglobal.shape[2]))
#         # ypre3 = self.softmax(self.relu(self.out2(fglobal)))
#         #
#         # ypre_cat = torch.cat([ypre1, ypre2, ypre3], dim=1)
#         # ypre_all = self.softmax(self.relu(self.out3(ypre_cat)))
#
#         # return ypre1, ypre2, ypre3, ypre_all, fglobal
#
#         # neg_numpy = np.array(text_word_embedding)
#         # neg_tensor = torch.from_numpy(neg_numpy)
#
#
#
#
#         text_att = torch.reshape(text_att, (-1, text_att.shape[1] * text_att.shape[2]))
#         ypre = self.softmax(self.relu(self.out4(text_att)))
#         return ypre, text_att













# import torch
#
# import sys
#
# sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
# # sys.path.append('Cross_Guided_Self_Attention/Text')
# # sys.path.append('Cross_Guided_Self_Attention/Model')
# import Model.self_attention as self_attention
# import Model.cross_attention as cross_attention
# import Text.text_bert as text_bert
# import numpy as np
# import Model.average_pooling_usr as average_pooling_usr
#
# import warnings
# warnings.filterwarnings("ignore")
#
# class Fc(torch.nn.Module):
#     def __init__(self,input,output):
#         super(Fc, self).__init__()
#         self.out1=torch.nn.Linear(input,output)
#     def forward(self,x):
#         x=self.out1(x)
#         return x
#
# class Model_Net(torch.nn.Module):
#     def __init__(self, args):
#         super(Model_Net, self).__init__()
#         self.args = args
#         self.relu = torch.nn.ReLU()
#         self.softmax = torch.nn.Softmax()
#         self.out1 = torch.nn.Linear(768 * 768 * 4, 20)
#         self.out4 = torch.nn.Linear(768*768*2, 20)
#         self.out5 = torch.nn.Linear(768 * 768, 20)
#         self.out2 = torch.nn.Linear(768 * 768, 20)
#         self.out3 = torch.nn.Linear(60, 20)
#
#     def average_pool(self, crop_img_list, crop_img_list_shape):
#         global m2
#         list = []
#         for img1, img_shape1 in zip(crop_img_list, crop_img_list_shape):
#             result_list = []
#             for img, img_shape in zip(img1, img_shape1):
#                 num1 = img_shape[0]
#                 num2 = img_shape[1]
#                 re = 3 * num1 * num2
#                 avg_pooling = average_pooling_usr.average_pooling_usr(re, num2, num1)
#                 m1 = torch.reshape(torch.Tensor(img), (1, 3, num2, num1))
#                 # output = avg_pooling.avg_pool(m1)
#                 # m2 = torch.reshape(output, (1, re))
#                 # x = avg_pooling.MLP(m2)
#                 x = avg_pooling(m1)
#                 x1 = torch.reshape(x, (-1, 1))
#                 result_list.append(x1.tolist())
#                 result = torch.Tensor(result_list)
#                 # print(result)
#                 # print(result.shape)
#                 num3 = result.shape[0]
#                 num4 = result.shape[1]
#                 m2 = torch.reshape(torch.Tensor(result), (1, num4, num3))
#             img_self_attention = self_attention.main(m2)
#             list.append(img_self_attention)
#         all_tensor = torch.reshape(torch.stack(list), (-1, 768, 768))
#         return all_tensor
#
#     def get_attension(self, clases, text):
#         img_self_attention = text_bert.get_text_embedding(clases)
#         text_word_embedding = text_bert.get_text_embedding(text)
#         return img_self_attention, text_word_embedding
#
#     def cross_attension(self, img_self_attention, text_word_embedding):
#         img_out, text_out = cross_attention.cross_attention(img_self_attention, text_word_embedding)
#         return img_out, text_out
#
#     def CAT(self, text_word_embedding, text_out):
#         list_text_att = []
#         list_text_cross_att = []
#         for text_em, text_ot in zip(text_word_embedding, text_out):
#             text_att1 = torch.reshape(text_em, (768, text_em.shape[2]))
#             text_cross_att1 = torch.reshape(text_ot, (768, text_ot.shape[2]))
#             # fc = torch.nn.Linear()
#             fc=Fc(text_ot.shape[2], 768)
#             text_att = fc(text_att1)
#             text_cross_att = fc(text_cross_att1)
#             list_text_att.append(text_att)
#             list_text_cross_att.append(text_cross_att)
#         list_text_att = torch.reshape(torch.stack(list_text_att), (-1, 768, 768))
#         list_text_cross_att = torch.reshape(torch.stack(list_text_cross_att), (-1, 768, 768))
#         return list_text_att, list_text_cross_att
#
#     def fc(shape1, shape2):
#         return torch.nn.Linear(shape1 * shape2, 20)
#
#     def forward(self, clases, text):
#         img_self_attention, text_word_embedding = self.get_attension(clases, text)
#         img_out, text_out = self.cross_attension(img_self_attention, text_word_embedding)
#         # img_att = img_self_attention
#         # img_corss_att = img_out
#         img_att, img_corss_att = self.CAT(text_word_embedding, text_out)
#         text_att, text_cross_att = self.CAT(text_word_embedding, text_out)
#
#         # f_va_vc = np.dot(img_att.detach(), img_corss_att.detach())
#         f_va_vc = img_att * img_corss_att
#         # f_va_vc = torch.matmul(img_att.detach(), img_corss_att.detach())
#         f_ta_tc = text_att * text_cross_att
#         f_va_ta = text_att * img_att
#         f_vc_tc = text_cross_att * img_corss_att
#         fglobal1 = f_va_ta * f_vc_tc
#         fglobal2 = fglobal1 * f_ta_tc
#         fglobal = fglobal2 * f_va_vc
#
#         f_va_vc = torch.tensor(f_va_vc)
#         f_ta_tc = torch.tensor(f_ta_tc)
#         f_va_ta = torch.tensor(f_va_ta)
#         f_vc_tc = torch.tensor(f_vc_tc)
#         fglobal1 = torch.tensor(fglobal1)
#         fglobal2 = torch.tensor(fglobal2)
#         fglobal = torch.tensor(fglobal)
#
#         # att = text_bert.get_text_embedding(clases)
#         # att1 = torch.cat([att],dim=1)
#         # t_att = torch.reshape(clases, (-1, att1.shape[1]*att1.shape[2]))
#         # ypre = self.softmax(self.relu(self.out5(t_att)))
#
#         att = torch.cat([img_corss_att, text_att], dim=1)
#         t_att = torch.reshape(att, (-1, att.shape[1]*att.shape[2]))
#         ypre = self.softmax(self.relu(self.out4(t_att)))
#
#         # att_cat = torch.cat([img_att, img_corss_att, text_att, text_cross_att], dim=1)
#         # att_cat = torch.reshape(att_cat, (-1, att_cat.shape[1] * att_cat.shape[2]))
#         # ypre1 = self.softmax(self.relu(self.out1(att_cat)))
#         #
#         # vatt_cat = torch.cat([f_va_vc, f_ta_tc, f_va_ta, f_vc_tc], dim=1)
#         # vatt_cat = torch.reshape(vatt_cat, (-1, vatt_cat.shape[1] * vatt_cat.shape[2]))
#         # ypre2 = self.softmax(self.relu(self.out1(vatt_cat)))
#         #
#         # fglobal = torch.reshape(fglobal, (-1, fglobal.shape[1] * fglobal.shape[2]))
#         # ypre3 = self.softmax(self.relu(self.out2(fglobal)))
#         #
#         # ypre_cat = torch.cat([ypre1, ypre2, ypre3], dim=1)
#         # ypre_all = self.softmax(self.relu(self.out3(ypre_cat)))
#
#         return ypre




