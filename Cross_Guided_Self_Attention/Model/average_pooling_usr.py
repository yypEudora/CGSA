import torch
import torch.nn as nn
import numpy as np
import cv2


# import self_attention


class average_pooling_usr(torch.nn.Module):
    def __init__(self, re, num2, num1):
        super(average_pooling_usr, self).__init__()
        self.re = re
        self.MLP = torch.nn.Linear(re, 768)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((num2, num1))

    def forward(self, x):
        output = self.avg_pool(x)
        m2 = torch.reshape(output, (1, self.re))
        out = self.MLP(m2)
        return out
    # def average_pooling(self,imglist, imglist_shape):
    #     result_list = []
    #     for img, img_shape in zip(imglist, imglist_shape):
    #         num1 = img_shape[0]
    #         num2 = img_shape[1]
    #         re = 3 * num1 * num2
    #
    #         m1 = torch.reshape(torch.Tensor(img), (1, 3, num2, num1))
    #         avg_pool = torch.nn.AdaptiveAvgPool2d((num2, num1))
    #         output = self.avg_pool(m1)
    #         m2 = torch.reshape(output, (1, re))
    #         x = self.MLP(m2)
    #         x1 = torch.reshape(x, (-1, 1))
    #         result_list.append(x1.tolist())
    #         result = torch.Tensor(result_list)
    #         #print(result)
    #         #print(result.shape)
    #         num3 = result.shape[0]
    #         num4 = result.shape[1]
    #         m2 = torch.reshape(torch.Tensor(result), (1, num4, num3))
    #         self_attention.main(m2)     # 多个图像区域特征向量，用于交叉自注意力，维度是(1,768,n)
    #         return m2
    #
    # def forward(self,):
    #     output = self.avg_pool(m1)
    #     m2 = torch.reshape(output, (1, re))
    #     x = self.MLP(m2)
