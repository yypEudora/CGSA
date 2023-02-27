import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context

# 与predict的接口
# def main(imglist, imglist_shape):
#     img = imglist[1]
#     shape = imglist_shape[1]
#     num1 = shape[0]
#     num2 = shape[1]
#     m1 = torch.reshape(torch.Tensor(img), (3, num2, num1))
#     attention = selfAttention(2, num1, num2)
#     result = attention.forward(m1)
#     print(result.shape)


# 与average_pooling的接口
def main(imglist):
    shape = imglist.shape
    num1 = shape[1]
    num2 = shape[2]
    attention = selfAttention(1, num2, num1)
    result = attention.forward(imglist)   #图像自注意力特征向量，维度是(1,768,768)
    #print(result.shape)
    return result




    # features = torch.rand((32, 20, 10))
    # attention = selfAttention(2, 10, 20)
    # result = attention.forward(features)
    # print(result.shape)

