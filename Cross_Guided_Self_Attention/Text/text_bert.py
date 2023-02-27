import torch
from transformers import BertTokenizer, BertModel
from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

import sys
sys.path.append('D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text')
bert_path = 'D:\\Files\\PycharmProjects\\Cross_Guided_Self_Attention\\Text\\bert-base-uncased'
# sys.path.append('Cross_Guided_Self_Attention/Text')
# bert_path = 'bert-base-uncased'


# def get_text_embedding(text):
#     tokenizer = BertTokenizer.from_pretrained(bert_path)
#     bert = BertModel.from_pretrained(bert_path, return_dict=True, output_hidden_states=True)
#     list=[]
#     #inputs = tokenizer("I love Beijing", return_tensors="pt")
#     for txt in text:
#         inputs = tokenizer(txt, return_tensors="pt")
#         outputs = bert(**inputs)
#         num = outputs[0][0].shape[0]
#         out = torch.reshape(outputs[0][0], (1, 768, num))
#         list.append(out)
#     #print(outputs[0].shape)   # (1, n,768)
#     #print(outputs[0][0])  # 与图像进行交叉引导   文本特征向量，维度是(n,768)
#     # print(outputs[1][0]) # 文本全局嵌入
#     return list


def get_text_embedding(text):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BertModel.from_pretrained(bert_path, return_dict=True, output_hidden_states=True)
    list=[]
    #inputs = tokenizer("I love Beijing", return_tensors="pt")

    input_ids = None,
    attention_mask = None,
    position_ids = None,
    token_type_ids = None,
    head_mask = None,
    inputs_embeds = None,

    for txt in text:
        inputs = tokenizer(txt, return_tensors="pt")
        outputs = bert(**inputs)
        out = outputs[1]
        list.append(out.detach().numpy())
    return list


# def get_text_embedding(text):
#     tokenizer = BertTokenizer.from_pretrained(bert_path)
#     bert = BertModel.from_pretrained(bert_path, return_dict=True, output_hidden_states=True)
#     #inputs = tokenizer("I love Beijing", return_tensors="pt")
#
#     inputs = tokenizer(text, return_tensors="pt")
#     outputs = bert(**inputs)
#     out = outputs[1]
#     return out