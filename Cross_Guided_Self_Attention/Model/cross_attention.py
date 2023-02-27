import torch
from bidirectional_cross_attention import BidirectionalCrossAttention


def cross_attention(img_, text_):
    # img = torch.randn(1, 768, 768)
    # txt = torch.randn(1, 768, n)
    list_img_out=[]
    list_text_out=[]
    for img,text in zip(img_,text_):
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
        list_img_out.append(img_out)
        list_text_out.append((text_out))
    list_img_out=torch.reshape(torch.stack(list_img_out), (-1,768,768))
    return list_img_out, list_text_out









# import torch
# from bidirectional_cross_attention import BidirectionalCrossAttention
#
#
# def cross_attention(img_, text_):
#     # img = torch.randn(1, 768, 768)
#     # txt = torch.randn(1, 768, n)
#     list_img_out=[]
#     list_text_out=[]
#     for img,text in zip(img_,text_):
#         img_mask = torch.ones((1, 768)).bool()
#         text_mask = torch.ones((1, 768)).bool()
#
#         num_text_dimension = text.shape[2]
#         joint_cross_attn = BidirectionalCrossAttention(
#             dim=img.shape[2],
#             heads=16,
#             dim_head=48,
#             context_dim=num_text_dimension
#         )
#
#         img_out, text_out = joint_cross_attn(  # img交叉引导注意力特征向量，维度是(1,768,768)
#             img,  # txt交叉引导注意力特征向量，维度是(1, 768, n)
#             text,
#             mask=img_mask,
#             context_mask=text_mask
#         )
#         list_img_out.append(img_out)
#         list_text_out.append((text_out))
#     # list_img_out=torch.reshape(torch.stack(list_img_out), (-1,768,768))
#     return list_img_out, list_text_out


