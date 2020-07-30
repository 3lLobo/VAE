import torch

b = torch.rand((4,3,3,2,2))
# print(a.diagonal_(5, wrapped=True))
print(torch.diag_embed(b, dim1=2, dim2=3))

# a = torch.transpose(b, 4,-2)

# print((b@a).shape)
