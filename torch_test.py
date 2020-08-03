import torch

b = torch.rand((4,5,5))
# print(a.diagonal_(5, wrapped=True))
print(torch.diagonal(b, dim1=-2, dim2=-1))

# a = torch.transpose(b, 4,-2)

# print((b@a).shape)
