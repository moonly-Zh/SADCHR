import torch

print(torch.__version__)
print(torch.cuda.is_available())

import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/linear_experiment")
# default at runs folder if not sepecify path



#矩阵实验-到底是哪里出了问题呢？
# l = torch.randn(5,2)
# x = torch.randn(5,4)
# y = torch.randn(5,3)
# w = torch.randn(2,4)
# I = torch.ones(4,4)*1
# #term21 = torch.sum(torch.pow(torch.matmul(F,W1.transpose(0,1))-L,2))*aph+torch.sum(torch.pow(W1,2))*lamda
# #W1 = torch.matmul(torch.matmul(L.transpose(0,1),F),torch.inverse(torch.matmul(F.transpose(0,1),F)+I))
# # print(l,x,y)
# l = l.cuda()
# x = x.cuda()
# y = y.cuda()
# w = w.cuda()
# I = I.cuda()
# t1 = torch.matmul(x.transpose(0,1),x)+I
# print(t1)
# # t2 = torch.inverse(t1)
# # print(t2)
# # t3 = torch.matmul(l.transpose(0,1),x)
# # print(t3)
# t4 = l.t()
# print(l)
# print(t4)