from radnet import RadNet
import torch
import matplotlib.pyplot as plt

net = RadNet()

Z = torch.tensor([6, 1])
pos = torch.tensor([[5., 0, 0], [8, 0, 0]], requires_grad=True)
cell = torch.stack([torch.tensor([[10., 0,0], [0,10., 0], [0,0,10.]])]* 2)
index = torch.tensor([0, 0])
outs = net(pos, Z, cell, index)
print(outs.shape)
print(torch.autograd.grad(outs[0], pos))