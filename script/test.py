import torch
import torch.nn as nn


input = torch.randn(3, 2)
target = torch.FloatTensor([[0, 1], [1, 0], [0, 1]])


prop = [0.7, 0.3]
weight = torch.zeros(target.shape)

for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        weight[i][j] = prop[int(target[i][j])]

conf_loss_function = nn.BCELoss(weight=weight)
m = nn.Sigmoid()
loss = conf_loss_function(m(input), target)
print(loss)