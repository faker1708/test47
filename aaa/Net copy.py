
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle


class Net(nn.Module):
    def __init__(self, mlp_architecture):

        
        super(Net, self).__init__()



        lm = len(mlp_architecture)
        layer_count = lm -1
        # 输入的结构有4个数字，则矩阵共有3个。


        # 比如 mlp_architecture = [4,50,2]
        # lm = 3            
        # layer_count = 2
        # i = 0 1
        # 01 12
        layer_list = list()

        for i in range(layer_count):
            ic = mlp_architecture[i]    # 输入向量的尺寸
            oc = mlp_architecture[i+1]  # 输出向量的尺寸
            layer = nn.Linear(ic, oc)   # 建立全连接层
            layer.weight.data.normal_(0, 0.1)
            layer_list.append(layer)
            
        self.mlp_architecture = mlp_architecture
        self.layer_list = layer_list


    def forward(self, x):
        mlp_architecture = self.mlp_architecture
        
        for i in range():



        x = self.fc1(x)
        x = F.relu(x)

        actions_value = self.out(x)
        return actions_value

