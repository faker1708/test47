
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle


class Net(nn.Module):
    def __init__(self, ):




        super(Net, self).__init__()
        # 64 32

        N_ACTIONS = 2
        N_STATES = 4
        mid = 16

        self.fc1 = nn.Linear(N_STATES, mid)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization



        self.fc2 = nn.Linear(mid, mid)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization


        self.out = nn.Linear(mid, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        
        actions_value = self.out(x)
        return actions_value

