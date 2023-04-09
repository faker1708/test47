import mlp


import random
import copy
import torch

class dqn():
    def __init__(self,mlp_architecture,lr):
        
        self.memory_capacity = 2**10   # 内存容量
        

        class memory():
            pass
        
        mm = memory()
        mm.state = torch.tensor([])
        mm.action = torch.tensor([])
        mm.action_after_state = torch.tensor([])
        mm.reward = torch.tensor([])
        
        self.memory = mm


        self.protagonist_net  = mlp.mlp(mlp_architecture,lr)
        self.assistant_net = copy.deepcopy(self.protagonist_net)

        self.sync_step = 0
        self.epsilon = 0.2

    def value_f(self,state):
        # 输入状态，输出价值表，各个行动的价值

    def stack(self,experience):
        