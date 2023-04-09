
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random


class Net(nn.Module):
    def __init__(self, ):

        N_STATES  = 4
        N_ACTIONS = 5
        super(Net, self).__init__()
        # 64 32

        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class dqn_memory():
    def __init__(self):
        pass

class dqn():
    
    def __init__(self):
        
        N_STATES = 4
        MEMORY_CAPACITY = 2000
        LR = 0.01


        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        
        
        memory = list()        


        memory.append(torch.tensor([]))
        memory.append(torch.tensor([]))
        memory.append(torch.tensor([]))
        memory.append(torch.tensor([]))

        self.memory = memory

        self.m_count = 0
        
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        return


    def random_action(self):

        N_ACTIONS = 5
        action = random.randint(0,N_ACTIONS-1)
        return action

    def action_f(self,state):
        # 0123 up down left right
        action = self.random_action()
        # action = 4
        # aa = random.randint(0,1)
        # if(aa):
        #     action = 0
        # else:
        #     action = 2
        return action
    
    
    def choose_action(self, x):

        
        EPSILON = 0.9               # greedy policy
        EPSILON = 0.5


        # EPSILON = 1
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = int(action)
            # print('acc',action)
        else:   # random
            action = self.random_action()
            # print('rand')

        return action

    def stack(self, experience):
        # 输入的经验 experience 是tensor的列表。[,,,,]
        
        MEMORY_CAPACITY = 2000
        view = experience[0]
        action = experience[1]
        view_after_action = experience[2]
        evaluate = experience[3]

        # print('action',action)

        # view,action,view_after_action,evaluate
        for i in range(4):
            self.memory[i] = torch.cat((self.memory[i],experience[i]),0)



        # print(self.memory.action)
        
        capcity = 2**10

        self.m_count +=1
        if(self.m_count>=2*capcity): #清理周期
            self.m_count = 0
            # print(self.memory.evaluate.shape)
            for i in range(4):
                aa = self.memory[i] 
                self.memory[i] = aa [-capcity:]
                # print('32',self.memory[2].shape)


        return

    def learn(self):
        # 莫烦的代码，一条数据是一行，真tm奇葩。我们一般输入数据 是列向量。
        
        N_STATES  = 4
        BATCH_SIZE = 2**8
        GAMMA = 0.9                 # reward discount
        TARGET_REPLACE_ITER = 100   # target update frequency
        MEMORY_CAPACITY = 2000


        # print('dqn.learn()')

        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1


        capcity= BATCH_SIZE
        b_s= self.memory[0][-capcity:]
        b_aa= self.memory[1][-capcity:]
        b_s_= self.memory[2][-capcity:]
        b_r= self.memory[3][-capcity:]

        # print(b_s.shape)

        # print('b_r',b_r[-1])

        b_a = torch.tensor(b_aa,dtype=torch.int64)
        # print(b_a)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)



        # print('bsss',b_s_.shape)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate

        # print(q_next.shape,'xx,32')


        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
