
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle


# 本地类
# import Net
import mlp 

import copy

class DQN(object):
    def __init__(self,mlp_architecture):


        # mlp_architecture 是一个列表，描述了要求的神经网络的结构 每层几个神经元
        # N_STATES 是输入层神经元的个数
        # N_ACTIONS 是输出层神经元的个数
        
        self.MEMORY_CAPACITY = 2000
        self.N_STATES = mlp_architecture[0]
        self.N_ACTIONS = mlp_architecture[-1]
        lr = 0.01

        mode = 'cpu'
        self.eval_net = mlp.mlp(mlp_architecture,lr,mode)
        self.target_net  = mlp.mlp(mlp_architecture,lr,mode)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        
        # print('self.eval_net.parameters()',self.eval_net.parameters())

        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr)
        # self.loss_func = nn.MSELoss()

        #pid
        self.integral = 0  # 位移的积分
        self.max_i = 2**6 # 位移积分的上界
        self.ki= -3 #-1

    def random_action(self):

        N_ACTIONS = self.N_ACTIONS
        action = np.random.randint(0, N_ACTIONS)
        return action
    

    def choose_action(self, x):
        # x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # x = torch.
        x = torch.FloatTensor(x)
        x = torch.unsqueeze(x, dim=1)
        # print(x.shape)
        # exit()

        EPSILON = self.epsilon
        ENV_A_SHAPE = 0

        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)

            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = self.random_action()
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)


        return action

    def store_transition(self, state, a, r, s_):

        MEMORY_CAPACITY=2000

        transition = np.hstack((state, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print('dqn.learn()')
        TARGET_REPLACE_ITER = 100
        MEMORY_CAPACITY = 2000
        BATCH_SIZE = 32
        N_STATES = self.N_STATES


        GAMMA = 0.9


        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # self.target_net.load_state_dict(self.eval_net.state_dict())
            print('拷贝')
            self.target_net = copy.deepcopy(self.eval_net)
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])


        # q_eval w.r.t the action in experience

        # print(b_s.shape)
        # b_s = self.eval_net.trans(b_s)

        # 矩阵转置
        b_s = b_s.permute(1,0)     
        b_a = b_a.permute(1,0)      
        b_s_ = b_s_.permute(1,0)      
        b_r = b_r.permute(1,0)      

        # print(b_s.shape)
        
        xxx = self.eval_net.forward(b_s)
        # print(xxx)

        q_eval = xxx.gather(1, b_a)  # shape (batch, 1)
        # print('qes 1x32===',q_eval.shape)

        # print('qe',q_eval)

        q_next = self.target_net.test(b_s_)     # detach from graph, don't backpropagate
        
        # print(q_next.shape)


        max_next_q = q_next.max(0)[0].view( 1,BATCH_SIZE)
        # print('qn',q_next)
        # print('mnq 应该是 1x32',max_next_q)
        # print('mnq 应该是 1x32',max_next_q.shape)
        # print('br 应该是 1x32',b_r.shape)

        # print('qn',q_next)
        # yyy = q_next.max(1)
        # print('yyy',yyy)
        q_target = b_r + GAMMA* max_next_q
        
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)


        loss_f= self.eval_net.loss_f
        loss = loss_f(q_eval, q_target)
        # print('dqn learn() loss',float(loss))

        loss.backward()
        self.eval_net.update()


        # print('。。')
        # loss = self.loss_func(q_eval, q_target)

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()


    def get_i(self):
        # 解码
        integral = self.integral      
        error = self.error

        # 计算积分
        # integral += error
        error = abs(error)

        alpha = 0.5
        # integral = (1-alpha)*integral + alpha*error
        integral = (1-alpha)*integral + error

        # 无界变换成有界有多种函数 ，这里随便写一种简单的。不是我们的重点。

        # print(integral)
        if(abs(integral)>self.max_i):
            print('积分爆了')
            if integral>0:
                integral = self.max_i
            else:
                integral = -self.max_i

        # 更新
        self.integral = integral
        return integral

    def reward_f(self,next_state):

        x, x_dot, theta, theta_dot = next_state

        # pid算法
        self.error = x
        integral = self.get_i()
        ri = integral* self.ki
        
        
        r0 = 0.7
        r1 = -abs(x)/2.4
        r2 = -abs(theta)/0.209

        reward = r0 +r1 + r2 + ri
        if(reward< 2**-10):reward = 2**-10

        reward = reward**2

        return reward
    