"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle

# 现在在研究收敛


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
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


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        #pid
        self.i = 0  # 位移的积分
        self.max_i = 2**6 # 位移积分的上界
        self.ki= -3 #-1

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        # print(x.dtype)
        # print('x',x)
        # print('x.s',x.shape)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
        # if True:
            actions_value = self.eval_net.forward(x)

            # print(actions_value)
            # exit()

            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)


        return action

    def store_transition(self, state, a, r, s_):
        transition = np.hstack((state, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print('dqn.learn()')

        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        print(b_s)
        exit()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_i(self):
        # 解码
        i = self.i      
        error = self.error

        # 计算积分
        # i += error
        error = abs(error)

        alpha = 0.5
        i = (1-alpha)*i + alpha*error

        # 无界变换成有界有多种函数 ，这里随便写一种简单的。不是我们的重点。

        if(abs(i)>self.max_i):
            if i>0:
                i = self.max_i
            else:
                i = -self.max_i

        # 更新
        self.i = i
        return i

    def reward_f(self,next_state):

        x, x_dot, theta, theta_dot = next_state

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

        # 现在准备加入pid算法
        self.error = x
        i = self.get_i()
        ri = i* self.ki
        
        # print('ri',ri,'x',x)
        # ri = 0
        

        reward = r1 + r2 + ri
        if(reward<0):reward = 2**-10

        reward = reward**2

        # print('rr',r1+r2,ri,reward,x)
        # print('ratio',ri/reward)

        # print(reward,x)
        return reward
    

dqn_init = 1
if(dqn_init):
    dqn = DQN() # init 
    EPSILON = 0.9

else:
    with open('./model/cart_pole/a.pkl', "rb") as f:
        dqn = pickle.load(f)
        EPSILON = 1
        # EPSILON = 0.9
    print('读档',EPSILON)


plt_on = 1  # 是否显示图表

if(plt_on ==1):
    plt.ion()
    plt.figure(1)
    t_list = list()
    # result_list=list()

    t_list = list()
    
    x_list = list()
    xd_list = list()
    th_list = list()
    thd_list = list()



i_episode = 0

stt = 0
while(1):
    state,_ = env.reset()
    
    
    step = 0

    while(1):
        env.render()
        action = dqn.choose_action(state)

        x,xd,th,thd = state

        next_state, _, done, _, _ = env.step(action)

        
        reward = dqn.reward_f(next_state)

        dqn.store_transition(state, action, reward, next_state)

        if done:
            break
        state = next_state
        step +=1
        stt+=1

        if(step>2**16):
            
            break

        if(step%2**13==0):
            print(step,state,next_state)

            t_list.append(stt)
            x_list.append(abs(x))
            xd_list.append(xd)
            th_list.append(th)
            thd_list.append(thd)
            plt.plot(t_list,x_list,c='red')

            # plt.plot(t_list,xd_list,c='yellow')
            # plt.plot(t_list,th_list,c='green')
            # plt.plot(t_list,thd_list,c='blue')
            plt.pause(0.1)
    dqn.learn()

    
    
    i_episode+=1

    if(step>=2**12):    # 随便训练一个2万的模型就能稳定运行了.本实验可以宣布结束了.
        x,_,_,_ = state
        # if(abs(x)<1):
        if(abs(x)<0.24):
        # if(abs(x)<0.1):  # 必须对x有要求。
            if(step>2**16):
            
                # with open('./model/cart_pole/'+str(x)+'.pkl', "wb") as f:
                with open('./model/cart_pole/'+str('a')+'.pkl', "wb") as f:
                    pickle.dump(dqn, f)
                print('ep:',i_episode,'step',step,state)
                print('\a')
                exit()