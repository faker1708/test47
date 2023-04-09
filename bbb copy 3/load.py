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

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate


# EPSILON = 0.9               # greedy policy
EPSILON = 1              # greedy policy


GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000


# render_on = 1


env = gym.make('CartPole-v1', render_mode="human")
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

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
        # if True:
            actions_value = self.eval_net.forward(x)
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

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def reward_f(self,next_state):

        x, x_dot, theta, theta_dot = next_state

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

        reward = r1 + r2

        return reward


with open('./a.pkl', "rb") as f:
    dqn = pickle.load(f)
    # EPSILON = 0.9







plt_on = 1  # 是否显示图表

if(plt_on ==1):
    plt.ion()
    plt.figure(1)
    t_list = list()
    
    x_list = list()
    xd_list = list()
    th_list = list()
    thd_list = list()
    # result_list=list()

i_episode = 0
while(1):
    state,_ = env.reset()
    
    
    step = 0
    while(1):
        env.render()

        # 0左 1右
        action = dqn.choose_action(state)
        

        x,xd,th,thd = state

        # if(step%2**3==0):
            # if(x<2.4):
            # action = 1
            # elif(x>0.12):
            #     actioin = 0
        # action = 0  

        next_state, _, done, _, _ = env.step(action)

        if(plt_on ==1):
        
            if(step%2**13==2**10):
            # if(step%2**16==0):
                # print(step,state,next_state)
                
                t_list.append(step)
                x_list.append(abs(x)/2.4*100)
                xd_list.append(xd)
                th_list.append(abs(th)/0.2*100)
                thd_list.append(thd)

                plt.plot(t_list,x_list,c='red')
                # plt.plot(t_list,xd_list,c='yellow')
                plt.plot(t_list,th_list,c='green')
                # plt.plot(t_list,thd_list,c='blue')
                plt.pause(0.1)
                
                if(abs(x)>0.5):
                    print('\a')
                    print('小车偏离太远')
                    exit()

        # print(step,next_state)
            

        if done:

            break
        state = next_state
        step +=1




    print('ep:',i_episode,'step',step,state,next_state)
    
    
    
    i_episode+=1
    if(i_episode>2**2):
        break