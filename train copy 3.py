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


class Net(nn.Module):
    def __init__(self, ):




        super(Net, self).__init__()
        # 64 32

        N_ACTIONS = 2
        N_STATES = 4

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
    def __init__(self,mlp_architecture):


        # mlp_architecture 是一个列表，描述了要求的神经网络的结构 每层几个神经元
        # N_STATES 是输入层神经元的个数
        # N_ACTIONS 是输出层神经元的个数
        
        self.MEMORY_CAPACITY = 2000
        self.N_STATES = mlp_architecture[0]
        self.N_ACTIONS = mlp_architecture[-1]
        lr = 0.01

        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr)
        self.loss_func = nn.MSELoss()

        #pid
        self.integral = 0  # 位移的积分
        self.max_i = 2**6 # 位移积分的上界
        self.ki= -3 #-1

    def random_action(self):

        N_ACTIONS = self.N_ACTIONS
        action = np.random.randint(0, N_ACTIONS)
        return action
    

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

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
    

class cart_pole():
    def main(self):

        okg = 0
        enough = 2**13


        env = gym.make('CartPole-v1')
        env = env.unwrapped
        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]

        mlp_architecture = [N_STATES,50,N_ACTIONS]

        dqn = DQN(mlp_architecture) # init 


        EPSILON = 0.9

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



        tlun = 0
        while(1):
            stt = 0

            if(tlun ==0):
                EPSILON = 0.9
                qline = 2
            elif(tlun ==1):
                EPSILON = 0.9
                qline = 1
            elif(tlun ==2):
                EPSILON = 0.9
                qline = 0.5
            elif(tlun ==3):
                EPSILON = 0.9
                qline = 0.24
            elif(tlun ==4):
                EPSILON = 1
                qline = 0.24


            # self.epsilon = EPSILON
            dqn.epsilon = EPSILON


            i_episode = 0
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

                    if(step>enough):
                        okg = 1
                        break

                    if(step%2**9==0):
                        # print(step,state,next_state)

                        t_list.append(stt)
                        x_list.append(abs(x))
                        plt.plot(t_list,x_list,c='red')

                        plt.pause(0.001)
                    if(step%2**12==0):
                        pass
                        # dqn.learn()
                # for _ in range(2**6):
                dqn.learn()
                
                i_episode+=1

                if(step>=enough):    # 随便训练一个2万的模型就能稳定运行了.本实验可以宣布结束了.
                    x,_,_,_ = state
                    if(abs(x)<qline):
                        break
            if(okg==1):
                okg =0

                print('合格',tlun,abs(x))
                print('\a')
                print('ep:',i_episode,'step',step,state)

                if(tlun>=   4   ):
                    
                    with open('./a.pkl', "wb") as f:
                        pickle.dump(dqn, f)
                    break

                tlun +=1

cart_pole().main()