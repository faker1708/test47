
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle


# 本地类
import Net
import DQN

# 现在在研究收敛


class cart_pole():
    def main(self):

        okg = 0
        enough = 2**13


        env = gym.make('CartPole-v1')
        env = env.unwrapped
        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]

        mlp_architecture = [N_STATES,50,N_ACTIONS]

        dqn = DQN.DQN(mlp_architecture) # init 


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
            step_list = list()
            ep_list= list()


        tlun = 0
        stt = 0
        while(1):

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
                        okg = 0
                        break
                    state = next_state
                    step +=1
                    stt+=1

                    if(step>enough):
                        okg = 1
                        break

                    if(step%2**10==0):
                        # print(step,state,next_state)

                        t_list.append(stt)
                        x_list.append(abs(x))
                        # x_list.append(step)
                        # x_list.append(float(fl))
                        plt.plot(t_list,x_list,c='red')
                        # print(float(fl))

                        plt.pause(0.001)
                dqn.learn()
                # ep_list.append(stt)
                # step_list.append(step)
                # plt.plot(ep_list,step_list,c='blue')

                i_episode+=1
                if(i_episode%2**0==0):
                # print(fl)
                    print('ep',stt,'step',step,okg,state)

                if(step>=enough):    # 随便训练一个2万的模型就能稳定运行了.本实验可以宣布结束了.
                    x,_,_,_ = state
                    if(abs(x)<qline):
                        if(tlun>=3):
                            if(abs(x)>qline/4):
                                break
                            else:
                                print('x 太小也不好',x,qline)
                        else:
                            break
            if(okg==1):

                print('合格',tlun,abs(x))
                print('\a')
                print('ep:',i_episode,'step',step,state)

                if(tlun>=   4   ):
                    
                    with open('./a.pkl', "wb") as f:
                        pickle.dump(dqn, f)
                    break

                tlun +=1
                okg =0

cart_pole().main()