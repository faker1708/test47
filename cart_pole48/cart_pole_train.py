import dqn

import torch
import random
import gym


class cart_pole_train():

    def value_f(self,next_state):
        

        # print('value_f')
        # print(next_state)
        x = next_state[0]
        theta= next_state[2]


        r0  = 1
        r1 = 0
        # r1 = -abs(x)/2.4
        # r2 = -abs(theta)/0.209
        r2 = -abs(theta)/0.2

        r = r0+r1+r2
        r = r.reshape((1,1))
        # print(r)
        return r
    
    def fcf(self,ep):
        # 以 ep 的概率输出 1  
        # 1-ep 的概率输出 0
        xxc = 2**12
        gate = int(ep*xxc)  # 判决阈值
        rr = random.randint(0,xxc)
        if(rr < gate):
            fc = 1
        else:
            fc = 0
        return fc

    def random_action(self):
        action = random.randint(0,1)
        return action
    
    def action_f(self,state):
        fc = self.fcf(self.epsilon)
        if(fc):
            # print('服从')
            action_value = self.dqn.value_vector_f(state)
            # print('vt',vt)
            # print(action_value)

            action = torch.max(action_value, 0)[1]
            action = int(action)
            # action = vt .max(1)
            # print('action_f  int ',action)
            # print('\n\n')

        else:
            # 叛逆
            # print('叛逆')
            action = self.random_action()



        return action
    

    def __init__(self):

        
        env = gym.make('CartPole-v1')
        # env = gym.make('CartPole-v1',render_mode= 'human')
        env = env.unwrapped

        

        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]
        

        mlp_architecture = [N_STATES,32,N_ACTIONS]
        lr = 2**-1  # q——ref 这些 ，可能 会为nan，要注意减小lr


        self.epsilon = 0.9

        self.dqn = dqn.dqn(mlp_architecture,lr)
        self.env = env

        episode = 0 # 对局次数
        
        okg = 0
        enough = 2**13

        while(1):
            # 开启新的对局

            step = 0
            state,_ = env.reset()
            state = torch.from_numpy(state).reshape((4,1))

            # print(state)
            while(1):
                env.render()
                action = self.action_f(state)

                # print('action',action)
                next_state, _, done, _, _ = env.step(action)
                next_state = torch.from_numpy(next_state).reshape((4,1))
                action = torch.tensor(action).reshape((1,1))
                # print('action',action)


                reward = self.value_f(next_state)   # 用价值来估计奖励


                

                experience = [state, action, next_state , reward]
                # print('experience',experience)
                self.dqn.stack(experience)
            
                state = next_state
                step +=1
                if done:
                    break
                if(step>enough):
                    okg = 1
                    break
            
            episode +=1
            fl = self.dqn.learn()

            if(episode%2**8==0):
                print(fl,'episode',episode,'step',step,list(state))

            if(okg==1):
                print("合格")
                x,_,_,_ = state
                print(x)






if __name__ == "__main__":
    cart_pole_train()