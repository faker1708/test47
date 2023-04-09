import mlp


import random
import copy
import torch

class dqn():
    def __init__(self,mlp_architecture,lr):
        
        self.memory_capacity = 2**10   # 内存容量
        
        memory = list()        


        memory.append(torch.tensor([]))
        memory.append(torch.tensor([]))
        memory.append(torch.tensor([]))
        memory.append(torch.tensor([]))

        self.memory = memory

        self.protagonist_net  = mlp.mlp(mlp_architecture,lr)
        self.assistant_net = copy.deepcopy(self.protagonist_net)

        self.sync_step = 0
        self.epsilon = 0.2

    def value_f(self,state):
        # 输入状态，输出价值表，各个行动的价值
        return 
    
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
    
    def fetch(self,y,action):
        q = y.gather(1,action)

    def learn(self):
        random_index = random.randint(0,44)
        
        batch_size = self.batch_size
        batch_state = self.memory[0][random_index:random_index+batch_size]
        batch_action = self.memory[1][random_index:random_index+batch_size]
        batch_next_state = self.memory[2][random_index:random_index+batch_size]
        batch_reward = self.memory[3][random_index:random_index+batch_size]

        
        batch_state = batch_action.type(torch.int64)
        gamma = 0.9

        ff = self.fetch 


        


        next_y = self.assistant_net.test(x)
        next_q = next_y.max(1)[0]

        print(next_q)
        q_ref = batch_reward + gamma* (next_q)


        # 构造训练集，交给mlp
        x = batch_state


        self.protagonist_net.train(x,batch_action,ff,q_ref)

        

        self.sync_step += 1
        if(self.sync_step %2**7== 0):   # 128次学习，同步一次
            self.assistant_net = copy.deepcopy(self.protagonist_net)