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
        self.m_count = 0

        self.protagonist_net  = mlp.mlp(mlp_architecture,lr)
        self.assistant_net = copy.deepcopy(self.protagonist_net)

        self.sync_step = 0
        self.epsilon = 0.2

    def value_vector_f(self,state):
        # 输入状态，输出价值表，各个行动的价值
        value_vector = self.protagonist_net.test(state)
        return value_vector
    
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
            xx = torch.cat((self.memory[i],experience[i]),1)

            # if(i==0):
                # print('xx',xx)
                # exit()
            self.memory[i] = xx


        # print('stack()')
        # print(self.memory[0])
        # print(self.memory.action)
        
        capcity = 2**13

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
        random_index = random.randint(0,44)
        # random_index = 0
        
        # batch_size = self.batch_size
        batch_size = 32 # 32

        batch_state = self.memory[0][:,random_index:random_index+batch_size]
        batch_action = self.memory[1][:,random_index:random_index+batch_size]
        batch_next_state = self.memory[2][:,random_index:random_index+batch_size]
        batch_reward = self.memory[3][:,random_index:random_index+batch_size]


        batch_action = batch_action.type(torch.int64)
        gamma = 0.9

        # sss = self.memory[0]
        # print('sss[0]',sss[:,0:batch_size])

        # print('learn')
        # print('mem',len(self.memory))
        # print('state',self.memory[0])
        # print('batch_state',batch_state)


        gg = len(batch_state[0])
        if(gg==0):
            return 
        # exit()
        # print(gg)

        batch_state = batch_state.cuda().half()
        batch_action = batch_action.cuda().half()
        batch_action = batch_action.type(torch.int64)


        y = self.protagonist_net.forward(batch_state)
        # print('y',y)
        
        # print('ba',batch_action)
        q = y.gather(0,batch_action)    #从y中取出一个值
        # print('q',q)


        next_y = self.assistant_net.test(batch_next_state)
        # print('ny',next_y)

        next_q = torch.max(next_y, 0)[0].reshape((gg,1))
        # next_q = torch.max(next_y, 0)[1]
        # print('next_q',next_q)
        
        # next_q = next_y.max(1)[0].view(1,batch_size)
        # print('next——q 1 x 32 ==',next_q.shape)

        # print(next_q)
        q_ref = batch_reward + gamma* (next_q)
        q_ref = q_ref.cuda().half()
        
        # print('q',q)
        # print('q_ref',q_ref)


        # 构造训练集，交给mlp

        q = q/batch_size
        q_ref = q_ref/batch_size

        loss = self.protagonist_net.loss_f(q,q_ref)
        loss.backward()
        self.protagonist_net.update()

        fl = float(loss)
        # print(loss)

        

        self.sync_step += 1
        if(self.sync_step %2**7== 0):   # 128次学习，同步一次
            self.assistant_net = copy.deepcopy(self.protagonist_net)
        return fl


