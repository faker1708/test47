


import torch
import math


class mlp():

    def __init__(self,mlp_architecture,lr,mode = 'default'):
        # mlp_architecture = [2,2] # 超参是个列表，记录有几层神经元，每层有几个神经元，列表里的值 不是个数 ，而是个数的对数。具体构造规则见build函数。
        self.mlp_architecture = mlp_architecture
        self.lr = lr
        mode_list = ['gpu','cuda','cuda_half','cpu']

        if(mode =='default'):
            ck = torch.cuda.is_available()
            # self.ck = ck
            # ck = 0
            if(ck):
                self.mode = 'cuda_half'
                # self.mode = 'cuda'
            else:
                self.mode = 'cpu'
        elif(mode in mode_list):
            self.mode = mode
        else:
            print('warning from mlp init,unknown device type.')
            self.mode = 'cpu'

        # self.depth = len(mlp_architecture)
        # self.lr = lr
        self.non_linear = torch.nn.ReLU(inplace=False)   # 定义relu

        self.__build_nn()
    
    def __build_nn(self):
        mlp_architecture=self.mlp_architecture


        depth = len(mlp_architecture)
        w_list = list()
        b_list = list()
        for i,ele in enumerate(mlp_architecture):
            if(i<=depth -2):
                # kn = mlp_architecture[i]
                # km = mlp_architecture[i+1]

                # n = 2**kn
                # m = 2**km
                
                n = mlp_architecture[i]
                m = mlp_architecture[i+1]

                # print('mlp m',m)

                means = 0
                std = 0.1
                
                
                w = torch.normal(means,std,(m,n))
                b = torch.normal(means,std,(m,1))
                
                if(self.mode == 'cpu'):
                    w = w.cpu()
                    b = b.cpu()
                elif(self.mode == 'gpu' or self.mode == 'cuda'):
                    w = w.cuda()
                    b = b.cuda()
                elif(self.mode == 'cuda_half'):
                    # w = w.cuda().half()
                    # b = b.cuda().half()
                    
                    w = w.cuda()
                    b = b.cuda()
                    
                    w = w.type(torch.float16)
                    b = b.type(torch.float16)


                
                # 默认记录计算图
                w.requires_grad=True
                b.requires_grad=True


                w_list.append(w)
                b_list.append(b)
                    

        param = dict()
        param['w_list'] = w_list
        param['b_list'] = b_list
        # param['depth'] = depth


        self.param = param
        # return param

    def forward(self,x):
        # y = 0

        # if(gr==0):

        mlp_architecture = self.mlp_architecture 
        param = self.param




        w_list= param['w_list']
        b_list= param['b_list']

        # print('forward')

        depth = len(mlp_architecture)

        for i in range(depth-2):    # 如果 是4层，则只循环3次，分别 是012
            
            # print('forward',i)
            w = w_list[i]
            b = b_list[i]

            
            x = self.non_linear(w @ x + b)

        w = w_list[depth-2]
        b = b_list[depth-2]
        x = w @ x + b
        # 最后一层不要加非线性，加了relu会导致大量数据梯度为0，网络无法收敛了。


        y = x
        return y

    def loss_f(self,q,q_ref):

        dd = (q-q_ref)**2 /2
        
        loss = dd.sum()

        return loss
    
    def update(self):
        # lr batch_size 只是在这里用到，用来梯度下降。
        # 但 batch_size 和lr同时出现 ，所以感觉没必要有batch_size 这个量了。

        mlp_architecture = self.mlp_architecture 
        
        param = self.param
        # batch_size = self.batch_size
        lr = self.lr

        w_list= param['w_list']
        b_list= param['b_list']

        # print('update')

        with torch.no_grad():
            
            depth = len(mlp_architecture)
            for i in range(depth-1): 
                # print('update',i)
                w = w_list[i]
                b = b_list[i]

                # w -= lr * w.grad / batch_size
                # print('upd w_grad',w.grad)
                w -= lr * w.grad 
                w.grad.zero_()


                b -= lr * b.grad 
                # b -= lr * b.grad / batch_size
                b.grad.zero_()

    def test(self,x):

        # 测试与训练不同，这是确实要一个值出来 ，并且不要算梯度。
        in_dtype = x.dtype
        in_device = x.device
        # print(x.device)

        if(self.mode=='cuda_half'):
            x = x.cuda().half()
        elif(self.mode=='cuda'):
            x = x.cuda()
                
        with torch.no_grad():
            y = self.forward(x)
        
        
        y = y.to(in_device)
        y = y.type(in_dtype)
        return y







