


import torch


# 感觉batchsize可以删除了呀，因为需要它的地方也有 lr 可以协同控制。

# 如果batch_size 为1 一定要保证你的输入输出 都是列向量
# x2 = torch.unsqueeze(x1, dim=1)

class mlp():

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
                

                if(self.mode == 'cpu'):
                    
                    w = torch.normal(0,1,(m,n)).cpu()
                    b = torch.normal(0,1,(m,1)).cpu()
                elif(self.mode == 'gpu'):
                    w = torch.normal(0,1,(m,n)).cuda()
                    b = torch.normal(0,1,(m,1)).cuda()
                    
                
                # 默认记录计算图
                w.requires_grad=True
                b.requires_grad=True


                w_list.append(w)
                b_list.append(b)
                    

        param = dict()
        param['w_list'] = w_list
        param['b_list'] = b_list
        param['depth'] = depth


        self.param = param
        # return param


    def __init__(self,mlp_architecture,lr,mode = 'gpu'):
        # mlp_architecture = [2,2] # 超参是个列表，记录有几层神经元，每层有几个神经元，列表里的值 不是个数 ，而是个数的对数。具体构造规则见build函数。
        self.mlp_architecture = mlp_architecture
        self.mode = mode
        


        self.depth = len(mlp_architecture)
        self.lr = lr
        self.rl = torch.nn.ReLU(inplace=False)   # 定义relu

        self.__build_nn()
    

    def set_super_param(self,mlp_architecture):
        self.set_super_param = mlp_architecture


    def forward(self,x):
        # y = 0

        # if(gr==0):

        param = self.param

        w_list= param['w_list']
        b_list= param['b_list']

        # print('forward')

        depth = param['depth']
        for i in range(depth-1):    # 如果 是4层，则只循环3次，分别 是012
            
            # print('forward',i)
            w = w_list[i]
            b = b_list[i]

            
            x = self.rl(w @ x + b)

        y = x
        return y

    def loss_f(self,y,y_ref):

        dd = (y-y_ref)**2 /2
        
        loss = dd.sum()

        return loss
    
    def update(self):
        
        param = self.param
        # batch_size = self.batch_size
        lr = self.lr

        w_list= param['w_list']
        b_list= param['b_list']

        # print('update')

        with torch.no_grad():
            
            depth = param['depth']
            for i in range(depth-1): 
                # print('update',i)
                w = w_list[i]
                b = b_list[i]

                # w -= lr * w.grad / batch_size
                w -= lr * w.grad 
                w.grad.zero_()


                b -= lr * b.grad 
                # b -= lr * b.grad / batch_size
                b.grad.zero_()

    def test(self,x):

        # 测试与训练不同，这是确实要一个值出来 ，并且不要算梯度。


        with torch.no_grad():
            param = self.param

            w_list= param['w_list']
            b_list= param['b_list']

            # print('forward')

            depth = param['depth']
            for i in range(depth-1):    # 如果 是4层，则只循环3次，分别 是012
                
                # print('forward',i)
                w = w_list[i]
                b = b_list[i]

                # print(x.shape)
                
                x = self.rl(w @ x + b)
                # print(w.shape)
                # print(b.shape)
                # print('x 后 ',i,x.shape)

                # print('\nwewfe\n')

            y = x
        return y
