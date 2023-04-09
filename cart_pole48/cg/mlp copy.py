


import torch
import math


class mlp():

    def __init__(self,mlp_architecture,mode = 'default'):
        # mlp_architecture = [2,2] # 超参是个列表，记录有几层神经元，每层有几个神经元，列表里的值 不是个数 ，而是个数的对数。具体构造规则见build函数。
        self.mlp_architecture = mlp_architecture

        mode_list = ['gpu','cuda','cuda_half','cpu']

        if(mode =='default'):
            ck = torch.cuda.is_available()
            # self.ck = ck
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

    def __forward(self,x):
        # y = 0

        # if(gr==0):

        mlp_architecture = self.mlp_architecture 
        param = self.param




        w_list= param['w_list']
        b_list= param['b_list']

        # print('__forward')

        depth = len(mlp_architecture)

        for i in range(depth-2):    # 如果 是4层，则只循环3次，分别 是012
            
            # print('__forward',i)
            w = w_list[i]
            b = b_list[i]

            
            x = self.non_linear(w @ x + b)

        w = w_list[depth-2]
        b = b_list[depth-2]
        x = w @ x + b
        # 最后一层不要加非线性，加了relu会导致大量数据梯度为0，网络无法收敛了。


        y = x
        return y

    def __loss_f(self,y,y_ref):

        dd = (y-y_ref)**2 /2
        
        loss = dd.sum()

        return loss
    
    def __update(self):
        # lr batch_size 只是在这里用到，用来梯度下降。
        # 但 batch_size 和lr同时出现 ，所以感觉没必要有batch_size 这个量了。

        mlp_architecture = self.mlp_architecture 
        
        param = self.param
        # batch_size = self.batch_size
        lr = self.lr

        w_list= param['w_list']
        b_list= param['b_list']

        # print('__update')

        with torch.no_grad():
            
            depth = len(mlp_architecture)
            for i in range(depth-1): 
                # print('__update',i)
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
            y = self.__forward(x)
        
        
        y = y.to(in_device)
        y = y.type(in_dtype)
        return y

    def train(self,x,y_ref):
        nn = self

        if(self.mode=='cuda_half'):
            x = x.cuda().half()
            y_ref = y_ref.cuda().half()
        elif(self.mode=='cuda'):
            x = x.cuda()
            y_ref = y_ref.cuda()
                


        self.lr = 2**-9

        total_cost = 2**0 #2**16    # 无论如何，会在这么多轮训练后结束，如果不是正数，则此限制无效。

        precious = 20   # 精度越大，效果越好。
        patient = 2**10  # 连续这么多轮训练都不收敛。
        
        ceil_line = 2**20   # 损失的上界


        # 局部变量 草稿 temp
        bad_count = 0  # 坏情况计数 当出现相同时，加一
        min_loss = ceil_line
        self.loss_integral = 0
        now_precious = -100

        ep = 0
        while(1):
            # print('ep',ep)

            ep+=1
            # print(ep)
            y = nn.__forward(x)


            loss = nn.__loss_f(y,y_ref)
            loss.backward()

            nn.__update()

            fl = float(loss)

            
            if(math.isnan(fl)):
                print('error,nan')
                break
            elif(math.isinf(fl)):
                print('error,inf')
                break

            # 判断收敛与否，不收敛就打断
            if(fl>=min_loss):
                bad_count+=1
                # print('loss','ep==',ep,fl)#,2**-now_precious)
                # print(bad_count)
            else:
                bad_count = 0  #清零

                
                min_loss = fl
            
                # print('loss',ep,fl)#,2**-now_precious)
            # ep+=1



            if(fl > ceil_line):
                # raise(BaseException())
                print('loss 太大，有问题.考虑减小学习率')
                break
            elif(fl< 2**-precious ):
                print('损失达标')
                break

            if(total_cost>0):
                if(ep>total_cost):
                    print('超时')
                    break

            if(bad_count>patient):
                # print('不收敛了',ep,fl)
                break


            
        if(fl==0):
            now_precious = 10000
            # break
        elif(fl>0):
            now_precious = -int(math.log2(fl))
        else:
            raise(BaseException('erro from mlp train,loss<0'))
        print('now_precious',now_precious,ep)#,2**-precious,2**-now_precious)
        # return out