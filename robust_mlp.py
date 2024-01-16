import argparse
import os
import time
import numpy as np
import torch
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
from Dataset import Dataset
from evaluate import evaluate_model
from utils import (AverageMeter, BatchDataset, get_optimizer,
                   get_train_instances, get_train_matrix)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument("--path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="yelp",
                        help="Choose a dataset.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs.")
    parser.add_argument("--bsz", type=int, default=100,
                        help="Batch size.")
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]",
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,
                        help="Number of negative instances to pair with a positive instance.")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="Learning rate.")
    parser.add_argument("--optim", nargs="?", default="adam",
                        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--enable_lat", nargs="?", default=True)
    parser.add_argument("--epsilon", nargs="?", default=0.5)
    parser.add_argument("--alpha", nargs="?", default=1)
    parser.add_argument("--pro_num", nargs="?", default=1, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument("--layerlist", nargs="?", default="all")
    parser.add_argument("--adv", nargs="?", default=True)
    parser.add_argument("--adv_reg", nargs="?", default=1)
    parser.add_argument("--reg", nargs="?", default=1e-3)
    parser.add_argument("--adv_type", nargs="?", default="mim", choices=['fgsm', 'bim', 'pgd','mim'])
    parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
    return parser.parse_args()

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)


class MLP(nn.Module):
    def __init__(self, fcLayers, userMatrix, itemMatrix, alpha, epsilon, pro_num, enable_lat, layerlist, adv_type, adv_reg,norm,decay_factor, device):
        super(MLP, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.adv_type = adv_type
        self.norm = norm
        self.criterion = torch.nn.BCELoss()
        self.device = device
        self.adv_reg = adv_reg
        self.decay_factor = decay_factor
        self.pro_num = pro_num
        self.enable_lat = enable_lat
        self.seed1,_,_,_,_,_ = self.random()
        _,self.seed2,_,_,_,_ = self.random()
        _,_,self.seed3,_,_,_ = self.random()
        _,_,_,self.seed4,_,_ = self.random()
        _,_,_,_,self.seed5,_ = self.random()
        _,_,_,_,_,self.seed6 = self.random()
        self.maxseed = max(self.seed1, self.seed2,self.seed3,self.seed4,self.seed5,self.seed6)
        self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg','y4_reg','y5_reg','y6_reg']
        self.enable_list = [0 for i in range(7)]
        if enable_lat and layerlist != "all":
            self.layerlist = [int(x) for x in layerlist.split(',')]
            self.layerlist_digit = [int(x) for x in layerlist.split(',')]
        else:
            self.layerlist = "all"
            self.layerlist_digit = list(range(0,self.maxseed+1))
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)
        
        nUsers = self.userMatrix.size(0)
        nItems = self.itemMatrix.size(0)


        # In the official implementation, 
        # the first dense layer has no activation
        self.userFC = nn.Linear(nItems, fcLayers[0]//2)
        self.itemFC = nn.Linear(nUsers, fcLayers[0]//2)
        self.reg_size_list = list()
        layers = []
        self.register_buffer(self.y_list[0], torch.zeros([100, 1024]))
        y_index = 1
        for l1, l2 in zip(fcLayers[:-1], fcLayers[1:]):
            layers+=[nn.Linear(l1, l2)]
            layers+=[nn.ReLU(inplace=True)]
            self.register_buffer(self.y_list[y_index], torch.zeros([100, l2]))
            self.reg_size_list.append([100, l2])
            y_index+=1
        self.fcs = nn.Sequential(*layers)

        # In the official implementation, 
        # the final module is initialized using Lecun normal method.
        # Here, the Kaiming normal initialization is adopted.
        self.final = nn.Sequential(
            nn.Linear(fcLayers[-1], 1),
            nn.Sigmoid(),
        )
        self.choose_layer()

    def forward(self, user, item, label):
        userInput = self.userMatrix[user, :].to(self.device)         # (B, 3706)
        itemInput = self.itemMatrix[item, :].to(self.device)         # (B, 6040)
        userVector = self.userFC(userInput).to(self.device)          # (B, fcLayers[0]//2)
        itemVector = self.itemFC(itemInput).to(self.device)           # (B, fcLayers[0]//2)
        self.y0 = torch.cat((userVector, itemVector), -1).to(self.device)   # (B, fcLayers[0])
        
        self.y1 = self.fcs[0](self.y0).to(self.device)
        self.y1 = self.fcs[1](self.y1) 
        
        self.y2 = self.fcs[2](self.y1).to(self.device)
        self.y2 = self.fcs[3](self.y2)
        
        self.y3 = self.fcs[4](self.y2).to(self.device)
        self.y3 = self.fcs[5](self.y3)

        self.y4 = self.fcs[6](self.y3).to(self.device)
        self.y4 = self.fcs[7](self.y4)

        self.y5 = self.fcs[8](self.y4).to(self.device)
        self.y5 = self.fcs[9](self.y5)

        self.y6 = self.fcs[10](self.y5).to(self.device)
        self.y6 = self.fcs[11](self.y6)
                                     # (B, fcLayers[-1])
        self.y = self.final(self.y6).to(self.device)
        self.yc = self.y.squeeze()
        loss = self.criterion(self.yc, label)
        
        if self.enable_lat:
            self.y1 = self.fcs[0](self.y0).to(self.device)  #torch.Size([100, 100])         
            if self.enable_lat and 1 in self.enable_list1:
                self.y1.retain_grad()
                self.y1_add = self.y1.add(self.y1_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y1_add-self.y1, -self.epsilon, self.epsilon).to(self.device) 
                        self.y1_add = self.y1.add(delta).to(self.device) 
                if self.norm =="l2":
                    delta = self.y1_add-self.y1
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y1_add = self.y1.add(delta).to(self.device) 
            else:
                self.y1_add = self.y1
            self.z1 = self.fcs[1](self.y1_add).to(self.device) 
            
            self.y2 = self.fcs[2](self.z1).to(self.device) 
            if self.enable_lat and 2 in self.enable_list2:
                self.y2.retain_grad()
                self.y2_add = self.y2.add(self.y2_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y2_add-self.y2, -self.epsilon, self.epsilon).to(self.device) 
                        self.y2_add = self.y2.add(delta).to(self.device) 
                if self.norm =="l2":
                    delta = self.y2_add-self.y2
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y2_add = self.y2.add(delta).to(self.device) 
            else:
                self.y2_add = self.y2
            self.z2 = self.fcs[3](self.y2_add).to(self.device)  #torch.Size([100, 128])

            self.y3 = self.fcs[4](self.z2).to(self.device)   
            if self.enable_lat and 3 in self.enable_list3:
                self.y3.retain_grad()
                self.y3_add = self.y3.add(self.y3_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y3_add-self.y3, -self.epsilon, self.epsilon).to(self.device)  
                        self.y3_add = self.y3.add(delta).to(self.device) 
                if self.norm =="l2":
                    delta = self.y3_add-self.y3
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y3_add = self.y3.add(delta).to(self.device) 
            else:
                self.y3_add = self.y3
            self.z3 = self.fcs[5](self.y3_add).to(self.device) 

            self.y4 = self.fcs[6](self.z3).to(self.device)  
            if self.enable_lat and 4 in self.enable_list4:
                self.y4.retain_grad()
                self.y4_add = self.y4.add(self.y4_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y4_add-self.y4, -self.epsilon, self.epsilon).to(self.device) 
                        self.y4_add = self.y4.add(delta).to(self.device)
                if self.norm =="l2":
                    delta = self.y4_add-self.y4
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y4_add = self.y4.add(delta).to(self.device)
            else:
                self.y4_add = self.y4
                self.y4_add = self.y4_add.to(self.device)
            self.z4 = self.fcs[7](self.y4_add)

            self.y5 = self.fcs[8](self.z4).to(self.device)  
            if self.enable_lat and 5 in self.enable_list5:
                self.y5.retain_grad()
                self.y5_add = self.y5.add(self.y5_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y5_add-self.y5, -self.epsilon, self.epsilon).to(self.device) 
                        self.y5_add = self.y5.add(delta).to(self.device)
                if self.norm =="l2":
                    delta = self.y5_add-self.y5
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y5_add = self.y5.add(delta).to(self.device)
            else:
                self.y5_add = self.y5
                self.y5_add = self.y5_add.to(self.device)
            self.z5 = self.fcs[9](self.y5_add)

            self.y6 = self.fcs[10](self.z5).to(self.device)  
            if self.enable_lat and 6 in self.enable_list6:
                self.y6.retain_grad()
                self.y6_add = self.y6.add(self.y6_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y6_add-self.y6, -self.epsilon, self.epsilon).to(self.device) 
                        self.y6_add = self.y6.add(delta).to(self.device)
                if self.norm =="l2":
                    delta = self.y6_add-self.y6
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y6_add = self.y6.add(delta).to(self.device)
            else:
                self.y6_add = self.y6
                self.y6_add = self.y6_add.to(self.device)
            self.z6 = self.fcs[11](self.y6_add)

            # print(y3_add.data)                            
            self.y = self.final(self.z6).to(self.device)
            self.yc = self.y.squeeze()
            loss_adv = self.criterion(self.yc, label)
            loss = loss + self.adv_reg * loss_adv                           # (B, 1)
        return loss, self.y

    

    def grad_init(self):
        if self.adv_type=="naive":
            for i in range(0,self.maxseed+1): 
                exec('self.y{}_reg.data = torch.randn(self.y{})'.format(i,i))
        elif self.adv_type=="fgsm" or self.adv_type=="bim" or self.adv_type=="mim":
            for i in range(0,self.maxseed+1): 
                exec('self.y{}_reg.data = torch.zeros_like(self.y{}).detach()'.format(i,i))
        elif self.adv_type=="pgd":
            for i in range(0,self.maxseed+1):
                if self.norm=="linf": 
                    exec('self.y{}_reg.data = torch.empty_like(self.y{}).uniform_(-self.epsilon, self.epsilon)'.format(i,i,i))
                elif self.norm=="l2":
                    exec('delta{} = torch.empty_like(self.y{}).uniform_(-self.epsilon, self.epsilon)'.format(i,i))
                    exec('normval{} = torch.norm(delta{}.view(100, -1), 2, 1)'.format(i,i))
                    exec('mask{} = normVal{}<=self.epsilon'.format(i,i))
                    exec('scaling{} = self.epsilon/normVal{}'.format(i,i))
                    exec('scaling{}[mask{}] = 1'.format(i,i))
                    exec('delta{} = delta{}*scaling{}.view(100, 1)'.format(i,i,i))
                    exec('self.y{}_reg.data = delta{}'.format(i,i))


    def choose_layer(self):
        if self.enable_lat == False:
            return
        if self.layerlist == 'all':
            self.enable_list1 = list(range(0, self.seed1+1))
            self.enable_list2 = list(range(0, self.seed2+1))
            self.enable_list3 = list(range(0, self.seed3+1))
            self.enable_list4 = list(range(0, self.seed4+1))
            self.enable_list5 = list(range(0, self.seed5+1))
            self.enable_list6 = list(range(0, self.seed6+1))   # all True
        else:
            for i in self.layerlist_digit:
                self.enable_list[i] = 1

    def save_grad(self):
        if self.enable_lat:
            if 1 in self.enable_list1:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y1_reg.data = (self.epsilon / self.pro_num) * (self.y1.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y1_reg.data = (self.epsilon / self.pro_num) * (self.y1.grad / cal_lp_norm(self.y1.grad,p=2,dim_count=len(self.y1_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum1 = torch.zeros_like(self.y1).detach()
                    grad1 = self.y1.grad
                    grad_norm1 = torch.norm(nn.Flatten()(grad1), p=1, dim=1)
                    grad1 = grad1 / grad_norm1.view([-1]+[1]*(len(grad1.shape)-1))
                    grad1 = grad1 + self.decay_factor * momentum1
                    momentum1 = grad1
                    if self.norm=="linf":
                        self.y1_reg.data = (self.epsilon / self.pro_num) * (grad1.sign())

                    elif self.norm=="l2":
                        self.y1_reg.data = (self.epsilon / self.pro_num) * (grad1 / cal_lp_norm(grad1,p=2,dim_count=len(self.y1.grad.size())))
            
            if 2 in self.enable_list2:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y2_reg.data = (self.epsilon / self.pro_num) * (self.y2.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y2_reg.data = (self.epsilon / self.pro_num) * (self.y2.grad / cal_lp_norm(self.y2.grad,p=2,dim_count=len(self.y2_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum2 = torch.zeros_like(self.y2).detach()
                    grad2 = self.y2.grad
                    grad_norm2 = torch.norm(nn.Flatten()(grad2), p=1, dim=1)
                    grad2 = grad2 / grad_norm2.view([-1]+[1]*(len(grad2.shape)-1))
                    grad2 = grad2 + self.decay_factor * momentum2
                    momentum2 = grad2
                    if self.norm=="linf":
                        self.y2_reg.data = (self.epsilon / self.pro_num) * (grad2.sign())

                    elif self.norm=="l2":
                        self.y2_reg.data = (self.epsilon / self.pro_num) * (grad2 / cal_lp_norm(grad2,p=2,dim_count=len(self.y2.grad.size())))

            if 3 in self.enable_list3:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y3_reg.data = (self.epsilon / self.pro_num) * (self.y3.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y3_reg.data = (self.epsilon / self.pro_num) * (self.y3.grad / cal_lp_norm(self.y3.grad,p=2,dim_count=len(self.y3_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum3 = torch.zeros_like(self.y3).detach()
                    grad3 = self.y3.grad
                    grad_norm3 = torch.norm(nn.Flatten()(grad3), p=1, dim=1)
                    grad3 = grad3 / grad_norm3.view([-1]+[1]*(len(grad3.shape)-1))
                    grad3 = grad3 + self.decay_factor * momentum3
                    momentum3 = grad3
                    if self.norm=="linf":
                        self.y3_reg.data = (self.epsilon / self.pro_num) * (grad3.sign())

                    elif self.norm=="l2":
                        self.y3_reg.data = (self.epsilon / self.pro_num) * (grad3 / cal_lp_norm(grad3,p=2,dim_count=len(self.y3.grad.size())))

            if 4 in self.enable_list4:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y4_reg.data = (self.epsilon / self.pro_num) * (self.y4.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y4_reg.data = (self.epsilon / self.pro_num) * (self.y4.grad / cal_lp_norm(self.y4.grad,p=2,dim_count=len(self.y4_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum4 = torch.zeros_like(self.y4).detach()
                    grad4 = self.y4.grad
                    grad_norm4 = torch.norm(nn.Flatten()(grad4), p=1, dim=1)
                    grad4 = grad4 / grad_norm4.view([-1]+[1]*(len(grad4.shape)-1))
                    grad4 = grad4 + self.decay_factor * momentum4
                    momentum4 = grad4
                    if self.norm=="linf":
                        self.y4_reg.data = (self.epsilon / self.pro_num) * (grad4.sign())

                    elif self.norm=="l2":
                        self.y4_reg.data = (self.epsilon / self.pro_num) * (grad4 / cal_lp_norm(grad4,p=2,dim_count=len(self.y4.grad.size())))

            if 5 in self.enable_list5:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y5_reg.data = (self.epsilon / self.pro_num) * (self.y5.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y5_reg.data = (self.epsilon / self.pro_num) * (self.y5.grad / cal_lp_norm(self.y5.grad,p=2,dim_count=len(self.y5_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum5 = torch.zeros_like(self.y5).detach()
                    grad5 = self.y5.grad
                    grad_norm5 = torch.norm(nn.Flatten()(grad5), p=1, dim=1)
                    grad5 = grad5 / grad_norm5.view([-1]+[1]*(len(grad5.shape)-1))
                    grad5 = grad5 + self.decay_factor * momentum5
                    momentum5 = grad5
                    if self.norm=="linf":
                        self.y5_reg.data = (self.epsilon / self.pro_num) * (grad5.sign())

                    elif self.norm=="l2":
                        self.y5_reg.data = (self.epsilon / self.pro_num) * (grad5 / cal_lp_norm(grad5,p=2,dim_count=len(self.y5.grad.size())))

            if 6 in self.enable_list6:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y6_reg.data = (self.epsilon / self.pro_num) * (self.y6.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y6_reg.data = (self.epsilon / self.pro_num) * (self.y6.grad / cal_lp_norm(self.y6.grad,p=2,dim_count=len(self.y6_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum6 = torch.zeros_like(self.y6).detach()
                    grad6 = self.y6.grad
                    grad_norm6 = torch.norm(nn.Flatten()(grad6), p=1, dim=1)
                    grad6 = grad6 / grad_norm6.view([-1]+[1]*(len(grad6.shape)-1))
                    grad6 = grad6 + self.decay_factor * momentum6
                    momentum6 = grad6
                    if self.norm=="linf":
                        self.y6_reg.data = (self.epsilon / self.pro_num) * (grad6.sign())

                    elif self.norm=="l2":
                        self.y6_reg.data = (self.epsilon / self.pro_num) * (grad6 / cal_lp_norm(grad6,p=2,dim_count=len(self.y6.grad.size())))
    
    def update_seed(self):
        self.seed1,self.seed2,self.seed3,self.seed4,self.seed5,self.seed6 = self.random()

    def random(self):
        seed = torch.rand(6)*0.7
        zs1= int(torch.clamp(seed[0]*10, min=0, max=6))
        zs2 = int(torch.clamp(seed[1]*10, min=0, max=6))
        zs3 = int(torch.clamp(seed[2]*10, min=0, max=6))
        zs4 = int(torch.clamp(seed[3]*10, min=0, max=6))
        zs5 = int(torch.clamp(seed[4]*10, min=0, max=6))
        zs6 = int(torch.clamp(seed[5]*10, min=0, max=6))
        return zs1,zs2,zs3,zs4,zs5,zs6

    def update_noise(self,epsilon,pro_num):
        self.epsilon = epsilon
        self.pro_num = pro_num

def set_noise(epoch,adv_type):
    if 0<=epoch < 9:
        if adv_type=="fgsm":
            return 0.5,1
        else:
            return 0.5, 25
    elif 9<=epoch < 19:
        if adv_type=="fgsm":
            return 0.4,1
        else:
            return 0.4, 20
    elif 19<=epoch < 29:
        if adv_type=="fgsm":
            return 0.6,1
        else:
            return 0.6, 20
    elif 29<=epoch < 39:
        if adv_type=="fgsm":
            return 0.3,1
        else:
            return 0.3, 15
    
    elif 39<=epoch<= 49:
        if adv_type=="fgsm":
            return 0.2,1
        else:
            return 0.2, 10
    

if __name__ == "__main__":
    args = parse_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    
    print("MLP arguments: %s " %(args))
    os.makedirs("pretrained", exist_ok=True)
    modelPath = f"pretrained/{args.dataset}_MLP_{time.time()}.pth"

    isCuda = torch.cuda.is_available()
    print(f"Use CUDA? {isCuda}")

    # Loading data
    t1 = time.time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    nUsers, nItems = train.shape
    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    
    # Build model
    userMatrix = torch.Tensor(get_train_matrix(train))
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    
    model = MLP(fcLayers, userMatrix, itemMatrix, alpha=args.alpha, epsilon=args.epsilon, pro_num=args.pro_num,\
                enable_lat=args.enable_lat, layerlist=args.layerlist, adv_type=args.adv_type,adv_reg=args.adv_reg, norm=args.norm,decay_factor=args.decay_factor, device=device)
    
    model.to(device) 
    torch.save(model.state_dict(), modelPath)
    
    optimizer = get_optimizer(args.optim, args.lr, model.parameters())

    # Check Init performance
    t1 = time.time()
    hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
        np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
    print(f"Init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    bestHr10, bestNdcg10,bestmap10, bestmrr10,bestHr20, bestNdcg20,bestmap20, bestmrr20,bestHr50, bestNdcg50,bestmap50, bestmrr50, bestEpoch = hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 , -1
    
    # Train model
    model.train()
    if args.enable_lat:
        model.grad_init()

    with open('result/%s_%s_robusttime.txt' %(args.adv_type, args.dataset), 'a') as f:
        for epoch in range(args.epochs):
            t1 = time.time()
            if epoch+1%10==0:
                print(f"model's random seed: seed1={model.seed1},seed2={model.seed2},seed3={model.seed3},seed4={model.seed4},seed5={model.seed5},seed6={model.seed6}")
            # Generate training instances
            userInput, itemInput, labels = get_train_instances(train, args.nNeg)
            dst = BatchDataset(userInput, itemInput, labels)
            ldr = torch.utils.data.DataLoader(dst, batch_size=args.bsz, shuffle=True, drop_last=True)
            losses = AverageMeter("Loss")
            if args.enable_lat:
                model.update_seed() 

            if args.enable_lat:
                args.epsilon, args.pro_num = set_noise(epoch,adv_type=args.adv_type)
                model.update_noise(args.epsilon, args.pro_num)
            
            for ui, ii, lbl in ldr:
                # if args.enable_lat:
                #     model.update_seed() 
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
                for iter in range(args.pro_num):
                    loss,_ = model(ui, ii, lbl)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if args.enable_lat:
                        model.save_grad() 
                    losses.update(loss.item(), lbl.size(0))

            print(f"Epoch {epoch+1}: Loss={losses.avg:.4f} [{time.time()-t1:.1f}s]")

            # Evaluation
            t1 = time.time()
            hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
            hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
            np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
            filename = "%s_%s.txt" %(args.adv_type, args.dataset)
            # 设置文件对象
            f.write(f"Epoch {epoch+1}: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
            f.write("\n")
            print(f"Epoch {epoch+1}: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
            if hr10 > bestHr10:
                bestHr10, bestNdcg10,bestmap10, bestmrr10,bestHr20, bestNdcg20,bestmap20, bestmrr20,bestHr50, bestNdcg50,bestmap50, bestmrr50, bestEpoch= hr10, ndcg10,\
                    map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50, epoch
                torch.save(model.state_dict(), modelPath)
        f.write(f"Best epoch {bestEpoch+1}:  HR10={bestHr10:.4f}, NDCG10={bestNdcg10:.4f}, mrrs10={bestmrr10:.4f}, HR20={bestHr20:.4f}, NDCG20={bestNdcg20:.4f}, mrrs20={bestmrr20:.4f}, HR50={bestHr50:.4f}, NDCG50={bestNdcg50:.4f}, mrrs50={bestmrr50:.4f}")
        print(f"Best epoch {bestEpoch+1}:  HR10={bestHr10:.4f}, NDCG10={bestNdcg10:.4f}, mrrs10={bestmrr10:.4f}, HR20={bestHr20:.4f}, NDCG20={bestNdcg20:.4f}, mrrs20={bestmrr20:.4f}, HR50={bestHr50:.4f}, NDCG50={bestNdcg50:.4f}, mrrs50={bestmrr50:.4f}")
        print(f"The best DMF model is saved to {modelPath}")
        f.close()