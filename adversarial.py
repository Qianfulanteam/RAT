import argparse
import os
import time

import numpy as np
from MLP import MLP
import torch
import torch.nn as nn
from Dataset import Dataset
from evaluate_adv import evaluate_model
from utils import (MLP_new, BatchDataset, get_optimizer,
                   get_train_instances, get_train_matrix)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate.")
    parser.add_argument("--optim", nargs="?", default="adam",
                        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--enable_lat", nargs="?", default=True)
    parser.add_argument("--epsilon", nargs="?", default=0.5)
    parser.add_argument("--alpha", nargs="?", default=1)
    parser.add_argument("--pro_num", nargs="?", default=1, choices=[1,25])
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument("--layerlist", nargs="?", default="all")
    parser.add_argument("--adv_type", nargs="?", default="fgsm", choices=['fgsm', 'bim', 'pgd',"mim"])
    parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
    return parser.parse_args()

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)

grads = {} # 存储节点名称与节点的grad
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

# args.epsilon, args.alpha, args.pro_num

class advMLP(nn.Module):
    def __init__(self, model,new_model, userMatrix, itemMatrix, alpha, epsilon, pro_num, enable_lat, layerlist, norm, adv_type,decay_factor, device):
        super(advMLP, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.new_model = new_model
        self.pro_num = pro_num
        self.device = device
        self.decay_factor = decay_factor
        self.model = model
        self.adv_type = adv_type
        self.norm = norm
        self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg','y4_reg','y5_reg','y6_reg']
        self.enable_lat = enable_lat
        self.enable_list = [0 for i in range(7)]
        if enable_lat and layerlist != "all":
            self.layerlist = [int(x) for x in layerlist.split(',')]
            self.layerlist_digit = [int(x) for x in layerlist.split(',')]
        else:
            self.layerlist = "all"
            self.layerlist_digit = [0,1,2,3,4,5,6]
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)


    def forward(self, user, item,label):
        userInput = self.userMatrix[user, :].to(self.device)          # (B, 3706)
        itemInput = self.itemMatrix[item, :].to(self.device)          # (B, 6040)
        userVector = self.model.userFC(userInput).to(self.device)           # (B, fcLayers[0]//2)
        itemVector = self.model.itemFC(itemInput).to(self.device)           # (B, fcLayers[0]//2)

        self.y0 = torch.cat((userVector, itemVector), -1)  # (B, fcLayers[0])
        self.y1 = self.model.fcs[0](self.y0).to(self.device)  #torch.Size([100, 256])         
        if self.adv_type=="fgsm":
            criterion = torch.nn.BCELoss()
            self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
            ly = ly.squeeze()
            loss = criterion(ly, label)
            self.l1.register_hook(save_grad('1'))
            loss.register_hook(save_grad('loss'))
            loss.backward()
            y = self.y1
            y = y + grads['1'].sign()*self.epsilon
            delta = torch.clamp(y-self.y1, -self.epsilon, self.epsilon).to(self.device) 
            y = self.y1.add(delta).to(self.device)
        if self.adv_type=="bim":
            y = self.y1
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l1.register_hook(save_grad('1'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['1'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y1, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y1.add(delta).to(self.device)   
        if self.adv_type=="pgd":
            y = self.y1 + torch.empty_like(self.y1).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l1.register_hook(save_grad('1'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['1'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y1, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y1.add(delta).to(self.device)
        if self.adv_type=="mim":
            y = self.y1
            momentum = torch.zeros_like(self.y1).detach()
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l1.register_hook(save_grad('1'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                grad = grads['1']
                grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
                grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
                grad = grad + self.decay_factor * momentum
                momentum = grad
                y = y + grad.sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y1, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y1.add(delta).to(self.device)  
        self.y1_add = y
        
        self.z1 = self.model.fcs[1](self.y1_add).to(self.device) 
        self.y2 = self.model.fcs[2](self.z1).to(self.device) 
        if self.adv_type=="fgsm":
            criterion = torch.nn.BCELoss()
            self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
            ly = ly.squeeze()
            loss = criterion(ly, label)
            self.l3.register_hook(save_grad('2'))
            loss.register_hook(save_grad('loss'))
            loss.backward()
            y = self.y2
            y = y + grads['2'].sign()*self.epsilon
            delta = torch.clamp(y-self.y2, -self.epsilon, self.epsilon).to(self.device) 
            y = self.y2.add(delta).to(self.device)
        if self.adv_type=="bim":
            y = self.y2
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l3.register_hook(save_grad('2'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['2'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y2, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y2.add(delta).to(self.device)
        if self.adv_type=="pgd":
            y = self.y2 + torch.empty_like(self.y2).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l3.register_hook(save_grad('2'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['2'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y2, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y2.add(delta).to(self.device)
        if self.adv_type=="mim":
            y = self.y2
            momentum = torch.zeros_like(self.y2).detach()
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l3.register_hook(save_grad('2'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                grad = grads['2']
                grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
                grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
                grad = grad + self.decay_factor * momentum
                momentum = grad
                y = y + grad.sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y2, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y2.add(delta).to(self.device)  
        self.y2_add = y
        
        self.z2 = self.model.fcs[3](self.y2_add).to(self.device)  #torch.Size([100, 128])

        self.y3 = self.model.fcs[4](self.z2).to(self.device)   
        if self.adv_type=="fgsm":
            criterion = torch.nn.BCELoss()
            self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
            ly = ly.squeeze()
            loss = criterion(ly, label)
            self.l5.register_hook(save_grad('3'))
            loss.register_hook(save_grad('loss'))
            loss.backward()
            y = self.y3
            y = y + grads['3'].sign()*self.epsilon
            delta = torch.clamp(y-self.y3, -self.epsilon, self.epsilon).to(self.device) 
            y = self.y3.add(delta).to(self.device)
        if self.adv_type=="bim":
            y = self.y3
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l5.register_hook(save_grad('3'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['3'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y3, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y3.add(delta).to(self.device)
        if self.adv_type=="pgd":
            y = self.y3 + torch.empty_like(self.y3).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l5.register_hook(save_grad('3'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['3'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y3, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y3.add(delta).to(self.device)
        if self.adv_type=="mim":
            y = self.y3
            momentum = torch.zeros_like(self.y3).detach()
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l5.register_hook(save_grad('3'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                grad = grads['3']
                grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
                grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
                grad = grad + self.decay_factor * momentum
                momentum = grad
                y = y + grad.sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y3, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y3.add(delta).to(self.device)
        self.y3_add = y
        self.z3 = self.model.fcs[5](self.y3_add).to(self.device) 

        self.y4 = self.model.fcs[6](self.z3).to(self.device)  
        if self.adv_type=="fgsm":
            criterion = torch.nn.BCELoss()
            self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
            ly = ly.squeeze()
            loss = criterion(ly, label)
            self.l7.register_hook(save_grad('4'))
            loss.register_hook(save_grad('loss'))
            loss.backward()
            y = self.y4
            y = y + grads['4'].sign()*self.epsilon
            delta = torch.clamp(y-self.y4, -self.epsilon, self.epsilon).to(self.device) 
            y = self.y4.add(delta).to(self.device)
        if self.adv_type=="bim":
            y = self.y4
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l7.register_hook(save_grad('4'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['4'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y4, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y4.add(delta).to(self.device)
        if self.adv_type=="pgd":
            y = self.y4 + torch.empty_like(self.y4).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l7.register_hook(save_grad('4'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['4'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y4, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y4.add(delta).to(self.device)
        if self.adv_type=="mim":
            y = self.y4
            momentum = torch.zeros_like(self.y4).detach()
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l7.register_hook(save_grad('4'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                grad = grads['4']
                grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
                grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
                grad = grad + self.decay_factor * momentum
                momentum = grad
                y = y + grad.sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y4, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y4.add(delta).to(self.device)
        self.y4_add = y
        self.z4 = self.model.fcs[7](self.y4_add)

        self.y5 = self.model.fcs[8](self.z4).to(self.device)  
        if self.adv_type=="fgsm":
            criterion = torch.nn.BCELoss()
            self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
            ly = ly.squeeze()
            loss = criterion(ly, label)
            self.l9.register_hook(save_grad('5'))
            loss.register_hook(save_grad('loss'))
            loss.backward()
            y = self.y5
            y = y + grads['5'].sign()*self.epsilon
            delta = torch.clamp(y-self.y5, -self.epsilon, self.epsilon).to(self.device) 
            y = self.y5.add(delta).to(self.device)
        if self.adv_type=="bim":
            y = self.y5
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l9.register_hook(save_grad('5'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['5'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y5, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y5.add(delta).to(self.device)
        if self.adv_type=="pgd":
            y = self.y5 + torch.empty_like(self.y5).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l9.register_hook(save_grad('5'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['5'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y5, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y5.add(delta).to(self.device)
        if self.adv_type=="mim":
            y = self.y5
            momentum = torch.zeros_like(self.y5).detach()
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l9.register_hook(save_grad('5'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                grad = grads['5']
                grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
                grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
                grad = grad + self.decay_factor * momentum
                momentum = grad
                y = y + grad.sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y5, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y5.add(delta).to(self.device)
        self.y5_add = y
        self.z5 = self.model.fcs[9](self.y5_add)

        self.y6 = self.model.fcs[10](self.z5).to(self.device)  
        if self.adv_type=="fgsm":
            criterion = torch.nn.BCELoss()
            self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
            ly = ly.squeeze()
            loss = criterion(ly, label)
            self.l11.register_hook(save_grad('6'))
            loss.register_hook(save_grad('loss'))
            loss.backward()
            y = self.y6
            y = y + grads['6'].sign()*self.epsilon
            delta = torch.clamp(y-self.y6, -self.epsilon, self.epsilon).to(self.device) 
            y = self.y6.add(delta).to(self.device)
        if self.adv_type=="bim":
            y = self.y6
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l11.register_hook(save_grad('6'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['6'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y6, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y6.add(delta).to(self.device)
        if self.adv_type=="pgd":
            y = self.y6 + torch.empty_like(self.y6).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l11.register_hook(save_grad('6'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                y = y + grads['6'].sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y6, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y6.add(delta).to(self.device)
        if self.adv_type=="mim":
            y = self.y6
            momentum = torch.zeros_like(self.y6).detach()
            for i in range(self.pro_num):
                criterion = torch.nn.BCELoss()
                if i==0:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.model(user,item)
                else:
                    self.l1,self.l3,self.l5,self.l7,self.l9,self.l11, ly = self.new_model(y, user,item)
                ly = ly.squeeze()
                loss = criterion(ly, label)
                self.l11.register_hook(save_grad('6'))
                loss.register_hook(save_grad('loss'))
                loss.backward()
                grad = grads['6']
                grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
                grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
                grad = grad + self.decay_factor * momentum
                momentum = grad
                y = y + grad.sign()*(self.epsilon/self.pro_num)
                delta = torch.clamp(y-self.y6, -self.epsilon, self.epsilon).to(self.device) 
                y = self.y6.add(delta).to(self.device)
        self.y6_add = y
        self.z6 = self.model.fcs[11](self.y6_add)

                           
        y = self.model.final(self.z6)                            # (B, 1)
        return y

    
if __name__ == "__main__":
    args = parse_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    
    print("MLP arguments: %s " %(args))
    os.makedirs("pretrained", exist_ok=True)
    modelPath = f"mlp_pretrain/{args.dataset}_MLP_{args.lr}.pth"

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
    
    model = MLP(fcLayers, userMatrix, itemMatrix, device=device)
    pretrain_dict = torch.load(modelPath, map_location=device)
    model.load_state_dict(pretrain_dict['net'])
    model.to(device)
    new_model = MLP_new(model,userMatrix, itemMatrix, device)
    new_model.to(device)
    adversarial = advMLP(model,new_model,userMatrix, itemMatrix, alpha=args.alpha, epsilon=args.epsilon, pro_num=args.pro_num,\
         enable_lat=args.enable_lat, layerlist=args.layerlist, norm=args.norm, adv_type=args.adv_type, decay_factor=args.decay_factor, device=device )
    adversarial.to(device)
    
    t1 = time.time()
    hits, ndcgs,_,_,_,_,_,_,_,_,_,_ = evaluate_model(adversarial, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print(f"HR={hr:.4f}, NDCG={ndcg:.4f} [{time.time()-t1:.1f}s]")
