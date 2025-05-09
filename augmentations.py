import numpy as np
import pandas as pd
import torch
import torch.nn as nn



import torchaudio
import torchvision
import torch.nn as nn
import torch

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=.1, rand_std=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.rand_std = rand_std
        
    def forward(self, x):
        noise = torch.randn(x.size()).to(x.device)
        if self.rand_std:
            std = torch.rand(x.size(0)).to(x.device)*self.std
            std = std.view(-1, 1, 1, 1).float()
        else: std = self.std
        x = x + noise * std
        x = (x - x.min()) / (x.max() - x.min())
        return x


class Stretch(nn.Module):
    def __init__(self, factor=0, axis='both'):
        super().__init__()
        self.factor = factor
        self.axis = axis
        
    def forward(self, x):
        x.clone()
        resize = torchvision.transforms.Resize(size=x.shape[-2:])
        for i in range(x.size(0)):
            img = x[i]
            
            if self.axis=='both' or self.axis=='x':
                shift_factor = torch.rand(1)*self.factor
                shift = int(shift_factor*x.size(-2))
                if shift>0: img = img[:,shift:]
                elif shift<0: img = img[:,:shift]

            if self.axis=='both' or self.axis=='y':
                shift_factor = torch.rand(1)*self.factor
                shift = int(shift_factor*x.size(-1))
                if shift>0: img = img[:,:,shift:]
                elif shift<0: img = img[:,:,:shift]

            x[i] = resize(img)
            
        return x

        
class Shift(nn.Module):
    def __init__(self, factor=0, axis='both'):
        super().__init__()
        self.factor = factor
        self.axis = axis
        
    def forward(self, x):
        x = x.clone()
        
        for i in range(x.size(0)):
            img = x[i]
            
            if self.axis=='both' or self.axis=='y':
                shift_factor = (torch.rand(1)*2-1)*self.factor
                size = x.size(-2)
                shift = int(shift_factor*size)
                if shift>0: 
                    img[:,:size-shift] = img[:,shift:].clone()
                    img[:,size-shift:] *= 0
                elif shift<0: 
                    img[:,-shift:] = img[:,:shift].clone()
                    img[:,:-shift] *= 0

            if self.axis=='both' or self.axis=='y':
                shift_factor = (torch.rand(1)*2-1)*self.factor
                size = x.size(-1)
                shift = int(shift_factor*size)
                if shift>0: 
                    img[:,:,:size-shift] = img[:,:,shift:].clone()
                    img[:,:,size-shift:] *= 0
                elif shift<0: 
                    img[:,:,-shift:] = img[:,:,:shift].clone()
                    img[:,:,:-shift] *= 0

            img = (img-img.mean())/(torch.std(img)+1e-7)
            img = (img-img.min())/(img.max()-img.min())
            x[i] = img
        return x
        

class Mixup(nn.Module):
    def __init__(self, alpha=0.5, theta=1):
        super().__init__()
        self.alpha = alpha
        self.theta = theta

    def forward(self, x, y):
        batch_size = x.size(0)
        
        lam = np.random.beta(self.alpha,self.alpha)
        lam = max(lam, 1-lam)
        idx = torch.randperm(batch_size).to(x.device)

        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]
        y[y>self.theta] = 1
        
        return x, y

class CutMix(nn.Module):
    def __init__(self, alpha=8):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        batch_size = x.size(0)
        img_size = x.size(-1)
        
        lam = np.random.beta(self.alpha,self.alpha)
        lam = max(lam, 1-lam)
        idx = torch.randperm(batch_size).to(x.device)

        x = torch.cat([x[:,:,:,:int(lam*img_size)], x[idx,:,:,int(lam*img_size):]], dim=-1)
        y = lam * y + (1 - lam) * y[idx]
        
        return x, y


class Masking(nn.Module):
    def __init__(self, shape='square', N=4):
        super().__init__()
        self.shape = shape
        self.N = N

    def forward(self, x):
        return self.masking(x)
        
    def masking(self, x):
            batch_size = x.size(0)
            length = x.size(2)
            channels = x.size(1)
            device = x.device
            x = x.clone()
            
            W = torch.tensor(np.random.beta(4,32,(self.N,batch_size))*length).to(device=device, dtype=torch.int32).unsqueeze(-1)
            H = torch.tensor(np.random.beta(4,32,(self.N,batch_size))*length).to(device=device, dtype=torch.int32).unsqueeze(-1)
            X = torch.tensor(np.random.beta(1,2,(self.N,batch_size))*length).to(device=device, dtype=torch.int32).unsqueeze(-1)
            Y = torch.tensor(np.random.beta(1,2,(self.N,batch_size))*length).to(device=device, dtype=torch.int32).unsqueeze(-1)
    
            batch_indices = torch.arange(length, device=x.device).unsqueeze(0)
            
            mask_x = (batch_indices >= X) & (batch_indices <= X + W)
            mask_y = (batch_indices >= Y) & (batch_indices <= Y + H)
            
            mask_x = mask_x.unsqueeze(2).unsqueeze(3).repeat(1, 1, channels, length, 1)
            mask_y = mask_y.unsqueeze(2).unsqueeze(4).repeat(1, 1, channels, 1, length)

            if self.shape=='square':
                mask = ((mask_x & mask_y).sum(dim=0)>0)
            elif self.shape=='cross':
                mask = ((mask_x | mask_y).sum(dim=0)>0)
            elif self.shape=='horizontal': mask = (mask_y.sum(dim=0)>0)
            elif self.shape=='vertical': mask = (mask_x.sum(dim=0)>0)
            else: mask = (mask_y.sum(dim=0)>np.inf)

            x[mask] = 0
        
            return x