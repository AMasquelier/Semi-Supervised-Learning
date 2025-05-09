import numpy as np
import pandas as pd
import torch.nn as nn
import torch


class Focal_BCE(nn.Module):
    def __init__(self, smoothing=0, alpha=0.5, gamma=2.0, beta=0.5, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigmoid = nn.Sigmoid()
        self.LOSS = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, pred, target, reduction='mean'):
        n_classes = pred.size(1)
        pred = self.sigmoid(pred)

        pred = (1-self.smoothing) * pred + self.smoothing * 0.5
        
        epsilon = 1e-7
        pred = torch.clamp(pred, epsilon, 1 - epsilon)
        bce = -(target * torch.log(pred) + (1-target) * torch.log(1-pred))
    
        pt = target * pred + (1-target) * (1-pred)
        pt = torch.clamp(pt, epsilon, 1 - epsilon)
        focal_loss = bce * ((1-pt)**self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss

        if reduction=='mean': focal_loss = torch.mean(focal_loss)

        return focal_loss
        

class BCE(nn.Module):
    def __init__(self, smoothing=0, alpha=0.5, gamma=2.0, beta=0.5, theta=1, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.theta = theta
        self.sigmoid = nn.Sigmoid()
        self.LOSS = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, pred, target, w=1, reduction='mean'):
        n_classes = pred.size(1)
        pred = self.sigmoid(pred)

        pred = (1-self.smoothing) * pred + self.smoothing * 0.5
        
        epsilon = 1e-7
        pred = torch.clamp(pred, epsilon, 1 - epsilon)
        bce = -(target * torch.log(pred) + (1-target) * torch.log(1-pred)) * w
    
        pt = target * pred + (1-target) * (1-pred)
        pt = torch.clamp(pt, epsilon, 1 - epsilon)
        focal_loss = bce * ((1-pt)**self.gamma) * w

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss

        # t_bce = torch.quantile(bce, self.theta)
        # t_focal = torch.quantile(focal_loss, self.theta)
        # bce = bce * (bce <= t_bce)
        # focal_loss * (focal_loss <= t_focal)

        if reduction=='mean': 
            bce = torch.mean(bce)
            focal_loss = torch.mean(focal_loss)
        
        loss = self.beta * bce + (1-self.beta) * focal_loss

        return loss