import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import timm


class Model(nn.Module):
    def __init__(self, num_classes, config=None):
        super().__init__()
        self.config = {
            'scale':1,
            'backbone_pooling':'avg',
            'backbone':'tf_efficientnetv2_b0',
            'dropout':0.1,
            'pretrained':False,
            'n_channels':3
        }
        if config: self.config.update(config)

        self.training = True
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            self.config['backbone'], 
            pretrained=self.config['pretrained'],  
            num_classes=0,  
            global_pool=self.config['backbone_pooling'],
            in_chans=self.config['n_channels'],
            drop_rate=self.config['dropout'],
        )
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, 128*self.config['scale']),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self.config['dropout']),
            torch.nn.Linear(128*self.config['scale'], self.num_classes)
        )
            
        
    def forward(self, x):
        x = self.backbone(x)
        labels = self.head(x)
        return labels