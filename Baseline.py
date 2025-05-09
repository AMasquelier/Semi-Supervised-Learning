import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
from IPython.display import clear_output

from augmentations import Masking, GaussianNoise, Stretch



class Baseline:
    
    def __init__(self, weak_transform=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if weak_transform: self.weak = weak_transform
        else:
            self.weak = torchvision.transforms.Compose([
                Masking('square', N=16)
            ])
            

    def train_one_epoch(self, epoch, verbose=1):
        self.model.train()        
        Loss = 0
        n_steps = len(self.labeled_loader)
        total_steps = self.epochs * n_steps

        if verbose==2: pbar = tqdm(enumerate(self.labeled_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.labeled_loader)

        
        for batch_idx, (x, y) in pbar:
            self.optimizer.zero_grad()

            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = x.size(0)
            
            
            # Augment + logits
            x = self.weak(x)
            logits = self.model(x)
            
            L = self.loss_fn(logits, y)

            L.backward()
            self.optimizer.step()

            Loss += L.detach().item()

        return Loss

        
                
    def validate(self, verbose=1):
        self.model.eval()        
        Loss = 0

        n_steps = len(self.val_loader)

        if verbose==2: pbar = tqdm(enumerate(self.val_loader), total=n_steps, desc="Validation")
        else: pbar = enumerate(self.val_loader)

        pred = []
        target = []

        with torch.no_grad():
            for batch_idx, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                
                loss = self.loss_fn(logits, y)

                Loss += loss.detach().item()
                pred.append(nn.functional.softmax(logits, dim=1).detach().cpu().numpy())
                target.append(y.detach().cpu().numpy())

        target = np.concatenate(target)
        pred = np.concatenate(pred)
        scores = []
        for m in self.metrics:
            scores.append(m(target, pred))
        return scores, Loss

    
    def train(self, labeled_loader, val_loader, model, optimizer, loss, metrics, scheduler=None, epochs=16, verbose=1, val_freq=4):
        self.epochs = epochs
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss
        self.metrics = metrics

        self.labeled_loader = labeled_loader
        self.val_loader = val_loader

        log = pd.DataFrame(columns=['train_loss', 'val_loss', *[m.__name__ for m in self.metrics]])

        best = (np.inf,0,0)
        
        Epochs = range(epochs)
        
        for epoch in Epochs:
            train_losses = self.train_one_epoch(epoch,verbose=verbose)
            
            if epoch%val_freq==0 or epoch==epochs-1:
                val_scores, val_loss = self.validate(verbose=verbose)            
                if val_loss<best[0]: best = (val_loss, val_scores,epoch)
                    
                log.loc[len(log)] = [train_losses, val_loss, *val_scores] 
                
                clear_output(wait=True)
                print(f"Epoch {epoch+1}/{epochs}")
                print(f'\033[1m Training \t|\t loss={np.round(train_losses, 3)}' + '\033[0m')
                print(f'\033[1m Validation \t|\t loss={np.round(val_loss, 3)}  -  ' + '  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.metrics, val_scores)])+'\033[0m')
                print()
                print(f"\033[1m Best : {'  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.metrics, best[1])])} at epoch {best[2]}")

            if self.scheduler: self.scheduler.step()
                
        return model, log