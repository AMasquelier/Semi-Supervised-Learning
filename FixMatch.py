import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
from IPython.display import clear_output

from augmentations import Masking, GaussianNoise, Stretch


# Implementation of FixMatch : https://proceedings.neurips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf



class FixMatch:
    
    def __init__(self, weak_transform=None, strong_transform=None, tau=0.95):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tau = tau

        if weak_transform: self.weak = weak_transform
        else:
            self.weak = torchvision.transforms.Compose([
                Masking('square', N=16)
            ])

        if strong_transform: self.strong = strong_transform
        else:
            self.strong = torchvision.transforms.Compose([
                Masking('cross', N=2),
                GaussianNoise(std=0.1),
                Stretch(0.1)
            ])
        

    def train_one_epoch(self, epoch, verbose=1):
        self.model.train()        
        Loss_U, Loss_L = 0, 0
        n_steps = len(self.labeled_loader)
        total_steps = self.epochs * n_steps
        n_pseudo_labels = 0
        
        tau = self.tau

        if verbose==2: pbar = tqdm(enumerate(self.labeled_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.labeled_loader)

        unlabeled_iter = iter(self.unlabeled_loader)
        
        for batch_idx, (x_l, y_l) in pbar:
            # Load unlabeled data
            try:
                x_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                x_u = next(unlabeled_iter)

            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device)
            x_u = x_u.to(self.device)
            batch_size = x_l.size(0)
            batch_size_u = x_u.size(0)
            
            # Augment + logits
            x_l = self.weak(x_l)
            x_u = torch.cat([self.weak(x_u),self.strong(x_u)])
            x = torch.cat([x_l, x_u], dim=0)

            self.optimizer.zero_grad()
            logits = self.model(x)

            logits_l = logits[:batch_size]
            logits_u = logits[batch_size:]
            logits_u_weak = logits_u[:batch_size_u]
            logits_u_strong = logits_u[batch_size_u:]
            
            
            # Pseudo-labels
            pseudo_label = nn.functional.softmax(logits_u_weak, dim=1)
            max_values, pseudo_label = torch.max(pseudo_label, dim=1)
            mask = (max_values >= tau).float()
            
            # Loss
            L_l = self.loss_fn(logits_l, y_l)
            L_u = (self.loss_fn(logits_u_strong, pseudo_label.detach(), reduction='none') * mask).mean()
            L = L_l + L_u

            L.backward()
            self.optimizer.step()

            n_pseudo_labels += mask.detach().sum()
            Loss_L += L_l.detach().item()
            Loss_U += L_u.detach().item()
            
        return [Loss_L + Loss_U, Loss_L, Loss_U], n_pseudo_labels
            
                
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

    
    def train(self, labeled_loader, unlabeled_loader, val_loader, model, optimizer, loss, metrics, scheduler=None, epochs=16, verbose=1, val_freq=4):
        self.epochs = epochs
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss
        self.metrics = metrics

        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.val_loader = val_loader

        log = pd.DataFrame(columns=['train_loss', 'train_loss_l', 'train_loss_u', 'val_loss', *[m.__name__ for m in self.metrics]])

        best = (np.inf,0,0)

        Epochs = range(epochs)
        
        for epoch in Epochs:
            train_losses,n_pseudo_labels = self.train_one_epoch(epoch, verbose=verbose)

            if epoch%val_freq==0 or epoch==epochs-1:
                val_scores, val_loss = self.validate(verbose=verbose)            
                if val_loss<best[0]: best = (val_loss, val_scores,epoch)
                    
                log.loc[len(log)] = [*train_losses, val_loss, *val_scores] 
                
                clear_output(wait=True)
                print(f"Epoch {epoch+1}/{epochs}")
                print(f'\033[1m Training \t|\t loss={np.round(train_losses, 3)}  |  {n_pseudo_labels} Pseudo-labels' + '\033[0m')
                print(f'\033[1m Validation \t|\t loss={np.round(val_loss, 3)}  -  ' + '  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.metrics, val_scores)])+'\033[0m')
                print()
                print(f"\033[1m Best : {'  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.metrics, best[1])])} at epoch {best[2]}")

            if self.scheduler: self.scheduler.step()
            
        return model, log