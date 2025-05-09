import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
from IPython.display import clear_output

from augmentations import Masking, GaussianNoise, Stretch


# Implementation of AdaMactch : https://arxiv.org/pdf/2106.04732



class AdaMatch:
    
    def __init__(self, weak_transform=None, strong_transform=None, mu=5, tau=0.95):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mu = mu
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
        

    def train_one_epoch(self, epoch, verbose=2):
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
            self.optimizer.zero_grad()

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
            x_l = torch.cat([self.weak(x_l),self.strong(x_l)])
            x_u = torch.cat([self.weak(x_u),self.strong(x_u)])
            
            x = torch.cat([x_l, x_u], dim=0)
            logits = self.model(x)

            with torch.no_grad():
                logits_lpp = self.model(x_l)

            logits_lp = logits[:2*batch_size]
            logits_u = logits[2*batch_size:]
            

            # Random Logit Interpolation
            lam = torch.rand_like(logits_lp)
            logits_l = lam * logits_lp + (1-lam) * logits_lpp

            
            # Distribution Alignement
            logits_l_weak = logits_l[:batch_size]
            logits_l_strong = logits_l[batch_size:]
            pseudo_label_l = nn.functional.softmax(logits_l_weak, dim=1)

            logits_u = logits[2*batch_size:]
            logits_u_weak = logits_u[:batch_size_u]
            logits_u_strong = logits_u[batch_size_u:]
            pseudo_label_u = nn.functional.softmax(logits_u_weak, dim=1)
            
            expectation_ratio = (1e-7 + torch.mean(pseudo_label_l, dim=0)) / (1e-6 + torch.mean(pseudo_label_u, dim=0))
            pseudo_label_u = torch.nn.functional.normalize(pseudo_label_u * expectation_ratio)


            # Relative Confidence Threshold
            max_values, _ = torch.max(pseudo_label_l, dim=1)
            ct = tau * max_values.mean(dim=0)

            max_values, pseudo_label_u = torch.max(pseudo_label_u, dim=1)
            mask = (max_values >= ct).float()

            
            # Loss
            pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
            step = torch.tensor(epoch * n_steps + batch_idx, dtype=torch.float).to(self.device)
            mut = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2
            
            L_l = self.loss_fn(logits_l_weak, y_l) + self.loss_fn(logits_l_strong, y_l)
            L_u = (self.loss_fn(logits_u_strong, pseudo_label_u.detach(), reduction='none') * mask).mean()
            L = L_l + mut * L_u

            L.backward()
            self.optimizer.step()
            
            n_pseudo_labels += mask.detach().sum()
            Loss_L += L_l.detach().item()
            Loss_U += L_u.detach().item()
            

        return [Loss_L + Loss_U, Loss_L, Loss_U], n_pseudo_labels

        

    def train_one_epoch_multilabel(self, epoch, verbose=2):
        self.model.train()        
        Loss_U, Loss_L = 0, 0
        n_steps = len(self.labeled_loader)
        total_steps = self.epochs * n_steps
        
        batch_size = self.batch_size
        mu = self.mu
        tau = self.tau

        if verbose==2: pbar = tqdm(enumerate(self.labeled_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.labeled_loader)

        unlabeled_iter = iter(self.unlabeled_loader)
        
        for batch_idx, (x_l, y_l) in pbar:
            self.optimizer.zero_grad()

            # Load unlabeled data
            try:
                x_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                x_u = next(unlabeled_iter)

            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device)
            x_u = x_u.to(self.device)
            batch_size = x.size(0)

            
            # Augment + logits
            x_l = torch.cat([self.weak(x_l),self.strong(x_l)])
            x_u = torch.cat([self.weak(x_u),self.strong(x_u)])
            
            x = torch.cat([x_l, x_u], dim=0)
            logits = self.model(x)

            with torch.no_grad():
                logits_lpp = self.model(x_l)

            logits_lp = logits[:2*batch_size]
            logits_u = logits[2*batch_size:]
            

            # Random Logit Interpolation
            lam = torch.rand_like(logits_lp)
            logits_l = lam * logits_lp + (1-lam) * logits_lpp

            
            # Distribution Alignement
            logits_l_weak = logits_l[:batch_size]
            logits_l_strong = logits_l[batch_size:]
            pseudo_label_l = nn.functional.sigmoid(logits_l_weak)

            logits_u = logits[2*batch_size:]
            logits_u_weak = logits_u[:mu * batch_size]
            logits_u_strong = logits_u[mu * batch_size:]
            pseudo_label_u = nn.functional.sigmoid(logits_u_weak)
            
            expectation_ratio = (1e-7 + torch.mean(pseudo_label_l, dim=0)) / (1e-6 + torch.mean(pseudo_label_u, dim=0))
            pseudo_label_u = (pseudo_label_u * expectation_ratio) / torch.max(expectation_ratio)


            # Relative Confidence Threshold
            max_values, _ = torch.max(pseudo_label_l, dim=1)
            ct = tau * max_values.mean(dim=0)

            max_values, _ = torch.max(pseudo_label_u, dim=1)
            mask = (max_values >= ct).float().unsqueeze(1).repeat(1,logits_u_weak.size(1))

            pseudo_label_u = (pseudo_label_u>0.5).float()

            
            # Loss
            pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
            step = torch.tensor(epoch * n_steps + batch_idx, dtype=torch.float).to(self.device)
            mut = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2
            
            L_l = self.loss_fn(logits_l_weak, y_l) + self.loss_fn(logits_l_strong, y_l)
            L_u = (self.loss_fn(logits_u_strong, pseudo_label_u.detach(), reduction='none') * mask).mean()
            L = L_l + mut * L_u

            L.backward()
            self.optimizer.step()

            Loss_L += L_l.detach().item()
            Loss_U += L_u.detach().item()
            

        return [Loss_L + Loss_U, Loss_L, Loss_U], n_pseudo_labels
            
                
    def validate(self, verbose=2):
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
                pred.append(nn.functional.sigmoid(logits).detach().cpu().numpy())
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