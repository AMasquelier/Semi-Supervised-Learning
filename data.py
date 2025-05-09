from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch



class Unlabeled(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image

class Labeled(Dataset):
    def __init__(self, dataset, n_classes):
        self.dataset = dataset
        self.n_classes = n_classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        y = torch.zeros(self.n_classes)
        image, label = self.dataset[idx]
        y[label-1]=1
        return image, y
        
class Dataset_OH(Dataset):
    def __init__(self, dataset, n_classes):
        self.dataset = dataset
        self.n_classes = n_classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        y = torch.zeros(self.n_classes)
        image, label = self.dataset[idx]
        y[label-1]=1
        return image, y

def get_datasets(root, name='mnist', transform=None, download=True, n_labeled=128, seed=2):
    if name.lower()=='mnist': 
        train = datasets.MNIST(root, train=True, transform=transform, download=download)
        test = datasets.MNIST(root, train=False, transform=transform, download=download)
        n_classes = 10
    if name.lower()=='cifar10': 
        train = datasets.CIFAR10(root, train=True, transform=transform, download=download)
        test = datasets.CIFAR10(root, train=False, transform=transform, download=download)
        n_classes = 10
        
    n = len(train)
    if n_labeled<1: n_labeled=int(n_labeled*n)
    
    labeled, unlabeled = torch.utils.data.random_split(
        train, [n_labeled, n-n_labeled], generator=torch.Generator().manual_seed(seed)
    )

    return Labeled(labeled, n_classes), Unlabeled(unlabeled), Dataset_OH(test, n_classes)