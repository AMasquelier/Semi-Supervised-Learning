from torchvision.datasets import CocoDetection
from torchvision.datasets import VOCDetection
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch


class COCO(Dataset):
    def __init__(self, dataset='train'):
        root = f"../../Datasets/COCO/{dataset}2017" 
        annFile = f"../../Datasets/COCO/annotations/instances_{dataset}2017.json"
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = CocoDetection(root=root, annFile=annFile, transform=transform)
        self.n_classes = 80
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        y = torch.zeros(self.n_classes)
        image, labels = self.dataset[idx]

        for lbl in labels: 
            y[lbl-1] = 1
            
        return image, y

class VOC(Dataset):
    def __init__(self, image_set='train', label_policy='all'):
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.label_set = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.dataset = VOCDetection(root="./data", year="2012", image_set=image_set, download=True, transform=transform)
        self.n_classes = 20
        self.label_policy = label_policy
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        y = torch.zeros(self.n_classes)
        image, labels = self.dataset[idx]

        if self.label_policy=="first":
            y[self.label_set.index(labels['annotation']['object'][0]['name'])-1] = 1
        else:
            for lbl in labels['annotation']['object']:
                y[self.label_set.index(lbl['name'])-1] = 1
                
        return image, y


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
        y[label]=1
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
        y[label]=1
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