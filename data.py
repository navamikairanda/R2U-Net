import torch
from torchvision import datasets
import torchvision.transforms as transforms
import pdb

num_workers = 8
batch_size = 32

# Load Cityscapes train and test datasets 
def load_dataset(split='train', transform=transforms.ToTensor()):
    #transform = transforms.ToTensor() 
    train_data = datasets.Cityscapes(root='cityscapes', split=split, mode='fine', target_type='semantic', transform=transform)
    #test_data = datasets.Cityscapes(root='cityscapes', split='test', mode='fine', target_type='semantic', transform=transform)
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    return train_loader, train_data #test_loader
    
    
train_loader, train_data = load_dataset()