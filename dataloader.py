import torch
#from torchvision import datasets
from dataset import Cityscapes
import torchvision.transforms as transforms
import pdb
import numpy as np
from collections import namedtuple

ignoreClassId = 19
img_size = 256

def targetToTensor(target):
    """
    A util function for transforming target segmentation masks
    Args:
        target: (N, H, W) PIL images
    Returns:
        torch tensor of dimensions (N, H, W)     
    """
    target = np.array(target)
    target = np.where(target == 255, ignoreClassId, target)
    target = torch.as_tensor(target, dtype=torch.int64)
    return target
    
image_transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize([img_size,]), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])       
])

target_transform = transforms.Compose([
    transforms.Resize([img_size,], interpolation=0),#0:InterpolationMode.NEAREST
    targetToTensor        
])     
        
# Load Cityscapes train and test datasets 
def load_dataset(batch_size, num_workers, split='train'):

    train_data = Cityscapes(root='cityscapes', split=split, mode='fine', target_type='semantic_basic', transform=image_transform, target_transform=target_transform)
    #test_data = datasets.Cityscapes(root='cityscapes', split='test', mode='fine', target_type='semantic', transform=transform)
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    return train_loader, train_data #test_loader
    
'''
train_loader, train_data = load_dataset(8, 8)
data = train_data[0]
pdb.set_trace()
seg = data[1].detach().cpu().numpy()
train_data.decode_segmap(seg)
'''
