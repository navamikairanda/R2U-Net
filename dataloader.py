import torch
#from torchvision import datasets

import torchvision.transforms as transforms
import pdb
import numpy as np

from dataset import Cityscapes
from dataset import ignoreClassId

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

    data_set = Cityscapes(root='cityscapes', split=split, mode='fine', target_type='semantic_basic', transform=image_transform, target_transform=target_transform)
    
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    return data_loader, data_set
    