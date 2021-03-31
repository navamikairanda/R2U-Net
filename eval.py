import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import pdb
import sys
import os

from fcn import Segnet
from r2unet import U_Net, R2U_Net, RecU_Net, ResU_Net
from deeplabv3 import DeepLabV3
from dataloader import load_dataset
from metrics import Metrics
from vis import Vis

expt_logdir = sys.argv[1]
ckpt_name = sys.argv[2] 

#Dataset parameters
num_workers = 8
batch_size = 16
n_classes = 20
img_size = 224 
test_split = 'val'

# Logging options
rows, cols = 5, 2 #Show 10 images in the dataset along with target and predicted masks

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
num_gpu = list(range(torch.cuda.device_count()))  

testloader, test_dst = load_dataset(batch_size, num_workers, split=test_split)

# Creating an instance of the model 
#model = Segnet(n_classes) #Fully Convolutional Networks
#model = U_Net(img_ch=3,output_ch=n_classes) #U Network
#model = R2U_Net(img_ch=3,output_ch=n_classes,t=2) #Residual Recurrent U Network, R2Unet (t=2)
#model = R2U_Net(img_ch=3,output_ch=n_classes,t=3) #Residual Recurrent U Network, R2Unet (t=3)
#model = RecU_Net(img_ch=3,output_ch=n_classes,t=2) #Recurrent U Network, RecUnet (t=2)
#model = ResU_Net(img_ch=3,output_ch=n_classes) #Residual U Network, ResUnet 
#model = DeepLabV3(n_classes, 'vgg') #DeepLabV3 VGG backbone
model = DeepLabV3(n_classes, 'resnet') #DeepLabV3 Resnet backbone

print('Evaluation logs for model: {}'.format(model.__class__.__name__))

model = nn.DataParallel(model, device_ids=num_gpu).to(device)
model_params = torch.load(os.path.join(expt_logdir, "{}".format(ckpt_name)))
model.load_state_dict(model_params) 

#Visualization of test data
test_vis = Vis(test_dst, expt_logdir, rows, cols)
#Metrics calculator for test data
test_metrics = Metrics(n_classes, testloader, test_split, device, expt_logdir)

model.eval()
for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    predictions = model(inputs)

epoch = ckpt_name

test_metrics.compute(epoch, model)
test_metrics.plot_roc(epoch) 
test_vis.visualize(epoch, model)