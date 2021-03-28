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
from r2unet import U_Net, R2U_Net
from dataloader import load_dataset
from metrics import Metrics
from vis import Vis

expt_logdir = sys.argv[1]
os.makedirs(expt_logdir, exist_ok=True)

#Dataset parameters
num_workers = 8
batch_size = 8 #TODO mulit-gpu, increase
n_classes = 20
img_size = 224 
test_split = 'val'

# Training parameters
epochs = 51 #use 200 
lr = 0.001
#TODO weight decay, plot results for validation data

# Logging options
i_save = 50#save model after every i_save epochs
i_vis = 1
rows, cols = 5, 2 #Show 10 images in the dataset along with target and predicted masks

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
num_gpu = list(range(torch.cuda.device_count()))  

trainloader, train_dst = load_dataset(batch_size, num_workers, split='train')
testloader, test_dst = load_dataset(batch_size, num_workers, split=test_split)

# Creating an instance of the model defined in net.py 
#model = nn.DataParallel(Segnet(n_classes), device_ids=num_gpu).to(device) #Fully Convolutional Networks
model = nn.DataParallel(U_Net(img_ch=3,output_ch=n_classes), device_ids=num_gpu).to(device) #U Network
#model = nn.DataParallel(R2U_Net(img_ch=3,output_ch=n_classes,t=2), device_ids=num_gpu).to(device) #Residual Recurrent U Network

# loss function
loss_f = nn.CrossEntropyLoss() #TODO s ignore_index required? ignore_index=19

# optimizer variable
opt = optim.Adam(model.parameters(), lr=lr) 

#TODO random seed

train_vis = Vis(train_dst, expt_logdir, rows, cols)
test_vis = Vis(test_dst, expt_logdir, rows, cols)

train_metrics = Metrics(n_classes, trainloader, 'train', device, expt_logdir)
test_metrics = Metrics(n_classes, testloader, test_split, device, expt_logdir)

epoch = -1
'''
train_metrics.compute(epoch, model)
train_metrics.plot_scalar_metrics(epoch)
train_metrics.plot_roc(epoch) 
'''
train_vis.visualize(epoch, model)

test_metrics.compute(epoch, model)
test_metrics.plot_scalar_metrics(epoch)
test_metrics.plot_roc(epoch) 
test_vis.visualize(epoch, model)

losses = []
for epoch in range(epochs):
    st = time.time()
    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        opt.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        predictions = model(inputs)
        loss = loss_f(predictions, labels)
        loss.backward()
        opt.step()
        if i % 20 == 0:
            print("Finish iter: {}, loss {}".format(i, loss.data))
    losses.append(loss)
    print("Training epoch: {}, loss: {}, time elapsed: {},".format(epoch, loss, time.time() - st))
    
    #train_metrics.compute(epoch, model)
    test_metrics.compute(epoch, model)
    
    if epoch % i_save == 0:
        torch.save(model.state_dict(), os.path.join(expt_logdir, "{}.tar".format(epoch)))
    if epoch % i_vis == 0:
        '''
        train_metrics.plot_scalar_metrics(epoch) 
        train_metrics.plot_roc(epoch) 
        train_vis.visualize(epoch, model)
        '''        
        train_metrics.plot_loss(epoch, losses) 
        
        test_metrics.plot_scalar_metrics(epoch) 
        test_metrics.plot_roc(epoch) 
        test_vis.visualize(epoch, model)
