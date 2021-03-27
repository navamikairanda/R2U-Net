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
from net import U_Net, R2U_Net
from dataloader import load_dataset
from metrics import Metrics
#from vis import visualize
from vis import Vis

expt_logdir = sys.argv[1]
os.makedirs(expt_logdir, exist_ok=True)

#local_path = 'VOCdevkit/VOC2012/' # modify it according to your device
#Dataset parameters
num_workers = 8
batch_size = 16 #TODO mulit-gpu, increase
n_classes = 20
img_size = 224 

# Training parameters
epochs = 51 #use 200 
lr = 0.001
#TODO weight decay, plot results for validation data

# Logging options
i_save = 50#save model after every i_save epochs
i_vis = 5
rows, cols = 5, 2 #Show 10 images in the dataset along with target and predicted masks

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
num_gpu = list(range(torch.cuda.device_count()))  

trainloader, train_dst = load_dataset(batch_size, num_workers)
testloader, test_dst = load_dataset(batch_size, num_workers, split='test')

# Creating an instance of the model defined in net.py 
#model = nn.DataParallel(Segnet(n_classes), device_ids=num_gpu).to(device)
model = nn.DataParallel(U_Net(img_ch=3,output_ch=n_classes), device_ids=num_gpu).to(device)
#model = nn.DataParallel(R2U_Net(img_ch=3,output_ch=n_classes,t=2), device_ids=num_gpu).to(device)

# loss function
loss_f = nn.CrossEntropyLoss() #TODO s ignore_index required? ignore_index=19
#softmax = nn.Softmax(dim=1)

# optimizer variable
opt = optim.Adam(model.parameters(), lr=lr) #Try SGD like in paper.. 

train_metrics = Metrics(n_classes, trainloader, 'train', device, expt_logdir)
test_metrics = Metrics(n_classes, testloader, 'test', device, expt_logdir)

#TODO random seed
image_ids = np.random.randint(len(train_dst), size=rows*cols)
train_vis = Vis(train_dst, 'train', image_ids, expt_logdir, rows, cols)

epoch = -1
train_vis.visualize(epoch, model)

train_metrics.evaluate(epoch, model)
train_metrics.plot(epoch)
#visualize(epoch, train_dst, model, image_ids, rows, cols, expt_logdir)

test_metrics.evaluate(epoch, model)
test_metrics.plot(epoch)
visualize(epoch, test_dst, model, image_ids, rows, cols, expt_logdir)

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
        print("Finish iter: {}, loss {}".format(i, loss.data))
    losses.append(loss)
    print("Training epoch: {}, loss: {}, time elapsed: {},".format(epoch, loss, time.time() - st))
    
    train_metrics.evaluate(epoch, model)
    if epoch % i_save == 0:
        torch.save(model.state_dict(), os.path.join(expt_logdir, "{}.tar".format(epoch)))
    if epoch % i_vis == 0:
        train_metrics.plot(epoch) #TODO plot losses
        visualize(epoch, train_dst, model, image_ids, rows, cols, expt_logdir)
        test_metrics.plot(epoch) #TODO plot losses
        visualize(epoch, test_dst, model, image_ids, rows, cols, expt_logdir)

