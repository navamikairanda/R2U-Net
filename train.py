import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os.path import join as pjoin
import time
import pdb

from net import Segnet
from dataloader import load_dataset
from metrics import Metrics#setup_metrics, plot_metrics, visualize
from vis import visualize

expt_logdir = sys.argv[1]
os.makedirs(expt, exist_ok=True)

#local_path = 'VOCdevkit/VOC2012/' # modify it according to your device
#Dataset parameters
num_workers = 8
batch_size = 16 #TODO mulit-gpu, increase
n_classes = 20
img_size = 224 

# Training parameters
epochs = 1 #use 200 
lr = 0.001
#TODO weight decay, plot results for validation data

# Logging options
i_save = 50#save model after every i_save epochs
i_vis = 1
rows, cols = 5, 2 #Show 10 images in the dataset along with target and predicted masks

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
num_gpu = list(range(torch.cuda.device_count()))  

trainloader, train_dst = load_dataset(batch_size, num_workers)

# Creating an instance of the model defined in net.py 
#model = Segnet().to(device)
model = nn.DataParallel(Segnet(n_classes), device_ids=num_gpu).to(device)

# loss function
loss_f = nn.CrossEntropyLoss(ignore_index=19) #TODO s ignore_index required?
softmax = nn.Softmax(dim=1)

# optimizer variable
opt = optim.Adam(model.parameters(), lr=lr) #Try SGD like in paper.. 

train_metrics = Metrics(n_classes, trainloader, device)
#TODO random seed
image_ids = np.random.randint(len(train_dst), size=rows*cols)
epoch = -1
#train_metrics.evaluate(epoch, model)
#train_metrics.plot(epoch, losses)
visualize(epoch, train_dst, model, image_ids, rows, cols)
train_metrics.plot(epoch)

losses = []
for epoch in range(epochs):
    st = time.time()
    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        opt.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        #pdb.set_trace()
        predictions = model(inputs)
        loss = loss_f(predictions, labels)
        loss.backward()
        opt.step()
        print("Finish iter: {}, loss {}".format(i, loss.data))
    losses.append(loss)
    print("Training epoch: {}, loss: {}, time elapsed: {},".format(epoch, loss, time.time() - st))
    
    train_metrics.evaluate(epoch, model)
    if epoch % i_save == 0:
        torch.save(model.state_dict(), pjoin(expt, "{}.tar".format(epoch)))
    if epoch % i_vis == 0:
        train_metrics.plot(epoch, losses)
        visualize(epoch, train_dst, model, image_ids, rows, cols)

