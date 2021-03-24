import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os.path import join as pjoin
from net import Segnet
from data import load_dataset
import time
import pdb
from evaluate import setup_metrics, plot_metrics, visualize

#local_path = 'VOCdevkit/VOC2012/' # modify it according to your device
#bs = 32 #TODO increase
#num_workers = 8 
n_classes = 30 #TODO change?
img_size = 224 #'same'
#TODO weight decay, plot results for validation data
# Training parameters
epochs = 1 #use 200 
lr = 0.001

# Logging options
i_save = 50#save model after every i_save epochs
i_vis = 1
rows, cols = 5, 2 #Show 10 images in the dataset along with target and predicted masks

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
num_gpu = list(range(torch.cuda.device_count()))  

trainloader, train_dst = load_dataset()

# Creating an instance of the model defined above. 
# You can modify it incase you need to pass paratemers to the constructor.
#model = Segnet().to(device)
model = nn.DataParallel(Segnet(n_classes), device_ids=num_gpu).to(device)
# loss function
loss_f = nn.CrossEntropyLoss() 
softmax = nn.Softmax(dim=1)

# optimizer variable
opt = optim.Adam(model.parameters(), lr=lr) #Try SGD like in paper.. 

train_metrics = setup_metrics(n_classes)
#image_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
image_ids = np.random.randint(len(train_dst), size=rows*cols)
#epoch = -1
#evaluate(epoch, trainloader, train_metrics)
#visualize(epoch, train_dst, image_ids)

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
        print("Finish iter {}, loss {}".format(i, loss.data))
    losses.append(loss)
    print("Training epoch: {}, loss: {}, time elapsed: {},".format(epoch, loss, time.time() - st))
    
    evaluate(epoch, trainloader, model, train_metrics)
    if epoch % i_save == 0:
        torch.save(model.state_dict(), pjoin(expt, "{}.tar".format(epoch)))
    if epoch % i_vis == 0:
        plot_metrics(epoch, train_metrics, losses)
        visualize(epoch, train_dst, model, image_ids, rows, cols)

