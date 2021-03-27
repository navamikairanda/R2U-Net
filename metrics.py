import torch
import torch.nn as nn
from pytorch_lightning import metrics
import matplotlib.pyplot as plt
import os
import pdb

class Metrics():
    def __init__(self, n_classes, dataloader, split, device, expt_logdir):
        self.dataloader = dataloader
        self.device = device
        accuracy = metrics.Accuracy().to(self.device) #TODO ignore_index doesn't work ignore_index=255
        #auroc = metrics.AUROC(num_classes=n_classes, ignore_index=255).to(self.device)
        #f1 = metrics.F1(num_classes=n_classes, ignore_index=255).to(self.device)
        iou = metrics.IoU(num_classes=n_classes).to(self.device)
           
        #maintain all metrics required in this dictionary- these are used in the training and evaluation loops
        self.eval_metrics = {'accuracy': {'module': accuracy, 'values': []}, 
                        #'f1': {'module': f1, 'values': []}, #TODO, results are exactly same as accuracy, why? 
                        'iou': {'module': iou, 'values': []}
                        #'auroc':{'module': auroc, 'values': []}
                        }
        self.softmax = nn.Softmax(dim=1)
        self.expt_logdir = expt_logdir
        self.split = split
        
    def compute(self, epoch, model): 
        #st = time.time()
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)#N, H, W
                labels = labels.to(self.device) #N, H, W
                #mask = (labels != self.255)
                #labels = labels[mask]
                #pdb.set_trace()
                predictions = model(inputs) #N, C, H, W
                predictions = self.softmax(predictions)
                for key in self.eval_metrics: 
                    self.eval_metrics[key]['module'].update(predictions, labels)
            for key in self.eval_metrics: 
                value = self.eval_metrics[key]['module'].compute()
                self.eval_metrics[key]['values'].append(value.item())
                self.eval_metrics[key]['module'].reset()
        metrics_string = " ; ".join("{}: {:05.3f}".format(key, self.eval_metrics[key]['values'][-1])
                                    for key in self.eval_metrics)
        print("Split: {}, epoch: {}, metrics: ".format(self.split, epoch) + metrics_string) 

    def plot(self, epoch): 
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        #pdb.set_trace()
        for k, l in self.eval_metrics.items():
            ax.plot(l['values'], label=k)
        #ax.plot(losses, label='loss')
        ax.legend(fontsize="16")
        ax.set_xlabel("Epochs", fontsize="16")
        ax.set_ylabel("Metric", fontsize="16")
        ax.set_title("Evaluation metric vs epochs", fontsize="16")
        plt.savefig(os.path.join(self.expt_logdir, 'metric_{}_{}.png'.format(self.split, epoch)))
    
