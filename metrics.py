import torch
import torch.nn as nn
from pytorch_lightning import metrics
import matplotlib.pyplot as plt
import os
import pdb

class Dice(metrics.Metric): 
    
    def __init__(self): 
        super().__init__()
        self.add_state("dice_score", default=[])
    
    def update(self, pred, target):
        dice_score_val = metrics.functional.classification.dice_score(pred, target, bg=True)
        self.dice_score.append(dice_score_val.item())
    
    def compute(self):
        self.dice_score = torch.tensor(self.dice_score)
        return torch.mean(self.dice_score)

    
class Metrics():
    def __init__(self, n_classes, dataloader, split, device, expt_logdir):
        self.dataloader = dataloader
        self.device = device
        #TODO, ROC-curve, Accuracy, AUC, SE, SP, IOU, F1, Dice
        accuracy = metrics.Accuracy().to(self.device) #TODO ignore_index doesn't work ignore_index=255
        dice = Dice().to(self.device)
        #auroc = metrics.AUROC(num_classes=n_classes).to(self.device)

        iou = metrics.IoU(num_classes=n_classes).to(self.device)
           
        #maintain all metrics required in this dictionary- these are used in the training and evaluation loops
        self.eval_metrics = {'accuracy': {'module': accuracy, 'values': []}, 
                        'iou': {'module': iou, 'values': []}, 
                        'dice': {'module': dice, 'values': []}
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
        for k, l in self.eval_metrics.items():
            ax.plot(l['values'], label=k)
        ax.legend(fontsize="16")
        ax.set_xlabel("Epochs", fontsize="16")
        ax.set_ylabel("Metric", fontsize="16")
        ax.set_title("Evaluation metric vs epochs", fontsize="16")
        plt.savefig(os.path.join(self.expt_logdir, 'metric_{}_{}.png'.format(self.split, epoch)))
        plt.clf()
    
