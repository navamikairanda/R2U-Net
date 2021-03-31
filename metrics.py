import torch
import torch.nn as nn
from pytorch_lightning import metrics
import matplotlib.pyplot as plt
import os
import pdb

class Dice(metrics.Metric): 
    '''
    Module to calcuate Dice metric
    
    Args:
        None
        
    Returns:
        None
        
    '''
    def __init__(self): 
        super().__init__()
        self.add_state("dice_score", default=[])
    
    def update(self, pred, target):
        '''
        Updates the parameters of dice coefficient
        Args:
            pred: The predicted value from the net
            target: The target value given
            
         Returns:
            None
        '''
        dice_score_val = metrics.functional.classification.dice_score(pred, target, bg=True)
        self.dice_score.append(dice_score_val.item())
    
    def compute(self):
        '''
        Computes the dice coefficient for the given parameters
        Args:
            None
         Returns:
            dice_score
        '''
        self.dice_score = torch.tensor(self.dice_score)
        return torch.mean(self.dice_score)

    
class Metrics():
    '''
    Metrics Calculator
    
    Calculates the required metrics for the given dataset and model.
    The metrics calculated are accuracy, iou, dice score, sensitivity, aucroc.
    Args:
        n_classes: Number of classes to predict
        dataloader: Dataloader of the dataset for which metric calculation is performed.
        split: Takes string input of 'train' or 'val' or any split provided by dataloader for training or validation data respectively
        device: Device value that contains the model.
        expt_logdir: Path to store the plots
        
     Returns: 
        None
    '''
    def __init__(self, n_classes, dataloader, split, device, expt_logdir):
        self.dataloader = dataloader
        self.device = device
        accuracy = metrics.Accuracy().to(self.device) 
        iou = metrics.IoU(num_classes=n_classes).to(self.device)
        dice = Dice().to(self.device)
        recall = metrics.Recall(num_classes=n_classes,average='macro', mdmc_average='global').to(self.device)
        roc = metrics.ROC(num_classes=n_classes,dist_sync_on_step=True).to(self.device)
        
        self.eval_metrics = {'accuracy': {'module': accuracy, 'values': []}, 
                        'iou': {'module': iou, 'values': []}, 
                        'dice': {'module': dice, 'values': []},
                        'sensitivity': {'module': recall, 'values': []},
                        'auroc': {'module': roc, 'values': []}
                        }
        self.softmax = nn.Softmax(dim=1)
        self.expt_logdir = expt_logdir
        self.split = split
    
    def compute_auroc(self, value):  #computes aucroc
        self.fpr, self.tpr, _ = value
        auc_scores = [torch.trapz(y, x) for x, y in zip(self.fpr, self.tpr)]
        return torch.mean(torch.stack(auc_scores))
        
    def compute(self, epoch, model):    #computes the metrics
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)#N, H, W
                labels = labels.to(self.device) #N, H, W

                predictions = model(inputs) #N, C, H, W
                predictions = self.softmax(predictions)

                for key in self.eval_metrics: 
                    #Evaluate AUC/ROC on subset of the training data, otherwise leads to OOM errors on GPU
                    #Full evaluation on validation/test data
                    if key == 'auroc' and i > 20: 
                        continue
                    self.eval_metrics[key]['module'].update(predictions, labels)
                    
            for key in self.eval_metrics: 
                value = self.eval_metrics[key]['module'].compute()
                if key == 'auroc':
                    value = self.compute_auroc(value)
                self.eval_metrics[key]['values'].append(value.item())
                self.eval_metrics[key]['module'].reset()
                
        metrics_string = " ; ".join("{}: {:05.3f}".format(key, self.eval_metrics[key]['values'][-1]) for key in self.eval_metrics)
        print("Split: {}, epoch: {}, metrics: ".format(self.split, epoch) + metrics_string) 

    def plot_scalar_metrics(self, epoch):  #for ploting the scalar metrics against epochs
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        for key, metric in self.eval_metrics.items():
            ax.plot(metric['values'], label=key)
        ax.legend(fontsize="16")
        ax.set_xlabel("Epochs", fontsize="16")
        ax.set_ylabel("Metric", fontsize="16")
        ax.set_title("Evaluation metric vs epochs", fontsize="16")
        plt.savefig(os.path.join(self.expt_logdir, 'metric_{}_{}.png'.format(self.split, epoch))) #example file name: 'metric_seg_val_100.png'
        plt.clf()
    
    def plot_roc(self, epoch): #for plotting roc
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        trainId2Name = self.dataloader.dataset.trainId2Name
        for class_idx, (x, y) in enumerate(zip(self.fpr, self.tpr)):
            class_idx = 255 if class_idx == 19 else class_idx
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), label=trainId2Name[class_idx])
        ax.legend(fontsize="8", ncol=2, loc='lower right')
        ax.set_xlabel("FPR (False Positive Rate)", fontsize="16")
        ax.set_ylabel("TPR (True Positive Rate)", fontsize="16")
        ax.set_title("ROC Curve", fontsize="16")
        plt.savefig(os.path.join(self.expt_logdir, 'roc_{}_{}.png'.format(self.split, epoch)))  #example file name: 'roc_seg_val_100.png'
        plt.clf()
        
    def plot_loss(self, epoch, losses):         #for plotting losses against epochs
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        ax.plot(losses)   
        ax.set_xlabel("Epochs", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Training loss vs. epochs", fontsize="16")
        plt.savefig(os.path.join(self.expt_logdir, 'loss_{}.png'.format(epoch))) #example file name: 'loss_100.png'
        plt.clf()
    
