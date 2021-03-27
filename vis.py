import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
 
 
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

def image_grid(images, rows=None, cols=None, fill=True, show_axes=False):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))

    for ax, im in zip(axarr.ravel(), images):
        # only render RGB channels
        ax.imshow(im[..., :3])
        if not show_axes:
            ax.set_axis_off()

class Vis():
    
    def __init__(self, dst, split, image_ids, expt_logdir, rows, cols):
        
        self.dst = dst
        self.split = split
        #self.image_ids = image_ids
        self.expt_logdir = expt_logdir
        self.rows = rows
        self.cols = cols
        self.images = []
        self.images_vis = []
        self.labels_vis = []
        #pdb.set_trace()
        for image_id in image_ids: 
            image, label = dst[image_id][0], dst[image_id][1]
            image = image[None, ...]
            #TODO create batch of images
            self.images.append(image)
            
            image = torch.squeeze(image)  
            image = image * std[:, None, None] + mean[:, None, None]
            image = torch.movedim(image, 0, -1) # (3,H,W) to (H,W,3) 
            image = image.cpu().numpy()
            
            label = label.cpu().numpy()
            label = dst.decode_segmap(label) 
            self.labels_vis.append(label)
            
            #image_vis = np.array([image, label])#, prediction])
            #self.images_vis.append(image_vis)
            self.images_vis.append(image)
        
        self.images = torch.cat(self.images, axis=0)
        #pdb.set_trace()
        #self.images_vis = np.concatenate(self.images_vis, axis=0)
        #self.labels_vis = np.concatenate(self.labels_vis, axis=0)
        self.images_vis = np.array(self.images_vis)
        self.labels_vis = np.array(self.labels_vis)
       
    # TODO - plot for test images      
    def visualize(self, epoch, model): #TODO make this a class
        #images_vis = []
        pdb.set_trace()
        prediction = model(self.images)
        
        prediction = torch.argmax(prediction, dim=1)
        #prediction = torch.squeeze(prediction)
        prediction = prediction.cpu().numpy()
        #prediction = self.dst.decode_segmap(prediction)
        #prediction = np.array([self.dst.decode_segmap(i_prediction) for i_prediction in prediction])
        #labels_vis
        rgb_vis = []
        for image, label, pred in zip(self.images_vis, self.labels_vis, prediction):
            pred = self.dst.decode_segmap(pred)
            rgb_vis.extend([image, label, pred])
        rgb_vis = np.array(rgb_vis)
        pdb.set_trace()
        #rgb_vis = np.array(rgb_vis)
        #images_print = np.append(self.images_vis, prediction, axis=0)
        #images_print = np.stack((A,B)).ravel('F')
        '''
        for image_id in image_ids: 
            image, label = self.dst[image_id][0], self.dst[image_id][1]
            image = image[None, ...]

            prediction = model(image)
            prediction = torch.argmax(prediction, dim=1)

            prediction = torch.squeeze(prediction)
            image = torch.squeeze(image)
            
            image = image * std[:, None, None] + mean[:, None, None]
            image = torch.movedim(image, 0, -1) # (3,H,W) to (H,W,3) 

            image = image.cpu().numpy()
            label = label.cpu().numpy()
            prediction = prediction.cpu().numpy()

            #pdb.set_trace()
            label = self.dst.decode_segmap(label)
            prediction = self.dst.decode_segmap(prediction)
             
            image_vis = np.array([image, label, prediction])
            images_vis.append(image_vis)
        
        images_vis = np.concatenate(images_vis, axis=0)
        '''
        image_grid(rgb_vis, rows=self.rows, cols=3*self.cols) 
        plt.savefig(os.path.join(self.expt_logdir, 'seg_{}_{}.png'.format(self.dst.split, epoch)))
