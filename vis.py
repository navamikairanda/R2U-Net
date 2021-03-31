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
    """
    Visualization module
    Saves the visualized segmentation images of dataset provided.

    Args:
        dst: train or validation dataset
        expt_logdir: number of rows in the grid
        rows: number of rows of the image
        cols: number of columns of the image
     

    Returns:
        None
    """
    def __init__(self, dst, expt_logdir, rows, cols):
        
        self.dst = dst
        self.expt_logdir = expt_logdir
        self.rows = rows
        self.cols = cols
        self.images = []
        self.images_vis = []
        self.labels_vis = []
        image_ids = np.random.randint(len(dst), size=rows*cols)
        
        for image_id in image_ids: 
            image, label = dst[image_id][0], dst[image_id][1]
            image = image[None, ...]
            self.images.append(image)
            
            image = torch.squeeze(image)  
            image = image * std[:, None, None] + mean[:, None, None]
            image = torch.movedim(image, 0, -1) # (3,H,W) to (H,W,3) 
            image = image.cpu().numpy()
            self.images_vis.append(image)
            
            label = label.cpu().numpy()
            label = dst.decode_segmap(label) 
            self.labels_vis.append(label)
                    
        self.images = torch.cat(self.images, axis=0)
        
    def visualize(self, epoch, model): 

        prediction = model(self.images) #TODO move to device?
        prediction = torch.argmax(prediction, dim=1)
        prediction = prediction.cpu().numpy()
        
        rgb_vis = []
        for image, label, pred in zip(self.images_vis, self.labels_vis, prediction):
            pred = self.dst.decode_segmap(pred)
            rgb_vis.extend([image, label, pred])
        rgb_vis = np.array(rgb_vis)
        
        image_grid(rgb_vis, rows=self.rows, cols=3*self.cols) 
        plt.savefig(os.path.join(self.expt_logdir, 'seg_{}_{}.png'.format(self.dst.split, epoch)))  #example file name: seg_val_0.png
