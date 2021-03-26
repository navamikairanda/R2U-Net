import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
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


# TODO - plot for test images      
def visualize(epoch, dst, model, image_ids, rows, cols, expt_logdir): #TODO make this a class
    images_vis = []
    for image_id in image_ids: 
        image, label = dst[image_id][0], dst[image_id][1]
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
        label = dst.decode_segmap(label)
        prediction = dst.decode_segmap(prediction)
         
        image_vis = np.array([image, label, prediction])
        images_vis.append(image_vis)
    
    images_vis = np.concatenate(images_vis, axis=0)
    image_grid(images_vis, rows=rows, cols=3*cols) 
    plt.savefig(pjoin(expt_logdir, 'seg_{}_{}.png'.format(dst.split, epoch)))
