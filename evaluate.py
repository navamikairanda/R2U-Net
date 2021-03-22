from os.path import join as pjoin
from pytorch_lightning import metrics

def setup_metrics():
    auroc = metrics.AUROC(num_classes=n_classes).to(device)
    f1 = metrics.F1(num_classes=n_classes).to(device)
    iou = metrics.IoU(num_classes=n_classes).to(device)
    accuracy = metrics.Accuracy().to(device)
    #pdb.set_trace()
    #maintain all metrics required in this dictionary- these are used in the training and evaluation loops
    eval_metrics = {'accuracy': {'module': accuracy, 'values': []}, 
                    #'f1': {'module': f1, 'values': []}, #TODO, results are exactly same as accuracy, why? 
                    'iou': {'module': iou, 'values': []},
                    'auroc':{'module': auroc, 'values': []}
                    }
    return eval_metrics
                    
def evaluate(epoch, dataloader, eval_metrics): 
    #st = time.time()
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)#N, H, W
            labels = labels.to(device) #N, H, W
            predictions = model(inputs) #N, C, H, W
            predictions = softmax(predictions)
            '''
            #_, labels = torch.max(predictions, dim=1)
            auroc.update(predictions, labels)
            '''
            for key in eval_metrics: 
                eval_metrics[key]['module'].update(predictions, labels)
        #pdb.set_trace()
        for key in eval_metrics: 
            value = eval_metrics[key]['module'].compute()
            eval_metrics[key]['values'].append(value.item())
            eval_metrics[key]['module'].reset()
    metrics_string = " ; ".join("{}: {:05.3f}".format(key, eval_metrics[key]['values'][-1])
                                for key in eval_metrics)
    print("Training epoch: {}, Eval metrics - ".format(epoch) + metrics_string) 

def plot_metrics(epoch, eval_metrics, losses): 
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    #pdb.set_trace()
    for k, l in eval_metrics.items():
        ax.plot(l['values'], label=k)
    ax.plot(losses, label='loss')
    ax.legend(fontsize="16")
    ax.set_xlabel("Epochs", fontsize="16")
    ax.set_ylabel("Metric/Loss", fontsize="16")
    ax.set_title("Evaluation metric/Loss vs epochs", fontsize="16")
    plt.savefig(pjoin(expt, 'metric_{}_{}.png'.format('train', epoch)))
    
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
def visualize(epoch, dst, image_ids, rows, cols):
    images_vis = []
    for image_id in image_ids: 
        image, label = dst[image_id][0], dst[image_id][1]
        image = image[None, ...]

        prediction = model(image)
        prediction = torch.argmax(prediction, dim=1)

        prediction = torch.squeeze(prediction)
        image = torch.squeeze(image)
        
        image = image * dst.std[:, None, None] + dst.mean[:, None, None]
        image = torch.movedim(image, 0, -1) # (3,H,W) to (H,W,3) 

        image = image.cpu().numpy()
        label = label.cpu().numpy()
        prediction = prediction.cpu().numpy()

        label = dst.decode_segmap(label)
        prediction = dst.decode_segmap(prediction)
         
        image_vis = np.array([image, label, prediction])
        images_vis.append(image_vis)
    
    images_vis = np.concatenate(images_vis, axis=0)
    image_grid(images_vis, rows=rows, cols=3*cols) 
    plt.savefig(pjoin('vis', 'seg_{}_{}.png'.format(dst.split, epoch)))
