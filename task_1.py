import os
from os.path import join as pjoin
import collections
import json
import torch
import imageio
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

import pdb
import time
import torch.nn as nn
import torchvision.models.vgg as vgg
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

class pascalVOCDataset(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(
        self,
        root,
        sbd_path=None,
        split="train_aug",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.sbd_path = sbd_path
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 21
        #self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        if not self.test_mode:
            for split in ["train", "val", "trainval"]:
                path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list
            self.setup_annotations()

        self.tf = transforms.Compose(
            [
                # add more trasnformations as you see fit 
                #transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
        im = Image.open(im_path) 
        lbl = Image.open(lbl_path) 
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, torch.clamp(lbl, max=20)

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST) #TODO train with and without this
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0
        return img, lbl

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        # print(np.unique(label_mask))
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = self.sbd_path
        target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        train_aug = self.files["train"]

        # keep unique elements (stable)
        train_aug = [train_aug[i] for i in sorted(np.unique(train_aug, return_index=True)[1])]
        self.files["train_aug"] = train_aug
        set_diff = set(self.files["val"]) - set(train_aug)  # remove overlap
        self.files["train_aug_val"] = list(set_diff)

        pre_encoded = glob.glob(pjoin(target_path, "*.png"))
        expected = np.unique(self.files["train_aug"] + self.files["val"]).size

        if len(pre_encoded) != expected:
            print("Pre-encoding segmentation masks...")

            for ii in tqdm(self.files["trainval"]):
                fname = ii + ".png"
                lbl_path = pjoin(self.root, "SegmentationClass", fname)
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(pjoin(target_path, fname), lbl)

        assert expected == 2913, "unexpected dataset sizes"
        
class Segnet(nn.Module):
  
  def __init__(self, n_classes):
    super(Segnet, self).__init__()
    #define the layers for your model
    self.vgg_model = vgg.vgg16(pretrained=True, progress=True)#.to(device)
    #del self.vgg_model.classifier
    self.relu    = nn.ReLU(inplace=True)
    self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn1     = nn.BatchNorm2d(512) #TODO BN not mentioned in paper
    self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn2     = nn.BatchNorm2d(256)
    self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn3     = nn.BatchNorm2d(128)
    self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn4     = nn.BatchNorm2d(64)
    self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn5     = nn.BatchNorm2d(32)
    self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

  def forward(self, x):
    #define the forward pass
    #pdb.set_trace()
    x = self.vgg_model.features(x) # B, 
    output = self.vgg_model.avgpool(x) # B, 512, 512, 7
    output_zero = torch.zeros([4, 21, 512, 512], requires_grad=True) #always background
    score = self.bn1(self.relu(self.deconv1(x)))     # size=(N, 512, x.H/16, x.W/16)
    score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
    score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
    score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
    score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
    score = self.classifier(score)                    # size=(N, n_classes, x.H/1, x.W/1)
    return score  # size=(N, n_class, x.H/1, x.W/1)
    #return output_zero
    
    
expt = sys.argv[1]
os.makedirs(expt, exist_ok=True)
# Hyper-parameters
# Dataset options
local_path = 'VOCdevkit/VOC2012/' # modify it according to your device
bs = 32 #TODO increase
num_workers = 8 
n_classes = 21
img_size = 224 #'same'
#TODO weight decay, plot results for validation data
# Training parameters
epochs = 500 #use 200 
lr = 0.001

# Logging options
i_save = 50#save mode after every i_save epochs
i_vis = 5
rows, cols = 5, 2 #Show 10 images in the dataset along with target and predicted masks

# dataset variable
train_dst = pascalVOCDataset(local_path, split="train", is_transform=True, img_size=img_size)
val_dst = pascalVOCDataset(local_path, split="val", is_transform=True, img_size=img_size)

# dataloader variable
trainloader = torch.utils.data.DataLoader(train_dst, batch_size=bs, num_workers=num_workers, pin_memory=True, shuffle=True) 
valloader = torch.utils.data.DataLoader(val_dst, batch_size=bs, num_workers=num_workers, pin_memory=True, shuffle=True) 

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
num_gpu = list(range(torch.cuda.device_count()))  

# Creating an instance of the model defined above. 
# You can modify it incase you need to pass paratemers to the constructor.
#model = Segnet().to(device)
model = nn.DataParallel(Segnet(n_classes), device_ids=num_gpu).to(device)

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
def visualize(epoch, dst, image_ids):
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

#ckpt = torch.load(, pjoin(model_path, "{}.tar".format(epoch)))
#model.load_state_dict(ckpt)
        
# loss function
loss_f = nn.CrossEntropyLoss() 
softmax = nn.Softmax(dim=1)

# optimizer variable
opt = optim.Adam(model.parameters(), lr=lr) #Try SGD like in paper.. 

train_metrics = setup_metrics()
epoch = -1
#evaluate(epoch, trainloader, train_metrics)
#image_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
image_ids = np.random.randint(len(train_dst), size=rows*cols)
#visualize(epoch, train_dst, image_ids)

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
    losses.append(loss)
    print("Training epoch: {}, loss: {}, time elapsed: {},".format(epoch, loss, time.time() - st))
    
    evaluate(epoch, trainloader, train_metrics)
    if epoch % i_save == 0:
        torch.save(model.state_dict(), pjoin(expt, "{}.tar".format(epoch)))
    if epoch % i_vis == 0:
        plot_metrics(epoch, train_metrics, losses)
        visualize(epoch, train_dst, image_ids)

