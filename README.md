# R2U-net
Pytorch Implementation of "Fully Convolutional Network", "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)" and "DeepLabV3" on PascalVOC and Cityscapes dataset.

## Contributors

Navami Kairanda
Priyanka Mohanta

## Requirements

Following packages are used

* python 3.8
* pytorch 1.7
* torchvision 0.8.1
* pytorch-lightning 1.2.3

## Prerequisites


For tasks 2 and 3, 
### Dataset preparation
Download and unzip gtFine_trainvaltest.zip (241MB) and leftImg8bit_trainvaltest.zip (11GB) from cityscapes site
https://www.cityscapes-dataset.com/downloads/

Generate trainId labels for the dataset, using the scripts provided by Cityscape authors https://github.com/mcordts/cityscapesScripts 
```
git clone https://github.com/mcordts/cityscapesScripts.git
pip install cityscapesScripts
CITYSCAPES_DATASET_PATH=/HPS/Navami/work/code/nnti/R2U-Net/cityscapes/
export CITYSCAPES_DATASET=$CITYSCAPES_DATASET_PATH
python /HPS/Navami/work/code/nnti/cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
Download resnet pretraineed model from https://download.pytorch.org/models/resnet50-19c8e357.pth and update corresponding path in resnet.py

## Train and Test

For task 1, run Vision_task_1.ipynb jupyter notebook

For tasks 2 and 3, 
```
python main.py /path/to/expt/logdir
```

## Test

For tasks 2 and 3, download model from Microsoft Teams

```
python eval.py /path/to/expt/logdir {model_name}.tar
```



## References


Task 1:
Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully Convolutional Networks for
Semantic Segmentation. arXiv e-prints, page arXiv:1411.4038, November 2014.

Task 2:
Md Zahangir Alom, Mahmudul Hasan, Chris Yakopcic, Tarek M Taha, and Vijayan K Asari.
Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image
segmentation. arXiv preprint arXiv:1802.06955, 2018.

Task 3:
Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous
convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.
