# Python modules remain same as task 1

# Dataset preparation
# Download and unzip gtFine_trainvaltest.zip (241MB) and leftImg8bit_trainvaltest.zip (11GB) from cityscapes site
https://www.cityscapes-dataset.com/downloads/

# Generate trainId labels for the dataset, using the scripts provided by Cityscape authors https://github.com/mcordts/cityscapesScripts 
git clone https://github.com/mcordts/cityscapesScripts.git
pip install cityscapesScripts
CITYSCAPES_DATASET_PATH=/HPS/Navami/work/code/nnti/R2U-Net/cityscapes/
export CITYSCAPES_DATASET=$CITYSCAPES_DATASET_PATH
python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py