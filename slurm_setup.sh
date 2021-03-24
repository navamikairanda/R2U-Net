wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

pip install scipy==1.1.0
conda install -c conda-forge imageio tqdm
conda install pytorch-lightning -c conda-forge

mkdir data

git clone https://github.com/mcordts/cityscapesScripts.git
pip install cityscapesScripts
CITYSCAPES_DATASET_PATH=/HPS/Navami/work/code/nnti/R2U-Net/cityscapes/
export CITYSCAPES_DATASET=$CITYSCAPES_DATASET_PATH
python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py