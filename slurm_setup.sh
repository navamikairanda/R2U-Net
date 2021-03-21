wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

pip install scipy==1.1.0
conda install -c conda-forge imageio tqdm
conda install pytorch-lightning -c conda-forge