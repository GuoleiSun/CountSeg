save_path_2007='/raid/xiaoke/pascal_2007/'
save_path_2012='/raid/xiaoke/pascal_2012/'

if [[ ! -e $save_path_2007 ]]; then
    mkdir $save_path_2007
elif [[ ! -e $save_path_2012 ]]; then
    mkdir $save_path_2012
fi


cd $save_path_2007

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar


cd $save_path_2012

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_18-May-2011.tar

