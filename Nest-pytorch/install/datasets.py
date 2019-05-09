import os
import sys
import torch

from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Optional

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from nest import register
import xml.etree.ElementTree as ET # changed
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json
import pickle

@register
def image_transform(
    image_size: Union[int, List[int]],
    augmentation: dict,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """Image transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    # data augmentations
    horizontal_flip = augmentation.pop('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1

    vertical_flip = augmentation.pop('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1

    random_crop = augmentation.pop('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)

    center_crop = augmentation.pop('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))

    if len(augmentation) > 0:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))
    
    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0], **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    
    return transforms.Compose([v for v in t if v is not None])

@register
def fetch_data(
    dataset: Callable[[str], Dataset],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    train_splits: List[str] = [],
    test_splits: List[str] = [],
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    train_augmentation: dict = {},
    test_augmentation: dict = {},
    batch_size: int = 1,
    dataloader_flag: str = 'counting',
    test_batch_size: int = 1) -> Tuple[List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """
    currently, only support test_batch_size=1
    """

    # fetch training data
    train_transform = transform(augmentation=train_augmentation) if transform else None
    train_loader_list = []
    for split in train_splits:
        train_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = train_transform,
                target_transform = target_transform,
                dataloader_flag = dataloader_flag),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = train_shuffle)))
    
    # fetch testing data
    test_transform = transform(augmentation=test_augmentation) if transform else None
    test_loader_list = []
    for split in test_splits:
        test_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = test_transform,
                target_transform = target_transform,
                dataloader_flag = dataloader_flag),
            batch_size = batch_size if test_batch_size is None else test_batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = test_shuffle)))

    return train_loader_list, test_loader_list

@register
def pascal_voc_object_categories(query: Optional[Union[int, str]] = None) -> Union[int, str, List[str]]:
    """PASCAL VOC dataset class names.
    """

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']
        
    if query is None:
        return categories
    else:
        for idx, val in enumerate(categories):
            if isinstance(query, int) and idx == query:
                return val
            elif val == query:
                return idx


class VOC_Classification(Dataset):
    """Dataset for PASCAL VOC.
    """

    def __init__(self, data_dir, dataset, split, classes, dataloader_flag, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.dataset = dataset
        self.split = split
        self.image_dir = os.path.join(data_dir, dataset, 'JPEGImages')
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(self.data_dir, self.dataset, 'ImageSets', 'Main')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        if dataloader_flag=='counting':
            ## counting training dataloader
            self.image_labels = self._read_annotations_07_regression(self.split)
        elif dataloader_flag=='ins_seg':
            ## instance segmentation training dataloader
            self.image_labels = self._read_annotations_124_plus_segtrain_rm_val_regression(self.split)
        else:
            print("error, dataloader_flag is neither counting or ins_seg")

        print("number of images for %s: %d" %(split,len(self.image_labels)))

    def _read_annotations_124_plus_segtrain_rm_val_regression(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        dim=14
        with open(self.poc_path+'/Data/Datasets/Pascal_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt','r') as f:
            val_list=[]
            for ima in f:
                ima=ima.strip('\n')
                val_list.append(ima)
        with open('/media/guolei/DISK1TB/Datasets/SBD/trainval.txt','r') as f:
            val_list2=[]
            for ima in f:
                ima=ima.strip('\n')
                val_list2.append(ima)
        if os.path.exists(os.path.join(self.gt_path, split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.gt_path, self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if (name not in val_list) and ('2007_'+name not in val_list) and (name in val_list2):
                            if name not in class_labels:
                                class_labels[name] = [np.zeros(num_classes),np.zeros((num_classes,dim,dim),dtype=int)]
                            if int(label)!=-1:
                                count=self.return_count_obj(os.path.join(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/Annotations/',name+'.xml'),
                                    self.classes[class_idx])
                                mask_obj=self.return_mask_obj(os.path.join(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/Annotations/',name+'.xml'),
                                    self.classes[class_idx],dim)
                                if count!=np.sum(mask_obj):
                                    print(count,np.sum(mask_obj))
                                    import matplotlib.pyplot as plt
                                    plt.imshow(mask_obj)
                                    print(name)
                                    print(mask_obj)
                                    print('error')
                                    dd
                                class_labels[name][0][class_idx] = int(count)
                                class_labels[name][1][class_idx] = mask_obj
                            else:
                                class_labels[name][0][class_idx] = int(0)
        if os.path.exists(os.path.join(self.poc_path+'/Datasets/Pascal_2007/VOCdevkit/VOC2007/ImageSets/Main/', split + '.txt')):
            #for class_idx in range(num_classes):
             #   filename = os.path.join(
              #      '/media/rao/Data/Datasets/Pascal_2007/VOCdevkit/VOC2007/ImageSets/Main/', self.classes[class_idx] + '_' + split + '.txt')
            with open(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', 'r') as f:
                for line in f:
                    name = line.strip('\n')
                    if (name not in val_list) and ('2007_'+name not in val_list):
                        if name not in class_labels:
                            class_labels[name] = [np.zeros(num_classes),np.zeros((num_classes,dim,dim),dtype=int)]
                        # class_labels[name][class_idx] = int(label)
                            class_labels[name][0]=self.return_segtrain_gt(num_classes,name)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.dataset))

        return list(class_labels.items())

    def _read_annotations_07_regression(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        if os.path.exists(os.path.join(self.data_dir,self.dataset,'ImageSets/Main/', split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.data_dir,self.dataset,'ImageSets/Main/', self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if name not in class_labels:
                            class_labels[name] = np.zeros(num_classes)
                        class_labels[name][class_idx] = int(label)
                        if int(label)!=-1:
                            count=self.return_count_obj_rm_diff(os.path.join(self.data_dir,self.dataset,'Annotations',name+'.xml'),
                                self.classes[class_idx])
                            class_labels[name][class_idx] = int(count)
                        else:
                            class_labels[name][class_idx] = int(0)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.dataset))

        return list(class_labels.items())

    def return_count_obj_rm_diff(self,xml_file,class_name):
        count=0
        tree = ET.parse(xml_file)
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            if obj.find('name').text==class_name and int(obj.find('difficult').text)==0:
                count+=1
        return count

    def return_segtrain_gt(self,num_classes,ima):
        gt_one=np.zeros(num_classes,dtype=int)
        im = Image.open(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/SegmentationObject/'+ima+'.png') # Replace with your image name here
        indexed = np.int64(np.array(im))
        im2 = Image.open(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/SegmentationClass/'+ima+'.png') # Replace with your image name here
        indexed2 = np.int64(np.array(im2))
        indexed[indexed==255]=-1
        indexed2[indexed2==255]=-1
        uniq_ins=set(list(indexed.flatten()))
        for i in uniq_ins:
            if i >0:
                ins=indexed==i
                x,y=np.where(ins==True)
                label=indexed2[x[0],y[0]]
                gt_one[int(label)-1]+=1
        return gt_one
        
    def return_mask_obj(self,xml_file,class_name,dim):
        # dim=14
        mask_obj=np.zeros((dim,dim), dtype =int)
        tree = ET.parse(xml_file)
        objs = tree.findall('object')
        size = tree.findall('size')
        # print(size)
        width=float(size[0].find('width').text)
        height=float(size[0].find('height').text)
        for ix, obj in enumerate(objs):
            if obj.find('name').text.lower().strip()==class_name:
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)-1
                y1 = float(bbox.find('ymin').text)-1
                x2 = float(bbox.find('xmax').text)-1
                y2 = float(bbox.find('ymax').text)-1
                mask_obj[int(np.round((x1+x2)/2*dim/width-0.5)),int(np.round((y1+y2)/2*dim/height-0.5))]+=1
        return mask_obj      

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        if len(target)==2:
            target0=target[0]
            target1=target[1]
        elif isinstance(target,np.ndarray):
            target0=target
            target1=np.array([1])

        target0 = torch.from_numpy(target0).float()
        # print(type(1*target1))
        target1 = torch.from_numpy(1*target1).float()
        # target = torch.from_numpy(target).float()
        img = Image.open(os.path.join(
            self.image_dir, filename + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target0,target1

    def __len__(self):
        return len(self.image_labels)

@register
def pascal_voc_classification(
    split: str,
    data_dir: str,
    year: int = 2007,
    dataloader_flag: str = 'counting',
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None) -> Dataset:
    """PASCAL VOC dataset.
    """

    object_categories = pascal_voc_object_categories()
    dataset = 'VOC' + str(year)
    return VOC_Classification(data_dir, dataset, split, object_categories,dataloader_flag, transform, target_transform)

class coco_Classification(Dataset):
    """Dataset for COCO
    """

    def __init__(self, data_dir,split,year,dataloader_flag, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.split = split
        ## data_dir: /media/rao/Data/Datasets/MSCOCO/coco/
        self.image_dir = os.path.join(data_dir,'images',split+str(year))
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(self.data_dir, 'annotations')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        if dataloader_flag=='counting':
            ## use coco 2017 train data
            self.image_labels = self._read_annotations(split,year)
        else:
            print('error, dataloader_flag should be counting')
        if split=='val':
            index=int(len(self.image_labels)/2)
            self.image_labels=self.image_labels[:index]
        print(len(self.image_labels))

    def _read_annotations(self,split,year):
        gt_file=os.path.join(self.gt_path,'instances_'+split+str(year)+'.json')
        cocoGt=COCO(gt_file)
        catids=cocoGt.getCatIds()
        num_classes=len(catids)
        catid2index={}
        for i,cid in enumerate(catids):
            catid2index[cid]=i
        annids=cocoGt.getAnnIds()
        class_labels = OrderedDict()
        for id in annids:
            anns=cocoGt.loadAnns(id)
            for i in range(len(anns)):
                ann=anns[i]
                name=ann['image_id']
                if name not in class_labels:
                    class_labels[name]=np.zeros(num_classes)
                category_id=ann['category_id']
                class_labels[name][catid2index[category_id]]+=1
        return list(class_labels.items())            

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        target0=target
        target1=np.array([1])
        target0 = torch.from_numpy(target0).float()
        # print(type(1*target1))
        target1 = torch.from_numpy(1*target1).float()
        # target = torch.from_numpy(target).float()
        # 000000291625.jpg
        filename='0'*(12-len(str(filename)))+str(filename)
        img = Image.open(os.path.join(
            self.image_dir, 'COCO_train2014_'+ filename + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target0,target1

    def __len__(self):
        return len(self.image_labels)

@register
def coco_classification(
    split: str,
    data_dir: str,
    year: int = 2014,
    dataloader_flag: str = 'counting',
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None) -> Dataset:
    """COCO dataset.
    """

    return coco_Classification(data_dir, split,year,dataloader_flag, transform, target_transform)