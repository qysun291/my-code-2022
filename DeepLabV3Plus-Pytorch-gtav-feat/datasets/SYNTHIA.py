import json
import os
from collections import namedtuple
import random

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os.path as osp
import imageio
import cv2
import albumentations as A

class SYNTHIA(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('sky',                  1, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('building',             2, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('road',                 3, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             4, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('fence',                5, 4, 'construction', 2, False, False, (64, 64, 128)),
        CityscapesClass('vegetation',           6, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('pole',                 7, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('car',                  8, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('traffic sign',         9, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('person',               10, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('bicycle',              11, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('motorcycle',           12, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('parking',              13, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('Road-work',            14, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('traffic light',        15, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('terrain',              16, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('rider',                17, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('truck',                18, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  19, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('train',                20, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('wall',                 21, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('Lanemarking',          22, 255, 'construction', 2, False, False, (102, 102, 156)),
    ]


    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root='/root/Desktop/DeepLabV3Plus-Pytorch-master/datasets/data/SYNTHIA', 
                 list_path='/root/Desktop/DeepLabV3Plus-Pytorch-master/datasets/data/SYNTHIA/train.txt', transform=None):
        self.root = root
        self.list_path = list_path
        self.transform = transform
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        # self.id_to_trainid = {3: 0 , 4 : 1,  2 : 2 , 21: 3 , 5 : 4 , 7 : 5,
        #             15: 6 , 9 : 7,  6 : 8 , 16: 9 , 1 : 10, 10: 11, 17: 12,
        #             8 : 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        # for split in ["train", "trainval", "val"]:
        # self.images = []
        # self.targets = []
        self.files = []
        for name in self.img_ids:
            # img_file = osp.join(self.root, "RGB/%s" % name)
            # label_file = osp.join(self.root, "GT/LABELS/%s" % name)
            # # self.files.append({
            # #     "img": img_file,
            # #     "label": label_file,
            # #     "name": name
            # # })
            # self.images.append(img_file)
            # self.targets.append(label_file)
            
            img_file = osp.join(self.root, "RGB/%s" % name)
            label_file = osp.join(self.root, "GT/LABELS/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file
            })

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        # index = index + 1000  
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        # target = Image.open(datafiles["label"])

        mask = cv2.imread(datafiles["label"], -1)[:, :, -1]
        mask = Image.fromarray(mask)
        mask = mask.convert('L')

        # target = np.asarray(imageio.imread(datafiles["label"], format='PNG-FI'))[:,:,0]  # uint16
        
        # target = Image.fromarray(target)
        
        target = mask
        
        
        # weather_transforms = [A.RandomSunFlare(), A.RandomSnow(), A.RandomShadow(), A.RandomRain(), A.RandomFog()] 
        
        # color_transform = A.Compose([
        #             # A.RGBShift(p=0.5),
        #             # A.HueSaturationValue(p=0.5),
        #             # A.ChannelShuffle(p=0.5),
        #             # A.CLAHE(p=0.5),
        #             # A.RandomContrast(p=0.5),
        #             A.RandomGamma(p=0.5),
        #             # A.RandomBrightness(p=0.5),
                    
        #         ])
        
        # weather = random.randint(0, len(weather_transforms)-1)
        
        
        # transformed = color_transform(image=np.array(image))
        # image = Image.fromarray(transformed["image"])
        
        # transformed = weather_transforms[weather](image=np.array(image))
        # image = Image.fromarray(transformed["image"])
            
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.files)

    