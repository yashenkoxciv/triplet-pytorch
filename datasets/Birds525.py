#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

from random_erasing import RandomErasing
from sklearn.preprocessing import LabelEncoder


class Birds525(Dataset):
    '''
    a wrapper of Birds525 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(Birds525, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = glob.glob(data_path + '/**/*.jpg', recursive=True)
        #self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        
        #self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        #self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.lb_ids = []
        self.lb_cams = []
        raw_idxs = []
        for el in self.imgs:
            base, file_name = os.path.split(el)
            file_cam = int(file_name.split('.')[0])
            self.lb_cams.append(file_cam)

            base, file_dir = os.path.split(base)
            raw_idxs.append(file_dir)
        le = LabelEncoder()
        le.fit(raw_idxs)
        self.lb_ids = le.transform(raw_idxs)

        #self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        if is_train:
            self.trans = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
                RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
            ])
        else:
            self.trans_tuple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])
            self.Lambda = transforms.Lambda(
                lambda crops: [self.trans_tuple(crop) for crop in crops])
            self.trans = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.TenCrop((256, 128)),
                self.Lambda,
            ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx]



if __name__ == "__main__":
    ds = Birds525('./train', is_train = True)
    im, _, _ = ds[1]
    print(len(ds))
    print(im.shape)
    print(im.max())
    print(im.min())
    ran_er = RandomErasing()
    im = ran_er(im)
    cv2.imshow('erased', im)
    cv2.waitKey(0)
