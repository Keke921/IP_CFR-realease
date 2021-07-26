import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class ImageDataset(Dataset):
    def __init__(self, root_base, root_txt, root_img, root_gallery,
                 transforms_=None, 
                 img_size   =None, 
                 mask_size  =None, 
                 train_rate = 0.9,
                 mode       ="train"):
        self.transform    = transforms.Compose(transforms_)
        self.img_size     = img_size
        self.mask_size    = mask_size
        self.mode         = mode
        self.root_base    = root_base
        self.root_gallery = root_gallery
        self.root_img     = root_img
        
        txtFiles = open(root_base + '/' + root_txt , 'r')
        files = []
        for line in txtFiles:
            line = line.rstrip()
            words = line.split()
            files.append((words[0], int(words[1])))
            self.files = files 
        
        gtFiles = open(root_base + '/' + root_gallery , 'r')
        gt_files = []
        for line in gtFiles:
            line = line.rstrip()
            words = line.split()
            gt_files.append(words[0])
            self.gt_files = gt_files 
        
        num_train = int(len(files)*train_rate)
        sorted(self.files)                 #random.shuffle(self.files)   
        
        if mode == "train":
            self.files = self.files[:num_train]
        elif mode == "test":
            self.files = self.files[num_train:]
        else: 
            self.files = self.files[num_train:]
        
    

    def apply_random_crop(self, img):
        """Randomly crop image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        croped_img = img[:, y1:y2, x1:x2]
        mask = torch.ones(3,self.img_size, self.img_size)
        mask[:, y1:y2, x1:x2] = 0
        mask_inv = torch.neg(mask)+torch.ones(3,self.img_size, self.img_size)
        #z = np.random.normal(0, 1, (3,self.img_size, self.img_size)).astype(np.float32)
        #z = torch.from_numpy(z)
        masked_img = img*mask_inv #masked_img = z*mask + img*mask_inv
        #masked_img = img.clone()
        #masked_img[:, y1:y2, x1:x2] = 1
 
        return masked_img,  croped_img

    def apply_center_crop(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        #masked_img = img.clone()
        #croped_img = img[:, i : i + self.mask_size, i : i + self.mask_size]
        mask = torch.ones(3,self.img_size, self.img_size)
        mask[:, i : i + self.mask_size, i : i + self.mask_size] = 0
        mask_inv = torch.neg(mask)+torch.ones(3,self.img_size, self.img_size)
        #z = np.random.normal(0, 1, (3,self.img_size, self.img_size)).astype(np.float32)
        #z = torch.from_numpy(z)
        masked_img = img*mask_inv #masked_img = z*mask + img*mask_inv
    
        return masked_img, i

    def __getitem__(self, index):
        fn, label = self.files[index % len(self.files)]
        root_fn = self.root_base + '/' + self.root_img + '/' + fn
        img = Image.open(root_fn)
        
        gt_fn = self.gt_files[(label-1)]
        root_gt = self.root_base + '/' + self.root_img + '/' + gt_fn
        img_gt = Image.open(root_gt)
        
        img_mode = 'RGB'
        if (img.mode!=img_mode):
            img = img.convert("RGB")
        img = self.transform(img)
        img_gt = self.transform(img_gt)
        if self.mode == "train":
            # For training data perform random mask
            croped_img, aux = self.apply_random_crop(img)
        else:
            # For test data mask the center of the image
            croped_img, aux = self.apply_center_crop(img)

        return img_gt, croped_img, label, img

    def __len__(self):
        return len(self.files)
