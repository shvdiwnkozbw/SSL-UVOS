import os
import cv2
import glob
import torch
import random
import einops
import numpy as np
import glob as gb
from utils import read_flo
from torch.utils.data import Dataset

def readRGB(sample_dir, resolution):
    rgb = cv2.imread(sample_dir)
    try:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    except:
        print(sample_dir)
    rgb = rgb / 255
    if resolution[0] == -1:
        h = (resolution[0] // 8) * 8
        w = (resolution[1] // 8) * 8
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    return einops.rearrange(rgb, 'h w c -> c h w')

def readSeg(sample_dir, resolution=None):
    gt = cv2.imread(sample_dir) / 255
    if resolution:
        gt = cv2.resize(gt, (resolution[1], resolution[0]), interpolation=cv2.INTER_NEAREST)
    return einops.rearrange(gt, 'h w c -> c h w')

    
class Dataloader(Dataset):
    def __init__(self, data_dir, resolution, dataset, seq_length=3, gap=4, train=True, val_seq=None):
        self.dataset = dataset
        self.eval = eval
        self.data_dir = data_dir
        self.img_dir = data_dir[1]
        self.gap = gap
        self.resolution = resolution
        self.seq_length = seq_length
        if train:
            self.train = train
            self.img_dir = '/path/to/ytvis/train'
            self.seq = list([os.path.basename(x) for x in gb.glob(os.path.join(self.img_dir, '*'))])
        else:
            self.train = train
            self.seq = val_seq
        

    def __len__(self):
        if self.train:
            return 10000
        else:
            return len(self.seq)

    def __getitem__(self, idx):
        if self.train:
            seq_name = random.choice(self.seq)
            seq = os.path.join(self.img_dir, seq_name, '*.jpg')
            imgs = gb.glob(seq)
            imgs.sort()
            length = len(imgs)
            gap = self.gap
            while gap*(self.seq_length//2) >= length-gap*(self.seq_length//2)-1:
                gap = gap-1
            ind = random.randint(gap*(self.seq_length//2), length-gap*(self.seq_length//2)-1)
            
            seq_ids = [ind+gap*(i-(self.seq_length//2)) for i in range(self.seq_length)]

            rgb_dirs = [imgs[i] for i in seq_ids]
            rgbs = [readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs]
            out_rgb = np.stack(rgbs, 0) ## T, C, H, W 
            return out_rgb

        else:
            if self.dataset == 'FBMS':
                seq_name = self.seq[idx]
                rgb_dirs = sorted(os.listdir(os.path.join(self.data_dir[1], seq_name)))
                rgb_dirs = [os.path.join(self.data_dir[1], seq_name, x) for x in rgb_dirs if x.endswith(".jpg")]
                rgbs = np.stack([readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs], axis=0)
                gt_dirs = os.listdir(os.path.join(self.data_dir[2], seq_name))
                gt_dirs = sorted([gt for gt in gt_dirs if gt.endswith(".png")])
                val_idx = [int(x[:-4])-int(gt_dirs[0][:-4]) for x in gt_dirs if x.endswith(".png")]
                gt_dirs = [os.path.join(self.data_dir[2], seq_name, x) for x in gt_dirs if x.endswith(".png")]  
                gts = np.stack([readSeg(gt_dir) for gt_dir in gt_dirs], axis=0)
                return rgbs, gts, seq_name, val_idx
            else:
                seq_name = self.seq[idx]
                tot = len(glob.glob(os.path.join(self.data_dir[1], seq_name, '*')))
                rgb_dirs = [os.path.join(self.data_dir[1], seq_name, str(i).zfill(5)+'.jpg') for i in range(tot-1)]
                gt_dirs = [os.path.join(self.data_dir[2], seq_name, str(i).zfill(5)+'.png') for i in range(tot-1)]
                rgbs = np.stack([readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs], axis=0)
                gts = np.stack([readSeg(gt_dir) for gt_dir in gt_dirs], axis=0)
                return rgbs, gts, seq_name, [i for i in range(tot-1)]
                
