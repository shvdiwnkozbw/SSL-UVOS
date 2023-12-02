from matplotlib.pyplot import sca
import torchvision
import skimage

import torch
from torchvision import transforms

import numpy as np
from PIL import Image
import random

IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD  = (0.2023, 0.1994, 0.2010)
NORM = [transforms.ToTensor(), 
        transforms.Normalize(IMG_MEAN, IMG_STD)]


class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])
    
def n_patches(x, n, transform, shape=(64, 64, 3), scale=[0.2, 0.8]):
    ''' unused '''
    if shape[-1] == 0:
        shape = np.random.uniform(64, 128)
        shape = (shape, shape, 3)

    crop = transforms.Compose([
        lambda x: Image.fromarray(x) if not 'PIL' in str(type(x)) else x,
        transforms.RandomResizedCrop(shape[0], scale=scale)
    ])    

    if torch.is_tensor(x):
        x = x.numpy().transpose(1,2, 0)
    
    P = []
    for _ in range(n):
        xx = transform(crop(x))
        P.append(xx)

    return torch.cat(P, dim=0)


def patch_grid(transform, shape=(64, 64, 3), stride=[0.5, 0.5]):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]
    
    spatial_jitter = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomResizedCrop(shape[0], scale=(0.7, 0.9))
    ])

    def aug(x):
        if torch.is_tensor(x):
            x = x.numpy().transpose(1, 2, 0)
        elif 'PIL' in str(type(x)):
            x = np.array(x)#.transpose(2, 0, 1)
        
        winds = skimage.util.view_as_windows(x, shape, step=stride)
        winds = winds.reshape(-1, *winds.shape[-3:])

        P = [transform(spatial_jitter(w)) for w in winds]
        return torch.cat(P, dim=0)

    return aug


def get_frame_aug(frame_aug, patch_size):
    train_transform = []

    if 'cj' in frame_aug:
        _cj = 0.1
        train_transform += [
            #transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(_cj, _cj, _cj, 0),
        ]
    if 'flip' in frame_aug:
        train_transform += [transforms.RandomHorizontalFlip()]

    train_transform += NORM
    train_transform = transforms.Compose(train_transform)

    print('Frame augs:', train_transform, frame_aug)

    if 'grid' in frame_aug:
        aug = patch_grid(train_transform, shape=np.array(patch_size))
    else:
        aug = train_transform

    return aug


def get_frame_transform(frame_transform_str, img_size):
    tt = []
    fts = frame_transform_str
    norm_size = torchvision.transforms.Resize((img_size, img_size))

    if 'all' in fts:
        tt = [torchvision.transforms.RandomResizedCrop(
            img_size, scale=(0.4, 1), interpolation=Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomGrayscale(p=0.2)
        ]
        print('Frame transforms:', tt, fts)
        return tt

    if 'crop' in fts:
        tt.append(torchvision.transforms.RandomResizedCrop(
            img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),)
    else:
        tt.append(norm_size)

    if 'cj' in fts:
        _cj = 0.1
        # tt += [#transforms.RandomGrayscale(p=0.2),]
        tt += [transforms.ColorJitter(_cj, _cj, _cj, 0),]

    if 'flip' in fts:
        tt.append(torchvision.transforms.RandomHorizontalFlip())

    print('Frame transforms:', tt, fts)

    return tt

def get_train_transforms(args):
    norm_size = torchvision.transforms.Resize((args.resolution[0], args.resolution[1]))
    radius_0 = int(0.1*args.resolution[0])//2*2 + 1
    radius_1 = int(0.1*args.resolution[1])//2*2 + 1
    sigma = random.uniform(0.1, 2)

    train_transform = torchvision.transforms.Compose([
        norm_size, 
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        torchvision.transforms.GaussianBlur((radius_0, radius_1), (sigma, sigma)),
        *NORM,
    ])

    train_transform = MapTransform(train_transform)

    return train_transform

def get_val_transforms(args):
    norm_size = torchvision.transforms.Resize((args.resolution[0], args.resolution[1]))
    val_transform = torchvision.transforms.Compose([
        norm_size, 
        *NORM,
    ])

    val_transform = MapTransform(val_transform)

    return val_transform