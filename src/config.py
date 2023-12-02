import os
import torch
import itertools
import glob as gb
import numpy as np
import json
from datetime import datetime
from data import Dataloader

def setup_dataset(args):
    resolution = args.resolution  # h,w
    res = ""
    with_gt = True
    if args.dataset == 'DAVIS':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages/480p'
        gt_dir = basepath + '/Annotations/480p'
        val_seq = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                    'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                    'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch']       
        val_data_dir = [img_dir, img_dir, gt_dir]
        res = "480p"

    elif args.dataset == 'DAVIS2017':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages/480p'
        gt_dir = basepath + '/Annotations/480p'
        val_seq = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                    'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                    'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch',
                    'bike-packing', 'dogs-jump', 'gold-fish', 'india', 'judo', 'lab-coat', 'loading', 'mbike-trick', 
                    'pigs', 'shooting']
        val_data_dir = [img_dir, img_dir, gt_dir]
        res = "480p"

    elif args.dataset == 'FBMS':
        basepath = args.basepath
        img_dir = args.basepath + '/FBMS/'
        gt_dir = args.basepath + '/Annotations/'    

        val_seq = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06', 
                    'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04', 
                    'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9', 
                    'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']
        val_img_dir = args.basepath + '/FBMS/'
        val_gt_dir =args.basepath + '/FBMS_annotation/'
        val_data_dir = [val_img_dir, val_img_dir, val_gt_dir]
        with_gt = False

    elif args.dataset == 'STv2':
        basepath = args.basepath
        img_dir = basepath + '/STv2_img/JPEGImages/'
        gt_dir = basepath + '/STv2_gt&pred/STv2_gt/GroundTruth/'

        val_seq = ['drift', 'birdfall', 'girl', 'cheetah', 'worm', 'parachute', 'monkeydog',
                    'hummingbird', 'soldier', 'bmx', 'frog', 'penguin', 'monkey', 'bird_of_paradise']
        val_data_dir = [img_dir, img_dir, gt_dir]

    elif args.dataset == 'YTVIS':
        basepath = args.basepath
        img_dir = basepath + '/val/JPEGImages'
        gt_dir = basepath + '/val/Annotations'
        val_seq = os.listdir(img_dir)
        val_data_dir = [img_dir, img_dir, gt_dir]

    else:
        raise ValueError('Unknown Setting.')
    
    flow_dir = basepath
    data_dir = [flow_dir, img_dir, gt_dir]
    trn_dataset = Dataloader(data_dir=data_dir, dataset=args.dataset, resolution=resolution, gap=args.gap, seq_length=args.num_frames,
                             train=True)
    val_dataset = Dataloader(data_dir=val_data_dir, dataset=args.dataset, resolution=resolution, gap=args.gap,
                             train=False, val_seq=val_seq)
    in_out_channels = 3
    
    return [trn_dataset, val_dataset, resolution, in_out_channels]
