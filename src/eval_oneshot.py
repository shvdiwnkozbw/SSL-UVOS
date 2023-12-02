import os
import time
import einops
import sys
import cv2
import numpy as np
import utils as ut
import config as cg
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from model.model_cluster import AttEncoder
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia
from kornia.augmentation.container import VideoSequential
from sklearn.cluster import SpectralClustering

def kl_distance(final_set, attn):
    ## kl divergence between two distributions
    self_entropy = - torch.einsum('nc,nc->n', final_set, torch.log(final_set)).unsqueeze(-1) - torch.einsum('mc,mc->m', attn, torch.log(attn)).unsqueeze(0)
    cross_entropy = - torch.einsum('nc,mc->nm', final_set, torch.log(attn)) - torch.einsum('mc,nc->nm', attn, torch.log(final_set))
    distance = cross_entropy - self_entropy
    return distance

def hierarchical_cluster(attn, tau, num_iter):
    # attn t hw c
    ts = 3
    bs = 10000
    attn = attn / attn.sum(dim=-1, keepdim=True)
    final_set = []

    ## use a temporal window for clustering of the first hierarchy to speed up clustering process
    for t in range(0, attn.shape[0], ts):
        sample = attn[t:t+ts].view(-1, attn.shape[-1])
        distance = kl_distance(sample, sample)
        keep_set = []
        for i in range(0, distance.shape[0], bs):
            indices = (distance[i:i+bs] <= tau) # bs hw
            dist = torch.einsum('bn,nc->bc', indices.float(), sample) # bs hw
            dist = dist / dist.sum(dim=1, keepdim=True)
            keep_set.append(dist)
        keep_set = torch.cat(keep_set, dim=0)
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(attn.device)
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance[i] <= tau) * (indicator > 0) # K
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
    keep_set = final_set

    ## clustering on all frames at the following hierarchies until no change
    for t in range(num_iter):
        final_set = []
        keep_set = torch.stack(keep_set, dim=0) # K hw
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(attn.device) # K
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance[i] <= tau) * (indicator > 0) # K
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
        if len(final_set) == len(keep_set):
            break
        keep_set = final_set
    final_set = torch.stack(final_set, dim=0)

    ## calculate cluster assignments as object segmentation masks
    distance = kl_distance(final_set, attn.view(-1, attn.shape[-1]))
    nms_set = torch.argmin(distance, dim=0)
    final_mask = torch.zeros(final_set.shape[0], attn.shape[0]*attn.shape[1]).to(attn.device)
    print('cluster centroids:', final_set.shape)
    for i in range(final_mask.shape[0]):
        final_mask[i, nms_set==i] = 1
    return final_mask

def inference(masks_collection, rgbs, gts, model, T, ratio, tau, device):
    bs = 8
    feats = []
    ## extract frame-wise dino features
    for i in range(0, T, bs):
        input = rgbs[:, i:i+bs]
        input = einops.rearrange(input, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            _, _, _, feat = model.encoder(input)
            feats.append(feat.cpu())
    feats = torch.cat(feats, 0).to(device) # t c h w

    ## calculate the spatio-temporal attention, use sparse sampling on keys to reduce computational cost
    T, C, H, W = feats.shape
    num_heads = model.temporal_transformer[0].attn.num_heads
    feats = einops.rearrange(feats, 't c h w -> t (h w) c')
    feats = model.temporal_transformer[0].norm1(feats) # t hw c
    qkv = model.temporal_transformer[0].attn.qkv(feats)
    qkv = qkv.reshape(T, H*W, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4) # 3 t h hw c
    q, k, _ = qkv.cpu().unbind(0) # t h hw c
    key_indices = torch.arange(T//ratio) * ratio # sparse sampling the keys with a sparsity ratio
    k = k[key_indices]
    attention = torch.einsum('qhnc,khmc->qkhnm', q, k) * model.temporal_transformer[0].attn.scale
    attention = einops.rearrange(attention, 'q k h n m -> (q n) h (k m)')
    attention = attention.softmax(dim=-1)
    attention = attention.mean(dim=1) # thw khw
    print('spatio-temporal attention matrix:', attention.shape)

    ## clustering on the spatio-temporal attention maps and produce segmentation for the whole video
    dist = hierarchical_cluster(attention.view(T, H*W, -1).to(device), tau=tau, num_iter=10000)
    dist = einops.rearrange(dist, '(s p) (t h w) -> t s p h w', t=T, p=1, h=H)
    mask = dist.unsqueeze(1)
    for i in range(T):
        masks_collection[i].append(mask[i])
    return masks_collection

def eval(val_loader, model, device, ratio, tau, save_path=None, writer=None, train=False):
    with torch.no_grad():
        t = time.time()
        model.eval()
        mean = torch.tensor([0.43216, 0.394666, 0.37645])
        std = torch.tensor([0.22803, 0.22145, 0.216989])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        aug_list = VideoSequential(
            normalize_video,
            data_format="BTCHW",
            same_on_frame=True)
        print(' --> running inference')
        for idx, val_sample in enumerate(val_loader):
            rgbs, gts, category, val_idx = val_sample
            rgbs = rgbs.float().to(device)  # b t c h w
            rgbs = aug_list(rgbs)
            gts = gts.float().to(device)  # b t c h w
            T = rgbs.shape[1]
            masks_collection = {}
            for i in range(T):
                masks_collection[i] = []
            masks_collection = inference(masks_collection, rgbs, gts, model, T, ratio, tau, device)
            torch.save(masks_collection, save_path+'/%s.pth' % category[0])

def main(args):
    epsilon = 1e-5
    batch_size = args.batch_size 
    resume_path = args.resume_path
    attn_drop_t = args.attn_drop_t
    path_drop = args.path_drop
    num_t = args.num_t
    args.resolution = tuple(args.resolution)

    # setup log and model path, initialize tensorboard,
    # initialize dataloader
    trn_dataset, val_dataset, resolution, in_out_channels = cg.setup_dataset(args)
    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=8, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('======> start inference {}, {}, use {}.'.format(args.dataset, args.verbose, device))

    model = AttEncoder(resolution=resolution,
                        path_drop=path_drop,
                        attn_drop_t=attn_drop_t,
                        num_t=num_t)
    model.to(device)

    it = 0
    if resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(resume_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        it = checkpoint['iteration']
        loss = checkpoint['loss']
        model.eval()
    else:
        print('no checkpouint found')
        sys.exit(0)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    eval(val_loader, model, device, args.ratio, args.tau, save_path=args.save_path, train=False)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--num_train_steps', type=int, default=3e5) #30k
    #data
    parser.add_argument('--dataset', type=str, default='DAVIS2017')
    parser.add_argument('--resolution', nargs='+', type=int)
    #architecture
    parser.add_argument('--num_frames', type=int, default=3)
    parser.add_argument('--path_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop_t', type=float, default=0.4)
    parser.add_argument('--num_t', type=int, default=1)
    parser.add_argument('--gap', type=int, default=2, help='the sampling stride of frames')
    parser.add_argument('--ratio', type=int, default=10, help='key frame sampling rate in inference')
    parser.add_argument('--tau', type=float, default=1.0, help='distance threshold in clustering')
    #misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--basepath', type=str, default="/home/ma-user/work/shuangrui/DAVIS-2016")
    parser.add_argument('--output_path', type=str, default="./OUTPUT/")
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    args = parser.parse_args()
    main(args)
