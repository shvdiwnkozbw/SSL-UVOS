import os
import torch
import einops
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from cvbase.optflow.visualize import flow2rgb
import inspect
import torch.distributed as dist

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

# B = 1
# H = 12
# W = 15
# base_grid = mesh_grid(B, H, W)
# flow12 = torch.randn(B, 2, H, W)
# flow12 = einops.rearrange(flow12, 'b c h w -> b c (h w)')
# flow12[:,0] = (flow12[:,0] / flow12[:,0].abs().max(dim=-1)[0])*0.33*base_grid[:,0].max()
# flow12[:,1] = (flow12[:,1] / flow12[:,1].abs().max(dim=-1)[0])*0.33*base_grid[:,1].max()
# flow12 = einops.rearrange(flow12, 'b c (h w) -> b c h w', h=H)

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def loss_photomatric(im1_scaled, im1_recons, occu_mask1):
    loss = []

    loss += [0.15 * (im1_scaled - im1_recons).abs() * occu_mask1]
    loss += [0.85 * SSIM(im1_recons * occu_mask1, im1_scaled * occu_mask1, 1)]


    #loss = [torch.exp(l) for l in loss]
    return sum([l.mean() for l in loss]) / occu_mask1.mean()

def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    flow12 = einops.rearrange(flow12, 'b c h w -> b c (h w)')
#     flow = torch.ones_like(flow12)
    flow0 = (flow12[:,0] / flow12[:,0].abs().max(dim=-1, keepdim=True)[0]).clamp(-0.33,0.33)*W
    flow1 = (flow12[:,1] / flow12[:,1].abs().max(dim=-1, keepdim=True)[0]).clamp(-0.33,0.33)*H
    flow = einops.rearrange(torch.stack([flow0,flow1], dim=1), 'b c (h w) -> b c h w', h=H)
    
    v_grid = norm_grid(base_grid + flow)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons

def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    flow21 = einops.rearrange(flow21, 'b c h w -> b c (h w)')
#     flow = torch.ones_like(flow21)
    flow0 = (flow21[:,0] / flow21[:,0].abs().max(dim=-1, keepdim=True)[0]).clamp(-0.33,0.33)*W
    flow1 = (flow21[:,1] / flow21[:,1].abs().max(dim=-1, keepdim=True)[0]).clamp(-0.33,0.33)*H
    flow = einops.rearrange(torch.stack([flow0,flow1], dim=1), 'b c (h w) -> b c h w', h=H)

    corr_map = get_corresponding_map(base_grid + flow)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()

def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)
    
def warp_loss(rgb, flow_idxs, recon_flow):
# 'rgb' has shape B, 7, C(3), H, W 
# 'flow_idxs' has shape B, 14
# 'recon_flow' has shape B, 7, 2, C(2), H, W 
    bs, T = rgb.shape[0], rgb.shape[1]
    idx = torch.arange(0, bs)
    rgb_pair = []
    for i in range(T):
        rgb_pair.append(torch.stack([rgb[idx, i], rgb[idx, flow_idxs[:, 2*i]]], dim=1))
        rgb_pair.append(torch.stack([rgb[idx, i], rgb[idx, flow_idxs[:, 2*i+1]]], dim=1))
    rgb_pair = torch.stack(rgb_pair, dim=1) #B, 14, 2, 3, H, W
    recon_flow = einops.rearrange(recon_flow, 'b t p c h w -> b (t p) c h w')
    warp_losses = []
    for i in range(2*T):
        # resize images to match the size of layer
        im1_scaled, im2_scaled = rgb_pair[:, i, 0], rgb_pair[:, i, 1] 

        im1_recons = flow_warp(im2_scaled, recon_flow[:, i], pad='border')
        # im2_recons = flow_warp(im1_scaled, recon_flow[:, 2:], pad='border')


        occu_mask1 = 1 - get_occu_mask_backward(recon_flow[:, i//2*2+(1-i%2)], th=0.2) ## reverse order
        loss_warp = loss_photomatric(im1_scaled, im1_recons, occu_mask1)
        warp_losses.append(loss_warp)
    warp_loss = sum(warp_losses) / (2*T)
    return warp_loss


def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def temporal_loss(slots):
    bs, nf, ns, ds = slots.shape 
    device = slots.device
    slot0 = slots[:,:-1,:,:].reshape(bs*(nf-1), ns, ds)
    slot1 = slots[:,1:,:,:].reshape(bs*(nf-1), ns, ds)
    slot1 = slot1.permute(0,2,1)
    scores = torch.einsum('bij,bjk->bik', slot0, slot1)
    scores = scores.softmax(dim=1) + 0.0001
    gt = torch.eye(ns).unsqueeze(0).repeat(bs*(nf-1),1,1).to(device)
    return torch.mean((gt-scores)**2)

def convert_for_vis(inp, use_flow=False):
    dim = len(inp.size())
    if not use_flow:
        return torch.clamp((0.5*inp+0.5)*255,0,255).type(torch.ByteTensor)
    else:
        if dim == 4:
            inp = einops.rearrange(inp, 'b c h w -> b h w c').detach().cpu().numpy()
            rgb = [flow2rgb(inp[x]) for x in range(np.shape(inp)[0])]
            rgb = np.stack(rgb, axis=0)
            rgb = einops.rearrange(rgb, 'b h w c -> b c h w')
        if dim == 5:
            b, s, w, h, c = inp.size()
            inp = einops.rearrange(inp, 'b s c h w -> (b s) h w c').detach().cpu().numpy()
            rgb = [flow2rgb(inp[x]) for x in range(np.shape(inp)[0])]
            rgb = np.stack(rgb, axis=0)
            rgb = einops.rearrange(rgb, '(b s) h w c -> b s c h w', b=b, s=s)
        return torch.Tensor(rgb*255).type(torch.ByteTensor)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def heuristic_fg_bg(mask):
    mask = mask.copy()
    h, w = mask.shape
    mask[1:-1, 1:-1] = 0
    borders = 2*h+2*w-4
    return np.sum(mask>0.5)/borders

def rectangle_iou(masks, gt):
    t, s, c, H_, W_ = masks.size()
    H, W = gt.size()
    masks = F.interpolate(masks, size=(1, H, W))
    ms = []
    for t_ in range(t):
        m = masks[t_,0,0] #h w
        m = m.detach().cpu().numpy()
        if heuristic_fg_bg(m) > 0.5: m = 1-m
        ms.append(m)
    masks = np.stack(ms, 0)
    gt = gt.detach().cpu().numpy()
    for idx, m in enumerate([masks[0], masks.mean(0)]):
        m[m>0.1]=1
        m[m<=0.1]=0
        contours = cv2.findContours((m*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        area = 0
        for cnt in contours:
            (x_,y_,w_,h_) = cv2.boundingRect(cnt)
            if w_*h_ > area:
                x=x_; y=y_; w=w_; h=h_;
                area = w_ * h_
        if area>0:
            bbox = np.array([x, y, x+w, y+h],dtype=float)
            #if the size reference for the annotation (the original jpg image) is different than the size of the mask
#             import pdb; pdb.set_trace()
            i, j = np.where(gt>=0.1)
            if len(i) != 0:
                bbox_gt = np.array([min(j), min(i), max(j)+1, max(i)+1],dtype=float)
                iou = bb_intersection_over_union(bbox_gt, bbox)
            else: iou = 1.
        else:
            iou = 0.
        if idx == 0: iou_single = iou
        if idx == 1: iou_mean = iou
    masks = np.expand_dims(masks, 1)
    return masks, masks.mean(0), [1.0 if iou_mean > 0.5 else 0.0], iou_single

def iou(masks, gt, thres=0.5):
    masks = (masks>thres).float()
    intersect = torch.tensordot(masks, gt, dims=([-2, -1], [0, 1]))
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    return intersect/(union + 1e-12)

def ensemble_hungarian_iou(masks, sudo, gt, thres=0.5, moca=False):
    ####### input
    #  'masks' has shape: t, s, 1, h, w
    #  'gt' has shape: 1, h, w
    ####### output
    #  'masks' has shape: t, s, 1, h, w
    #  'mean_mask' has shape: 1, h, w
    #  'iou_mean'
    thres = thres
    b, c, h, w = gt.size()
    gt = gt[0, 0,:,:] #h, w
    sudo = F.interpolate(sudo, size=(h, w))
    
    # masks = masks / masks.max()
    if moca:
        #return masks, masks.mean(0), 0, rectangle_iou(masks[0], gt) 
        masks, mean_mask, iou_mean, iou_ther = rectangle_iou(masks, gt)
    else:
        masks = F.interpolate(masks, size=(1, h, w))  # t s 1 h w
        match_mask = masks[:, :, 0]
        match_mask = einops.rearrange(match_mask, 't s h w -> t s (h w)')
        match_mask = (match_mask-torch.min(match_mask, dim=-1, keepdim=True)[0]) / (torch.max(match_mask, dim=-1, keepdim=True)[0]-torch.min(match_mask, dim=-1, keepdim=True)[0]+1e-10)
        match_mask = einops.rearrange(match_mask, 't s (h w) -> t s h w', h=h)
        mask_iou = iou(match_mask, gt, thres)  # t s h w # h w
        back_iou = iou(match_mask, 1-gt, thres)
        iou_max, slot_max = mask_iou.max(dim=1)
        indices = (mask_iou > back_iou).float()
        masks = torch.sum(masks*indices[...,None,None,None], dim=1)
        # masks = masks[torch.arange(masks.size(0)), slot_max]  # pick the slot for each mask
        # top_idx = iou_max.topk(masks.size(0)//2)[1]
        # print(top_idx)
        # max_mask = masks[top_idx].mean(dim=0)
        mean_mask = masks.mean(0)
        mean_mask = (mean_mask-mean_mask.min()) / (mean_mask.max()-mean_mask.min()+1e-10)
        sudo_mask = (sudo[:, 0]-sudo[:, 0].min()) / (sudo[:, 0].max()-sudo[:, 0].min())
       
        # mean_mask_entro = (entropy[..., None, None] * masks).sum(0)
        # gap_1_mask = masks[0]  # note last frame will use gap of -1, not major
        
        iou_dino = iou(sudo_mask, gt, thres).detach().cpu().numpy()
        iou_mean = iou(mean_mask, gt, thres).detach().cpu().numpy()
        
        # iou_ther = iou_max.max().detach().cpu().numpy()
        # iou_single_gap = iou(gap_1_mask, gt, thres).detach().cpu().numpy()
        mean_mask = mean_mask.detach().cpu().numpy()  # c h w
        # masks = masks.detach().cpu().numpy()
    return masks, mean_mask, iou_mean, iou_dino

# 'masks' has shape B, 3, 2(flow_pos/neg), 2(num_slot), 1, H, W 
# 'gt' has shape B, 3, C, H, W 
def hungarian_iou(masks, gt):
    gt = einops.rearrange(gt, 'b n c h w -> (b n) c h w')
    masks = einops.rearrange(masks, 'b n t s c h w -> (b n) (t s) c h w')
    thres = 0.5
    masks = (masks>thres).float()
    gt = gt[:,0:1,:,:]
    b, c, h, w = gt.size()
    iou_max = []
    for i in range(masks.size(1)):
        mask = masks[:,i]
        mask = F.interpolate(mask, size=(h, w))
        #IOU
        intersect = (mask*gt).sum(dim=[-2, -1])
        union = mask.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
        iou = intersect/(union + 1e-12)
        iou_max += [iou]
    iou_max, slot_max = torch.cat(iou_max, -1).max(dim=-1)
    return iou_max.mean(), slot_max


TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def produce_loss(data_tuple, gt, reduce_dim, criterion):
    import pdb; pdb.set_trace() 
    recon_combined, recons, masks= data_tuple
    recon_loss = criterion(gt, recon_combined)
    entropy_loss = -(masks * torch.log(masks + epsilon)).sum(dim=reduce_dim).mean()
    return recon_loss, entropy_loss

import random
import kornia
from kornia.augmentation.container import VideoSequential

def Augment_GPU_pre(args):
#     crop_size = args.crop_size
    resolution = args.resolution
    radius_0 = int(0.1*resolution[0])//2*2 + 1
    radius_1 = int(0.1*resolution[1])//2*2 + 1
    sigma = random.uniform(0.1, 2)
    # For k400 parameter:
    # mean = torch.tensor([0.43216, 0.394666, 0.37645])
    # std = torch.tensor([0.22803, 0.22145, 0.216989])
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        # kornia.augmentation.RandomResizedCrop(size=resolution, scale=(0.8, 1.0)),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        # kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomGaussianBlur((radius_0, radius_1), (sigma, sigma), p=0.5),
        normalize_video,
        data_format="BTCHW",
        same_on_frame=True)
    return aug_list

import itertools

def hungarian_entropy(mask, mask_t, epsilon):
    # mask b t p s hw
    permutation = torch.LongTensor(list(itertools.permutations(np.arange(mask.shape[-2]))))
    permutation = permutation.to(mask.device)
    mask_t = mask_t[:, :, :, permutation]
    entropy = - torch.sum(mask_t * torch.log(mask.unsqueeze(3) + epsilon), dim=-2)
    entropy = torch.mean(entropy, dim=-1)
    entropy = torch.min(entropy, dim=-1)[0]
    return entropy.mean()

def random_walk(x, mask, mask_t, valid_mask=None, epsilon=1e-10):
    # x b t hw c
    # mask b t 1 s hw
    # sudo_mask b t hw
    x = F.normalize(x, dim=-1)
    aff = torch.einsum('btnc,bpmc->btpnm', x, x)
    # comb = torch.stack([aff[:, 0, 1], aff[:, 0, 2], aff[:, 1, 0], aff[:, 1, 2], aff[:, 2, 0], aff[:, 2, 1]], dim=1)
    # comb = einops.rearrange(comb, 'b (t p) n m -> b t p n m', t=x.shape[1])
    aff[torch.rand_like(aff)<0.1] = -1e10
    comb = F.softmax(aff/0.07, dim=-2)
    mask = F.softmax(mask, dim=-2) + 1e-8
    mask_warp = torch.einsum('btsn,btpnm->btpsm', mask.squeeze(2), comb)
    mask_warp = mask_warp / torch.sum(mask_warp, dim=-2, keepdim=True)
    mask_warp = mask_warp.permute(0, 2, 1, 3, 4).contiguous()
    # mask_warp = F.softmax(mask_warp, dim=-2) + 1e-8
    # mask_warp = torch.stack(
    #     [mask_warp[:, 1, 0], mask_warp[:, 2, 0], mask_warp[:, 0, 0], mask_warp[:, 2, 1], mask_warp[:, 0, 1], mask_warp[:, 1, 1]], dim=1)
    # mask_warp = einops.rearrange(mask_warp, 'b (t p) s n -> b t p s n', t=x.shape[1])
    # entropy = hungarian_entropy(mask_warp, mask_t, epsilon)
    entropy = - torch.sum(mask_t * torch.log(mask_warp + epsilon), dim=-2)
    # entropy = torch.sum(entropy * valid_mask.unsqueeze(2)) / torch.sum(valid_mask.unsqueeze(2))
    return entropy.mean()

def motion_center(motion, motion_t, mask, valid_mask=None):
    # motion b t hw c
    # mask b t s hw
    # valid mask b t hw
    mask = mask.squeeze(2)
    assign = F.softmax(mask/0.01, dim=-2)
    mask = F.softmax(mask, dim=-2) + 1e-8
    mask = mask / mask.sum(dim=-1, keepdim=True)
    center = torch.einsum('btnc,btsn->btsc', motion_t, mask)
    motion = F.normalize(motion, dim=-1)
    center = F.normalize(center, dim=-1)
    similarity = torch.einsum('btnc,btsc->btsn', motion, center)
    pos = torch.einsum('btsn,btsn->btn', similarity, assign)
    neg = torch.min(similarity, dim=-2)[0]
    loss = F.relu(neg-pos+0.5).mean()
    # loss = torch.sum(loss * valid_mask) / torch.sum(valid_mask+1e-10)
    return loss

def motion_gt(motion, mask):
    # motion b t hw c
    # mask b t s hw
    mask = mask.squeeze(2)
    center = torch.einsum('btnc,btsn->btsc', motion, mask)
    motion = F.normalize(motion, dim=-1)
    center = F.normalize(center, dim=-1)
    similarity = torch.einsum('btnc,btsc->btsn', motion, center)
    pos = torch.einsum('btsn,btsn->btn', similarity, mask)
    neg = torch.einsum('btsn,btsn->btn', similarity, 1-mask)
    loss = F.relu(neg-pos+0.8).mean()
    return loss

def filter_slot(sudo, mask):
    # sudo b t hw
    # mask b t 1 s hw
    # valid slot b t s
    # valid mask b t hw
    sudo = F.normalize(sudo, dim=-1)
    mask = F.softmax(mask, dim=-2)[:, :, 0] + 1e-8
    norm_mask = F.normalize(mask, dim=-1)
    similarity = torch.einsum('btn,btsn->bts', sudo, norm_mask)
    valid_slot = (similarity > 0.5).float()
    valid_slot[:, :, torch.argmax(similarity, -1)] = 1
    valid_mask = torch.zeros_like(mask[:, :, 0])
    sum_mask = torch.einsum('bts,btsn->btn', valid_slot, mask)
    valid_mask[sum_mask>0.5] = 1
    max_slot = torch.argmax(mask, dim=-2)
    max_slot = torch.gather(valid_slot, -1, max_slot)
    valid_mask[max_slot>0] = 1
    return valid_slot, valid_mask

def slot_contrast(slot_warp, slot_warp_t, valid_slot):
    pos = torch.einsum('btsc,bpsc->btps', slot_warp, slot_warp_t)
    pos = torch.sum(pos * valid_slot.unsqueeze(-2), -1) / torch.sum(valid_slot.unsqueeze(-2), -1)
    pos = pos.mean(-1)
    neg = F.relu(torch.einsum('btsc,ptsc->bpts', slot_warp, slot_warp_t)-0.2)
    neg = torch.sum(neg * valid_slot.unsqueeze(1), -1) / torch.sum(valid_slot.unsqueeze(1), -1)
    mask = torch.ones_like(neg)
    mask[torch.arange(mask.shape[0]), torch.arange(mask.shape[1])] = 0
    neg = torch.sum(neg * mask, dim=1) / torch.sum(mask, dim=1)
    neg_s = F.relu(torch.einsum('btsc,btpc->btsp', slot_warp, slot_warp_t)-0.2)
    mask = torch.ones_like(neg_s)
    mask[:, :, torch.arange(mask.shape[-2]), torch.arange(mask.shape[-1])] = 0
    neg_s = torch.sum(neg_s * mask, dim=-1) / torch.sum(mask, dim=-1)
    neg_s = torch.sum(neg_s * valid_slot, -1) / torch.sum(valid_slot, -1)
    slot_loss = (neg + 1 - pos).mean()
    return slot_loss

def slot_consistency(slot, slot_t):
    # slot b t s c
    b, t, s = slot.shape[:3]
    slot = F.normalize(slot, dim=-1)
    slot_t = F.normalize(slot_t, dim=-1)
    similarity = torch.einsum('btsc,bpkc->btpsk', slot, slot_t) # b t t s s
    pos_indicator = torch.zeros_like(similarity)
    pos_indicator[:, :, :, torch.arange(s), torch.arange(s)] = 1
    pos = similarity[pos_indicator>0].view(b, t, t, s)
    neg = similarity[pos_indicator==0].view(b, t, t, s, s-1)
    loss = F.relu(neg-pos[...,None]+0.8).mean()
    return loss

def cluster_attn(attn, tau, num_iter):
    # attn thw thw
    attn = attn / attn.sum(dim=1, keepdim=True)
    distance = kl_distance(attn, attn)
    keep_set = []
    for i in range(distance.shape[0]):
        indices = (distance[i] <= tau) # hw
        dist = torch.mean(attn[indices], dim=0) # hw
        keep_set.append(dist)
    for t in range(num_iter):
        final_set = []
        keep_set = torch.stack(keep_set, dim=0) # K hw
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(attn.device) # K
        if len(keep_set) >= 8:
            threshold = torch.sort(distance.view(-1), descending=False)[0][indicator.shape[0]]
            if threshold > tau:
                cur_tau = threshold
            else:
                cur_tau = tau
        else:
            cur_tau = tau
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
    distance = kl_distance(final_set, attn)
    final_mask = torch.softmax(-distance/0.1, dim=0) # k thw
    return final_mask

def kl_distance(final_set, attn, epsilon=1e-10):
    self_entropy = - torch.einsum('bnc,bnc->bn', final_set, torch.log(final_set+epsilon)).unsqueeze(-1) - torch.einsum('bmc,bmc->bm', attn, torch.log(attn+epsilon)).unsqueeze(1)
    cross_entropy = - torch.einsum('bnc,bmc->bnm', final_set, torch.log(attn+epsilon)) - torch.einsum('bmc,bnc->bnm', attn, torch.log(final_set+epsilon))
    distance = cross_entropy - self_entropy
    return distance

def kmeans_attn(attn, k, num_iter):
    # attn b thw c
    attn = attn / attn.sum(dim=2, keepdim=True)
    anchor = attn[:, torch.randperm(attn.shape[1]//3)[:k]] # b k c
    for _ in range(num_iter):
        distance = kl_distance(anchor, attn) # b k thw
        weight = torch.softmax(-distance/1e-5, dim=1) # b k thw
        anchor = torch.einsum('bkn,bnc->bkc', weight, attn)
        anchor = anchor / torch.sum(anchor+1e-10, dim=-1, keepdim=True)
    distance = kl_distance(anchor, attn)
    final_mask = - distance / 1e-5
    return final_mask.detach()

def motion_align(vector, weight):
    # vector b t hw c
    # weight b thw thw
    num_frame = vector.shape[1]
    vector = F.normalize(vector, dim=-1)
    weight = einops.rearrange(weight, 'b (t m) (p q) -> b t m p q', t=num_frame, p=num_frame)
    weight = weight[:, torch.arange(num_frame), :, torch.arange(num_frame)] # t b hw hw
    weight = weight.transpose(0, 1).contiguous() # b t hw hw
    weight = weight / torch.sum(weight, dim=-1, keepdim=True)
    weight = torch.softmax(torch.log(weight)/0.5, dim=-1)
    vector_dist = torch.einsum('btmc,btnc->btmn', vector, vector) # b t hw hw
    vector_dist = torch.softmax(vector_dist/0.1, dim=-1)
    # import pdb; pdb.set_trace()
    loss = - torch.sum(weight.detach() * torch.log(vector_dist+1e-10), dim=-1) \
        - torch.sum(vector_dist.detach() * torch.log(weight+1e-10), dim=-1)
    return loss.mean()