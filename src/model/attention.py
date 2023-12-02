import torch
import torch.nn as nn
import einops
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, f):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(f).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class InstanceNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.InstanceNorm1d(dim, affine=True)
    def forward(self, x):
        ######input has shape: [b n c]
        x = einops.rearrange(x, 'b n c -> b c n')
        x = self.norm1(x)
        x = einops.rearrange(x, 'b c n -> b n c')
        return x

class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_token=5*16*28):
        super().__init__()
        # self.encoder_pos = nn.Parameter(torch.randn(1, n_token, dim) * .02)
        self.norm1 = norm_layer(dim)
        # self.norm1 = InstanceNorm1d(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        # self.norm2 = InstanceNorm1d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # encoder_pos = self.encoder_pos.unsqueeze(2).repeat([1, 1, x.shape[1]//self.encoder_pos.shape[1], 1])
        # encoder_pos = encoder_pos.view(1, x.shape[1], x.shape[2])
        # x = x + encoder_pos
        inter, attn = self.attn(self.norm1(x))
        x = x + inter
        x = x + self.mlp(self.norm2(x))
        return x, attn

class CrossBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_token=5*16*28):
        super().__init__()
        self.encoder_pos = nn.Parameter(torch.randn(1, n_token, dim) * .02)
        self.norm1 = norm_layer(dim)
#         self.norm1 = InstanceNorm1d(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
#         self.norm2 = InstanceNorm1d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, f):
        x = x + self.encoder_pos
        f = f + self.encoder_pos
        x = x + self.attn(self.norm1(x), self.norm1(f))
        x = x + self.mlp(self.norm2(x))
        return x

class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False, padding_mode='replicate'))

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MEBlock, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.conv1 = nn.Conv2d(in_channels=self.channel, 
                               out_channels=self.channel//self.reduction, 
                               kernel_size=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                               out_channels=self.channel//self.reduction,
                               kernel_size=3,
                               padding=1,
                               groups=self.channel//self.reduction,
                               bias=False)
        
        self.conv3 = nn.Conv2d(in_channels=self.channel//self.reduction,
                               out_channels=self.channel,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)
        
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
    
    def forward(self, x):
        n, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(n*t, c, h, w)
        xr = self.conv1(x)
        xr = self.bn1(xr)
        xrp = self.conv2(xr)
        xr = xr.view(n, t, c//self.reduction, h, w)[:, :t-1]
        xrp = xrp.view(n, t, c//self.reduction, h, w)[:, 1:]
        m = xrp - xr
        m = F.pad(m, self.pad, mode='constant', value=0)
        m = m.view(n*t, c//self.reduction, h, w)
        m = F.adaptive_avg_pool2d(m, (1, 1))
        m = self.conv3(m)
        m = self.bn3(m)
        m = torch.sigmoid(m) - 0.5
        out = m * x
        out = out.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return out