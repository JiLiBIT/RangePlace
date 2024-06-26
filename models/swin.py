import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops import rearrange
from typing import Optional
from tools.utils.utils import *

class OutputPredict(nn.Module):
    r""" Image to Patch Embedding

    Args:
        height= 64 (int): Image height.  Default: 64.
        width = 900 (int): Image width.  Default: 900.
        patch_size (tuple): Patch token size. Default: 4,1.
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 16.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, height= 64, width = 896, patch_size=[2,2], in_chans=1, embed_dim=16, proj_type="mlp",
                 norm_layer=None,drop = 0.):
        super().__init__()
        patches_resolution = [height // patch_size[0] , width // patch_size[1]]
        self.img_size = [height , width]
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.proj_type = proj_type
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.embed1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=[3,1], stride=[2,1]), #60
            nn.ReLU(inplace=True),) # mixvpr 
        self.embed2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=[3,1], stride=[2,1]), #29
            nn.ReLU(inplace=True),) # mixvpr 
        self.embed3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=[3,1], stride=[2,1]), #14
            nn.ReLU(inplace=True),) # mixvpr 
        self.embed4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim//4, in_chans, kernel_size=[5,1], stride=[1,1]), #6
            nn.ReLU(inplace=True),) # mixvpr 
            
            # self.proj = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=patch_size, stride=patch_size)

            # self.embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    def upsample1(self, x):
        # 在self.upsample方法中使用F.upsample
        return F.upsample(x, size=(14, 896), mode='bilinear')
    def upsample2(self, x):
        # 在self.upsample方法中使用F.upsample
        return F.upsample(x, size=(29, 896), mode='bilinear')
    def upsample3(self, x):
        # 在self.upsample方法中使用F.upsample
        return F.upsample(x, size=(60, 896), mode='bilinear')
    def upsample4(self, x):
        # 在self.upsample方法中使用F.upsample
        return F.upsample(x, size=(64, 896), mode='bilinear')
    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0,2,1).reshape(B,C,6,self.img_size[1]).contiguous()
        x = self.embed1(x)
        x = self. upsample1(x)
        x = self.embed2(x)
        x = self. upsample2(x)
        x = self.embed3(x)
        x = self. upsample3(x)
        x = self.embed4(x)
        x = self. upsample4(x)
        # x = func.pad(x, (0, 0, 1, 0, 0, 0))
        _, _, Ph, Pw = x.shape 
        x = x.flatten(2).transpose(1, 2).squeeze(2)  # B Ph*Pw C
        # if self.norm is not None:
        #     x = self.norm(x)
        return Ph, Pw, x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor
        return x


# class PatchEmbedding(nn.Module):
#     def __init__(self, patch_size: int = [2,1], in_c: int = 1, embed_dim: int = 16, norm_layer: nn.Module = None):
#         super().__init__()
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size,) * 2, stride=(patch_size,) * 2)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def padding(self, x: torch.Tensor) -> torch.Tensor:
#         _, _, H, W = x.shape
#         if H % self.patch_size != 0 or W % self.patch_size != 0:
#             x = func.pad(x, (0, self.patch_size - W % self.patch_size,
#                              0, self.patch_size - H % self.patch_size,
#                              0, 0))
#         return x

#     def forward(self, x):
#         x = self.padding(x)
#         x = self.proj(x)
#         x = rearrange(x, 'B C H W -> B H W C')
#         x = self.norm(x)
#         return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        height= 64 (int): Image height.  Default: 64.
        width = 900 (int): Image width.  Default: 900.
        patch_size (tuple): Patch token size. Default: 4,1.
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 16.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, height= 64, width = 900, patch_size=[2,2], in_chans=1, embed_dim=16, proj_type="mlp",
                 norm_layer=None,drop = 0.):
        super().__init__()
        patches_resolution = [height // patch_size[0] , width // patch_size[1]]
        self.img_size = [height , width]
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.proj_type = proj_type
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_chans,embed_dim//4),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(embed_dim//4,embed_dim//2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(embed_dim//2,embed_dim),
                nn.GELU(),
                nn.Dropout(drop),)
        elif  proj_type == "conv":
            # self.mlp = nn.Sequential(
            #     nn.Linear(in_chans,embed_dim//4),
            #     nn.GELU(),
            #     nn.Dropout(drop),
            #     nn.Linear(embed_dim//4,embed_dim//2),
            #     nn.GELU(),
            #     nn.Dropout(drop),)
            
            # self.embed = nn.Sequential(
            #     nn.Conv2d(in_chans, embed_dim//4, kernel_size=1, stride=1),
            #     nn.GELU(),
            #     nn.Dropout(drop),
            #     nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=[5,1], stride=[1,1]),
            #     nn.GELU(),
            #     nn.Dropout(drop),
            #     nn.Conv2d(embed_dim//2, embed_dim, kernel_size=[5,1], stride=[1,1]),
            #     nn.GELU(),) # 7
            
            # self.embed = nn.Sequential(
            #     nn.Conv2d(in_chans, embed_dim//8, kernel_size=[5,1], stride=[2,1]),
            #     nn.GELU(),
            #     nn.Conv2d(embed_dim//8, embed_dim//4, kernel_size=[3,1], stride=[1,1]),
            #     nn.GELU(),
            #     nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=[3,1], stride=[1,1]),
            #     nn.GELU(),
            #     nn.Conv2d(embed_dim//2, embed_dim, kernel_size=[3,1], stride=[1,1]),
            #     nn.GELU(),) # 3
            
            self.embed = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=[5,1], stride=[2,1]), #30
                nn.GELU(),
                nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=[5,1], stride=[1,1]), #26
                nn.GELU(),
                nn.Conv2d(embed_dim//2, embed_dim, kernel_size=[3,1], stride=[1,1]), #24
                nn.GELU(),) # mixvpr 

            
            # self.proj = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=patch_size, stride=patch_size)

            # self.embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        if self.proj_type == "mlp":
            x = x.view(B,H*W,C)
            x = self.proj(x)
            x = x.view(B,self.embed_dim,H,W)
        elif self.proj_type == "conv":
            # x = x.view(B,H*W,C)
            # x = self.mlp(x)
            # x = x.view(B,self.embed_dim//2,H,W)
            # x = self.proj(x)
            
            x = self.embed(x)
            # x = func.pad(x, (0, 0, 1, 0, 0, 0))
        _, _, Ph, Pw = x.shape 
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        
        if self.norm is not None:
            x = self.norm(x)
        return Ph, Pw, x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)

        x =self.padding(x)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops
class OverlapPatchMeriging(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, down1D, img_size=224, kernel_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.kernel_size = kernel_size
        # self.H, self.W = img_size[0] // stride, img_size[1] // stride
        # self.num_patches = self.H * self.W
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
        #                       padding=(patch_size[0] // 2, patch_size[1] // 2))
        if down1D == False:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride),
                nn.ReLU(inplace=True))
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=[kernel_size,1], stride=[stride,1]),
                nn.ReLU(inplace=True))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = rearrange(x, 'B C H W -> B (H W) C')
        x = x.reshape(B,C,-1).permute(0,2,1).contiguous()
        # x = self.norm(x)

        return x, H, W

class OverlapPatchExpanding(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, down1D, img_size=224, kernel_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.kernel_size = kernel_size
        # self.H, self.W = img_size[0] // stride, img_size[1] // stride
        # self.num_patches = self.H * self.W
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
        #                       padding=(patch_size[0] // 2, patch_size[1] // 2))
        if down1D == False:
            self.proj = nn.Sequential(
                nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride),
                nn.ReLU(inplace=True))
        else:
            self.proj = nn.Sequential(
                nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=[kernel_size,1], stride=[stride,1]),
                nn.ReLU(inplace=True))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = rearrange(x, 'B C H W -> B (H W) C')
        x = x.reshape(B,C,-1).permute(0,2,1).contiguous()
        # x = self.norm(x)

        return x, H, W
    
class PatchMerging1D(nn.Module):
    """ Patch Merging 1D Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim,embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.embed_dim = embed_dim
        self.reduction = nn.Linear(2 * dim, embed_dim, bias=False)  # Reduce to 2 * C
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, -1, C)
        # x = pad_to_multiples_of(x, [2, 1])  # Pad only in H direction

        x0 = x[:, 0::2, :, :]  # B H/2 W C
        x1 = x[:, 1::2, :, :]  # B H/2 W C
        x = torch.cat([x0, x1], -1)  # B H/2 W 2*C
        x = x.view(B, -1, 2 * C)  # B H/2*W 2*C
        x = self.reduction(x)
        x = self.norm(x)
        
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        C = self.dim
        flops = 2 * H * W * self.dim * 2 * self.dim  # Linear transformation
        flops += H * W * self.dim // 2  # LayerNorm
        return flops

class PatchMerging2(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, embed_dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.embed_dim = embed_dim
        self.reduction = nn.Linear(4 * dim, embed_dim, bias=False)
        self.norm = norm_layer(embed_dim)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)

        x =self.padding(x)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)
        return x


class PatchMerging3(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, embed_dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.embed_dim = embed_dim
        self.reduction = nn.Sequential(
                            nn.Conv2d(dim, embed_dim,kernel_size=[2,1],stride=[2,1]),
                            nn.GELU(),
                            )
        self.norm = norm_layer(embed_dim)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, C, H, W)

        x =self.padding(x)

        x = self.reduction(x)
        x = x.view(B, -1, self.embed_dim)
        # x = self.norm(x)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x

# class PatchExpanding1D(nn.Module):
#     def __init__(self, dim: int, norm_layer=nn.LayerNorm):
#         super(PatchExpanding1D, self).__init__()
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2 * dim, bias=False)
#         self.norm = norm_layer(dim // 2)

#     def forward(self, x: torch.Tensor):
#         x = self.expand(x)
#         x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=1)
#         x = self.norm(x)
#         return x
    
# Upsample Block

# class Upsample1D(nn.Module):
#     def __init__(self, input_resolution, in_channel, out_channel, norm_layer):
#         super(Upsample1D, self).__init__()
#         self.input_resolution = input_resolution
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(in_channel, out_channel, kernel_size=[2,1], stride=[2,1]),
#         )
#         self.in_channel = in_channel
#         self.out_channel = out_channel
        
#     def forward(self, x):
#         B, L, C = x.shape
#         H, W = self.input_resolution
#         x = x.transpose(1, 2).contiguous().view(B, C, H, W)
#         out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
#         return out


class Upsample1D(nn.Module):
    def __init__(self, input_resolution, in_channel, out_channel, norm_layer):
        super(Upsample1D, self).__init__()
        self.input_resolution = input_resolution
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.expand = nn.Linear(self.in_channel, self.in_channel, bias=False)
        self.norm = norm_layer(self.out_channel)

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B,H,W,C)
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1 W P2) C', P1=2, P2=1)
        if self.norm is not None:  
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops
    


# class FinalPatchExpanding(nn.Module):
#     def __init__(self, dim: int, norm_layer=nn.LayerNorm):
#         super(FinalPatchExpanding, self).__init__()
#         self.dim = dim
#         self.expand = nn.Linear(dim, 16 * dim, bias=False)
#         self.norm = norm_layer(dim)

#     def forward(self, x: torch.Tensor):
#         x = self.expand(x)
#         x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4, P2=4)
#         x = self.norm(x)
#         return x
    
# class FinalPatchExpanding1D(nn.Module):
#     def __init__(self, dim: int, norm_layer=nn.LayerNorm):
#         super(FinalPatchExpanding1D, self).__init__()
#         self.dim = dim
#         self.expand = nn.Linear(dim, 16 * dim, bias=False)
#         self.norm = norm_layer(dim)

#     def forward(self, x: torch.Tensor):
#         x = self.expand(x)
#         x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4, P2=1)
#         x = self.norm(x)
#         return x


# Output Projection
class OutputProj1D(nn.Module):
    def __init__(self, input_resolution, in_channel=16, out_channel=1, kernel_size=[3,1], stride=[2,1], norm_layer=None,act_layer=None):
        super().__init__()
        self.input_resolution = input_resolution
        # self.proj = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        self.expand = nn.Linear(in_channel, 4 * in_channel, bias=False)
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B,H,W,C)
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4, P2=4)
        if self.norm is not None:
            x = self.norm(x)    
        x = x.transpose(1, 2).view(B, C, H, W)
        
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, win_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 qk_scale=None, use_Dropkey = False, token_projection='linear', pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_Dropkey = use_Dropkey

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.win_size[0] - 1), self.win_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.win_size[1] - 1), self.win_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1) + 1e-12
            # relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1) + 1e-5
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1) + 1e-12
        else:
            relative_coords_table[:, :, :, 0] /= (self.win_size[0] - 1) + 1e-12
            # relative_coords_table[:, :, :, 1] /= (self.win_size[1] )
            # relative_coords_table[:, :, :, 1]  /= (self.win_size[1] - 1) + 1e-5
            relative_coords_table[:, :, :, 1] /= (self.win_size[1] - 1) + 1e-12
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if token_projection =='conv':
            nn.Linear(dim, dim * 3, bias=False)
            #self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
        else:
            raise Exception("Projection error!") 
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    
        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

            
        # logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).cuda()).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if self.use_Dropkey == True:
            m_r = torch.ones_like(attn) * 0.3
            attn = attn + torch.bernoulli(m_r) * -1e12

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.win_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


# class WindowAttention1(nn.Module):
#     def __init__(self, dim: int, window_size: tuple, num_heads: int, qkv_bias: Optional[bool] = True,
#                  attn_drop: Optional[float] = 0., proj_drop: Optional[float] = 0., shift: bool = False):
#         super().__init__()
#         self.window_size = window_size
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5

#         if shift:
#             self.shift_size = window_size // 2
#         else:
#             self.shift_size = 0

#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size - 1) ** 2, num_heads))
#         nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

#         coords_size = torch.arange(self.window_size)
#         coords = torch.stack(torch.meshgrid([coords_size, coords_size], indexing="ij"))
#         coords_flatten = torch.flatten(coords, 1)

#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += self.window_size - 1
#         relative_coords[:, :, 1] += self.window_size - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size - 1
#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)

#     def window_partition(self, x: torch.Tensor) -> torch.Tensor:
#         _, H, W, _ = x.shape

#         x = rearrange(x, 'B (Nh Mh) (Nw Mw) C -> (B Nh Nw) Mh Mw C', Nh=H // self.window_size, Nw=W // self.window_size)
#         return x

#     def create_mask(self, x: torch.Tensor) -> torch.Tensor:
#         _, H, W, _ = x.shape

#         assert H % self.window_size == 0 and W % self.window_size == 0, "H or W is not divisible by window_size"

#         img_mask = torch.zeros((1, H, W, 1), device=x.device)
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1

#         mask_windows = self.window_partition(img_mask)
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         return attn_mask

#     def forward(self, x):
#         _, H, W, _ = x.shape

#         if self.shift_size > 0:
#             x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#             mask = self.create_mask(x)
#         else:
#             mask = None

#         x = self.window_partition(x)
#         Bn, Mh, Mw, _ = x.shape
#         x = rearrange(x, 'Bn Mh Mw C -> Bn (Mh Mw) C')
#         qkv = rearrange(self.qkv(x), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)
#         q, k, v = qkv.unbind(0)
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size ** 2, self.window_size ** 2, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(Bn // nW, nW, self.num_heads, Mh * Mw, Mh * Mw) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)
#         attn = self.softmax(attn)
#         attn = self.attn_drop(attn)
#         x = attn @ v
#         x = rearrange(x, 'Bn Nh (Mh Mw) C -> Bn Mh Mw (Nh C)', Mh=Mh)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         x = rearrange(x, '(B Nh Nw) Mh Mw C -> B (Nh Mh) (Nw Mw) C', Nh=H // Mh, Nw=H // Mw)

#         if self.shift_size > 0:
#             x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

class LeFF(nn.Module):
    def __init__(self, dim=32, height=64, width=900,hidden_dim=128, act_layer=nn.GELU, 
                 drop_path = 0., drop = 0, layer_scale_init_value =1e-6):
        super().__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=True),
        #     # act_layer(),
        #     # nn.LayerNorm(hidden_dim, eps=1e-5),
        # )
        
        self.dwconv = nn.Conv2d(dim,dim,groups=dim,kernel_size=[1,3],stride=[1,1],padding=[0,1])                            
        self.act = act_layer()
        self.dwconv_bn = nn.BatchNorm2d(hidden_dim, eps=1e-6)
        self.dwconv_ln = LayerNorm(hidden_dim, eps=1e-6)
        self.conv_ln = LayerNorm(dim, eps=1e-6)
        # self.conv2 = nn.Sequential(
        #                 nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=True),
        #                 # nn.BatchNorm2d(dim, eps=1e-5),
        # )
        self.conv1 = nn.Linear(dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.dim = dim
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # bs x hw x c
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.height, self.width)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv_ln(x)
        # x = self.dwconv_bn(x)
        # print("before",x.shape)
        x = self.conv1(x)
        # x = self.drop(x)
        x = self.act(x)
        # x = self.dwconv_ln(x)
        x = self.conv2(x)
        # x = self.drop(x)
        # x = self.act(x)
        # x = self.conv_ln(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        x = x.flatten(2).permute(0, 2, 1)
        

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim 
        # dwconv
        flops += H*W*self.hidden_dim*3*1
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        # eca 
        if hasattr(self.eca, 'flops'): 
            flops += self.eca.flops()
        return flops



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    
    # B, H, W, C = x.shape
    # x = pad_to_multiples_of(x, [4,4]) # V13
    # x = pad_to_multiples_of(x, [4,1]) # V14
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class LWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=[4,1], shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                 modulator=False, cross_modulator=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        # if min(self.input_resolution) <= min(self.win_size):
        #     self.shift_size = 0
        #     self.win_size[1] = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.win_size[0], "shift_size must in 0-win_size"
        # modulator
        if modulator:
            self.modulator = nn.Embedding(win_size[0]*win_size[1], dim) 
        else:
            self.modulator = None
        # cross_modulator
        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size[0]*win_size[1], dim) 
            self.cross_attn = Attention(dim,num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    token_projection=token_projection,)
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None
        if norm_layer is not None:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, self.win_size, num_heads=num_heads,
            qkv_bias=qkv_bias,  attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)

        #token
        if token_mlp in ['ffn','mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) 
        elif token_mlp=='leff':
            print("leff")
            self.mlp =  LeFF(dim=dim,height=self.input_resolution[0],width=self.input_resolution[1],hidden_dim=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop, drop_path=drop_path)
        
        elif token_mlp=='fastleff':
            self.mlp =  FastLeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)    
        else:
            raise Exception("FFN error!") 
        

        if self.shift_size != 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            # H, W = add_to_multiples_of(self.input_resolution[0],2), add_to_multiples_of(self.input_resolution[1],2)

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.win_size[0]),
                        slice(-self.win_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.win_size[1]),
                        slice(-self.win_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.win_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.win_size[0] * self.win_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))           
            attn_mask = None
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
        
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x):
        H, W = self.input_resolution
        # H, W = add_to_multiples_of(self.input_resolution[0],self.win_size[0]), add_to_multiples_of(self.input_resolution[1],self.win_size[1])
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = pad_to_multiples_of(x, [4,4]) #V13
        # x = pad_to_multiples_of(x, [4,1]) # V14 V12 4,1 V15 2,1
        # B, H, W, C = x.shape
        shortcut = x
        x = x.view(B, H, W, C)
        # shortcut = shortcut.view(B, H*W, C)

        # cyclic shift
        if self.shift_size != 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size[0] * self.win_size[1], C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size[0], self.win_size[1], C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size != 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        # FFN
        
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        # print("after SWinTransformer Block")
        # print(x.shape)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H*W, self.win_size[0]*self.win_size[1])

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H,W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops
    
class BasicLayer(nn.Module):
    def __init__(self, index: int, embed_dim: int = 16, input_resolution: int = [64, 900], win_size= [8,10], depths: tuple = (2, 2, 2, 2),
                 num_heads: tuple = (1, 2, 4 , 8), mlp_ratio: float = 2., qkv_bias: bool = True,dim = [128,128,320,512],
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_merging = True, patch_size = 3, 
                 stride = [1,2,1,1],kernel_size=[3,2,1,1],
                 token_projection='linear',token_mlp='leff', shift_flag=True, down1D = True,
                 modulator=False,cross_modulator=False):
        super(BasicLayer, self).__init__()
        self.patch_merging = patch_merging
        self.input_resolution = input_resolution
        depth = depths[index]
        self.index = index
        # dim = embed_dim * 2 ** index
        self.input_dim = input_dim = dim[index]
        if patch_merging:
            embed_dim = dim[index+1]
        num_head = num_heads[index]
        token_mlp = token_mlp
        win_size =win_size
        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]
        self.blocks = nn.ModuleList([
        
            
            LWinTransformerBlock(
                dim=input_dim,
                input_resolution=input_resolution,
                num_heads=num_head,
                win_size=win_size,
                shift_size=0 if (i % 2 == 0) else [win_size[0] // 2, win_size[1] // 2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    modulator=modulator,cross_modulator=cross_modulator)
                
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if self.patch_merging:
            # self.downsample = PatchMerging(input_resolution=input_resolution,dim=embed_dim * 2 ** index, norm_layer=norm_layer)
            # self.downsample = PatchMerging2(input_resolution=input_resolution,dim=input_dim, embed_dim=embed_dim, norm_layer=norm_layer)
            # self.downsample = PatchMerging1D(input_resolution=input_resolution,dim=input_dim, embed_dim=embed_dim, norm_layer=norm_layer)
            # self.downsample = PatchMerging3(input_resolution=input_resolution,dim=input_dim, embed_dim=embed_dim, norm_layer=norm_layer)
            self.downsample = OverlapPatchMeriging( down1D=down1D,
                                                    in_chans=input_dim, 
                                                    embed_dim=embed_dim,
                                                    kernel_size= kernel_size[index],
                                                    stride=stride[index])
        else:
            self.downsample = None

    def forward(self, x):
        B  = x.shape[0]
        H,W = self.input_resolution
        for layer in self.blocks:
            y = x
            x = layer(x)
            # x = x + y
        # x = rearrange(y, 'B (H W) C  -> B C H W',H=H,W=W)

        
        # x = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = x.view(-1,self.input_dim,H,W)
        f = x
        if self.downsample is not None:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x,H,W = self.downsample(x)
        
        return x,f



class BasicBlockUp(nn.Module):
    def __init__(self, index: int, embed_dim: int = 16, input_resolution: int = [64, 900], win_size= [4,1], depths: tuple = (2, 2, 2, 2),
                 kernel_size=[3,2,1,1],stride = [1,2,1,1],
                 num_heads: tuple = (1, 2, 4, 8), mlp_ratio: float = 2., qkv_bias: bool = True,dim = [128,128,320,512],
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 token_projection='linear',token_mlp='leff', shift_flag=True, down1D = True,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm):
        super(BasicBlockUp, self).__init__()
        self.patch_expanding = patch_expanding
        self.input_resolution = input_resolution
        index = index
        input_dim = dim[index]
        depth = depths[index]
        if patch_expanding:
            embed_dim = dim[index-1]
        num_head = num_heads[index]
        self.win_size = win_size
        token_mlp = token_mlp
        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]
        
        self.blocks = nn.ModuleList([
            LWinTransformerBlock(
                dim=input_dim,
                input_resolution=input_resolution,
                num_heads=num_head,
                win_size=self.win_size,
                shift_size=0 if (i % 2 == 0) else [win_size[0] // 2, win_size[1] // 2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer,
                token_projection=token_projection,token_mlp=token_mlp)
            for i in range(depth)])
        print("patch_expanding",patch_expanding)
        if patch_expanding:
            self.upsample = OverlapPatchExpanding(
                                       down1D=down1D,
                                       in_chans=input_dim,
                                       embed_dim = embed_dim,
                                       kernel_size= kernel_size[index-1],
                                       stride=stride[index-1])
            # self.upsample = Upsample1D(input_resolution = input_resolution,
            #                            in_channel=input_dim,
            #                            out_channel = embed_dim,
            #                            norm_layer = norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        B  = x.shape[0]
        H,W = self.input_resolution
        for layer in self.blocks:
            y = x
            x = layer(x)
            # x = x + y
        if self.upsample is not None:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x,H,W = self.upsample(x)
        return x

class Swin(nn.Module):
    def __init__(self, patch_size: int = [1,1], in_chans: int = 1, num_classes: int = 1000, embed_dim: int = 16,
                 window_size=[8,10], depths: tuple = (2, 2, 2, 2), num_heads: tuple = (1, 2, 4, 8),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, patch_norm: bool = True):
        super().__init__()

        self.window_size = window_size
        
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.norm = norm_layer(self.num_features)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = self.build_layers()
        self.first_patch_expanding = Upsample1D(input_resolution = (get_input_resolution(self.patches_resolution[0], 3),
                                                 self.patches_resolution[1] ),
                                                in_channel=embed_dim * 2 ** (len(depths) - 1),
                                                out_channel = embed_dim * 2 ** (len(depths) - 2),
                                                norm_layer = nn.LayerNorm)

        self.layers_up = self.build_layers_up()
        self.skip_connection_layers = self.skip_connection()
        self.norm_up = norm_layer(embed_dim)
        self.final_patch_expanding = OutputProj1D(input_resolution=(patches_resolution[0], 
                                               patches_resolution[1]),
                                               in_channel=embed_dim, out_channel=in_chans, kernel_size=[2,1], stride=[2,1])
        self.head = nn.Conv2d(in_channels=1, out_channels=num_classes, kernel_size=(1, 1), bias=False)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            
            layer = BasicLayer(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                input_resolution = (get_input_resolution(self.patches_resolution[0], i),
                                                 self.patches_resolution[1] ),
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                win_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                input_resolution = (get_input_resolution(self.patches_resolution[0], self.num_layers -i -2),
                                                 self.patches_resolution[1] ),
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer)
            layers_up.append(layer)
        return layers_up

    def skip_connection(self):
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            layer = nn.Linear(dim * 2, dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers

    def forward(self, x):
        Hp,Wp,x = self.patch_embed(x)
        features = []
        features.append(x)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            # print("layer",x.shape)
        return Hp,Wp,tuple(features)



if __name__ == '__main__':
    # load config ================================================================
    #config_filename = '../config/config.yml'
    #config = yaml.safe_load(open(config_filename))
    #seqs_root = config["data_root"]["data_root_folder"]
    # ============================================================================
    # seqs_root = "F:/Dataset/OT/data_root_folder/"
    # combined_tensor = read_one_need_from_seq(seqs_root, "000000","00")
    # print(combined_tensor.shape)
    # combined_tensor = torch.cat((combined_tensor,combined_tensor), dim=0)
    # print(combined_tensor.shape)
    
    combined_tensor = torch.randn([2,1,64,900])
    swin_unet=Swin(patch_size=[2,1],window_size=[4,2],in_chans=1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SwinUnet.to(device)
    swin_unet
    swin_unet.eval()


    #print("model architecture: \n")
    #print(feature_extracter)
    print("combined_tensor" , combined_tensor.device)
    _,_,pretrained_descriptor = swin_unet(combined_tensor)

    print("size of gloabal descriptor: \n")
    print(pretrained_descriptor.size())
