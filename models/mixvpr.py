import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=4,drop=0.2):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # return self.mix(x)
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=256,
                 in_h=4,
                 in_w=900,
                 out_channels=256,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=1,
                 init_eps = 1e-3,
                 drop = 0.2
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.hw = hw
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=in_w, mlp_ratio=4)
            for _ in range(self.mix_depth)
        ])
        self.proj_in = nn.Sequential(
            nn.Linear(in_channels,in_channels*2),
            nn.GELU()
        )
        self.channel_proj = nn.Linear(in_channels , out_channels)

        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels // 2 , self.in_channels // 2  * self.mlp_ratio, kernel_size=1),
                nn.LayerNorm([self.in_channels // 2 * self.mlp_ratio, 1, hw]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.in_channels // 2 * self.mlp_ratio, self.in_channels // 2 , kernel_size=1))
        self.norm = nn.LayerNorm(self.in_channels)
        self.drop = nn.Dropout(drop)
        self.spatial_proj = nn.Conv1d(hw, hw, 1)
        self.act = nn.Identity()
        init_eps /= hw
        # nn.init.uniform_(self.spatial_proj.weight, -init_eps, init_eps)
        # nn.init.constant_(self.spatial_proj.bias, 1.)

        self.row_proj = nn.Linear(in_w , out_rows)
        # self.input_proj = nn.Linear(hw , in_w)

    def spatial_pooling(self, x):
        B, C, H, W = x.shape
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(B, C, H * W)
        # [N, 1, C, H * W] # [N, 1, H * W, 1]
        input_x = input_x.unsqueeze(1)
        print("input_x",input_x.shape)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        print("context_mask",context_mask.shape)
        # [N, 1, H * W]
        context_mask = context_mask.view(B, 1, H * W)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        print("input_x",input_x.shape)
        print("context_mask",context_mask.shape)
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(B, C, 1, 1)
        
        return context
    
    def spatial_gating(self, x, gate_res = None):
        x = self.proj_in(x)
        res, gate = x.chunk(2, dim=-1) # B  H*W C/2
        print("gate",gate.shape)
        print("gate_res",gate_res.shape)
        gate = self.norm(gate)
        weight, bias = self.spatial_proj.weight , self.spatial_proj.bias
        gate = F.conv1d(gate,weight,bias)
        # print("gate",gate.shape)
        if gate_res is not None:
            gate = gate + gate_res
        print("gate",gate.shape)
        # print("res",res.shape)
        # print("self.act(gate)",(self.act(gate)).shape)
        # v = self.row_proj(v) # B C R_OUT
        return self.act(gate) * res

        # u,v = x.chunk(2, dim=-1) # B  H*W C/2
        # v=self.norm(v)
        # v=self.row_proj(v)
        # print(u.shape)
        # print(v.shape)
        # return u*v
        

        
        
    def forward(self, x):
        # B C H W

        ####GATING
        
        # x = torch.mean(x, dim=2, keepdim=True)
        # x = x.squeeze(2).permute(0, 2, 1)
        # shorcut = x      
        # x = self.mix(x) # B C H*W
        # x = self.spatial_gating(x,gate_res=shorcut) # B 1 C
        # x = self.channel_proj(x) # B H*W C_OUT
        # x = x.permute(0, 2, 1) # B C_OUT H*W
        # x = self.row_proj(x) # B C_OUT R_OUT
        # x = F.normalize(x.flatten(1), p=2, dim=-1) # B C_OUT*R_OUT 
        
        B,C,H,W = x.shape
        x = x.flatten(2) # B C H*W
        if self.hw > self.in_w:
            x = self.input_proj(x) # B C R_OUT
        x = self.mix(x) # B C H*W
        x = x.permute(0, 2, 1) # B R_OUT C
        x = self.channel_proj(x) # B R_OUT C_OUT
        x = self.drop(x)
        x = x.permute(0, 2, 1) # B H*W C
        x = self.row_proj(x) # B C R_OUT
        x = self.drop(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1) # B C_OUT*R_OUT 
        
        
        # x = x.squeeze(3).permute(0, 2, 1) # B H*W C
        # x = self.channel_proj(x) # B H*W C_OUT
        # print(x.shape)
        # x = x.permute(0, 2, 1) # B C_OUT H*W
        # x = self.row_proj(x) # B C_OUT R_OUT
        return x


# -------------------------------------------------------------------------------

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(2, 64, 16, 224)
    agg = MixVPR(
        in_channels=64,
        in_h=16,
        in_w=224,
        out_channels=8,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=8)

    print_nb_params(agg)
    output = agg(x)
    print(output.shape)


if __name__ == '__main__':
    main()
