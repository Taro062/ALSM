import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven import surrogate
from timm.layers import to_2tuple, trunc_normal_, DropPath
from timm.models import register_model
from timm.models.vision_transformer import _cfg
import snntorch as snn
import math

__all__ = ['spikformerLSM_Interactive']

class SNN_LIF(nn.Module):

    def __init__(self):
        super().__init__()
        self.node = MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, backend='cupy', 
            surrogate_function=surrogate.ATan()
        )
    def forward(self, x): return self.node(x)
    def reset(self): self.node.reset()

class ARIG_Fusion(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 交互权重
        self.w_att = nn.Linear(dim, dim)
        self.w_lsm = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        
        # 融合后的输出层
        self.proj = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.lif = SNN_LIF()

    def forward(self, x_attn, x_lsm):
        # x_attn, x_lsm: [T, B, N, C]
        T, B, N, C = x_attn.shape
        
        # 1. 扁平化处理
        attn_flat = x_attn.flatten(0, 1) # [TB, N, C]
        lsm_flat = x_lsm.flatten(0, 1)   # [TB, N, C]

        # gate_lsm: 用 Attention 的信息去过滤 LSM 的噪声
        gate_lsm = self.sigmoid(self.w_att(attn_flat)) 
        lsm_filtered = lsm_flat * gate_lsm
        
        # gate_attn: 用 LSM 的时序惯性去平滑 Attention 的突变
        gate_attn = self.sigmoid(self.w_lsm(lsm_flat))
        attn_filtered = attn_flat * gate_attn
        
        # 3. 融合
        fused = attn_filtered + lsm_filtered
        
        # 4. 激活输出
        out = self.proj(fused).transpose(1, 2) # [TB, C, N]
        out = self.bn(out).transpose(1, 2).reshape(T, B, N, C)
        out = self.lif(out)
        
        return out
    
    def reset(self):
        self.lif.reset()

class InteractiveBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop)

        hidden_dim = int(dim * mlp_ratio)
        self.lsm_branch = LSM_Branch(in_features=dim, hidden_features=hidden_dim, out_features=dim)
        
        # --- Interaction: ARIG ---
        self.fusion = ARIG_Fusion(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_norm = x

        res_attn = self.attn(x)
        

        res_lsm = self.lsm_branch(x)

        fused_features = self.fusion(res_attn, res_lsm)
        
        x = x + self.drop_path(fused_features)
        
        return x

class LSM_Branch(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, th=1.0):
        super().__init__()
        # 输入映射
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.lsm = snn.RSynaptic(alpha=0.9, beta=0.8, all_to_all=True, linear_features=hidden_features, threshold=th)

        W = torch.randn(hidden_features, hidden_features) * 0.1
        try:
            spec_rad = torch.max(torch.abs(torch.linalg.eigvals(W.to(torch.complex64))))
            if spec_rad > 0: W *= 0.95 / spec_rad
        except: pass
        self.lsm.recurrent.weight = nn.Parameter(W)
        
        # 输出映射
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.lif = SNN_LIF()
        self.hidden_features = hidden_features

    def forward(self, x):
        T, B, N, C = x.shape
        x_flat = x.flatten(0, 1) # [TB, N, C]
        
        # Input
        curr = self.fc1(x_flat) # [TB, N, Hidden]
        curr = curr.reshape(T, B*N, self.hidden_features)
        
        # LSM Recurrence
        spk_rec = []
        spk = torch.zeros(B*N, self.hidden_features, device=x.device)
        syn = torch.zeros(B*N, self.hidden_features, device=x.device)
        mem = torch.zeros(B*N, self.hidden_features, device=x.device)
        
        for t in range(T):
            spk, syn, mem = self.lsm(curr[t], spk, syn, mem)
            spk_rec.append(spk)
            
        out_spk = torch.stack(spk_rec) # [T, BN, Hidden]
        
        # Output
        out_spk = out_spk.view(T*B, N, self.hidden_features)
        out = self.fc2(out_spk) # [TB, N, Out]
        
        # BN & LIF
        out = out.transpose(1, 2) # [TB, Out, N] for BN
        out = self.bn(out).transpose(1, 2).reshape(T, B, N, -1)
        out = self.lif(out)
        
        return out

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_lif = SNN_LIF()
        self.k_lif = SNN_LIF()
        self.v_lif = SNN_LIF()
        self.attn_lif = SNN_LIF()
        self.proj = nn.Linear(dim, dim)
        self.proj_lif = SNN_LIF()

    def forward(self, x):
        T, B, N, C = x.shape
        x_flat = x.flatten(0, 1)
        
        q = self.q_lif(self.q_conv(x_flat)).reshape(T, B, N, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        k = self.k_lif(self.k_conv(x_flat)).reshape(T, B, N, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        v = self.v_lif(self.v_conv(x_flat)).reshape(T, B, N, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        
        x = x.transpose(3, 4).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj(x.flatten(0, 1))).reshape(T, B, N, C)
        return x

class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=8, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)

        self.dim1 = embed_dims // 8
        self.dim2 = embed_dims // 4
        self.dim3 = embed_dims // 2
        self.dim4 = embed_dims

        self.proj_conv = nn.Conv2d(in_channels, self.dim1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(self.dim1)
        self.proj_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(self.dim1, self.dim2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(self.dim2)
        self.proj_lif1 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(self.dim2, self.dim3, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(self.dim3)
        self.proj_lif2 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(self.dim3, self.dim4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(self.dim4)
        self.proj_lif3 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.rpe_conv = nn.Conv2d(self.dim4, self.dim4, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(self.dim4)
        self.rpe_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.H = img_size_h // patch_size[0]
        self.W = img_size_w // patch_size[1]
        self.num_patches = self.H * self.W

    def forward(self, x):
        T, B, C, H, W = x.shape
        
        # Stage 1
        x = self.proj_lif(self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, -1, H, W)).flatten(0, 1)
        x = self.maxpool(x)
        
        # Stage 2
        x = self.proj_lif1(self.proj_bn1(self.proj_conv1(x)).reshape(T, B, -1, x.shape[2], x.shape[3])).flatten(0, 1)
        x = self.maxpool1(x)
        
        # Stage 3
        x = self.proj_lif2(self.proj_bn2(self.proj_conv2(x)).reshape(T, B, -1, x.shape[2], x.shape[3])).flatten(0, 1)
        x = self.maxpool2(x)
        
        # Stage 4
        x = self.proj_lif3(self.proj_bn3(self.proj_conv3(x)).reshape(T, B, -1, x.shape[2], x.shape[3])).flatten(0, 1)
        
        # RPE
        x_rpe = self.rpe_lif(self.rpe_bn(self.rpe_conv(x)).reshape(T, B, -1, x.shape[2], x.shape[3])).flatten(0, 1)
        x = x + x_rpe
        
        # Flatten
        x = x.flatten(2).transpose(1, 2).reshape(T, B, -1, x.shape[1]).contiguous()
        return x

class SpikformerLSM_Interactive(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=8, in_channels=2, num_classes=10,
                 embed_dims=256, num_heads=8, mlp_ratios=4, drop_path_rate=0.1, depths=2, T=10):
        super().__init__()
        self.T = T
        
        self.patch_embed = SPS(img_size_h, img_size_w, patch_size, in_channels, embed_dims)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dims))
        trunc_normal_(self.pos_embed, std=.02)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.block = nn.ModuleList([
            InteractiveBlock(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, 
                             drop_path=dpr[j])
            for j in range(depths)
        ])
        
        self.norm = nn.LayerNorm(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4) # [T, B, C, H, W]

        x = self.patch_embed(x)

        T, B, N, C = x.shape
        pos = self.pos_embed.unsqueeze(0).expand(T, -1, -1, -1)
        x = x + pos

        for blk in self.block:
            x = blk(x)

        x = x.mean(2).mean(0) # [B, C]
        
        x = self.norm(x)
        x = self.head(x)
        return x
    
    def reset(self):
        for m in self.modules():
            if hasattr(m, 'reset') and m is not self:
                try: m.reset()
                except: pass

@register_model
def spikformerLSM(pretrained=False, **kwargs):
    model = SpikformerLSM_Interactive(
        img_size_h=128, img_size_w=128, patch_size=8, embed_dims=256, num_heads=8,
        mlp_ratios=4, in_channels=2, num_classes=11,
        drop_path_rate=0.1, depths=2, T=10,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model