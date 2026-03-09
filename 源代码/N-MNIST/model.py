import torch
import torch.nn as nn

from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import surrogate

from timm.layers import trunc_normal_, DropPath
from timm.models import register_model
from timm.models.vision_transformer import _cfg

__all__ = ['spikformerLSM']


class SNN_LIF(nn.Module):

    def __init__(self, backend='cupy'):
        super().__init__()
        self.node = MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, backend=backend,
            surrogate_function=surrogate.ATan()
        )

    def forward(self, x): return self.node(x)

    def reset(self): self.node.reset()


class ARIG_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w_att = nn.Linear(dim, dim)
        self.w_lsm = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.lif = SNN_LIF()

    def forward(self, x_attn, x_lsm):
        # x_attn, x_lsm: [T, B, N, C]
        T, B, N, C = x_attn.shape
        attn_flat = x_attn.flatten(0, 1)
        lsm_flat = x_lsm.flatten(0, 1)

        # 交互门控
        gate_lsm = self.sigmoid(self.w_att(attn_flat))
        lsm_filtered = lsm_flat * gate_lsm

        gate_attn = self.sigmoid(self.w_lsm(lsm_flat))
        attn_filtered = attn_flat * gate_attn

        fused = attn_filtered + lsm_filtered

        # BN & LIF
        out = self.proj(fused).transpose(1, 2)
        out = self.bn(out).transpose(1, 2).reshape(T, B, N, C)
        out = self.lif(out)
        return out

    def reset(self):
        self.lif.reset()

class LSM_Branch(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, backend='cupy'):
        super().__init__()
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.rec = nn.Linear(hidden_features, hidden_features)
        nn.init.orthogonal_(self.rec.weight)

        self.lif = MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, backend=backend,
            surrogate_function=surrogate.ATan()
        )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        T, B, N, C = x.shape
        x_flat = x.flatten(0, 1)
        curr_in = self.fc1(x_flat).reshape(T, B * N, self.hidden_features)

        spk_rec = []
        spk = torch.zeros(B * N, self.hidden_features, device=x.device)

        for t in range(T):

            membrane_potential = curr_in[t] + self.rec(spk)
            spk = self.lif(membrane_potential)
            spk_rec.append(spk)

        out_spk = torch.stack(spk_rec).reshape(T * B, N, self.hidden_features)
        out = self.fc2(out_spk)
        out = out.transpose(1, 2)
        out = self.bn(out).transpose(1, 2).reshape(T, B, N, -1)
        return out

    def reset(self):
        self.lif.reset()


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125


        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, N, C = x.shape

        x_for_qkv = x.flatten(0, 1)


        q_linear_out = self.q_linear(x_for_qkv)
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        return x

    def reset(self):
        self.q_lif.reset()
        self.k_lif.reset()
        self.v_lif.reset()
        self.attn_lif.reset()
        self.proj_lif.reset()

class InteractiveBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        hidden_dim = int(dim * mlp_ratio)
        self.lsm_branch = LSM_Branch(in_features=dim, hidden_features=hidden_dim, out_features=dim)
        self.fusion = ARIG_Fusion(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_norm = self.norm1(x)
        res_attn = self.attn(x_norm)
        res_lsm = self.lsm_branch(x_norm)
        fused_features = self.fusion(res_attn, res_lsm)
        x = x + self.drop_path(fused_features)
        return x

    def reset(self):
        self.attn.reset()
        self.lsm_branch.reset()
        self.fusion.reset()


class SPS(nn.Module):
    def __init__(self, img_size_h=28, img_size_w=28, patch_size=4, in_channels=1, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]

        self.dim1 = embed_dims // 8
        self.dim2 = embed_dims // 4
        self.dim3 = embed_dims // 2
        self.dim4 = embed_dims

        self.proj_conv = nn.Conv2d(in_channels, self.dim1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(self.dim1)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv1 = nn.Conv2d(self.dim1, self.dim2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(self.dim2)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv2 = nn.Conv2d(self.dim2, self.dim3, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(self.dim3)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(self.dim3, self.dim4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(self.dim4)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(self.dim4, self.dim4, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(self.dim4)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.H = img_size_h // 4
        self.W = img_size_w // 4
        self.num_patches = self.H * self.W

    def forward(self, x):
        T, B, C, H, W = x.shape

        # Stage 1
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        # Stage 2
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        # Stage 3
        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)
        H, W = H // 2, W // 2

        # Stage 4
        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)
        H, W = H // 2, W // 2

        # RPE (Output shape is 5D: [T, B, C, H, W])
        x_feat = x.reshape(T, B, -1, H, W).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat


        x = x.flatten(-2).transpose(-1, -2).contiguous()

        return x

    def reset(self):
        for m in [self.proj_lif, self.proj_lif1, self.proj_lif2, self.proj_lif3, self.rpe_lif]:
            m.reset()


class SpikformerLSM_Interactive(nn.Module):
    def __init__(self, img_size_h=32, img_size_w=32, patch_size=4, in_channels=1, num_classes=10,
                 embed_dims=128, num_heads=8, mlp_ratios=4, drop_path_rate=0.1, depths=2, T=4):
        super().__init__()
        self.T = T
        self.patch_embed = SPS(img_size_h, img_size_w, patch_size, in_channels, embed_dims)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dims))
        trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.block = nn.ModuleList([
            InteractiveBlock(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, drop_path=dpr[j])
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
        # Input x: [B, T, C, H, W] -> [T, B, C, H, W]
        x = x.permute(1, 0, 2, 3, 4)

        x = self.patch_embed(x)

        T, B, N, C = x.shape
        pos = self.pos_embed.unsqueeze(0).expand(T, -1, -1, -1)
        x = x + pos

        for blk in self.block:
            x = blk(x)

        x = x.mean(2)  # Pooling spatial dimension [T, B, C]
        x = self.norm(x)
        x = self.head(x)  # [T, B, Num_classes]

        return x

    def reset(self):
        self.patch_embed.reset()
        for blk in self.block:
            blk.reset()


@register_model
def spikformerLSM(**kwargs):
    model = SpikformerLSM_Interactive(**kwargs)
    model.default_cfg = _cfg()
    return model