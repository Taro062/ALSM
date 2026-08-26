from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple

_NUMPY_COMPAT_ALIASES = {
    "int": int,
    "float": float,
    "complex": complex,
    "bool": bool,
    "object": object,
}
for _name, _value in _NUMPY_COMPAT_ALIASES.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_

try:
    from spikingjelly.activation_based import neuron, surrogate
    _SJ_API = "activation_based"
except ImportError:
    from spikingjelly.clock_driven import neuron, surrogate
    _SJ_API = "clock_driven"


def _safe_construct(cls, **kwargs):

    try:
        return cls(**kwargs)
    except TypeError:
        kwargs = dict(kwargs)
        kwargs.pop("step_mode", None)
        try:
            return cls(**kwargs)
        except TypeError:
            kwargs.pop("backend", None)
            return cls(**kwargs)


def _make_lif(
    *,
    multi_step: bool,
    parametric: bool,
    tau: float = 2.0,
    threshold: float = 1.0,
    backend: str = "torch",
):
    common = dict(
        detach_reset=True,
        v_threshold=threshold,
        surrogate_function=surrogate.ATan(),
    )

    if _SJ_API == "activation_based":
        cls = neuron.ParametricLIFNode if parametric else neuron.LIFNode
        if parametric:
            common["init_tau"] = tau
        else:
            common["tau"] = tau
        common["step_mode"] = "m" if multi_step else "s"
        common["backend"] = backend if multi_step else "torch"
        return _safe_construct(cls, **common)

    if multi_step:
        cls = (
            neuron.MultiStepParametricLIFNode
            if parametric
            else neuron.MultiStepLIFNode
        )
    else:
        cls = neuron.ParametricLIFNode if parametric else neuron.LIFNode

    if parametric:
        common["init_tau"] = tau
    else:
        common["tau"] = tau
    common["backend"] = backend if multi_step else "torch"
    return _safe_construct(cls, **common)


class ConvBNPoolLIF(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pool: bool,
        backend: str,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if pool else nn.Identity()
        )
        self.lif = _make_lif(
            multi_step=True, parametric=False, tau=2.0, backend=backend
        )

    def forward(self, x: torch.Tensor, t: int, b: int) -> torch.Tensor:
        # x is [T*B, C, H, W]
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.pool(self.bn(self.conv(x)))
        _, c, h, w = x.shape
        x = x.reshape(t, b, c, h, w).contiguous()
        x = self.lif(x)
        return x.flatten(0, 1).contiguous()


class SpikingPatchStem(nn.Module):

    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 2,
        embed_dim: int = 256,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError("img_size must be divisible by 16.")
        if embed_dim % 8 != 0:
            raise ValueError("embed_dim must be divisible by 8.")

        dims = [embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim]
        self.stage1 = ConvBNPoolLIF(
            in_channels, dims[0], pool=True, backend=backend
        )
        self.stage2 = ConvBNPoolLIF(
            dims[0], dims[1], pool=True, backend=backend
        )
        self.stage3 = ConvBNPoolLIF(
            dims[1], dims[2], pool=True, backend=backend
        )
        self.stage4 = ConvBNPoolLIF(
            dims[2], dims[3], pool=True, backend=backend
        )

        self.rpe_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = _make_lif(
            multi_step=True, parametric=False, tau=2.0, backend=backend
        )

        self.grid_size = img_size // 16
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        if x.ndim != 5:
            raise ValueError(f"Expected 5-D input, got shape {tuple(x.shape)}")
        t, b, _, _, _ = x.shape
        x = x.flatten(0, 1).contiguous()

        x = self.stage1(x, t, b)
        x = self.stage2(x, t, b)
        x = self.stage3(x, t, b)
        x = self.stage4(x, t, b)

        _, c, h, w = x.shape
        residual = x.reshape(t, b, c, h, w).contiguous()

        rpe = self.rpe_bn(self.rpe_conv(x))
        rpe = rpe.reshape(t, b, c, h, w).contiguous()
        rpe = self.rpe_lif(rpe)

        x = residual + rpe
        # [T, B, C, H, W] -> [T, B, N, C]
        return x.flatten(-2).transpose(-1, -2).contiguous()


class SpikingSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        attn_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = (
            float(attn_scale)
            if attn_scale is not None
            else 1.0 / math.sqrt(head_dim)
        )

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.v_bn = nn.BatchNorm1d(dim)

        self.q_lif = _make_lif(
            multi_step=True, parametric=False, tau=2.0, backend=backend
        )
        self.k_lif = _make_lif(
            multi_step=True, parametric=False, tau=2.0, backend=backend
        )
        self.v_lif = _make_lif(
            multi_step=True, parametric=False, tau=2.0, backend=backend
        )
        self.attn_lif = _make_lif(
            multi_step=True, parametric=False, tau=2.0,
            threshold=0.5, backend=backend
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = _make_lif(
            multi_step=True, parametric=False, tau=2.0, backend=backend
        )

    @staticmethod
    def _linear_bn(
        x: torch.Tensor,
        linear: nn.Linear,
        bn: nn.BatchNorm1d,
        t: int,
        b: int,
        n: int,
        c: int,
    ) -> torch.Tensor:
        y = linear(x)
        y = bn(y.transpose(1, 2)).transpose(1, 2)
        return y.reshape(t, b, n, c).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        t, b, n, c = x.shape
        flat = x.flatten(0, 1)

        q = self.q_lif(
            self._linear_bn(flat, self.q, self.q_bn, t, b, n, c)
        )
        k = self.k_lif(
            self._linear_bn(flat, self.k, self.k_bn, t, b, n, c)
        )
        v = self.v_lif(
            self._linear_bn(flat, self.v, self.v_bn, t, b, n, c)
        )

        head_dim = c // self.num_heads
        q = q.reshape(t, b, n, self.num_heads, head_dim)
        k = k.reshape(t, b, n, self.num_heads, head_dim)
        v = v.reshape(t, b, n, self.num_heads, head_dim)
        q = q.permute(0, 1, 3, 2, 4).contiguous()
        k = k.permute(0, 1, 3, 2, 4).contiguous()
        v = v.permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn)
        y = attn @ v
        y = y.transpose(2, 3).reshape(t, b, n, c).contiguous()
        y = self.attn_lif(y)

        y = self.proj(y.flatten(0, 1))
        y = self.proj_bn(y.transpose(1, 2)).transpose(1, 2)
        y = y.reshape(t, b, n, c).contiguous()
        return self.proj_lif(y)


class LSMBranch(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        backend: str = "cupy",
        recurrent_gain: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.in_norm = nn.LayerNorm(dim)
        self.fc_in = nn.Linear(dim, hidden_dim, bias=False)
        self.in_bn = nn.BatchNorm1d(hidden_dim)

        self.rec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # tanh(raw_gain) is bounded and keeps the reservoir stable.
        recurrent_gain = min(max(recurrent_gain, 1e-4), 0.95)
        self.raw_rec_gain = nn.Parameter(
            torch.tensor(math.atanh(recurrent_gain), dtype=torch.float32)
        )

        self.rec_lif = _make_lif(
            multi_step=False, parametric=True, tau=2.0, backend="torch"
        )

        self.fc_out = nn.Linear(hidden_dim, dim, bias=False)
        self.out_bn = nn.BatchNorm1d(dim)
        self.out_lif = _make_lif(
            multi_step=True, parametric=True, tau=2.0, backend=backend
        )
        self.reset_special_parameters()

    def reset_special_parameters(self) -> None:
        nn.init.orthogonal_(self.rec.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        t, b, n, _ = x.shape
        x = self.in_norm(x)
        current = self.fc_in(x.flatten(0, 1))
        current = self.in_bn(current.transpose(1, 2)).transpose(1, 2)
        current = current.reshape(t, b * n, self.hidden_dim).contiguous()

        spike = torch.zeros(
            b * n,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )
        gain = torch.tanh(self.raw_rec_gain)
        spikes = []

        for step in range(t):
            recurrent_current = self.rec(spike)
            spike = self.rec_lif(current[step] + gain * recurrent_current)
            spikes.append(spike)

        y = torch.stack(spikes, dim=0).reshape(t * b, n, self.hidden_dim)
        y = self.fc_out(y)
        y = self.out_bn(y.transpose(1, 2)).transpose(1, 2)
        y = y.reshape(t, b, n, -1).contiguous()
        return self.out_lif(y)


class ARIGFusion(nn.Module):
    """Attention-Reservoir Interactive Gating."""

    def __init__(
        self,
        dim: int,
        *,
        gate_dropout: float = 0.0,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.lsm_norm = nn.LayerNorm(dim)

        # Attention controls reservoir; reservoir controls attention.
        self.attn_to_lsm = nn.Linear(dim, dim)
        self.lsm_to_attn = nn.Linear(dim, dim)

        self.gate_dropout = nn.Dropout(gate_dropout)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = _make_lif(
            multi_step=True, parametric=True, tau=2.0, backend=backend
        )

        # A moderate initial residual scale gives stable optimization while
        # allowing ARIG to contribute from the beginning.
        self.layer_scale = nn.Parameter(torch.full((dim,), 0.5))
        self.last_gate_stats = None
        self.reset_special_parameters()

    def reset_special_parameters(self) -> None:
        nn.init.zeros_(self.attn_to_lsm.weight)
        nn.init.zeros_(self.lsm_to_attn.weight)
        nn.init.constant_(self.attn_to_lsm.bias, 1.0)
        nn.init.constant_(self.lsm_to_attn.bias, 1.0)

    def forward(
        self,
        x_attn: torch.Tensor,
        x_lsm: torch.Tensor,
    ) -> torch.Tensor:
        t, b, n, c = x_attn.shape

        gate_lsm = torch.sigmoid(
            self.attn_to_lsm(self.attn_norm(x_attn))
        )
        gate_attn = torch.sigmoid(
            self.lsm_to_attn(self.lsm_norm(x_lsm))
        )

        gate_lsm = self.gate_dropout(gate_lsm)
        gate_attn = self.gate_dropout(gate_attn)

        fused = x_lsm * gate_lsm + x_attn * gate_attn
        fused = fused * self.layer_scale

        y = self.proj(fused.flatten(0, 1))
        y = self.proj_bn(y.transpose(1, 2)).transpose(1, 2)
        y = y.reshape(t, b, n, c).contiguous()
        y = self.proj_lif(y)

        # Scalar summaries can be logged for the paper without retaining
        # large activation tensors.
        self.last_gate_stats = {
            "lsm_gate_mean": gate_lsm.detach().mean(),
            "attn_gate_mean": gate_attn.detach().mean(),
            "lsm_gate_std": gate_lsm.detach().std(),
            "attn_gate_std": gate_attn.detach().std(),
        }
        return y


class InteractiveALSMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        lsm_ratio: float = 4.0,
        attn_scale: Optional[float] = 0.125,
        attn_drop: float = 0.0,
        gate_dropout: float = 0.0,
        drop_path: float = 0.0,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.attn = SpikingSelfAttention(
            dim,
            num_heads,
            attn_scale=attn_scale,
            attn_drop=attn_drop,
            backend=backend,
        )
        self.lsm = LSMBranch(
            dim,
            hidden_dim=int(dim * lsm_ratio),
            backend=backend,
        )
        self.arig = ARIGFusion(
            dim,
            gate_dropout=gate_dropout,
            backend=backend,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x)
        y_attn = self.attn(y)
        y_lsm = self.lsm(y)
        y = self.arig(y_attn, y_lsm)
        return x + self.drop_path(y)


class ALSMDVS128Gesture(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 128,
        in_channels: int = 2,
        num_classes: int = 11,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 2,
        lsm_ratio: float = 2.0,
        drop_path_rate: float = 0.10,
        attn_drop: float = 0.0,
        gate_dropout: float = 0.0,
        head_dropout: float = 0.1,
        attn_scale: Optional[float] = 0.125,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = SpikingPatchStem(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            backend=backend,
        )

        grid = self.patch_embed.grid_size
        self.pos_grid_size = grid
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid * grid, embed_dim)
        )
        trunc_normal_(self.pos_embed, std=0.02)

        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                InteractiveALSMBlock(
                    embed_dim,
                    num_heads,
                    lsm_ratio=lsm_ratio,
                    attn_scale=attn_scale,
                    attn_drop=attn_drop,
                    gate_dropout=gate_dropout,
                    drop_path=rates[i],
                    backend=backend,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

        # Generic initialization above must not overwrite the reservoir's
        # orthogonal recurrent matrix or ARIG's initially open gates.
        for module in self.modules():
            if isinstance(module, (LSMBranch, ARIGFusion)):
                module.reset_special_parameters()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _position_embedding(self, n: int) -> torch.Tensor:
        if n == self.pos_embed.shape[1]:
            return self.pos_embed

        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError("Token count must form a square grid.")
        pos = self.pos_embed.reshape(
            1, self.pos_grid_size, self.pos_grid_size, self.embed_dim
        ).permute(0, 3, 1, 2)
        pos = F.interpolate(
            pos, size=(side, side), mode="bicubic", align_corners=False
        )
        return pos.flatten(2).transpose(1, 2).contiguous()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Input [B, T, C, H, W] -> [T, B, C, H, W]
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        x = self.patch_embed(x)

        t, _, n, _ = x.shape
        pos = self._position_embedding(n).unsqueeze(0)
        x = x + pos.expand(t, -1, -1, -1)

        for block in self.blocks:
            x = block(x)

        # Spatial rate pooling, preserving time for temporal supervision.
        return self.norm(x.mean(dim=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(self.head_dropout(features))

    def gate_statistics(self):
        stats = []
        for i, block in enumerate(self.blocks):
            if block.arig.last_gate_stats is not None:
                item = {"block": i}
                item.update(block.arig.last_gate_stats)
                stats.append(item)
        return stats


def alsm_dvs128gesture(**kwargs) -> ALSMDVS128Gesture:
    return ALSMDVS128Gesture(**kwargs)


__all__ = ["ALSMDVS128Gesture", "alsm_dvs128gesture"]
