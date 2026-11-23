"""Standalone pure-PyTorch SimAM-ERes2Conformer speaker encoder (FlaxSpeaker).

Loads pretrained ``wq2012/univ_full_eres2conformer_*`` SafeTensors weights from
FlaxSpeaker (Apache-2.0) directly in PyTorch without requiring JAX or Flax.
Supports both pure cosine (`catp`) checkpoints (e.g.
``wq2012/univ_full_eres2conformer_xl`` and
``wq2012/univ_full_eres2conformer_v2_large``) and ``Hybrid-PFAS`` checkpoints
(``wq2012/univ_full_eres2conformer_*_pfas``) for attentive scoring.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
except ImportError:  # pragma: no cover
  torch = None  # type: ignore
  nn = None  # type: ignore
  F = None  # type: ignore


FLAXSPEAKER_PRESET_MODELS = (
    "wq2012/univ_full_eres2conformer_xl",
    "wq2012/univ_full_eres2conformer_v2_large",
    "wq2012/univ_full_eres2conformer_large",
    "wq2012/univ_full_eres2conformer_xl_pfas",
    "wq2012/univ_full_eres2conformer_v2_pfas",
    "wq2012/univ_full_eres2conformer_large_pfas",
)


def is_flaxspeaker_model_id(name: Optional[str]) -> bool:
  """Returns True when ``name`` points to a FlaxSpeaker ERes2Conformer model."""
  if not name:
    return False
  low = str(name).lower()
  if "eres2conformer" in low:
    return True
  cfg_path = os.path.join(str(name), "config.json")
  if os.path.isfile(cfg_path):
    try:
      with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
      return cfg.get("model_type") == "flaxspeaker"
    except Exception:
      return False
  return False


def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
  """Slaney mel scale matching librosa.hz_to_mel(..., htk=False)."""
  f_sp = 200.0 / 3.0
  min_log_hz = 1000.0
  min_log_mel = min_log_hz / f_sp
  logstep = np.log(6.4) / 27.0
  hz = np.asarray(hz, dtype=np.float64)
  mel = hz / f_sp
  mask = hz >= min_log_hz
  mel[mask] = min_log_mel + np.log(hz[mask] / min_log_hz) / logstep
  return mel


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
  """Inverse Slaney mel scale matching ``librosa.mel_to_hz(..., htk=False)``."""
  f_sp = 200.0 / 3.0
  min_log_hz = 1000.0
  min_log_mel = min_log_hz / f_sp
  logstep = np.log(6.4) / 27.0
  mel = np.asarray(mel, dtype=np.float64)
  hz = f_sp * mel
  mask = mel >= min_log_mel
  hz[mask] = min_log_hz * np.exp(logstep * (mel[mask] - min_log_mel))
  return hz


def build_slaney_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    fmin: float = 20.0,
    fmax: float = 7600.0,
) -> np.ndarray:
  """Builds Slaney mel filterbank matching librosa(..., norm='slaney')."""
  n_freqs = n_fft // 2 + 1
  fftfreqs = np.linspace(0.0, float(sample_rate) / 2.0, n_freqs)
  mel_min = float(_hz_to_mel_slaney(np.array([fmin]))[0])
  mel_max = float(_hz_to_mel_slaney(np.array([fmax]))[0])
  mels = np.linspace(mel_min, mel_max, n_mels + 2)
  mel_f = _mel_to_hz_slaney(mels)
  fdiff = np.diff(mel_f)
  ramps = mel_f[:, None] - fftfreqs[None, :]
  weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
  for i in range(n_mels):
    lower = -ramps[i] / fdiff[i]
    upper = ramps[i + 2] / fdiff[i + 1]
    weights[i] = np.maximum(0.0, np.minimum(lower, upper))
  enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
  weights *= enorm[:, None]
  return weights.astype(np.float32)


def compute_flaxspeaker_log_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    fmin: float = 20.0,
    fmax: float = 7600.0,
    preemph: float = 0.97,
    log_offset: float = 1e-6,
    apply_agc: bool = True,
    apply_cmvn: bool = True,
) -> np.ndarray:
  """Computes FlaxSpeaker 80-D natural-log Slaney Mel features ``(T, 80)``."""
  wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
  if wav.size == 0:
    return np.zeros((0, n_mels), dtype=np.float32)
  if apply_agc:
    peak = float(np.max(np.abs(wav)))
    if peak > 1e-8:
      wav = wav / peak
  if preemph > 0.0 and wav.shape[0] > 1:
    wav = np.concatenate([wav[:1], wav[1:] - preemph * wav[:-1]])
  if wav.shape[0] < win_length:
    wav = np.pad(wav, (0, win_length - wav.shape[0]))

  mel_basis = build_slaney_mel_filterbank(
      sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
  )
  if torch is not None:
    wav_t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    # librosa hann window matches scipy/torch hann_window(periodic=True)
    win_t = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
    # Match librosa.stft(center=True, pad_mode='constant') zero-padding
    pad_len = n_fft // 2
    padded = F.pad(wav_t.unsqueeze(0), (pad_len, pad_len), mode="constant")[0]
    stft = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=win_t,
        center=False,
        return_complex=True,
    )
    power = stft.real.pow(2) + stft.imag.pow(2)  # (n_freqs, T)
    mel_t = torch.from_numpy(mel_basis) @ power
    feats = torch.log(mel_t + float(log_offset)).transpose(0, 1).numpy()
  else:  # pragma: no cover
    pad_len = n_fft // 2
    padded = np.pad(wav, (pad_len, pad_len), mode="constant")
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    win = np.hanning(win_length + 1)[:-1].astype(np.float32)
    left = (n_fft - win_length) // 2
    win_pad = np.zeros(n_fft, dtype=np.float32)
    win_pad[left:left + win_length] = win
    idx = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop_length
    frames = padded[idx] * win_pad[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    feats = np.log(power @ mel_basis.T + float(log_offset)).astype(np.float32)

  if apply_cmvn and feats.shape[0] > 0:
    feats = feats - np.mean(feats, axis=0, keepdims=True)
  return np.asarray(feats, dtype=np.float32)


def compute_hybrid_pfas_similarity_matrix(
    packed_embs: np.ndarray,
    cosine_dim: int = 256,
    num_keys: int = 8,
    key_dim: int = 32,
    value_dim: int = 96,
    hybrid_alpha: float = 0.25,
    scale_factor: float = 16.0,
    center_cosine: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
  """Computes ``(N, N)`` Hybrid-PFAS similarity matrix mapped to ``[-1, 1]``."""
  arr = np.asarray(packed_embs, dtype=np.float32)
  cos_e = arr[:, :cosine_dim]
  if center_cosine and cos_e.shape[0] >= 2:
    cos_e = cos_e - np.mean(cos_e, axis=0, keepdims=True)
  cos_e = cos_e / np.maximum(
      np.linalg.norm(cos_e, axis=-1, keepdims=True), eps
  )
  cos_mat = cos_e @ cos_e.T

  if hybrid_alpha <= 0.0 or arr.shape[1] <= cosine_dim:
    return cos_mat

  tail = arr[:, cosine_dim:]
  k_tot = num_keys * key_dim
  keys = tail[:, :k_tot].reshape(arr.shape[0], num_keys, key_dim)
  vals = tail[:, k_tot:].reshape(arr.shape[0], num_keys, value_dim)
  keys = keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + eps)

  n = arr.shape[0]
  pfas_mat = np.zeros((n, n), dtype=np.float32)
  v_sq = np.sum(vals * vals, axis=-1)  # (N, M)
  k_flat = keys.reshape(n * num_keys, key_dim)
  v_flat = vals.reshape(n * num_keys, value_dim)
  blk = max(1, min(n, 64))
  for i0 in range(0, n, blk):
    i1 = min(n, i0 + blk)
    b = i1 - i0
    ks = keys[i0:i1].reshape(b * num_keys, key_dim)
    kd = (ks @ k_flat.T).reshape(b, num_keys, n, num_keys).swapaxes(1, 2)
    logits = (scale_factor * kd).reshape(b, n, num_keys * num_keys)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    w = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + eps)
    w = w.reshape(b, n, num_keys, num_keys) + eps
    w = w / (np.sum(w, axis=(-2, -1), keepdims=True) + eps)
    vs = vals[i0:i1].reshape(b * num_keys, value_dim)
    vd = (vs @ v_flat.T).reshape(b, num_keys, n, num_keys).swapaxes(1, 2)
    num = np.sum(w * vd, axis=(-2, -1))
    w_t = np.sum(w, axis=-1)
    w_e = np.sum(w, axis=-2)
    nt = np.sqrt(np.einsum("tep,tp->te", w_t, v_sq[i0:i1]) + eps)
    ne = np.sqrt(np.einsum("teq,eq->te", w_e, v_sq) + eps)
    pfas_mat[i0:i1] = num / np.maximum(nt * ne, eps)

  pfas_mat = 0.5 * (pfas_mat + pfas_mat.T)
  return (1.0 - float(hybrid_alpha)) * cos_mat + float(hybrid_alpha) * pfas_mat


if nn is not None:

  def _subsample_mask_pt(
      mask: Optional[torch.Tensor], factor: int
  ) -> Optional[torch.Tensor]:
    if mask is None or factor <= 1:
      return mask
    seq_len = mask.shape[1]
    pad = (-seq_len) % factor
    if pad > 0:
      mask = F.pad(mask, (0, pad), value=0.0)
    b = mask.shape[0]
    return mask.view(b, mask.shape[1] // factor, factor).max(dim=-1).values

  def _apply_simam_2d_pt(
      h: torch.Tensor,
      mask_f: Optional[torch.Tensor] = None,
      eps: float = 1e-4,
  ) -> torch.Tensor:
    # h: (B, C, T, F), mask_f: (B, 1, T, 1)
    if mask_f is not None:
      freq = float(h.shape[3])
      denom = (mask_f.sum(dim=(2, 3), keepdim=True) * freq).clamp(min=1.0)
      mu = (h * mask_f).sum(dim=(2, 3), keepdim=True) / denom
      diff_sq = ((h - mu) ** 2) * mask_f
      var = diff_sq.sum(dim=(2, 3), keepdim=True) / denom
    else:
      mu = h.mean(dim=(2, 3), keepdim=True)
      diff_sq = (h - mu) ** 2
      var = diff_sq.mean(dim=(2, 3), keepdim=True)
    e_inv = diff_sq / (4.0 * (var + eps)) + 0.5
    out = h * torch.sigmoid(e_inv)
    if mask_f is not None:
      out = out * mask_f
    return out

  def _apply_rope_4d_pt(x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, D_h)
    seq_len = x.shape[1]
    head_dim = x.shape[-1]
    half = head_dim // 2
    if half <= 0:
      return x
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, half, device=x.device, dtype=torch.float32)
            / float(half)
        )
    )
    pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]
    cos_a = torch.cos(angles)[None, :, None, :].to(x.dtype)
    sin_a = torch.sin(angles)[None, :, None, :].to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:2 * half]
    rot = torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1)
    if 2 * half < head_dim:
      rot = torch.cat([rot, x[..., 2 * half:]], dim=-1)
    return rot

  class _BN1dNoBias(nn.Module):
    """1D BatchNorm inference matching Flax MaskedBatchNorm1D."""

    def __init__(self, features: int, use_bias: bool = True, eps: float = 1e-5):
      super().__init__()
      self.eps = eps
      self.use_bias = use_bias
      self.register_buffer("running_mean", torch.zeros(features))
      self.register_buffer("running_var", torch.ones(features))
      self.scale = nn.Parameter(torch.ones(features))
      if use_bias:
        self.bias = nn.Parameter(torch.zeros(features))
      else:
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      # x: (B, ..., C)
      inv = torch.rsqrt(self.running_var.to(x.dtype) + self.eps)
      y = (x - self.running_mean.to(x.dtype)) * (inv * self.scale.to(x.dtype))
      if self.bias is not None:
        y = y + self.bias.to(x.dtype)
      return y

  class _SimAMSERes2Block2D(nn.Module):
    """Pure-PyTorch port of FlaxSpeaker's ``SimAMSERes2Block2D``."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        in_freq: int,
        stride: Tuple[int, int] = (1, 1),
        scale: int = 4,
        width_mult: float = 2.0,
        use_simam: bool = True,
        use_se: bool = True,
        use_fwse: bool = False,
        use_aff: bool = True,
    ):
      super().__init__()
      self.stride = stride
      self.scale = max(2, int(scale))
      self.is_strided = stride[0] > 1 or stride[1] > 1
      self.use_simam = use_simam
      self.use_se = use_se
      self.use_fwse = use_fwse
      self.use_aff = use_aff
      self.out_ch = out_ch
      grp_w = max(8, int(round(out_ch * float(width_mult) / self.scale)))
      inner_ch = self.scale * grp_w
      self.grp_w = grp_w

      self.pw1 = nn.Conv2d(in_ch, inner_ch, 1, bias=False)
      self.pw1_bn = nn.BatchNorm2d(inner_ch, eps=1e-5)

      self.res2_convs = nn.ModuleDict()
      self.res2_bns = nn.ModuleDict()
      self.aff1 = nn.ModuleDict()
      self.aff2 = nn.ModuleDict()
      if self.is_strided:
        for s_idx in range(self.scale):
          k = str(s_idx)
          self.res2_convs[k] = nn.Conv2d(
              grp_w, grp_w, 3, stride=stride, padding=1, bias=False
          )
          self.res2_bns[k] = nn.BatchNorm2d(grp_w, eps=1e-5)
      else:
        aff_b = max(8, grp_w // 2)
        for s_idx in range(1, self.scale):
          k = str(s_idx)
          self.res2_convs[k] = nn.Conv2d(
              grp_w, grp_w, 3, stride=1, padding=1, bias=False
          )
          self.res2_bns[k] = nn.BatchNorm2d(grp_w, eps=1e-5)
          if s_idx >= 2 and use_aff:
            self.aff1[k] = nn.Linear(grp_w, aff_b)
            self.aff2[k] = nn.Linear(aff_b, grp_w)

      self.pw2 = nn.Conv2d(inner_ch, out_ch, 1, bias=False)
      self.pw2_bn = nn.BatchNorm2d(out_ch, eps=1e-5)

      out_freq = (
          (in_freq + 2 - 3) // stride[1] + 1 if self.is_strided else in_freq
      )
      if use_fwse:
        fc_bot = max(16, (out_freq + out_ch) // 4)
        self.fwse_fc1 = nn.Linear(out_freq + out_ch, fc_bot)
        self.fwse_f_out = nn.Linear(fc_bot, out_freq)
        self.fwse_c_out = nn.Linear(fc_bot, out_ch)
      elif use_se:
        bot = max(16, out_ch // 4)
        self.se1 = nn.Linear(out_ch, bot)
        self.se2 = nn.Linear(bot, out_ch)

      need_ds = self.is_strided or (in_ch != out_ch)
      if need_ds:
        self.downsample = nn.Conv2d(
            in_ch, out_ch, 1, stride=stride, padding=0, bias=False
        )
        self.downsample_bn = nn.BatchNorm2d(out_ch, eps=1e-5)
      else:
        self.downsample = None
        self.downsample_bn = None

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
      # x: (B, C, T, F)
      in_mf = (
          (mask > 0).to(x.dtype)[:, None, :, None] if mask is not None else None
      )
      if in_mf is not None:
        x = x * in_mf
      out_mask = (
          _subsample_mask_pt(mask, self.stride[0])
          if (mask is not None and self.stride[0] > 1)
          else mask
      )
      mf = (
          (out_mask > 0).to(x.dtype)[:, None, :, None]
          if out_mask is not None
          else None
      )
      residual = x
      h = F.relu(self.pw1_bn(self.pw1(x)))
      if in_mf is not None:
        h = h * in_mf
      splits = torch.split(h, self.grp_w, dim=1)
      outs = []
      if self.is_strided:
        for s_idx in range(self.scale):
          k = str(s_idx)
          yk = F.relu(self.res2_bns[k](self.res2_convs[k](splits[s_idx])))
          if mf is not None:
            yk = yk * mf
          outs.append(yk)
      else:
        outs.append(splits[0])
        prev = splits[0]
        for s_idx in range(1, self.scale):
          k = str(s_idx)
          cur_in = splits[s_idx]
          if s_idx == 1:
            fused = cur_in
          elif self.use_aff:
            sum_xy = cur_in + prev
            if mf is not None:
              denom = (
                  mf.sum(dim=(2, 3)) * float(sum_xy.shape[3])
              ).clamp(min=1.0)
              g_ctx = (sum_xy * mf).sum(dim=(2, 3)) / denom
            else:
              g_ctx = sum_xy.mean(dim=(2, 3))
            gate = torch.sigmoid(
                self.aff2[k](F.relu(self.aff1[k](g_ctx)))
            )[:, :, None, None]
            fused = gate * cur_in + (1.0 - gate) * prev
          else:
            fused = cur_in + prev
          yk = F.relu(self.res2_bns[k](self.res2_convs[k](fused)))
          if mf is not None:
            yk = yk * mf
          outs.append(yk)
          prev = yk

      h = self.pw2_bn(self.pw2(torch.cat(outs, dim=1)))
      if mf is not None:
        h = h * mf
      if self.use_simam:
        h = _apply_simam_2d_pt(h, mask_f=mf)
      if self.use_fwse:
        if mf is not None:
          denom_t = mf.sum(dim=2).clamp(min=1.0)  # (B, 1, 1)
          h_fc = (h * mf).sum(dim=2) / denom_t  # (B, C, F')
        else:
          h_fc = h.mean(dim=2)  # (B, C, F')
        f_desc = h_fc.mean(dim=1)  # (B, F')
        c_desc = h_fc.mean(dim=2)  # (B, C)
        fc_hid = F.relu(self.fwse_fc1(torch.cat([f_desc, c_desc], dim=-1)))
        gate_f = torch.sigmoid(self.fwse_f_out(fc_hid))[:, None, None, :]
        gate_c = torch.sigmoid(self.fwse_c_out(fc_hid))[:, :, None, None]
        h = h * gate_f * gate_c
      elif self.use_se:
        if mf is not None:
          denom = (mf.sum(dim=(2, 3)) * float(h.shape[3])).clamp(min=1.0)
          se = (h * mf).sum(dim=(2, 3)) / denom
        else:
          se = h.mean(dim=(2, 3))
        se = torch.sigmoid(self.se2(F.relu(self.se1(se))))[:, :, None, None]
        h = h * se

      if self.downsample is not None:
        residual = self.downsample_bn(self.downsample(residual))
      out = F.relu(residual + h)
      if mf is not None:
        out = out * mf
      return out

  class _RoPEConformerBlock(nn.Module):
    """Pure-PyTorch port of ``RoPEBatchNormConformerBlock``."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        conv_kernel_size: int = 15,
    ):
      super().__init__()
      self.dim = dim
      self.num_heads = num_heads
      self.head_dim = max(2, dim // num_heads)
      self.inner_d = self.num_heads * self.head_dim
      ffn_dim = 2 * dim

      self.ffn1_norm = _BN1dNoBias(dim)
      self.ffn1_up = nn.Linear(dim, ffn_dim)
      self.ffn1_down = nn.Linear(ffn_dim, dim)
      self.ls_ffn1 = nn.Parameter(torch.full((dim,), 0.2))

      self.mhsa_norm = _BN1dNoBias(dim)
      self.qkv_proj = nn.Linear(dim, 3 * self.inner_d, bias=False)
      self.qk_log_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))
      self.mhsa_out = nn.Linear(self.inner_d, dim)
      self.ls_mhsa = nn.Parameter(torch.full((dim,), 0.2))

      self.conv_pre_norm = _BN1dNoBias(dim)
      self.conv_pw1 = nn.Linear(dim, 2 * dim)
      self.conv_dw = nn.Conv1d(
          dim,
          dim,
          conv_kernel_size,
          padding=conv_kernel_size // 2,
          groups=dim,
          bias=False,
      )
      self.conv_bn = _BN1dNoBias(dim)
      self.conv_pw2 = nn.Linear(dim, dim)
      self.ls_conv = nn.Parameter(torch.full((dim,), 0.2))

      self.ffn2_norm = _BN1dNoBias(dim)
      self.ffn2_up = nn.Linear(dim, ffn_dim)
      self.ffn2_down = nn.Linear(ffn_dim, dim)
      self.ls_ffn2 = nn.Parameter(torch.full((dim,), 0.2))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
      # x: (B, T, D)
      bsz, seq_len, _ = x.shape
      mf = (mask > 0).to(x.dtype)[..., None] if mask is not None else None

      h = self.ffn1_down(F.silu(self.ffn1_up(self.ffn1_norm(x))))
      x = x + (0.5 * self.ls_ffn1[None, None, :]) * h
      if mf is not None:
        x = x * mf

      qkv = self.qkv_proj(self.mhsa_norm(x))
      q, k, v = torch.chunk(qkv, 3, dim=-1)
      q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
      k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
      v = v.view(bsz, seq_len, self.num_heads, self.head_dim)
      q = q * torch.rsqrt((q * q).mean(dim=-1, keepdim=True) + 1e-6)
      k = k * torch.rsqrt((k * k).mean(dim=-1, keepdim=True) + 1e-6)
      q = _apply_rope_4d_pt(q)
      k = _apply_rope_4d_pt(k)
      temp = torch.exp(self.qk_log_scale).clamp(max=50.0).to(x.dtype)
      attn = torch.matmul(
          q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)
      ) * temp[None, :, :, :]
      if mask is not None:
        key_mask = (mask > 0)[:, None, None, :]
        attn = attn.masked_fill(~key_mask, -1e9)
      w = torch.softmax(attn, dim=-1)
      ctx = torch.matmul(w, v.permute(0, 2, 1, 3))
      ctx = ctx.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.inner_d)
      x = x + self.ls_mhsa[None, None, :] * self.mhsa_out(ctx)
      if mf is not None:
        x = x * mf

      h_a, h_b = torch.chunk(
          self.conv_pw1(self.conv_pre_norm(x)), 2, dim=-1
      )
      h_c = h_a * torch.sigmoid(h_b)
      if mf is not None:
        h_c = h_c * mf
      h_c = self.conv_dw(h_c.transpose(1, 2)).transpose(1, 2)
      h_c = self.conv_pw2(F.silu(self.conv_bn(h_c)))
      x = x + self.ls_conv[None, None, :] * h_c
      if mf is not None:
        x = x * mf

      h2 = self.ffn2_down(F.silu(self.ffn2_up(self.ffn2_norm(x))))
      x = x + (0.5 * self.ls_ffn2[None, None, :]) * h2
      if mf is not None:
        x = x * mf
      return x

  class _CATPPooling(nn.Module):
    """Pure-PyTorch port of ``ContextAttentiveTemporalPooling``."""

    def __init__(
        self,
        feat_dim: int,
        num_heads: int = 16,
        attention_dim: int = 128,
        use_global_context: bool = True,
        eps: float = 1e-4,
    ):
      super().__init__()
      self.num_heads = num_heads
      self.use_global_context = use_global_context
      self.eps = eps
      self.catp_frame = nn.Linear(feat_dim, attention_dim)
      if use_global_context:
        self.catp_ctx = nn.Linear(2 * feat_dim, attention_dim, bias=False)
      else:
        self.catp_ctx = None
      self.catp_out = nn.Linear(attention_dim, num_heads)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
      bsz, seq_len, feat_dim = x.shape
      h_f = self.catp_frame(x)
      if self.catp_ctx is not None:
        if mask is not None:
          mf = (mask > 0).to(x.dtype)[..., None]
          denom = mf.sum(dim=1).clamp(min=1.0)
          mu_g = (x * mf).sum(dim=1) / denom
          var_g = (((x - mu_g[:, None, :]) ** 2) * mf).sum(dim=1) / denom
        else:
          mu_g = x.mean(dim=1)
          var_g = x.var(dim=1, unbiased=False)
        std_g = torch.sqrt(var_g.clamp(min=self.eps))
        h_c = self.catp_ctx(torch.cat([mu_g, std_g], dim=-1))
        h = torch.tanh(h_f + h_c[:, None, :])
      else:
        h = torch.tanh(h_f)
      w = torch.sigmoid(self.catp_out(h)) + self.eps
      if mask is not None:
        w = w * (mask > 0).to(x.dtype)[..., None]
      eta = w.sum(dim=1, keepdim=True).clamp(min=self.eps)
      nw = w / eta
      hd = feat_dim // self.num_heads
      xh = x.view(bsz, seq_len, self.num_heads, hd)
      wh = nw[..., None]
      mean = (wh * xh).sum(dim=1)
      second = (wh * (xh * xh)).sum(dim=1)
      std = torch.sqrt((second - mean * mean).clamp(min=self.eps))
      return torch.cat(
          [mean.reshape(bsz, feat_dim), std.reshape(bsz, feat_dim)], dim=-1
      )

  class _PFASSNHead(nn.Module):
    """Pure-PyTorch port of ``PfasRepresentationHead`` (shared_nonlinear)."""

    def __init__(
        self,
        feat_dim: int,
        num_keys: int = 8,
        key_dim: int = 32,
        value_dim: int = 96,
        attention_dim: int = 128,
        bottleneck_dim: int = 256,
        atp_num_heads: int = 16,
        local_value_scale: float = 0.25,
        apply_l2_norm_to_keys: bool = True,
        apply_l2_norm_to_values: bool = False,
        apply_global_l2_norm_to_concat_form: bool = True,
        eps: float = 1e-6,
    ):
      super().__init__()
      self.num_keys = num_keys
      self.key_dim = key_dim
      self.value_dim = value_dim
      self.atp_num_heads = atp_num_heads
      self.local_value_scale = local_value_scale
      self.apply_l2_norm_to_keys = apply_l2_norm_to_keys
      self.apply_l2_norm_to_values = apply_l2_norm_to_values
      self.apply_global_l2_norm = apply_global_l2_norm_to_concat_form
      self.eps = eps

      self.atp_hid = nn.Linear(feat_dim, attention_dim)
      self.atp_ffn = nn.Linear(attention_dim, atp_num_heads)
      self.pfas_g_ln1 = nn.LayerNorm(2 * feat_dim, eps=1e-6)
      self.pfas_g_fc = nn.Linear(2 * feat_dim, bottleneck_dim)
      self.pfas_g_ln2 = nn.LayerNorm(bottleneck_dim, eps=1e-6)
      self.pfas_g_val = nn.Linear(bottleneck_dim, value_dim)

      self.pfas_key_proj = nn.Linear(feat_dim, num_keys * key_dim)
      self.pfas_val_proj = nn.Linear(feat_dim, num_keys * value_dim)
      self.pfas_att_hid = nn.Linear(feat_dim, attention_dim)
      self.pfas_att_out = nn.Linear(attention_dim, num_keys)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
      bsz, seq_len, feat_dim = x.shape
      atp_w = torch.sigmoid(
          self.atp_ffn(torch.tanh(self.atp_hid(x)))
      ) + 1e-4
      if mask is not None:
        atp_w = atp_w * (mask > 0).to(x.dtype)[..., None]
      atp_w = atp_w / atp_w.sum(dim=1, keepdim=True).clamp(min=1e-4)
      hd = feat_dim // self.atp_num_heads
      xh = x.view(bsz, seq_len, self.atp_num_heads, hd)
      wh = atp_w[..., None]
      mu = (wh * xh).sum(dim=1)
      sec = (wh * (xh * xh)).sum(dim=1)
      std = torch.sqrt((sec - mu * mu).clamp(min=1e-4))
      pooled = torch.cat(
          [mu.reshape(bsz, feat_dim), std.reshape(bsz, feat_dim)], dim=-1
      )
      g = F.relu(self.pfas_g_ln2(self.pfas_g_fc(self.pfas_g_ln1(pooled))))
      g_val = self.pfas_g_val(g)

      fk = self.pfas_key_proj(x).view(bsz, seq_len, self.num_keys, self.key_dim)
      fv = self.pfas_val_proj(x).view(
          bsz, seq_len, self.num_keys, self.value_dim
      )
      logits = self.pfas_att_out(torch.tanh(self.pfas_att_hid(x)))
      if mask is not None:
        logits = logits.masked_fill(~(mask > 0)[..., None], -1e9)
      alpha = torch.softmax(logits, dim=1)
      keys = torch.einsum("btm,btmd->bmd", alpha, fk)
      l_vals = torch.einsum("btm,btmd->bmd", alpha, fv)
      vals = g_val[:, None, :] + self.local_value_scale * l_vals
      if self.apply_l2_norm_to_keys:
        keys = keys / (torch.linalg.norm(keys, dim=-1, keepdim=True) + self.eps)
      if self.apply_l2_norm_to_values:
        vals = vals / (torch.linalg.norm(vals, dim=-1, keepdim=True) + self.eps)
      flat_k = keys.reshape(bsz, self.num_keys * self.key_dim)
      flat_v = vals.reshape(bsz, self.num_keys * self.value_dim)
      if self.apply_global_l2_norm and not self.apply_l2_norm_to_values:
        flat_v = flat_v / (
            torch.linalg.norm(flat_v, dim=-1, keepdim=True) + self.eps
        )
      return torch.cat([flat_k, flat_v], dim=-1)

  class FlaxSpeakerERes2Conformer(nn.Module):
    """Pure-PyTorch SimAM-ERes2Conformer encoder loading FlaxSpeaker weights."""

    is_flaxspeaker_encoder: bool = True

    def __init__(self, cfg: Dict[str, Any], atp_num_heads: int = 16):
      super().__init__()
      exp_d = cfg.get("experiment_dict", {})
      m_cfg = exp_d.get("model", {})
      e_cfg = m_cfg.get("eres2conformer", {})
      p_cfg = m_cfg.get("pooling", {})
      f_cfg = exp_d.get("frontend", {})

      self.n_mels = int(f_cfg.get("n_mels", cfg.get("n_mels", 80)))
      self.sample_rate = int(
          f_cfg.get("sample_rate", cfg.get("sample_rate", 16000))
      )
      self.n_fft = int(f_cfg.get("n_fft", 512))
      self.fmin = float(f_cfg.get("fmin", 20.0))
      self.fmax = float(f_cfg.get("fmax", 7600.0))
      self.preemph = float(f_cfg.get("preemph", 0.97))
      self.log_offset = float(f_cfg.get("log_offset", 1e-6))

      self.base_channels = int(e_cfg.get("base_channels", 64))
      self.num_blocks = tuple(
          int(v) for v in e_cfg.get("num_blocks", (3, 4, 6, 3))
      )
      self.scale = int(e_cfg.get("scale", 4))
      self.width_mult = float(e_cfg.get("width_mult", 2.0))
      self.use_simam = bool(e_cfg.get("use_simam", True))
      self.use_se = bool(e_cfg.get("use_se", True))
      self.use_fwse = bool(e_cfg.get("use_fwse", False))
      self.use_aff = bool(e_cfg.get("use_aff", True))
      self.use_mfa_stage23 = bool(e_cfg.get("use_mfa_stage23", True))
      self.use_mfa_stage123 = bool(e_cfg.get("use_mfa_stage123", False))
      self.conformer_dim = int(e_cfg.get("conformer_dim", 512))
      self.num_conformer_layers = int(e_cfg.get("num_conformer_layers", 2))
      self.num_heads = int(e_cfg.get("num_heads", 8))
      self.conv_kernel_size = int(e_cfg.get("conv_kernel_size", 15))
      self.temporal_strides = tuple(
          int(v) for v in e_cfg.get("temporal_strides", (1, 1, 2, 2))
      )
      self.embedding_dim = int(m_cfg.get("embedding_dim", 256))
      self.pooling_type = str(
          p_cfg.get("pooling_type", cfg.get("pooling_type", "catp"))
      ).lower()

      self.stem_conv = nn.Conv2d(
          1, self.base_channels, 3, padding=1, bias=False
      )
      self.stem_bn = nn.BatchNorm2d(self.base_channels, eps=1e-5)

      self.stages = nn.ModuleList()
      cur_ch = self.base_channels
      cur_freq = self.n_mels
      stage_chs: List[int] = []
      for s_idx, n_blk in enumerate(self.num_blocks):
        ch = self.base_channels * (2 ** s_idx)
        s_t = max(1, self.temporal_strides[s_idx])
        s_f = 2 if s_idx > 0 else 1
        blks = nn.ModuleList()
        for b_idx in range(n_blk):
          st = (s_t, s_f) if b_idx == 0 else (1, 1)
          blks.append(
              _SimAMSERes2Block2D(
                  in_ch=cur_ch,
                  out_ch=ch,
                  in_freq=cur_freq,
                  stride=st,
                  scale=self.scale,
                  width_mult=self.width_mult,
                  use_simam=self.use_simam,
                  use_se=self.use_se,
                  use_fwse=self.use_fwse,
                  use_aff=self.use_aff,
              )
          )
          cur_ch = ch
          if b_idx == 0 and (st[0] > 1 or st[1] > 1):
            cur_freq = (cur_freq + 2 - 3) // st[1] + 1
        self.stages.append(blks)
        stage_chs.append(ch)

      n_st = len(self.num_blocks)
      if self.use_mfa_stage123 and n_st >= 3:
        s_t2 = max(1, self.temporal_strides[n_st - 2])
        s_f2 = 2
        self.mfa_stage12_align = nn.Conv2d(
            stage_chs[-3],
            stage_chs[-3],
            3,
            stride=(s_t2, s_f2),
            padding=1,
            bias=False,
        )
        self.mfa_align12_bn = nn.BatchNorm2d(stage_chs[-3], eps=1e-5)
        prev_ch = stage_chs[-3] + stage_chs[-2]
      else:
        self.mfa_stage12_align = None
        self.mfa_align12_bn = None
        prev_ch = stage_chs[-2]

      if self.use_mfa_stage123 or self.use_mfa_stage23:
        last_st = max(1, self.temporal_strides[n_st - 1])
        self.mfa_stage_align = nn.Conv2d(
            prev_ch,
            stage_chs[-2],
            3,
            stride=(last_st, 2),
            padding=1,
            bias=False,
        )
        self.mfa_align_bn = nn.BatchNorm2d(stage_chs[-2], eps=1e-5)
        self.mfa_2d_fuse = nn.Conv2d(
            stage_chs[-2] + stage_chs[-1], stage_chs[-1], 1, bias=False
        )
        self.mfa_fuse_bn = nn.BatchNorm2d(stage_chs[-1], eps=1e-5)
      else:
        self.mfa_stage_align = None

      local_dim = cur_freq * stage_chs[-1]
      if self.num_conformer_layers > 0 and self.conformer_dim > 0:
        self.conf_in_proj = nn.Linear(local_dim, self.conformer_dim, bias=False)
        self.conf_in_bn = _BN1dNoBias(self.conformer_dim)
        self.conformer_blocks = nn.ModuleList([
            _RoPEConformerBlock(
                dim=self.conformer_dim,
                num_heads=self.num_heads,
                conv_kernel_size=self.conv_kernel_size,
            )
            for _ in range(self.num_conformer_layers)
        ])
        feat_dim = local_dim + self.conformer_dim
      else:
        self.conf_in_proj = None
        self.conformer_blocks = nn.ModuleList()
        feat_dim = local_dim

      self.pool = _CATPPooling(
          feat_dim=feat_dim,
          num_heads=int(p_cfg.get("num_heads", 16)),
          attention_dim=int(p_cfg.get("attention_dim", 128)),
          use_global_context=bool(p_cfg.get("use_global_context", True)),
          eps=float(p_cfg.get("epsilon", 1e-4)),
      )
      self.neck_bn1 = _BN1dNoBias(2 * feat_dim, use_bias=True)
      self.embedding_proj = nn.Linear(
          2 * feat_dim, self.embedding_dim, bias=False
      )
      self.neck_bn2 = _BN1dNoBias(self.embedding_dim, use_bias=False)

      self.pfas_cfg = m_cfg.get("pfas", {})
      self.hybrid_alpha = float(self.pfas_cfg.get("hybrid_alpha", 0.25))
      self.pfas_scale = float(self.pfas_cfg.get("scale_factor", 16.0))
      if self.pooling_type == "hybrid_pfas":
        self.pfas_head = _PFASSNHead(
            feat_dim=feat_dim,
            num_keys=int(self.pfas_cfg.get("num_keys", 8)),
            key_dim=int(self.pfas_cfg.get("key_dim", 32)),
            value_dim=int(self.pfas_cfg.get("value_dim", 96)),
            attention_dim=int(self.pfas_cfg.get("attention_dim", 128)),
            bottleneck_dim=int(p_cfg.get("bottleneck_dim", 256)),
            atp_num_heads=atp_num_heads,
            local_value_scale=float(
                self.pfas_cfg.get("local_value_scale", 0.25)
            ),
            apply_l2_norm_to_keys=bool(
                self.pfas_cfg.get("apply_l2_norm_to_keys", True)
            ),
            apply_l2_norm_to_values=bool(
                self.pfas_cfg.get("apply_l2_norm_to_values", False)
            ),
            apply_global_l2_norm_to_concat_form=bool(
                self.pfas_cfg.get("apply_global_l2_norm_to_concat_form", True)
            ),
        )
      else:
        self.pfas_head = None

    def forward(
        self,
        feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hybrid_pfas: bool = False,
    ) -> torch.Tensor:
      # feats: (B, T, n_mels)
      cur_mask = mask
      if cur_mask is not None:
        feats = feats * (cur_mask > 0).to(feats.dtype)[..., None]
      h = feats.unsqueeze(1)  # (B, 1, T, F)
      h = F.relu(self.stem_bn(self.stem_conv(h)))
      if cur_mask is not None:
        h = h * (cur_mask > 0).to(h.dtype)[:, None, :, None]

      stage_outs: List[torch.Tensor] = []
      stage_masks: List[Optional[torch.Tensor]] = []
      for s_idx, blks in enumerate(self.stages):
        for b_idx, blk in enumerate(blks):
          h = blk(h, mask=cur_mask)
          if b_idx == 0 and blk.stride[0] > 1 and cur_mask is not None:
            cur_mask = _subsample_mask_pt(cur_mask, blk.stride[0])
        stage_outs.append(h)
        stage_masks.append(cur_mask)

      if self.mfa_stage12_align is not None:
        s1_to_s2 = F.relu(
            self.mfa_align12_bn(self.mfa_stage12_align(stage_outs[-3]))
        )
        sm = stage_masks[-2]
        if sm is not None:
          s1_to_s2 = s1_to_s2 * (sm > 0).to(h.dtype)[:, None, :, None]
        prev_h = torch.cat([s1_to_s2, stage_outs[-2]], dim=1)
      elif self.mfa_stage_align is not None:
        prev_h = stage_outs[-2]
      else:
        prev_h = None

      if prev_h is not None and self.mfa_stage_align is not None:
        prev_al = F.relu(self.mfa_align_bn(self.mfa_stage_align(prev_h)))
        h = F.relu(
            self.mfa_fuse_bn(
                self.mfa_2d_fuse(torch.cat([prev_al, h], dim=1))
            )
        )
      if cur_mask is not None:
        h = h * (cur_mask > 0).to(h.dtype)[:, None, :, None]

      # Flatten (B, C, T, F) -> (B, T, F*C) matching Flax (B, T, F, C) reshape
      bsz, ch, seq_len, freq_b = h.shape
      z_local = h.permute(0, 2, 3, 1).reshape(bsz, seq_len, freq_b * ch)

      if self.conf_in_proj is not None:
        h_c = F.silu(self.conf_in_bn(self.conf_in_proj(z_local)))
        if cur_mask is not None:
          h_c = h_c * (cur_mask > 0).to(h_c.dtype)[..., None]
        for c_blk in self.conformer_blocks:
          h_c = c_blk(h_c, mask=cur_mask)
        frames = torch.cat([z_local, h_c], dim=-1)
        if cur_mask is not None:
          frames = frames * (cur_mask > 0).to(frames.dtype)[..., None]
      else:
        frames = z_local

      pooled = self.pool(frames, mask=cur_mask)
      emb = self.neck_bn2(self.embedding_proj(self.neck_bn1(pooled)))
      emb = emb / (torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-6)
      if return_hybrid_pfas and self.pfas_head is not None:
        pfas_out = self.pfas_head(frames, mask=cur_mask)
        return torch.cat([emb, pfas_out], dim=-1)
      return emb

    def extract_log_mel(
        self, waveform: np.ndarray, apply_cmvn: bool = True
    ) -> np.ndarray:
      return compute_flaxspeaker_log_mel(
          waveform,
          sample_rate=self.sample_rate,
          n_mels=self.n_mels,
          n_fft=self.n_fft,
          fmin=self.fmin,
          fmax=self.fmax,
          preemph=self.preemph,
          log_offset=self.log_offset,
          apply_agc=True,
          apply_cmvn=apply_cmvn,
      )

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_dir: str = "wq2012/univ_full_eres2conformer_xl",
        device: str = "cpu",
    ) -> "FlaxSpeakerERes2Conformer":
      """Loads FlaxSpeaker ``model.safetensors`` + ``config.json`` weights."""
      def _read_safetensors_np(path: str) -> Dict[str, np.ndarray]:
        try:
          from safetensors.numpy import load_file as _st_load

          return _st_load(path)
        except Exception:
          import struct

          dt_map = {
              "F32": np.float32,
              "F16": np.float16,
              "F64": np.float64,
              "I32": np.int32,
              "I64": np.int64,
              "U8": np.uint8,
              "BOOL": np.bool_,
          }
          with open(path, "rb") as sfb:
            hdr_len = struct.unpack("<Q", sfb.read(8))[0]
            hdr = json.loads(sfb.read(hdr_len).decode("utf-8"))
            blob = memoryview(sfb.read())
          out: Dict[str, np.ndarray] = {}
          for tk, meta in hdr.items():
            if tk == "__metadata__":
              continue
            s_off, e_off = meta["data_offsets"]
            dt_name = meta["dtype"]
            shape = tuple(meta["shape"])
            if dt_name == "BF16":
              u16 = np.frombuffer(blob[s_off:e_off], dtype=np.uint16)
              u32 = u16.astype(np.uint32) << 16
              out[tk] = u32.view(np.float32).reshape(shape).copy()
            else:
              out[tk] = (
                  np.frombuffer(blob[s_off:e_off], dtype=dt_map[dt_name])
                  .reshape(shape)
                  .copy()
              )
          return out

      if os.path.isdir(repo_id_or_dir):
        cfg_path = os.path.join(repo_id_or_dir, "config.json")
        st_path = os.path.join(repo_id_or_dir, "model.safetensors")
      else:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(
            repo_id=repo_id_or_dir, filename="config.json"
        )
        st_path = hf_hub_download(
            repo_id=repo_id_or_dir, filename="model.safetensors"
        )

      with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
      raw: Dict[str, np.ndarray] = _read_safetensors_np(st_path)
      return cls.from_config_and_tensors(cfg, raw, device=device)

    @classmethod
    def from_config_and_tensors(
        cls,
        cfg: Dict[str, Any],
        raw: Dict[str, np.ndarray],
        device: str = "cpu",
    ) -> "FlaxSpeakerERes2Conformer":
      """Constructs a model directly from in-memory config and tensor dict."""
      atp_heads = 16
      atp_b_key = "pfas_head.pfas_global_atp.atp_ffn.bias"
      if atp_b_key in raw:
        atp_heads = int(raw[atp_b_key].shape[0])

      model = cls(cfg, atp_num_heads=atp_heads)
      loaded_keys = set()

      def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

      def _load_conv2d(mod: nn.Conv2d, flax_prefix: str) -> None:
        k = f"{flax_prefix}.kernel"
        loaded_keys.add(k)
        # Flax (Kh, Kw, Cin, Cout) -> PyTorch (Cout, Cin, Kh, Kw)
        w = np.transpose(raw[k], (3, 2, 0, 1))
        with torch.no_grad():
          mod.weight.copy_(_t(w))

      def _load_conv1d_dw(mod: nn.Conv1d, flax_prefix: str) -> None:
        k = f"{flax_prefix}.kernel"
        loaded_keys.add(k)
        # Flax depthwise (K, 1, C) -> PyTorch (C, 1, K)
        w = np.transpose(raw[k], (2, 1, 0))
        with torch.no_grad():
          mod.weight.copy_(_t(w))

      def _load_linear(mod: nn.Linear, flax_prefix: str) -> None:
        kk = f"{flax_prefix}.kernel"
        loaded_keys.add(kk)
        with torch.no_grad():
          mod.weight.copy_(_t(raw[kk].T))
          bk = f"{flax_prefix}.bias"
          if mod.bias is not None and bk in raw:
            loaded_keys.add(bk)
            mod.bias.copy_(_t(raw[bk]))

      def _load_bn2d(mod: nn.BatchNorm2d, flax_prefix: str) -> None:
        sk = f"{flax_prefix}.scale"
        bk = f"{flax_prefix}.bias"
        mk = f"_batch_stats.{flax_prefix}.running_mean"
        vk = f"_batch_stats.{flax_prefix}.running_var"
        loaded_keys.update((sk, bk, mk, vk))
        with torch.no_grad():
          mod.weight.copy_(_t(raw[sk]))
          mod.bias.copy_(_t(raw[bk]))
          mod.running_mean.copy_(_t(raw[mk]))
          mod.running_var.copy_(_t(raw[vk]))

      def _load_bn1d(mod: _BN1dNoBias, flax_prefix: str) -> None:
        sk = f"{flax_prefix}.scale"
        mk = f"_batch_stats.{flax_prefix}.running_mean"
        vk = f"_batch_stats.{flax_prefix}.running_var"
        loaded_keys.update((sk, mk, vk))
        with torch.no_grad():
          mod.scale.copy_(_t(raw[sk]))
          mod.running_mean.copy_(_t(raw[mk]))
          mod.running_var.copy_(_t(raw[vk]))
          if mod.bias is not None:
            bk = f"{flax_prefix}.bias"
            loaded_keys.add(bk)
            mod.bias.copy_(_t(raw[bk]))

      def _load_ln(mod: nn.LayerNorm, flax_prefix: str) -> None:
        sk = f"{flax_prefix}.scale"
        bk = f"{flax_prefix}.bias"
        loaded_keys.update((sk, bk))
        with torch.no_grad():
          mod.weight.copy_(_t(raw[sk]))
          mod.bias.copy_(_t(raw[bk]))

      _load_conv2d(model.stem_conv, "backbone.stem_conv")
      _load_bn2d(model.stem_bn, "backbone.stem_bn")
      for s_idx, blks in enumerate(model.stages):
        for b_idx, blk in enumerate(blks):
          bp = f"backbone.stage{s_idx}_block{b_idx}"
          _load_conv2d(blk.pw1, f"{bp}.pw1")
          _load_bn2d(blk.pw1_bn, f"{bp}.pw1_bn")
          for k_str, conv_m in blk.res2_convs.items():
            _load_conv2d(conv_m, f"{bp}.res2_conv_{k_str}")
            _load_bn2d(blk.res2_bns[k_str], f"{bp}.res2_bn_{k_str}")
          for k_str, l_m in blk.aff1.items():
            _load_linear(l_m, f"{bp}.aff1_{k_str}")
            _load_linear(blk.aff2[k_str], f"{bp}.aff2_{k_str}")
          _load_conv2d(blk.pw2, f"{bp}.pw2")
          _load_bn2d(blk.pw2_bn, f"{bp}.pw2_bn")
          if blk.use_fwse:
            _load_linear(blk.fwse_fc1, f"{bp}.fwse_fc1")
            _load_linear(blk.fwse_f_out, f"{bp}.fwse_f_out")
            _load_linear(blk.fwse_c_out, f"{bp}.fwse_c_out")
          elif blk.use_se:
            _load_linear(blk.se1, f"{bp}.se1")
            _load_linear(blk.se2, f"{bp}.se2")
          if blk.downsample is not None:
            _load_conv2d(blk.downsample, f"{bp}.downsample")
            _load_bn2d(blk.downsample_bn, f"{bp}.downsample_bn")

      if model.mfa_stage12_align is not None:
        _load_conv2d(model.mfa_stage12_align, "backbone.mfa_stage12_align")
        _load_bn2d(model.mfa_align12_bn, "backbone.mfa_align12_bn")
      if model.mfa_stage_align is not None:
        _load_conv2d(model.mfa_stage_align, "backbone.mfa_stage_align")
        _load_bn2d(model.mfa_align_bn, "backbone.mfa_align_bn")
        _load_conv2d(model.mfa_2d_fuse, "backbone.mfa_2d_fuse")
        _load_bn2d(model.mfa_fuse_bn, "backbone.mfa_fuse_bn")

      if model.conf_in_proj is not None:
        _load_linear(model.conf_in_proj, "backbone.conf_in_proj")
        _load_bn1d(model.conf_in_bn, "backbone.conf_in_bn")
        for c_idx, c_blk in enumerate(model.conformer_blocks):
          cp = f"backbone.rope_conformer_{c_idx}"
          _load_bn1d(c_blk.ffn1_norm, f"{cp}.ffn1_norm")
          _load_linear(c_blk.ffn1_up, f"{cp}.ffn1_up")
          _load_linear(c_blk.ffn1_down, f"{cp}.ffn1_down")
          _load_bn1d(c_blk.mhsa_norm, f"{cp}.mhsa_norm")
          _load_linear(c_blk.qkv_proj, f"{cp}.qkv_proj")
          _load_linear(c_blk.mhsa_out, f"{cp}.mhsa_out")
          _load_bn1d(c_blk.conv_pre_norm, f"{cp}.conv_pre_norm")
          _load_linear(c_blk.conv_pw1, f"{cp}.conv_pw1")
          _load_conv1d_dw(c_blk.conv_dw, f"{cp}.conv_dw")
          _load_bn1d(c_blk.conv_bn, f"{cp}.conv_bn")
          _load_linear(c_blk.conv_pw2, f"{cp}.conv_pw2")
          _load_bn1d(c_blk.ffn2_norm, f"{cp}.ffn2_norm")
          _load_linear(c_blk.ffn2_up, f"{cp}.ffn2_up")
          _load_linear(c_blk.ffn2_down, f"{cp}.ffn2_down")
          with torch.no_grad():
            for pname in (
                "ls_ffn1",
                "ls_mhsa",
                "ls_conv",
                "ls_ffn2",
                "qk_log_scale",
            ):
              full_k = f"{cp}.{pname}"
              loaded_keys.add(full_k)
              getattr(c_blk, pname).copy_(_t(raw[full_k]))

      _load_linear(model.pool.catp_frame, "pool.catp_frame")
      if model.pool.catp_ctx is not None:
        _load_linear(model.pool.catp_ctx, "pool.catp_ctx")
      _load_linear(model.pool.catp_out, "pool.catp_out")
      _load_bn1d(model.neck_bn1, "neck_bn1")
      _load_linear(model.embedding_proj, "embedding_proj")
      _load_bn1d(model.neck_bn2, "neck_bn2")

      if model.pfas_head is not None:
        ph = model.pfas_head
        _load_linear(ph.atp_hid, "pfas_head.pfas_global_atp.atp_hid")
        _load_linear(ph.atp_ffn, "pfas_head.pfas_global_atp.atp_ffn")
        _load_ln(ph.pfas_g_ln1, "pfas_head.pfas_g_ln1")
        _load_linear(ph.pfas_g_fc, "pfas_head.pfas_g_fc")
        _load_ln(ph.pfas_g_ln2, "pfas_head.pfas_g_ln2")
        _load_linear(ph.pfas_g_val, "pfas_head.pfas_g_val")
        _load_linear(ph.pfas_key_proj, "pfas_head.pfas_key_proj")
        _load_linear(ph.pfas_val_proj, "pfas_head.pfas_val_proj")
        _load_linear(ph.pfas_att_hid, "pfas_head.pfas_att_hid")
        _load_linear(ph.pfas_att_out, "pfas_head.pfas_att_out")
        if "_aux_hybrid_logit" in raw:
          lv = float(np.asarray(raw["_aux_hybrid_logit"]).reshape(-1)[0])
          model.hybrid_alpha = float(1.0 / (1.0 + math.exp(-lv)))
          loaded_keys.add("_aux_hybrid_logit")
        if "_aux_pfas_log_scale" in raw:
          ls = float(np.asarray(raw["_aux_pfas_log_scale"]).reshape(-1)[0])
          model.pfas_scale = float(math.exp(ls))
          loaded_keys.add("_aux_pfas_log_scale")

      extra = sorted(
          k for k in raw.keys()
          if k not in loaded_keys and not k.startswith("_aux_")
      )
      if extra:
        raise ValueError(
            f"Unloaded FlaxSpeaker tensors ({len(extra)}): {extra[:10]}"
        )
      model.eval()
      model.to(device)
      return model
