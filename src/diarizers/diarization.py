"""Open-source Apache-2.0 Speaker Diarization backends."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import wave

import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter

try:
  import soundfile as sf
except ImportError:  # pragma: no cover
  sf = None  # type: ignore

try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
except ImportError:  # pragma: no cover
  torch = None  # type: ignore

  class _DummyModule:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      pass

    def __call__(self, x: Any) -> Any:
      return x

  class _DummyNN:
    Module = _DummyModule
    ModuleList = list
    Conv1d = _DummyModule
    BatchNorm1d = _DummyModule

  nn = _DummyNN()  # type: ignore
  F = None  # type: ignore

from diarizers import configs
from diarizers import flaxspeaker_encoder
from diarizers import spectral
from diarizers import types


class Conv1dSame(nn.Module):
  """1D convolution with symmetric reflection padding."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      dilation: int = 1,
      groups: int = 1,
  ):
    super().__init__()
    self.conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        dilation=dilation,
        groups=groups,
    )
    self.pad = (kernel_size - 1) * dilation

  def forward(self, x: Any) -> Any:
    if self.pad > 0:
      left = self.pad // 2
      right = self.pad - left
      x = F.pad(x, (left, right), mode="reflect")
    return self.conv(x)


class TDNNBlock(nn.Module):
  """Time-Delay Neural Network block with Conv1d, ReLU, and BatchNorm1d."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      dilation: int = 1,
      groups: int = 1,
  ):
    super().__init__()
    self.conv = Conv1dSame(
        in_channels, out_channels, kernel_size, dilation, groups
    )
    self.norm = nn.BatchNorm1d(out_channels, eps=1e-5)

  def forward(self, x: Any) -> Any:
    return self.norm(F.relu(self.conv(x)))


class Res2NetBlock(nn.Module):
  """Multi-scale Res2Net block with hierarchical residual connections."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      scale: int = 8,
      kernel_size: int = 3,
      dilation: int = 1,
  ):
    super().__init__()
    in_ch = in_channels // scale
    out_ch = out_channels // scale
    self.blocks = nn.ModuleList([
        TDNNBlock(in_ch, out_ch, kernel_size, dilation=dilation)
        for _ in range(scale - 1)
    ])
    self.scale = scale

  def forward(self, x: Any) -> Any:
    y: List[Any] = []
    y_i = x
    for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
      if i == 0:
        y_i = x_i
      elif i == 1:
        y_i = self.blocks[i - 1](x_i)
      else:
        y_i = self.blocks[i - 1](x_i + y_i)
      y.append(y_i)
    return torch.cat(y, dim=1)


class SEBlock(nn.Module):
  """Squeeze-and-Excitation channel attention block."""

  def __init__(self, in_channels: int, se_channels: int, out_channels: int):
    super().__init__()
    self.conv1 = Conv1dSame(in_channels, se_channels, kernel_size=1)
    self.conv2 = Conv1dSame(se_channels, out_channels, kernel_size=1)

  def forward(self, x: Any) -> Any:
    s = x.mean(dim=2, keepdim=True)
    s = F.relu(self.conv1(s))
    s = torch.sigmoid(self.conv2(s))
    return s * x


class SERes2NetBlock(nn.Module):
  """Squeeze-and-Excitation Res2Net block for ECAPA-TDNN."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      res2net_scale: int = 8,
      se_channels: int = 128,
      kernel_size: int = 1,
      dilation: int = 1,
  ):
    super().__init__()
    self.tdnn1 = TDNNBlock(
        in_channels, out_channels, kernel_size=1, dilation=1
    )
    self.res2net_block = Res2NetBlock(
        out_channels, out_channels, res2net_scale, kernel_size, dilation
    )
    self.tdnn2 = TDNNBlock(
        out_channels, out_channels, kernel_size=1, dilation=1
    )
    self.se_block = SEBlock(out_channels, se_channels, out_channels)

  def forward(self, x: Any) -> Any:
    residual = x
    out = self.tdnn1(x)
    out = self.res2net_block(out)
    out = self.tdnn2(out)
    out = self.se_block(out)
    return out + residual


class AttentiveStatisticsPooling(nn.Module):
  """Channel- and context-dependent attentive statistics pooling."""

  def __init__(self, channels: int, attention_channels: int = 128):
    super().__init__()
    self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
    self.conv = Conv1dSame(attention_channels, channels, kernel_size=1)

  def forward(self, x: Any) -> Any:
    mean = x.mean(dim=2, keepdim=True).expand_as(x)
    std = x.std(dim=2, keepdim=True).clamp(min=1e-12).expand_as(x)
    attn = torch.cat([x, mean, std], dim=1)
    attn = F.softmax(self.conv(torch.tanh(self.tdnn(attn))), dim=2)
    w_mean = (attn * x).sum(dim=2)
    diff_sq = (x - w_mean.unsqueeze(2)).pow(2)
    w_std = ((attn * diff_sq).sum(dim=2).clamp(min=1e-12)).sqrt()
    return torch.cat([w_mean, w_std], dim=1).unsqueeze(2)


class SpeechBrainECAPATDNN(nn.Module):
  """Standalone PyTorch ECAPA-TDNN compatible with spkrec-ecapa-voxceleb."""

  def __init__(self, in_mels: int = 80, emb_dim: int = 192):
    super().__init__()
    channels = [1024, 1024, 1024, 1024, 3072]
    self.blocks = nn.ModuleList([
        TDNNBlock(in_mels, channels[0], kernel_size=5, dilation=1),
        SERes2NetBlock(
            channels[0], channels[1], 8, 128, kernel_size=3, dilation=2
        ),
        SERes2NetBlock(
            channels[1], channels[2], 8, 128, kernel_size=3, dilation=3
        ),
        SERes2NetBlock(
            channels[2], channels[3], 8, 128, kernel_size=3, dilation=4
        ),
    ])
    self.mfa = TDNNBlock(
        sum(channels[1:4]), channels[4], kernel_size=1, dilation=1
    )
    self.asp = AttentiveStatisticsPooling(channels[4], 128)
    self.asp_bn = nn.BatchNorm1d(channels[4] * 2, eps=1e-5)
    self.fc = Conv1dSame(channels[4] * 2, emb_dim, kernel_size=1)
    if torch is not None:
      self.register_buffer(
          "glob_mean", torch.zeros(emb_dim), persistent=False
      )
    else:
      self.glob_mean = None

  def forward(self, x: Any) -> Any:
    """Computes (B, emb_dim) embeddings from (B, T, in_mels) features."""
    x = x.transpose(1, 2)
    xl: List[Any] = []
    for i, layer in enumerate(self.blocks):
      x = layer(x)
      if i > 0:
        xl.append(x)
    x = self.mfa(torch.cat(xl, dim=1))
    x = self.asp_bn(self.asp(x))
    x = self.fc(x).squeeze(-1)
    if self.glob_mean is not None:
      x = x - self.glob_mean.unsqueeze(0)
    return x

  @classmethod
  def from_pretrained(
      cls,
      repo_or_path: str = "speechbrain/spkrec-ecapa-voxceleb",
      device: str = "cpu",
  ) -> "SpeechBrainECAPATDNN":
    """Loads pretrained ECAPA-TDNN checkpoint from local path or HF Hub."""
    model = cls()
    mvn_path: Optional[str] = None
    model_root = os.environ.get(
        "DIARIZERS_MODEL_DIR", "/tmp/diarizers_models"
    )
    if repo_or_path == "speechbrain/spkrec-ecapa-voxceleb":
      cand_dir = os.path.join(model_root, "spkrec_ecapa_voxceleb")
      if os.path.isdir(cand_dir) and os.path.isfile(
          os.path.join(cand_dir, "embedding_model.ckpt")
      ):
        repo_or_path = cand_dir

    if os.path.isfile(repo_or_path):
      ckpt_path = repo_or_path
      candidate = os.path.join(
          os.path.dirname(repo_or_path), "mean_var_norm_emb.ckpt"
      )
      if os.path.isfile(candidate):
        mvn_path = candidate
    elif os.path.isdir(repo_or_path) and os.path.isfile(
        os.path.join(repo_or_path, "embedding_model.ckpt")
    ):
      ckpt_path = os.path.join(repo_or_path, "embedding_model.ckpt")
      candidate = os.path.join(repo_or_path, "mean_var_norm_emb.ckpt")
      if os.path.isfile(candidate):
        mvn_path = candidate
    else:
      from huggingface_hub import hf_hub_download

      ckpt_path = hf_hub_download(
          repo_id=repo_or_path, filename="embedding_model.ckpt"
      )
      try:
        mvn_path = hf_hub_download(
            repo_id=repo_or_path, filename="mean_var_norm_emb.ckpt"
        )
      except Exception:
        mvn_path = None

    raw_sd = torch.load(ckpt_path, map_location=device)
    clean_sd = {
        k.replace(".norm.norm.", ".norm.").replace("asp_bn.norm.", "asp_bn."): v
        for k, v in raw_sd.items()
    }
    model.load_state_dict(clean_sd, strict=True)
    if mvn_path is not None and os.path.isfile(mvn_path):
      try:
        mvn_sd = torch.load(mvn_path, map_location=device)
        if "glob_mean" in mvn_sd and model.glob_mean is not None:
          model.glob_mean.copy_(mvn_sd["glob_mean"].to(device))
      except Exception:
        pass
    model.to(device)
    model.eval()
    return model


class ParamSincFB(nn.Module):
  """Parameterized Sinc filterbank (cos + sin filters) for PyanNet SincNet."""

  def __init__(
      self,
      n_filters: int = 80,
      kernel_size: int = 251,
      stride: int = 10,
      sample_rate: int = 16000,
      min_low_hz: int = 50,
      min_band_hz: int = 50,
  ):
    super().__init__()
    self.n_filters = n_filters
    self.kernel_size = kernel_size
    self.stride = stride
    self.sample_rate = sample_rate
    self.min_low_hz = min_low_hz
    self.min_band_hz = min_band_hz
    half_kernel = kernel_size // 2
    if torch is not None:
      self.low_hz_ = nn.Parameter(torch.zeros(n_filters // 2, 1))
      self.band_hz_ = nn.Parameter(torch.zeros(n_filters // 2, 1))
      win = np.hamming(kernel_size)[:half_kernel].astype(np.float32)
      self.register_buffer("window_", torch.from_numpy(win))
      n_vec = (
          2.0
          * np.pi
          * torch.arange(-half_kernel, 0.0).view(1, -1)
          / float(sample_rate)
      )
      self.register_buffer("n_", n_vec)

  def filters(self) -> Any:
    """Computes (n_filters, 1, kernel_size) bandpass filters."""
    low = self.min_low_hz + torch.abs(self.low_hz_)
    high = torch.clamp(
        low + self.min_band_hz + torch.abs(self.band_hz_),
        self.min_low_hz,
        self.sample_rate / 2.0,
    )
    band = (high - low)[:, 0]
    ft_low = torch.matmul(low, self.n_)
    ft_high = torch.matmul(high, self.n_)
    bp_left_cos = (
        (torch.sin(ft_high) - torch.sin(ft_low)) / (self.n_ / 2.0)
    ) * self.window_
    bp_center_cos = 2.0 * band.view(-1, 1)
    bp_right_cos = torch.flip(bp_left_cos, dims=[1])
    cos_f = torch.cat([bp_left_cos, bp_center_cos, bp_right_cos], dim=1) / (
        2.0 * band[:, None]
    )
    bp_left_sin = (
        (torch.cos(ft_low) - torch.cos(ft_high)) / (self.n_ / 2.0)
    ) * self.window_
    bp_center_sin = torch.zeros_like(bp_center_cos)
    bp_right_sin = -torch.flip(bp_left_sin, dims=[1])
    sin_f = torch.cat([bp_left_sin, bp_center_sin, bp_right_sin], dim=1) / (
        2.0 * band[:, None]
    )
    return torch.cat([cos_f, sin_f], dim=0).view(
        self.n_filters, 1, self.kernel_size
    )


class SincConv1d(nn.Module):
  """1D Sinc convolution layer wrapping ParamSincFB."""

  def __init__(self, stride: int = 10, sample_rate: int = 16000):
    super().__init__()
    self.filterbank = ParamSincFB(
        80, 251, stride=stride, sample_rate=sample_rate
    )
    self.stride = stride

  def forward(self, x: Any) -> Any:
    return F.conv1d(x, self.filterbank.filters(), stride=self.stride)


class SincNet(nn.Module):
  """SincNet acoustic frontend for PyanNet speaker segmentation."""

  def __init__(self, stride: int = 10, sample_rate: int = 16000):
    super().__init__()
    if torch is not None:
      self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
      self.conv1d = nn.ModuleList([
          SincConv1d(stride=stride, sample_rate=sample_rate),
          nn.Conv1d(80, 60, 5, stride=1),
          nn.Conv1d(60, 60, 5, stride=1),
      ])
      self.pool1d = nn.ModuleList([nn.MaxPool1d(3, stride=3) for _ in range(3)])
      self.norm1d = nn.ModuleList([
          nn.InstanceNorm1d(80, affine=True),
          nn.InstanceNorm1d(60, affine=True),
          nn.InstanceNorm1d(60, affine=True),
      ])

  def forward(self, x: Any) -> Any:
    x = self.wav_norm1d(x)
    for i, (conv, pool, norm) in enumerate(
        zip(self.conv1d, self.pool1d, self.norm1d)
    ):
      x = conv(x)
      if i == 0:
        x = torch.abs(x)
      x = F.leaky_relu(norm(pool(x)))
    return x


class _SafeStub:
  """Pickle stub for loading PyTorch Lightning checkpoints without pyannote."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    pass

  def __setstate__(self, state: Any) -> None:
    if isinstance(state, dict):
      self.__dict__.update(state)


def powerset_probs_to_multilabel(probs: np.ndarray) -> np.ndarray:
  """Converts (T, 7) powerset probabilities to (T, 3) speaker activity."""
  mapping = np.array(
      [
          [0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [1.0, 1.0, 0.0],
          [1.0, 0.0, 1.0],
          [0.0, 1.0, 1.0],
      ],
      dtype=np.float32,
  )
  return np.asarray(probs, dtype=np.float32) @ mapping


class PyanNetSegmentation(nn.Module):
  """Standalone PyTorch PyanNet powerset segmentation (segmentation-3.0)."""

  def __init__(self, num_classes: int = 7):
    super().__init__()
    self.num_classes = num_classes
    if torch is not None:
      self.sincnet = SincNet()
      self.lstm = nn.LSTM(
          60, 128, num_layers=4, bidirectional=True, batch_first=True
      )
      self.linear = nn.ModuleList([nn.Linear(256, 128), nn.Linear(128, 128)])
      self.classifier = nn.Linear(128, num_classes)
      mapping = torch.tensor([
          [0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [1.0, 1.0, 0.0],
          [1.0, 0.0, 1.0],
          [0.0, 1.0, 1.0],
      ])
      self.register_buffer("powerset_mapping", mapping, persistent=False)

  def forward(self, waveforms: Any) -> Any:
    """Computes (B, T_frames, 7) powerset log-probabilities from (B, T_wav)."""
    if waveforms.ndim == 2:
      waveforms = waveforms.unsqueeze(1)
    x = self.sincnet(waveforms).transpose(1, 2)
    x, _ = self.lstm(x)
    for lin in self.linear:
      x = F.leaky_relu(lin(x))
    return F.log_softmax(self.classifier(x), dim=-1)

  def to_multilabel(self, log_probs: Any, soft: bool = True) -> Any:
    """Converts (B, T_frames, 7) powerset log-probs to (B, T_frames, 3)."""
    if soft:
      probs = torch.exp(log_probs)
    else:
      probs = F.one_hot(
          torch.argmax(log_probs, dim=-1), log_probs.shape[-1]
      ).float()
    return torch.matmul(probs, self.powerset_mapping)

  @classmethod
  def from_pretrained(
      cls,
      repo_or_path: str = "pyannote/segmentation-3.0",
      device: str = "cpu",
  ) -> "PyanNetSegmentation":
    """Loads pretrained PyanNet segmentation weights in pure PyTorch."""
    import pickle

    model = cls()
    model_root = os.environ.get(
        "DIARIZERS_MODEL_DIR", "/tmp/diarizers_models"
    )
    if repo_or_path in ("pyannote/segmentation-3.0", "default"):
      cand_dir = os.path.join(model_root, "pyannote_segmentation_3_0")
      if os.path.isdir(cand_dir) and (
          os.path.isfile(os.path.join(cand_dir, "model.safetensors"))
          or os.path.isfile(os.path.join(cand_dir, "pytorch_model.bin"))
      ):
        repo_or_path = cand_dir

    state_dict = None
    if os.path.isdir(repo_or_path):
      st_path = os.path.join(repo_or_path, "model.safetensors")
      bin_path = os.path.join(repo_or_path, "pytorch_model.bin")
      if os.path.isfile(st_path):
        from safetensors.torch import load_file

        state_dict = load_file(st_path, device=device)
      elif os.path.isfile(bin_path):
        repo_or_path = bin_path

    if state_dict is None and os.path.isfile(repo_or_path):
      if repo_or_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(repo_or_path, device=device)
      else:

        class _Unpickler(pickle.Unpickler):

          def find_class(self, module: str, name: str) -> Any:
            try:
              return super().find_class(module, name)
            except Exception:
              return _SafeStub

        class _PickleMod:
          Unpickler = _Unpickler
          load = pickle.load
          loads = pickle.loads

        ckpt = torch.load(
            repo_or_path,
            map_location=device,
            pickle_module=_PickleMod,
            weights_only=False,
        )
        state_dict = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )

    if state_dict is None:
      from huggingface_hub import hf_hub_download

      # Try safetensors or pytorch_model.bin from repo_or_path, with fallback to
      # un-gated MIT mirror tensorlake/segmentation-3.0 or diarizers-community.
      diar_seg_repo = (
          "diarizers-community/speaker-segmentation-fine-tuned-callhome-eng"
      )
      candidates = [
          (repo_or_path, "model.safetensors"),
          (repo_or_path, "pytorch_model.bin"),
          ("tensorlake/segmentation-3.0", "pytorch_model.bin"),
          (diar_seg_repo, "model.safetensors"),
      ]
      for cand_repo, cand_file in candidates:
        try:
          dl_path = hf_hub_download(repo_id=cand_repo, filename=cand_file)
          if cand_repo != repo_or_path:
            warnings.warn(
                f"Primary segmentation repo '{repo_or_path}' unavailable; "
                f"falling back to '{cand_repo}/{cand_file}'.",
                RuntimeWarning,
            )
          return cls.from_pretrained(dl_path, device=device)
        except Exception:
          continue
      raise RuntimeError(
          f"Could not load PyanNetSegmentation from {repo_or_path}"
      )

    clean_sd = {
        k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()
    }
    model.load_state_dict(clean_sd, strict=True)
    model.to(device)
    model.eval()
    return model


def _build_mel_filterbank_np(
    n_stft: int,
    num_mel_bins: int = 80,
    sample_rate: int = 16000,
    f_min: float = 0.0,
) -> np.ndarray:
  """Builds an (n_stft, num_mel_bins) HTK triangular Mel filterbank matrix."""
  f_max = float(sample_rate) / 2.0
  mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
  mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
  mel_pts = np.linspace(mel_min, mel_max, num_mel_bins + 2)
  hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
  freqs = np.linspace(0.0, f_max, n_stft)
  fb = np.zeros((n_stft, num_mel_bins), dtype=np.float32)
  for m in range(num_mel_bins):
    f_l, f_c, f_r = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
    up = (freqs - f_l) / max(1e-8, f_c - f_l)
    down = (f_r - freqs) / max(1e-8, f_r - f_c)
    fb[:, m] = np.maximum(0.0, np.minimum(up, down))
  return fb


def compute_log_mel_fbank(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    apply_cmvn: bool = True,
) -> Any:
  """Computes CMVN-normalized 80-dim dB log-mel filterbank features (T, 80)."""
  wav_f32 = np.asarray(waveform, dtype=np.float32)
  if wav_f32.ndim > 1:
    wav_f32 = np.mean(wav_f32, axis=1, dtype=np.float32)
  if torch is not None:
    wav_t = torch.from_numpy(wav_f32)
    win_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    n_fft = max(400, win_length)
    window = torch.hamming_window(win_length)
    stft = torch.stft(
        wav_t,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = stft.abs().pow(2).transpose(0, 1)
    mel_mat = torch.from_numpy(
        _build_mel_filterbank_np(power.shape[1], num_mel_bins, sample_rate)
    ).to(power.device)
    fbank = 10.0 * torch.log10(torch.clamp(power @ mel_mat, min=1e-10))
    fbank = torch.maximum(fbank, fbank.max() - 80.0)
    if apply_cmvn:
      return fbank - fbank.mean(dim=0, keepdim=True)
    return fbank

  # Pure-NumPy fallback when PyTorch is unavailable
  hop = max(1, int(0.010 * sample_rate))
  win = max(1, int(0.025 * sample_rate))
  n_fft = max(400, win)
  if len(wav_f32) < win:
    return np.zeros((1, num_mel_bins), dtype=np.float32)
  n_frames = 1 + (len(wav_f32) - win) // hop
  frames = np.stack(
      [wav_f32[i * hop:i * hop + win] for i in range(n_frames)], axis=0
  )
  spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
  mel_proj = _build_mel_filterbank_np(spec.shape[1], num_mel_bins, sample_rate)
  fbank_np = (
      10.0 * np.log10(np.maximum(spec @ mel_proj, 1e-10))
  ).astype(np.float32)
  fbank_np = np.maximum(fbank_np, np.max(fbank_np) - 80.0)
  if apply_cmvn:
    return fbank_np - np.mean(fbank_np, axis=0, keepdims=True)
  return fbank_np


def stitch_powerset_probs(
    chunk_starts: np.ndarray,
    ml_soft: np.ndarray,
    cs_meta: np.ndarray,
    cs_embs_all: np.ndarray,
    centroids: np.ndarray,
    duration_sec: float,
    chunk_window_weight: str = "uniform",
    use_temporal_stitch: bool = True,
    stitch_weight: float = 0.50,
    prob_smooth_sec: float = 0.25,
) -> Tuple[np.ndarray, float]:
  """Stitches overlapping 10s local curves into global `(T, K)` soft probs.

  Uses two-pass Hungarian assignment combining cosine similarity against
  ``centroids`` and overlap correlation across adjacent 10s powerset windows.
  """
  n_chunks, n_frames, _ = ml_soft.shape
  frame_dt = 10.0 / float(max(1, n_frames))
  n_k = int(centroids.shape[0])
  if n_chunks == 0 or n_k == 0:
    return np.zeros((0, max(1, n_k)), dtype=np.float32), frame_dt
  max_end = max(float(duration_sec), float(chunk_starts[-1]) + 10.0)
  total_frames = max(n_frames, int(math.ceil(max_end / frame_dt)) + 2)
  if n_k == 1:
    probs = np.zeros((total_frames, 1), dtype=np.float32)
    den = np.zeros((total_frames,), dtype=np.float32)
    for ci in range(n_chunks):
      f0 = int(round(float(chunk_starts[ci]) / frame_dt))
      f1 = min(total_frames, f0 + n_frames)
      ln = f1 - f0
      if ln > 0:
        probs[f0:f1, 0] += np.max(ml_soft[ci, :ln, :], axis=1)
        den[f0:f1] += 1.0
    return probs / np.maximum(den[:, None], 1e-6), frame_dt

  if chunk_window_weight == "hann":
    w_chunk = np.maximum(
        np.hanning(n_frames + 2)[1:-1].astype(np.float32), 0.1
    )
  elif chunk_window_weight == "tri":
    ramp = np.linspace(0.1, 1.0, max(1, n_frames // 2), dtype=np.float32)
    w_chunk = np.concatenate([ramp, ramp[::-1]])
    if len(w_chunk) < n_frames:
      w_chunk = np.pad(w_chunk, (0, n_frames - len(w_chunk)), mode="edge")
    w_chunk = w_chunk[:n_frames]
  else:
    w_chunk = np.ones((n_frames,), dtype=np.float32)

  embs_n = (
      spectral.l2_normalize_rows(cs_embs_all)
      if len(cs_embs_all) > 0
      else np.asarray(cs_embs_all, dtype=np.float32)
  )
  cents_n = spectral.l2_normalize_rows(centroids)
  chunk_to_rows: Dict[int, List[int]] = {}
  for r_idx in range(len(cs_meta)):
    ci_row = int(round(float(cs_meta[r_idx, 0])))
    chunk_to_rows.setdefault(ci_row, []).append(r_idx)

  ml_f = np.asarray(ml_soft, dtype=np.float32)
  probs_num = np.zeros((total_frames, n_k), dtype=np.float32)
  probs_den = np.zeros((total_frames,), dtype=np.float32)

  n_passes = 2 if use_temporal_stitch else 1
  for pass_idx in range(n_passes):
    cur_num = np.zeros((total_frames, n_k), dtype=np.float32)
    cur_den = np.zeros((total_frames,), dtype=np.float32)
    for ci in range(n_chunks):
      f0 = int(round(float(chunk_starts[ci]) / frame_dt))
      f1 = min(total_frames, f0 + n_frames)
      ln = f1 - f0
      if ln <= 0:
        continue
      loc_p = ml_f[ci, :ln, :]
      wt = w_chunk[:ln]
      active_locals = [
          s for s in (0, 1, 2) if float(loc_p[:, s].sum()) > 0.5
      ]
      if not active_locals:
        cur_den[f0:f1] += wt
        continue

      score = np.full((3, n_k), -1e3, dtype=np.float32)
      has_emb = [False, False, False]
      for r_idx in chunk_to_rows.get(ci, []):
        sl = int(round(float(cs_meta[r_idx, 1])))
        dur_c = float(cs_meta[r_idx, 2])
        sim_vec = embs_n[r_idx] @ cents_n.T
        score[sl, :] = sim_vec * min(1.0, max(0.3, dur_c))
        has_emb[sl] = True

      if use_temporal_stitch and pass_idx == 1:
        prev_den = np.maximum(probs_den[f0:f1], 1e-6)
        prev_p = probs_num[f0:f1] / prev_den[:, None]
        ov_corr = loc_p.T @ prev_p
        loc_mass = np.maximum(loc_p.sum(axis=0, keepdims=True).T, 1e-4)
        ov_norm = ov_corr / loc_mass
        for s in active_locals:
          if has_emb[s]:
            score[s, :] = score[s, :] + float(stitch_weight) * ov_norm[s, :]
          else:
            score[s, :] = ov_norm[s, :]
      elif pass_idx == 0:
        for s in active_locals:
          if not has_emb[s]:
            if float(cur_den[f0:f1].max()) > 0.0:
              cur_p = cur_num[f0:f1] / np.maximum(cur_den[f0:f1, None], 1e-6)
              ov_c = loc_p[:, s] @ cur_p
              score[s, :] = ov_c / max(1e-4, float(loc_p[:, s].sum()))
            else:
              score[s, :] = 0.0

      if len(active_locals) <= n_k:
        sub = score[active_locals, :]
        ri, gi = optimize.linear_sum_assignment(sub, maximize=True)
        for r_l, g_k in zip(ri, gi):
          sl = active_locals[int(r_l)]
          if (
              pass_idx == 0
              and (not has_emb[sl])
              and float(cur_den[f0:f1].max()) == 0.0
          ):
            continue
          cur_num[f0:f1, int(g_k)] += loc_p[:, sl] * wt
      else:
        for sl in active_locals:
          gk = int(np.argmax(score[sl]))
          cur_num[f0:f1, gk] += loc_p[:, sl] * wt
      cur_den[f0:f1] += wt
    probs_num, probs_den = cur_num, cur_den

  global_p = probs_num / np.maximum(probs_den[:, None], 1e-6)
  if prob_smooth_sec > 0.0:
    sig = max(0.5, float(prob_smooth_sec) / frame_dt)
    global_p = gaussian_filter(global_p, sigma=(sig, 0.0))
  return global_p.astype(np.float32), frame_dt


def probs_to_speaker_turns(
    frame_probs: np.ndarray,
    frame_dt: float,
    duration_sec: float,
    min_active_prob: float = 0.05,
) -> List[types.SpeakerTurn]:
  """Converts continuous `(T, K)` frame probs into contiguous SpeakerTurns."""
  probs = np.asarray(frame_probs, dtype=np.float32)
  if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] == 0:
    return []
  max_p = probs.max(axis=1)
  top_k = probs.argmax(axis=1)
  n_frames = min(
      probs.shape[0],
      max(1, int(math.ceil(float(duration_sec) / max(1e-6, frame_dt)))),
  )
  turns: List[types.SpeakerTurn] = []
  cur_spk: Optional[str] = None
  cur_start = 0.0
  for fi in range(n_frames):
    if float(max_p[fi]) <= min_active_prob:
      if cur_spk is not None:
        t_end = min(float(duration_sec), fi * frame_dt)
        if t_end > cur_start:
          turns.append(
              types.SpeakerTurn(start=cur_start, end=t_end, speaker=cur_spk)
          )
        cur_spk = None
      continue
    spk_str = str(int(top_k[fi]) + 1)
    if cur_spk is None:
      cur_spk = spk_str
      cur_start = fi * frame_dt
    elif spk_str != cur_spk:
      t_mid = min(float(duration_sec), fi * frame_dt)
      if t_mid > cur_start:
        turns.append(
            types.SpeakerTurn(start=cur_start, end=t_mid, speaker=cur_spk)
        )
      cur_spk = spk_str
      cur_start = t_mid
  if cur_spk is not None:
    t_end = min(float(duration_sec), n_frames * frame_dt)
    if t_end > cur_start:
      turns.append(
          types.SpeakerTurn(start=cur_start, end=t_end, speaker=cur_spk)
      )
  if not turns and duration_sec > 0.0:
    turns.append(
        types.SpeakerTurn(start=0.0, end=float(duration_sec), speaker="1")
    )
  return turns


def load_audio_16k(
    audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]],
    target_sr: int = 16000,
) -> np.ndarray:
  """Loads mono float32 waveform at 16 kHz from path or array."""
  if isinstance(audio_input, str):
    if sf is not None:
      wav, sr = sf.read(audio_input, dtype="float32")
    else:
      with wave.open(audio_input, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
      pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
      wav = (
          np.mean(pcm.reshape(-1, n_ch), axis=1, dtype=np.float32)
          if n_ch > 1
          else pcm
      )
  elif isinstance(audio_input, tuple):
    wav, sr = np.asarray(audio_input[0], dtype=np.float32), int(audio_input[1])
  else:
    wav, sr = np.asarray(audio_input, dtype=np.float32), target_sr

  if wav.ndim > 1:
    wav = np.mean(wav, axis=1, dtype=np.float32)
  if sr != target_sr and wav.size > 0:
    from scipy import signal

    gcd = int(np.gcd(sr, target_sr))
    wav = signal.resample_poly(wav, target_sr // gcd, sr // gcd).astype(
        np.float32
    )
  return wav


class ECAPASpectralDiarizer:
  """Speaker Diarizer combining PyanNet, ECAPA-TDNN, and Spectral Clustering."""

  def __init__(
      self,
      preset: configs.DiarizationPreset = configs.FISHER_TELEPHONY_PRESET,
      device: str = "cpu",
      batch_size: int = 128,
      encoder: Optional[Any] = None,
      segmentation_model: Optional[Any] = "default",
      segmentation_fn: Optional[Any] = None,
  ):
    self.preset = preset
    self.device = device
    self.batch_size = batch_size
    self.encoder = encoder
    self.segmentation_fn = segmentation_fn
    if segmentation_model == "default":
      self.segmentation_model_spec = preset.segmentation_model
      self._seg_mod: Optional[Any] = None
    elif isinstance(segmentation_model, str):
      if segmentation_model in ("", "none", "None", "energy_vad"):
        self.segmentation_model_spec = None
        self._seg_mod = None
      else:
        self.segmentation_model_spec = segmentation_model
        self._seg_mod = None
    else:
      self.segmentation_model_spec = None
      self._seg_mod = segmentation_model

    self.clusterer = spectral.SpectralClusterer(
        min_clusters=preset.min_clusters,
        max_clusters=preset.max_clusters,
        p_percentile=preset.p_percentile,
        gaussian_blur_sigma=preset.gaussian_blur_sigma,
        eigen_threshold=preset.eigen_threshold,
    )
    self.last_frame_probs: Optional[np.ndarray] = None
    self.last_frame_dt: float = 10.0 / 295.0

  @property
  def segmentation_model(self) -> Optional[Any]:
    return self._seg_mod or self.segmentation_model_spec

  @staticmethod
  def _is_flaxspeaker(encoder: Any) -> bool:
    return bool(getattr(encoder, "is_flaxspeaker_encoder", False))

  def _ensure_encoder(self) -> Any:
    if self.encoder is None:
      if flaxspeaker_encoder.is_flaxspeaker_model_id(
          self.preset.encoder_name
      ):
        self.encoder = (
            flaxspeaker_encoder.FlaxSpeakerERes2Conformer.from_pretrained(
                self.preset.encoder_name, device=self.device
            )
        )
      else:
        self.encoder = SpeechBrainECAPATDNN.from_pretrained(
            self.preset.encoder_name, device=self.device
        )
    return self.encoder

  def _ensure_segmentation_model(self) -> Optional[Any]:
    if self._seg_mod is not None:
      return self._seg_mod
    if not self.segmentation_model_spec or self.segmentation_model_spec in (
        "none",
        "None",
        "energy_vad",
    ):
      return None
    # Skip loading real segmentation weights when a mock callable is passed.
    if self.encoder is not None and (
        getattr(self.encoder, "is_mock", False)
        or (nn is not None and not isinstance(self.encoder, nn.Module))
    ):
      return None
    if torch is None:
      return None
    try:
      self._seg_mod = PyanNetSegmentation.from_pretrained(
          self.segmentation_model_spec, device=self.device
      )
    except Exception as exc:
      warnings.warn(
          f"Could not load segmentation model '{self.segmentation_model_spec}'"
          f" ({exc}); continuing without neural boundary snapping.",
          RuntimeWarning,
      )
      self._seg_mod = None
    return self._seg_mod

  def extract_segmentation_curves(
      self,
      waveform: np.ndarray,
      sample_rate: int = 16000,
  ) -> Optional[np.ndarray]:
    """Runs PyanNetSegmentation on 10s chunks for (T_seg, 3) activity curves."""
    if waveform.size == 0:
      return None
    if self.segmentation_fn is not None:
      custom_seg = self.segmentation_fn(waveform)
      if custom_seg is None:
        return None
      if isinstance(custom_seg, np.ndarray) and custom_seg.ndim == 2:
        return np.asarray(custom_seg, dtype=np.float32)
      frame_dt = 10.0 / 589.0
      dur_sec = len(waveform) / float(sample_rate)
      num_seg_frames = max(1, int(np.ceil(dur_sec / frame_dt)))
      ml = np.zeros((num_seg_frames, 3), dtype=np.float32)
      for idx_seg, seg_item in enumerate(custom_seg):
        if isinstance(seg_item, (tuple, list)):
          s_sec, e_sec = float(seg_item[0]), float(seg_item[1])
        else:
          s_sec, e_sec = float(seg_item.start), float(seg_item.end)
        f0 = max(0, min(num_seg_frames, int(round(s_sec / frame_dt))))
        f1 = max(f0, min(num_seg_frames, int(round(e_sec / frame_dt))))
        ml[f0:f1, idx_seg % 3] = 1.0
      return ml
    seg_mod = self._ensure_segmentation_model()
    if seg_mod is None or torch is None:
      return None
    chunk_samples = 10 * sample_rate
    seg_chunks: List[Any] = []
    for s in range(0, len(waveform), chunk_samples):
      c = waveform[s:s + chunk_samples]
      if len(c) < chunk_samples:
        c = np.pad(c, (0, chunk_samples - len(c)))
      seg_chunks.append(torch.from_numpy(np.asarray(c, dtype=np.float32)))
    if not seg_chunks:
      return None
    ml_list: List[np.ndarray] = []
    with torch.no_grad():
      for i in range(0, len(seg_chunks), self.batch_size):
        b = torch.stack(seg_chunks[i:i + self.batch_size], dim=0).to(
            self.device
        )
        lp = seg_mod(b)
        if hasattr(seg_mod, "to_multilabel"):
          ml = seg_mod.to_multilabel(lp, soft=True)
        else:
          ml = lp
        ml_list.append(ml.cpu().numpy())
    return np.concatenate(ml_list, axis=0).reshape(-1, 3)

  def snap_turn_boundaries(
      self,
      times: List[Tuple[float, float]],
      labels: np.ndarray,
      ml_flat: Optional[np.ndarray],
  ) -> List[float]:
    """Snaps speaker transitions to PyanNet local speaker change peaks."""
    n = len(labels)
    if n == 0:
      return []
    boundaries: List[float] = [times[0][0]]
    frame_dt = 10.0 / 589.0
    tol = getattr(self.preset, "boundary_snap_tolerance_sec", 0.35)
    max_t = times[-1][1]
    for idx in range(1, n):
      t_mid = 0.5 * (times[idx - 1][0] + times[idx][1])
      if ml_flat is None or labels[idx] == labels[idx - 1]:
        snapped = t_mid
      else:
        f_lo = max(1, int(max(0.0, t_mid - tol) / frame_dt))
        f_hi = min(len(ml_flat) - 2, int((t_mid + tol) / frame_dt))
        if f_hi > f_lo + 2:
          best_f = int(t_mid / frame_dt)
          best_flux = -1.0
          for f in range(f_lo, f_hi + 1):
            l_mean = np.mean(ml_flat[max(0, f - 8):f], axis=0)
            r_mean = np.mean(ml_flat[f:min(len(ml_flat), f + 8)], axis=0)
            flux = float(np.sum(np.abs(r_mean - l_mean)))
            if flux > best_flux:
              best_flux = flux
              best_f = f
          snapped = best_f * frame_dt if best_flux > 0.3 else t_mid
        else:
          snapped = t_mid
      boundaries.append(max(boundaries[-1], min(max_t, snapped)))
    boundaries.append(max_t)
    return boundaries

  def extract_overlapping_powerset(
      self,
      waveform: np.ndarray,
      sample_rate: int = 16000,
      step_sec: Optional[float] = None,
  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Runs overlapping 10s PyanNet windows for powerset diarization."""
    if waveform.size == 0 or torch is None:
      return None
    seg_mod = self._ensure_segmentation_model()
    if seg_mod is None:
      return None

    step_s = (
        float(step_sec)
        if step_sec is not None
        else float(getattr(self.preset, "seg_step_sec", 2.5))
    )
    win_samp = 10 * sample_rate
    step_samp = max(1, int(round(step_s * sample_rate)))
    n_samp = len(waveform)
    starts = list(range(0, max(1, n_samp - win_samp + 1), step_samp))
    if n_samp > win_samp and (n_samp - (starts[-1] + win_samp)) >= sample_rate:
      starts.append(max(0, n_samp - win_samp))
    if not starts:
      starts = [0]

    chunks: List[np.ndarray] = []
    for s in starts:
      c = waveform[s:s + win_samp]
      if len(c) < win_samp:
        c = np.pad(c, (0, win_samp - len(c)))
      chunks.append(np.asarray(c, dtype=np.float32))

    argmax_list: List[np.ndarray] = []
    soft_list: List[np.ndarray] = []
    with torch.no_grad():
      for i0 in range(0, len(chunks), self.batch_size):
        arr = np.stack(chunks[i0:i0 + self.batch_size], axis=0)
        b_t = torch.from_numpy(arr).to(self.device)
        lp = seg_mod(b_t)
        cls_arg = torch.argmax(lp, dim=-1).cpu().numpy().astype(np.uint8)
        if hasattr(seg_mod, "to_multilabel"):
          ml = seg_mod.to_multilabel(lp, soft=True).cpu().numpy()
        else:
          ml = powerset_probs_to_multilabel(torch.exp(lp).cpu().numpy())
        argmax_list.append(cls_arg)
        soft_list.append(ml[:, ::2, :].astype(np.float32))

    chunk_starts = np.array(
        [s / float(sample_rate) for s in starts], dtype=np.float32
    )
    return (
        chunk_starts,
        np.concatenate(argmax_list, axis=0),
        np.concatenate(soft_list, axis=0),
    )

  def _encode_variable_tracks(
      self,
      encoder: Any,
      tracks: List[np.ndarray],
  ) -> np.ndarray:
    """Batch-encodes variable-length clean `(T_i, 80)` tracks."""
    if not tracks:
      return np.zeros((0, 192), dtype=np.float32)
    is_fs = self._is_flaxspeaker(encoder)
    order = sorted(range(len(tracks)), key=lambda i: tracks[i].shape[0])
    out_list: List[Optional[np.ndarray]] = [None] * len(tracks)
    bs = 32 if is_fs else min(64, self.batch_size)
    with torch.no_grad():
      for pos in range(0, len(order), bs):
        b_idx = order[pos:pos + bs]
        lengths = [tracks[i].shape[0] for i in b_idx]
        max_t = max(lengths)
        if is_fs:
          arr = np.zeros((len(b_idx), max_t, tracks[b_idx[0]].shape[1]),
                         dtype=np.float32)
          msk = np.zeros((len(b_idx), max_t), dtype=np.float32)
          for row, i in enumerate(b_idx):
            tl = tracks[i].shape[0]
            arr[row, :tl] = tracks[i]
            msk[row, :tl] = 1.0
          x_t = torch.from_numpy(arr).to(self.device)
          m_t = torch.from_numpy(msk).to(self.device)
          embs = encoder(x_t, mask=m_t).cpu().numpy()
          for row, i in enumerate(b_idx):
            out_list[i] = embs[row]
        elif max_t - min(lengths) > 25:
          for i in b_idx:
            x_t = torch.from_numpy(tracks[i]).unsqueeze(0).to(self.device)
            raw = encoder(x_t)
            if isinstance(raw, torch.Tensor):
              emb = F.normalize(raw, p=2, dim=1).cpu().numpy()[0]
            else:
              emb = spectral.l2_normalize_rows(
                  np.asarray(raw, dtype=np.float32)
              )[0]
            out_list[i] = emb
        else:
          feat_d = tracks[b_idx[0]].shape[1]
          arr = np.zeros((len(b_idx), max_t, feat_d), dtype=np.float32)
          for row, i in enumerate(b_idx):
            tl = tracks[i].shape[0]
            arr[row, :tl] = tracks[i]
            if tl < max_t:
              reps = (max_t - tl + tl - 1) // tl
              pad_src = np.tile(tracks[i], (reps, 1))[:max_t - tl]
              arr[row, tl:] = pad_src
          x_t = torch.from_numpy(arr).to(self.device)
          raw = encoder(x_t)
          if isinstance(raw, torch.Tensor):
            embs = F.normalize(raw, p=2, dim=1).cpu().numpy()
          else:
            embs = spectral.l2_normalize_rows(
                np.asarray(raw, dtype=np.float32)
            )
          for row, i in enumerate(b_idx):
            out_list[i] = embs[row]
    stacked = np.stack([e for e in out_list if e is not None], axis=0)
    return spectral.l2_normalize_rows(stacked)

  def _run_powerset_ward(
      self,
      waveform: np.ndarray,
      sample_rate: int = 16000,
      num_speakers: Optional[int] = None,
      min_speakers: Optional[int] = None,
      max_speakers: Optional[int] = None,
  ) -> Optional[List[types.SpeakerTurn]]:
    """Runs overlapping powerset-masked track diarization + Ward + EM."""
    ps_res = self.extract_overlapping_powerset(
        waveform, sample_rate=sample_rate
    )
    if ps_res is None:
      return None
    chunk_starts, ps_argmax, ml_soft_ds = ps_res
    n_seg_frames = ps_argmax.shape[1]
    if len(chunk_starts) == 0 or n_seg_frames == 0:
      return None

    encoder = self._ensure_encoder()
    is_fs = self._is_flaxspeaker(encoder)
    if is_fs:
      fb_raw = flaxspeaker_encoder.compute_flaxspeaker_log_mel(
          waveform, sample_rate=sample_rate, apply_agc=True, apply_cmvn=False
      )
    else:
      fb_raw_t = compute_log_mel_fbank(
          waveform, sample_rate=sample_rate, apply_cmvn=True
      )
      fb_raw = (
          fb_raw_t.numpy()
          if hasattr(fb_raw_t, "numpy")
          else np.asarray(fb_raw_t, dtype=np.float32)
      )

    n_fb = int(fb_raw.shape[0])
    frame_dt_full = 10.0 / float(n_seg_frames)
    if is_fs:
      active_10ms = np.zeros(n_fb, dtype=bool)
      for ci, cs_sec in enumerate(chunk_starts):
        f0 = int(round(float(cs_sec) * 100.0))
        f1 = min(n_fb, f0 + 1000)
        if f1 <= f0:
          continue
        t_rel = np.arange(f1 - f0, dtype=np.float32) * 0.01
        seg_i = np.clip(
            (t_rel / frame_dt_full).astype(np.int32), 0, n_seg_frames - 1
        )
        active_10ms[f0:f1] |= ps_argmax[ci, seg_i] > 0
      if int(active_10ms.sum()) >= 50:
        rec_mean = np.mean(fb_raw[active_10ms], axis=0, keepdims=True)
      else:
        rec_mean = np.mean(fb_raw, axis=0, keepdims=True)
      fb_rec = (fb_raw - rec_mean).astype(np.float32)
    else:
      fb_rec = np.asarray(fb_raw, dtype=np.float32)

    min_ext_sec = float(getattr(self.preset, "min_extract_sec", 0.4))
    min_ext_frames = max(20, int(round(min_ext_sec * 100.0)))
    tracks: List[np.ndarray] = []
    cs_meta_list: List[List[float]] = []
    for ci, cs_sec in enumerate(chunk_starts):
      f0 = int(round(float(cs_sec) * 100.0))
      f1 = min(n_fb, f0 + 1000)
      if f1 - f0 < min_ext_frames:
        continue
      t_rel = np.arange(f1 - f0, dtype=np.float32) * 0.01
      seg_i = np.clip(
          (t_rel / frame_dt_full).astype(np.int32), 0, n_seg_frames - 1
      )
      cls_10ms = ps_argmax[ci, seg_i]
      fb_chunk = fb_rec[f0:f1]
      for s_local in (0, 1, 2):
        mask_clean = cls_10ms == (s_local + 1)
        n_clean = int(mask_clean.sum())
        if n_clean < min_ext_frames:
          continue
        tracks.append(fb_chunk[mask_clean].astype(np.float32))
        cs_meta_list.append([float(ci), float(s_local), n_clean * 0.01])

    if not tracks:
      return None

    cs_meta = np.asarray(cs_meta_list, dtype=np.float32)
    emb_all = self._encode_variable_tracks(encoder, tracks)
    durs = cs_meta[:, 2]
    lo_k = (
        int(num_speakers)
        if num_speakers is not None
        else int(
            min_speakers
            if min_speakers is not None
            else self.preset.min_clusters
        )
    )
    hi_k = (
        int(num_speakers)
        if num_speakers is not None
        else int(
            max_speakers
            if max_speakers is not None
            else self.preset.max_clusters
        )
    )
    min_clean_sec = float(getattr(self.preset, "min_clean_sec", 1.0))
    use_mask = durs >= min_clean_sec
    if int(use_mask.sum()) < max(lo_k, 6):
      order = np.argsort(-durs)
      use_mask = np.zeros_like(durs, dtype=bool)
      keep_n = min(len(durs), max(lo_k * 2, 6))
      use_mask[order[:keep_n]] = True

    sel_emb = emb_all[use_mask]
    sel_dur = durs[use_mask]
    sel_cids = np.round(cs_meta[use_mask, 0]).astype(np.int32)
    merge_thr = float(
        getattr(
            self.preset,
            "ward_merge_threshold",
            0.70 if is_fs else 0.60,
        )
    )
    raw_min_spk_sec = float(getattr(self.preset, "min_cluster_sec", 8.0))
    min_spk_sec = min(raw_min_spk_sec, 0.08 * float(np.sum(sel_dur)))
    labels = spectral.ward_cluster_auto_k(
        sel_emb,
        min_clusters=lo_k,
        max_clusters=hi_k,
        merge_threshold=merge_thr,
        min_cluster_weight=min_spk_sec,
        weights=sel_dur,
        num_clusters=num_speakers,
    )
    em_iters = int(getattr(self.preset, "em_iters", 5))
    if len(set(labels.tolist())) > 1 and em_iters > 0:
      labels = spectral.refine_cosine_em(
          sel_emb,
          labels,
          weights=sel_dur,
          chunk_ids=sel_cids,
          n_iters=em_iters,
          enforce_cannot_link=True,
      )

    emb_c = emb_all
    sel_c = sel_emb
    bal_beta = float(getattr(self.preset, "balanced_centering_beta", 1.0))
    if bal_beta > 0.0 and len(set(labels.tolist())) >= 2:
      emb_c = spectral.balance_center_embeddings(
          emb_all, sel_emb, labels, weights=sel_dur, beta=bal_beta
      )
      sel_c = emb_c[use_mask]
      labels = spectral.refine_cosine_em(
          sel_c,
          labels,
          weights=sel_dur,
          chunk_ids=sel_cids,
          n_iters=3,
          enforce_cannot_link=True,
      )

    cents, _ = spectral.compute_weighted_centroids(
        sel_c, labels, weights=sel_dur
    )
    dur_sec = float(len(waveform)) / float(sample_rate)
    global_p, frame_dt = stitch_powerset_probs(
        chunk_starts=chunk_starts,
        ml_soft=ml_soft_ds,
        cs_meta=cs_meta,
        cs_embs_all=emb_c,
        centroids=cents,
        duration_sec=dur_sec,
        chunk_window_weight=str(
            getattr(self.preset, "chunk_window_weight", "uniform")
        ),
        use_temporal_stitch=True,
        stitch_weight=float(getattr(self.preset, "stitch_weight", 0.50)),
        prob_smooth_sec=float(getattr(self.preset, "prob_smooth_sec", 0.25)),
    )
    self.last_frame_probs = global_p
    self.last_frame_dt = frame_dt
    return probs_to_speaker_turns(global_p, frame_dt, dur_sec)

  def extract_window_embeddings(
      self,
      waveform: np.ndarray,
      sample_rate: int = 16000,
  ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Extracts L2-normalized sliding-window speaker embeddings."""
    encoder = self._ensure_encoder()
    is_fs = self._is_flaxspeaker(encoder)
    if is_fs:
      fbank_raw = flaxspeaker_encoder.compute_flaxspeaker_log_mel(
          waveform, sample_rate=sample_rate, apply_agc=True, apply_cmvn=True
      )
      fbank = (
          torch.from_numpy(fbank_raw) if torch is not None else fbank_raw
      )
    else:
      fbank = compute_log_mel_fbank(waveform, sample_rate=sample_rate)
    num_frames = int(fbank.shape[0])
    win = self.preset.window_frames
    step = self.preset.step_frames

    if num_frames <= 0:
      return np.zeros((0, 256 if is_fs else 192), dtype=np.float32), []

    if num_frames <= win:
      chunks = [fbank]
      times = [(0.0, num_frames * 0.01)]
    else:
      chunks = []
      times = []
      last_s = 0
      for s in range(0, num_frames - win + 1, step):
        chunks.append(fbank[s:s + win])
        times.append((s * 0.01, (s + win) * 0.01))
        last_s = s
      if num_frames - (last_s + win) >= max(2, step // 2):
        tail_s = num_frames - win
        chunks.append(fbank[tail_s:num_frames])
        times.append((tail_s * 0.01, num_frames * 0.01))

    if torch is not None and isinstance(chunks[0], torch.Tensor):
      batch_tensor = torch.stack(chunks, dim=0).to(self.device)
      embs_list: List[np.ndarray] = []
      with torch.no_grad():
        for i in range(0, len(batch_tensor), self.batch_size):
          out = encoder(batch_tensor[i:i + self.batch_size])
          if isinstance(out, torch.Tensor):
            out = F.normalize(out, p=2, dim=1).cpu().numpy()
          else:
            out = np.asarray(out, dtype=np.float32)
            norms = np.maximum(
                np.linalg.norm(out, axis=1, keepdims=True), 1e-12
            )
            out = out / norms
          embs_list.append(out)
      embs = np.concatenate(embs_list, axis=0)
      if isinstance(encoder, SpeechBrainECAPATDNN) and len(embs) >= 2:
        embs = embs - np.mean(embs, axis=0, keepdims=True)
        norms = np.maximum(
            np.linalg.norm(embs, axis=1, keepdims=True), 1e-12
        )
        embs = embs / norms
      return embs, times

    # NumPy mock encoder path
    batch_np = np.stack([np.asarray(c) for c in chunks], axis=0)
    raw_embs = np.asarray(encoder(batch_np), dtype=np.float32)
    norms = np.maximum(np.linalg.norm(raw_embs, axis=1, keepdims=True), 1e-12)
    return raw_embs / norms, times

  def __call__(
      self,
      audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]],
      num_speakers: Optional[int] = None,
      min_speakers: Optional[int] = None,
      max_speakers: Optional[int] = None,
  ) -> List[types.SpeakerTurn]:
    """Runs speaker diarization and returns contiguous SpeakerTurns."""
    self.last_frame_probs = None
    wav = load_audio_16k(audio_input, target_sr=16000)
    if wav.size == 0:
      return []

    mode = str(getattr(self.preset, "diarization_mode", "powerset_ward"))
    if mode == "powerset_ward" and self.segmentation_fn is None:
      ps_turns = self._run_powerset_ward(
          wav,
          sample_rate=16000,
          num_speakers=num_speakers,
          min_speakers=min_speakers,
          max_speakers=max_speakers,
      )
      if ps_turns is not None:
        return ps_turns

    embs, times = self.extract_window_embeddings(wav, sample_rate=16000)
    if len(embs) == 0:
      return []

    if num_speakers is not None:
      labels = self.clusterer.predict(embs, num_clusters=int(num_speakers))
    elif min_speakers is not None or max_speakers is not None:
      temp_clusterer = spectral.SpectralClusterer(
          min_clusters=min_speakers or self.preset.min_clusters,
          max_clusters=max_speakers or self.preset.max_clusters,
          p_percentile=self.preset.p_percentile,
          gaussian_blur_sigma=self.preset.gaussian_blur_sigma,
          eigen_threshold=self.preset.eigen_threshold,
      )
      labels = temp_clusterer.predict(embs)
    else:
      labels = self.clusterer.predict(embs)

    ml_flat = self.extract_segmentation_curves(wav, sample_rate=16000)
    boundaries = self.snap_turn_boundaries(times, labels, ml_flat)

    raw_turns: List[types.SpeakerTurn] = []
    n = len(labels)
    for idx in range(n):
      t_s = boundaries[idx]
      t_e = boundaries[idx + 1]
      spk_str = str(int(labels[idx]) + 1)
      if raw_turns and raw_turns[-1].speaker == spk_str:
        raw_turns[-1].end = t_e
      else:
        raw_turns.append(types.SpeakerTurn(start=t_s, end=t_e, speaker=spk_str))

    return raw_turns


PyanNetECAPASpectralDiarizer = ECAPASpectralDiarizer
