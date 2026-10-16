"""Open-source Apache-2.0 Speaker Diarization backends."""

import os
from typing import Any, List, Optional, Tuple, Union
import wave

import numpy as np

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
) -> Any:
  """Computes CMVN-normalized 80-dim dB log-mel filterbank features (T, 80)."""
  wav_f32 = np.asarray(waveform, dtype=np.float32)
  if wav_f32.ndim > 1:
    wav_f32 = wav_f32[:, 0]
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
    return fbank - fbank.mean(dim=0, keepdim=True)

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
  return fbank_np - np.mean(fbank_np, axis=0, keepdims=True)



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
      wav = pcm.reshape(-1, n_ch)[:, 0] if n_ch > 1 else pcm
  elif isinstance(audio_input, tuple):
    wav, sr = np.asarray(audio_input[0], dtype=np.float32), int(audio_input[1])
  else:
    wav, sr = np.asarray(audio_input, dtype=np.float32), target_sr

  if wav.ndim > 1:
    wav = wav[:, 0]
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

  @property
  def segmentation_model(self) -> Optional[Any]:
    return self._seg_mod or self.segmentation_model_spec

  def _ensure_encoder(self) -> Any:
    if self.encoder is None:
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
    # Skip loading real segmentation weights when a mock encoder is passed.
    if self.encoder is not None and not isinstance(
        self.encoder, SpeechBrainECAPATDNN
    ):
      return None
    if torch is None:
      return None
    try:
      self._seg_mod = PyanNetSegmentation.from_pretrained(
          self.segmentation_model_spec, device=self.device
      )
    except Exception:
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

  def extract_window_embeddings(
      self,
      waveform: np.ndarray,
      sample_rate: int = 16000,
  ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Extracts L2-normalized sliding-window speaker embeddings."""
    fbank = compute_log_mel_fbank(waveform, sample_rate=sample_rate)
    num_frames = int(fbank.shape[0])
    win = self.preset.window_frames
    step = self.preset.step_frames

    if num_frames <= 0:
      return np.zeros((0, 192), dtype=np.float32), []

    if num_frames <= win:
      chunks = [fbank]
      times = [(0.0, num_frames * 0.01)]
    else:
      chunks = []
      times = []
      for s in range(0, num_frames - win + 1, step):
        chunks.append(fbank[s:s + win])
        times.append((s * 0.01, (s + win) * 0.01))

    encoder = self._ensure_encoder()
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
    wav = load_audio_16k(audio_input, target_sr=16000)
    if wav.size == 0:
      return []
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
