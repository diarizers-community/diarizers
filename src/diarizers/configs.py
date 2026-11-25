"""Pre-configured settings for Diarizers ASR, Diarization, and LLM stages."""

from __future__ import annotations

import dataclasses
from typing import Optional


from diarizers import flaxspeaker_encoder


SPEAKER_PREFIX = "<speaker:"
SPEAKER_SUFFIX = ">"
PROMPT_PREFIX = ""
PROMPT_SUFFIX = " --> "
COMPLETION_SUFFIX = " [eod]"
EMIT_INPUT_LENGTH = 4000
EMIT_TARGET_LENGTH = 4000


@dataclasses.dataclass
class PromptConfig:
  """Configuration for DiarizationLM prompt and speaker formatting."""

  speaker_prefix: str = SPEAKER_PREFIX
  speaker_suffix: str = SPEAKER_SUFFIX
  prompt_prefix: str = PROMPT_PREFIX
  prompt_suffix: str = PROMPT_SUFFIX
  completion_suffix: str = COMPLETION_SUFFIX
  emit_input_length: int = EMIT_INPUT_LENGTH
  emit_target_length: int = EMIT_TARGET_LENGTH
  use_gemma_turn_template: bool = True
  use_locality_gate: bool = True
  max_local_span_words: int = 5
  max_two_spk_span_words: int = 15
  max_ghost_cluster_words: int = 15
  overlap_words: int = 0
  max_chunk_bytes: int = 5400
  use_byte_budget: bool = False
  max_chunk_word_edit_rate: float = 0.0


DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
DEFAULT_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_FLAXSPEAKER_LARGE_MODEL = "wq2012/univ_full_eres2conformer_large"
DEFAULT_FLAXSPEAKER_V2_MODEL = "wq2012/univ_full_eres2conformer_v2_large"
DEFAULT_FLAXSPEAKER_XL_MODEL = "wq2012/univ_full_eres2conformer_xl"
DEFAULT_FLAXSPEAKER_V2_DIAR_MODEL = "wq2012/univ_full_eres2conformer_v2_diar"
DEFAULT_FLAXSPEAKER_XL_DIAR_BASE_MODEL = (
    "wq2012/univ_full_eres2conformer_xl_diar"
)
DEFAULT_FLAXSPEAKER_XL_DIAR_MODEL = (
    "wq2012/univ_full_eres2conformer_xl_diar_adapt"
)
DEFAULT_FLAXSPEAKER_XL_DIAR_PFAS_MODEL = (
    "wq2012/univ_full_eres2conformer_xl_diar_pfas"
)
DEFAULT_LLM_MODEL_V1 = "diarizers-community/DiarizationLM-Gemma-4-E4B-v1"
DEFAULT_LLM_MODEL_V2 = "diarizers-community/DiarizationLM-Gemma-4-E4B-v2"
DEFAULT_LLM_MODEL_12B = "diarizers-community/DiarizationLM-Gemma-4-12B-v1"


@dataclasses.dataclass
class ASRConfig:
  """Configuration for word-timestamped Whisper ASR backend."""

  model_name_or_path: str = DEFAULT_ASR_MODEL
  device: str = "cpu"
  batch_size: int = 4
  chunk_seconds: int = 30
  num_threads: int = 8

  @property
  def model_name(self) -> str:
    return self.model_name_or_path


@dataclasses.dataclass
class DiarizationPreset:
  """Configuration preset for segmentation, speaker embeddings, clustering."""

  name: str = "telephony_2spk"
  segmentation_model: Optional[str] = DEFAULT_SEGMENTATION_MODEL
  encoder_name: str = DEFAULT_ECAPA_MODEL
  window_frames: int = 120
  step_frames: int = 40
  min_clusters: int = 2
  max_clusters: int = 2
  p_percentile: float = 0.85
  gaussian_blur_sigma: float = 1.5
  eigen_threshold: float = 0.45
  boundary_snap_tolerance_sec: float = 0.35
  diarization_mode: str = "powerset_ward"
  seg_step_sec: float = 2.5
  min_clean_sec: float = 1.0
  min_extract_sec: float = 0.4
  ward_merge_threshold: float = 0.60
  min_cluster_sec: float = 8.0
  balanced_centering_beta: float = 1.0
  stitch_weight: float = 0.50
  chunk_window_weight: str = "uniform"
  prob_smooth_sec: float = 0.25
  em_iters: int = 5
  flaxspeaker_cmvn_mode: str = "auto"
  secondary_encoder_name: Optional[str] = None
  secondary_fusion_weight: float = 0.35


@dataclasses.dataclass
class DiarizersConfig:
  """Top-level configuration for the 4-stage Diarizers pipeline."""

  asr: ASRConfig = dataclasses.field(default_factory=ASRConfig)
  diarization: DiarizationPreset = dataclasses.field(
      default_factory=lambda: FLAXSPEAKER_XL_DIAR_PRESET
  )
  prompt: PromptConfig = dataclasses.field(default_factory=PromptConfig)
  llm_model_or_path: Optional[str] = DEFAULT_LLM_MODEL_12B


@dataclasses.dataclass(frozen=True)
class PipelineProfile:
  """Named high-level quality/cost profile for DiarizersPipeline."""

  name: str
  asr_model: str
  diarization: DiarizationPreset
  llm_model_or_path: Optional[str]
  use_locality_gate: bool = True
  description: str = ""


DEFAULT_ASR_CONFIG = ASRConfig()
DEFAULT_PROMPT_CONFIG = PromptConfig()

FISHER_TELEPHONY_PRESET = DiarizationPreset(
    name="fisher_telephony_2spk",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name="speechbrain/spkrec-ecapa-voxceleb",
    window_frames=120,
    step_frames=40,
    min_clusters=2,
    max_clusters=2,
    p_percentile=0.85,
    gaussian_blur_sigma=1.5,
    eigen_threshold=0.45,
)

CALLHOME_TELEPHONY_PRESET = DiarizationPreset(
    name="callhome_telephony_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name="speechbrain/spkrec-ecapa-voxceleb",
    window_frames=120,
    step_frames=40,
    min_clusters=2,
    max_clusters=6,
    p_percentile=0.70,
    gaussian_blur_sigma=1.5,
    eigen_threshold=0.45,
)

AMI_MEETING_PRESET = DiarizationPreset(
    name="ami_meeting_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name="speechbrain/spkrec-ecapa-voxceleb",
    window_frames=200,
    step_frames=100,
    min_clusters=3,
    max_clusters=5,
    p_percentile=0.70,
    gaussian_blur_sigma=1.5,
    eigen_threshold=0.45,
)

MEETING_MULTISPK_PRESET = DiarizationPreset(
    name="meeting_multispk_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name="speechbrain/spkrec-ecapa-voxceleb",
    window_frames=200,
    step_frames=100,
    min_clusters=2,
    max_clusters=10,
    p_percentile=0.70,
    gaussian_blur_sigma=1.5,
    eigen_threshold=0.45,
    ward_merge_threshold=0.60,
)

FLAXSPEAKER_LARGE_PRESET = DiarizationPreset(
    name="flaxspeaker_large_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name=DEFAULT_FLAXSPEAKER_LARGE_MODEL,
    window_frames=200,
    step_frames=100,
    min_clusters=1,
    max_clusters=10,
    diarization_mode="powerset_ward",
    ward_merge_threshold=0.70,
    min_clean_sec=1.0,
    min_cluster_sec=8.0,
    balanced_centering_beta=1.0,
    chunk_window_weight="uniform",
    prob_smooth_sec=0.25,
)

FLAXSPEAKER_V2_PRESET = DiarizationPreset(
    name="flaxspeaker_v2_large_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name=DEFAULT_FLAXSPEAKER_V2_MODEL,
    window_frames=200,
    step_frames=100,
    min_clusters=1,
    max_clusters=10,
    diarization_mode="powerset_ward",
    ward_merge_threshold=0.70,
    min_clean_sec=1.0,
    min_cluster_sec=8.0,
    balanced_centering_beta=1.0,
    chunk_window_weight="uniform",
    prob_smooth_sec=0.25,
)

FLAXSPEAKER_XL_PRESET = DiarizationPreset(
    name="flaxspeaker_xl_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name=DEFAULT_FLAXSPEAKER_XL_MODEL,
    window_frames=200,
    step_frames=100,
    min_clusters=1,
    max_clusters=10,
    diarization_mode="powerset_ward",
    ward_merge_threshold=0.70,
    min_clean_sec=1.0,
    min_cluster_sec=8.0,
    balanced_centering_beta=1.0,
    chunk_window_weight="uniform",
    prob_smooth_sec=0.25,
    flaxspeaker_cmvn_mode="recording",
)

FLAXSPEAKER_V2_DIAR_PRESET = DiarizationPreset(
    name="flaxspeaker_v2_diar_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name=DEFAULT_FLAXSPEAKER_V2_DIAR_MODEL,
    window_frames=200,
    step_frames=100,
    min_clusters=1,
    max_clusters=10,
    diarization_mode="powerset_ward",
    ward_merge_threshold=0.75,
    min_clean_sec=1.0,
    min_cluster_sec=8.0,
    balanced_centering_beta=1.0,
    chunk_window_weight="uniform",
    prob_smooth_sec=0.25,
    flaxspeaker_cmvn_mode="segment",
)

FLAXSPEAKER_XL_DIAR_PRESET = DiarizationPreset(
    name="flaxspeaker_xl_diar_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name=DEFAULT_FLAXSPEAKER_XL_DIAR_MODEL,
    window_frames=200,
    step_frames=100,
    min_clusters=1,
    max_clusters=10,
    diarization_mode="powerset_ward",
    ward_merge_threshold=0.75,
    min_clean_sec=1.0,
    min_cluster_sec=8.0,
    balanced_centering_beta=1.0,
    chunk_window_weight="uniform",
    prob_smooth_sec=0.25,
    flaxspeaker_cmvn_mode="segment",
)

FLAXSPEAKER_DIAR_ECAPA_FUSION_PRESET = DiarizationPreset(
    name="flaxspeaker_diar_ecapa_fusion_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name=DEFAULT_FLAXSPEAKER_XL_DIAR_MODEL,
    window_frames=200,
    step_frames=100,
    min_clusters=1,
    max_clusters=10,
    diarization_mode="powerset_ward",
    ward_merge_threshold=0.72,
    min_clean_sec=1.0,
    min_cluster_sec=8.0,
    balanced_centering_beta=1.0,
    chunk_window_weight="uniform",
    prob_smooth_sec=0.25,
    flaxspeaker_cmvn_mode="segment",
    secondary_encoder_name=DEFAULT_ECAPA_MODEL,
    secondary_fusion_weight=0.35,
)

UNIVERSAL_AUTO_PRESET = DiarizationPreset(
    name="universal_auto",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name=DEFAULT_ECAPA_MODEL,
    window_frames=200,
    step_frames=100,
    min_clusters=1,
    max_clusters=10,
    diarization_mode="powerset_ward",
    ward_merge_threshold=0.60,
    min_clean_sec=1.0,
    min_cluster_sec=8.0,
    balanced_centering_beta=1.0,
    chunk_window_weight="uniform",
    prob_smooth_sec=0.25,
)


def adapt_preset_for_encoder(
    preset: DiarizationPreset,
    encoder_name: str,
    diarization_mode: Optional[str] = None,
) -> DiarizationPreset:
  """Updates ``preset`` with calibrated CMVN & Ward thresholds for encoder."""
  is_diar = flaxspeaker_encoder.is_flaxspeaker_diar_model_id(encoder_name)
  is_fs = flaxspeaker_encoder.is_flaxspeaker_model_id(encoder_name)
  thr = 0.75 if is_diar else (0.70 if is_fs else 0.60)
  cmvn = "segment" if is_diar else ("recording" if is_fs else "auto")
  kwargs = {
      "encoder_name": encoder_name,
      "ward_merge_threshold": thr,
      "flaxspeaker_cmvn_mode": cmvn,
      "secondary_encoder_name": None,
  }
  if diarization_mode is not None:
    kwargs["diarization_mode"] = diarization_mode
  return dataclasses.replace(preset, **kwargs)


BEST_QUALITY_PROFILE = PipelineProfile(
    name="best-quality",
    asr_model=DEFAULT_ASR_MODEL,
    diarization=FLAXSPEAKER_XL_DIAR_PRESET,
    llm_model_or_path=DEFAULT_LLM_MODEL_12B,
    use_locality_gate=True,
    description=(
        "Best single-encoder accuracy (3.82% micro / 5.16% macro WDER, "
        "0.177 SpkCntMAE): FlaxSpeaker-XL-Diar-Adapt + Gemma-4-12B-v1."
    ),
)

MAX_PROFILE = PipelineProfile(
    name="max",
    asr_model=DEFAULT_ASR_MODEL,
    diarization=FLAXSPEAKER_DIAR_ECAPA_FUSION_PRESET,
    llm_model_or_path=DEFAULT_LLM_MODEL_12B,
    use_locality_gate=True,
    description=(
        "Maximum SOTA dual-encoder accuracy (3.79% micro / 5.14% macro WDER, "
        "7.92% AMI): FlaxSpeaker-XL-Diar-Adapt + ECAPA fusion + Gemma-4-12B-v1."
    ),
)

BALANCED_PROFILE = PipelineProfile(
    name="balanced",
    asr_model=DEFAULT_ASR_MODEL,
    diarization=UNIVERSAL_AUTO_PRESET,
    llm_model_or_path=DEFAULT_LLM_MODEL_12B,
    use_locality_gate=True,
    description=(
        "15x faster CPU speaker encoder with 12B LLM (3.88% micro / 5.25% "
        "macro WDER): ECAPA powerset_ward + DiarizationLM-Gemma-4-12B-v1."
    ),
)

FAST_PROFILE = PipelineProfile(
    name="fast",
    asr_model=DEFAULT_ASR_MODEL,
    diarization=UNIVERSAL_AUTO_PRESET,
    llm_model_or_path=DEFAULT_LLM_MODEL_V2,
    use_locality_gate=True,
    description=(
        "2x faster LLM & ~50% lower RAM (~7.6 GiB GGUF, 4.13% micro WDER): "
        "ECAPA powerset_ward + DiarizationLM-Gemma-4-E4B-v2."
    ),
)

ACOUSTIC_ONLY_PROFILE = PipelineProfile(
    name="acoustic-only",
    asr_model=DEFAULT_ASR_MODEL,
    diarization=UNIVERSAL_AUTO_PRESET,
    llm_model_or_path=None,
    use_locality_gate=True,
    description=(
        "Lightweight acoustic powerset_ward diarization without LLM "
        "(5.65% micro / 6.96% macro WDER, <2 GiB RAM)."
    ),
)

PIPELINE_PROFILES = {
    "best-quality": BEST_QUALITY_PROFILE,
    "max": MAX_PROFILE,
    "balanced": BALANCED_PROFILE,
    "fast": FAST_PROFILE,
    "acoustic-only": ACOUSTIC_ONLY_PROFILE,
}

PIPELINE_PROFILE_ALIASES = {
    "best-quality": "best-quality",
    "best": "best-quality",
    "quality": "best-quality",
    "default": "best-quality",
    "flaxspeaker": "best-quality",
    "flaxspeaker-diar": "best-quality",
    "max": "max",
    "fusion": "max",
    "flaxspeaker-fusion": "max",
    "best-flaxspeaker": "max",
    "max-multispk": "max",
    "gpu": "max",
    "balanced": "balanced",
    "ecapa-12b": "balanced",
    "cpu-12b": "balanced",
    "fast": "fast",
    "lite": "fast",
    "cpu": "fast",
    "e4b": "fast",
    "acoustic-only": "acoustic-only",
    "no-llm": "acoustic-only",
    "realtime": "acoustic-only",
    "acoustic": "acoustic-only",
}


def resolve_pipeline_profile(
    profile: Optional[str] = "best-quality",
) -> PipelineProfile:
  """Resolves a profile name or alias into its canonical PipelineProfile."""
  key = (profile or "best-quality").strip().lower().replace("_", "-")
  canonical = PIPELINE_PROFILE_ALIASES.get(key)
  if canonical is None:
    valid = ", ".join(sorted(PIPELINE_PROFILES.keys()))
    raise ValueError(
        f"Unknown pipeline profile '{profile}'. "
        f"Choose one of: {valid} (or aliases: best, flaxspeaker, balanced, "
        "no-llm)."
    )
  return PIPELINE_PROFILES[canonical]


DEFAULT_DIARIZERS_CONFIG = DiarizersConfig()
