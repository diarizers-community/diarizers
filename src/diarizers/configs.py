"""Pre-configured settings for Diarizers ASR, Diarization, and LLM stages."""

import dataclasses


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
  max_ghost_cluster_words: int = 12
  overlap_words: int = 0


DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
DEFAULT_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


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
  segmentation_model: str | None = DEFAULT_SEGMENTATION_MODEL
  encoder_name: str = DEFAULT_ECAPA_MODEL
  window_frames: int = 120
  step_frames: int = 40
  min_clusters: int = 2
  max_clusters: int = 2
  p_percentile: float = 0.85
  gaussian_blur_sigma: float = 1.5
  eigen_threshold: float = 0.45
  boundary_snap_tolerance_sec: float = 0.35


@dataclasses.dataclass
class DiarizersConfig:
  """Top-level configuration for the 4-stage Diarizers pipeline."""

  asr: ASRConfig = dataclasses.field(default_factory=ASRConfig)
  diarization: DiarizationPreset = dataclasses.field(
      default_factory=lambda: FISHER_TELEPHONY_PRESET
  )
  prompt: PromptConfig = dataclasses.field(default_factory=PromptConfig)
  llm_model_or_path: str = (
      "diarizers-community/DiarizationLM-Gemma-4-E4B-v1"
  )


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
)

DEFAULT_DIARIZERS_CONFIG = DiarizersConfig()
