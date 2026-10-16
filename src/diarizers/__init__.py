"""Diarizers v1.0.0: Open-Source ASR, Speaker Diarization, and DiarizationLM."""

__version__ = "1.0.0"

from diarizers.alignment import (
    ChunkedPromptBuilder,
    UtteranceChunker,
    apply_locality_gated_speaker_transfer,
    assign_speakers,
    create_diarized_text,
    extract_text_and_spk,
    get_oracle_speakers,
    levenshtein_with_edits,
    locality_gated_speaker_transfer,
    normalize_text,
    reconcile_chunk_speakers_hungarian,
    speakers_transform,
    tpst_align,
    transcript_preserving_speaker_transfer,
)
from diarizers.asr import (
    AudioCppASR,
    PrecomputedASR,
    WhisperWordASR,
    turns_to_word_segments,
)
from diarizers.configs import (
    ASRConfig,
    CALLHOME_TELEPHONY_PRESET,
    DEFAULT_ASR_CONFIG,
    DEFAULT_PROMPT_CONFIG,
    DiarizationPreset,
    DiarizersConfig,
    FISHER_TELEPHONY_PRESET,
    MEETING_MULTISPK_PRESET,
    PromptConfig,
)
from diarizers.dataprep import (
    build_locality_preserving_oracle,
    build_sft_dataset,
    compute_utt_oracle_and_degraded,
    degrade_speakers,
    generate_synthetic_dataset,
)
from diarizers.diarization import (
    ECAPASpectralDiarizer,
    ParamSincFB,
    PyanNetECAPASpectralDiarizer,
    PyanNetSegmentation,
    SincConv1d,
    SincNet,
    SpeechBrainECAPATDNN,
    compute_log_mel_fbank,
    load_audio_16k,
)
from diarizers.evaluation import (
    bootstrap_paired_ci,
    evaluate_json_dict,
    evaluate_json_file,
    evaluate_utterance,
)
from diarizers.export import export_gguf_q4_k_m, export_merged_safetensors
from diarizers.llm import DiarizationLMRefiner
from diarizers.pipeline import (
    DiarizationPipeline,
    DiarizersPipeline,
    Pipeline,
)
from diarizers.spectral import SpectralClusterer
from diarizers.train import (
    TrainingConfig,
    prepare_sft_records_from_json,
    run_lora_sft_training,
)
from diarizers.types import (
    DiarizedTranscript,
    DiarizedWord,
    SpeakerTurn,
    WordSegment,
)

__all__ = [
    "__version__",
    "ASRConfig",
    "AudioCppASR",
    "CALLHOME_TELEPHONY_PRESET",
    "ChunkedPromptBuilder",
    "DEFAULT_ASR_CONFIG",
    "DEFAULT_PROMPT_CONFIG",
    "DiarizationLMRefiner",
    "DiarizationPipeline",
    "DiarizationPreset",
    "DiarizedTranscript",
    "DiarizedWord",
    "DiarizersConfig",
    "DiarizersPipeline",
    "ECAPASpectralDiarizer",
    "FISHER_TELEPHONY_PRESET",
    "MEETING_MULTISPK_PRESET",
    "ParamSincFB",
    "Pipeline",
    "PrecomputedASR",
    "PromptConfig",
    "PyanNetECAPASpectralDiarizer",
    "PyanNetSegmentation",
    "SincConv1d",
    "SincNet",
    "SpeakerTurn",
    "SpectralClusterer",
    "SpeechBrainECAPATDNN",
    "TrainingConfig",
    "UtteranceChunker",
    "WhisperWordASR",
    "WordSegment",
    "apply_locality_gated_speaker_transfer",
    "assign_speakers",
    "bootstrap_paired_ci",
    "build_locality_preserving_oracle",
    "build_sft_dataset",
    "compute_log_mel_fbank",
    "compute_utt_oracle_and_degraded",
    "create_diarized_text",
    "degrade_speakers",
    "evaluate_json_dict",
    "evaluate_json_file",
    "evaluate_utterance",
    "export_gguf_q4_k_m",
    "export_merged_safetensors",
    "extract_text_and_spk",
    "generate_synthetic_dataset",
    "get_oracle_speakers",
    "levenshtein_with_edits",
    "load_audio_16k",
    "locality_gated_speaker_transfer",
    "normalize_text",
    "prepare_sft_records_from_json",
    "reconcile_chunk_speakers_hungarian",
    "run_lora_sft_training",
    "speakers_transform",
    "tpst_align",
    "transcript_preserving_speaker_transfer",
    "turns_to_word_segments",
]

