# 🤗 Diarizers (`v1.2.0`) — Unified Open-Source ASR, Speaker Diarization, and DiarizationLM Post-Processing

[![Python package](https://github.com/diarizers-community/diarizers/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/diarizers-community/diarizers/actions/workflows/pythonapp.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Version](https://img.shields.io/pypi/v/diarizers.svg)](https://pypi.org/project/diarizers/)
[![Python Versions](https://img.shields.io/pypi/pyversions/diarizers.svg)](https://pypi.org/project/diarizers/)
[![Downloads](https://static.pepy.tech/badge/diarizers)](https://www.pepy.tech/projects/diarizers)
[![arxiv](https://img.shields.io/badge/arXiv-2401.03506-blue.svg)](https://arxiv.org/abs/2401.03506)
[![E4B-v2](https://img.shields.io/badge/Hugging_Face-DiarizationLM--Gemma--4--E4B--v2-blue.svg)](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v2)
[![12B-v1](https://img.shields.io/badge/Hugging_Face-DiarizationLM--Gemma--4--12B--v1-green.svg)](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1)
[![Audio Demo Space](https://img.shields.io/badge/Audio_Demo-spaces%2Fdiarizers-blue.svg)](https://huggingface.co/spaces/diarizers-community/diarizers)
[![Text Demo Space](https://img.shields.io/badge/Text_Demo-DiarizationLM--GGUF-blue.svg)](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF)

## Table of Contents

* [Overview](#overview)
* [Configurable Pipeline Profiles (`best-quality`, `max`, `balanced`, `fast`, `acoustic-only`)](#configurable-pipeline-profiles-best-quality-max-balanced-fast-acoustic-only)
* [Architecture](#architecture)
* [Why 2024 `DiarizersLM` Had 43.99%–56.79% WDER & How `diarizers` Solves It](#why-2024-diarizerslm-had-43995679-wder--how-diarizers-solves-it)
* [Evaluation Results](#evaluation-results)
  * [Table 0: End-to-End `diarizers` `v1.2.0` (`powerset_ward`) Benchmark (`211` Test Recordings)](#table-0-end-to-end-diarizers-v120-powerset_ward-benchmark-211-held-out-test-recordings-bfloat16)
  * [Table 1: Open-Source Frontend (`diarizers` v1.0.1 Legacy Baseline) + `DiarizationLM-Gemma-4-E4B-v1`](#table-1-open-source-frontend-diarizers-v101-legacy-baseline--diarizationlm-gemma-4-e4b-v1)
  * [Decomposed Stage-by-Stage Ablation, Model Reuse & Multi-Party Meeting Analysis](#decomposed-stage-by-stage-ablation-model-reuse--multi-party-meeting-analysis)
  * [Table 2: USM + Turn-to-Diarize Frontend + `DiarizationLM-Gemma-4-E4B-v1`](#table-2-usm--turn-to-diarize-frontend--diarizationlm-gemma-4-e4b-v1)
* [Installation](#installation)
* [How to Run the Pipeline](#how-to-run-the-pipeline)
  * [1. One-Liner End-to-End Python API (`DiarizersPipeline.from_pretrained`)](#1-one-liner-end-to-end-python-api-diarizerspipelinefrom_pretrained)
  * [2. Modular End-to-End Python API (`DiarizationPipeline`)](#2-modular-end-to-end-python-api-diarizationpipeline)
  * [3. Post-Processing Any Diarized Transcript with `DiarizationLM`](#3-post-processing-any-diarized-transcript-with-diarizationlm)
  * [4. Command-Line Interface (`diarizers` CLI & `scripts/`)](#4-command-line-interface-diarizers-cli--scripts)
  * [5. Training & Exporting Custom DiarizationLM Models](#5-training--exporting-custom-diarizationlm-models)
  * [6. Evaluating WER, WDER, cpWER, and SpkCntMAE](#6-evaluating-wer-wder-cpwer-and-spkcntmae)
  * [7. Interactive CPU Demos (`hf_space/app.py` & `hf_space/index.html`)](#7-interactive-cpu-demos-hf_spaceapppy--hf_spaceindexhtml)
* [Data Format](#data-format)
* [Repository Structure](#repository-structure)
* [Pretrained Models on Hugging Face](#pretrained-models-on-hugging-face)
* [Running Tests & Linter](#running-tests--linter)
* [Citation](#citation)

---

## Overview

**`diarizers` (`v1.2.0`)** is a unified, self-contained **Apache-2.0** Python library and CLI for end-to-end **speaker-attributed automatic speech recognition (ASR + Speaker Diarization + DiarizationLM Post-Processing)**, as well as training and evaluating custom DiarizationLM models.

It combines:
1. **Word-Timestamped Open-Source ASR (`diarizers.asr`)**: Defaulting to [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT License) with segment-level timestamp decoding and character-proportional word timestamp interpolation (and also supporting `Qwen/Qwen3-ASR` GGUF via `audio.cpp`, Apache-2.0 License, or precomputed word segments).
2. **Overlapping Powerset Speaker Segmentation + Diarization-Trained FlaxSpeaker / ECAPA Embeddings + Auto-`K` Clustering (`diarizers.diarization`, `diarizers.flaxspeaker_encoder` & `diarizers.spectral`)**:
   - Standalone pure-PyTorch implementation of [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) (`PyanNetSegmentation`: SincNet + 4-layer BiLSTM + 7-class powerset classifier, MIT License, with automatic fallback to un-gated mirrors) extracting clean overlapping single-speaker tracks (`powerset_ward`).
   - Standalone pure-PyTorch implementations of [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) **diarization-trained `SimAM-ERes2Conformer`** encoders (`wq2012/univ_full_eres2conformer_xl_diar_adapt` `64.4M`, `xl_diar_pfas` `77.1M`, `v2_diar` `36.3M`, and `xl_diar` `64.4M`, with `cos = 1.0000000` float32 parity against native JAX and automatic per-segment CMVN `flaxspeaker_cmvn_mode="segment"`), [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (`SpeechBrainECAPATDNN`, `20.8M`, `15×` faster on CPU), and **dual-encoder fusion** (`xl_diar_adapt + ECAPA`, `secondary_fusion_weight=0.35`).
   - **Ward Auto-`K` + Cannot-Link Cosine EM + Balanced Speaker-Centroid Centering + Soft Powerset Stitching** (`powerset_ward`, cutting multi-speaker meeting WDER on ICSI from `23.48% → 6.33%` and AMI from `20.68% → 7.92%`, and reducing `SpkCntMAE` to `0.174`), alongside the clean-room **Symmetric Normalized Laplacian Spectral Clusterer** (`diarization_mode="sliding_spectral"`).
3. **Sub-Millisecond Frame-Posterior & Turn Alignment + Chunked Prompting (`diarizers.alignment`)**: Soft frame-probability word assignment (`assign_speakers_from_probs`) and float64 turn-overlap fallback (`assign_speakers`), script-aware multilingual tokenization (`tokenize_multilingual_text`), and 4,000-character chunked prompt building (`ChunkedPromptBuilder`).
4. **Multilingual DiarizationLM Post-Processing & `LG-TPST` (`diarizers.llm` & `diarizers.alignment`)**: Post-processing via **[`diarizers-community/DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1)** (highest accuracy, default in `best-quality`, `max`, and `balanced`) and **[`diarizers-community/DiarizationLM-Gemma-4-E4B-v2`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v2)** (`2×` faster & half the RAM in `fast`), supporting automatic 4-bit `Q4_K_M` GGUF inference on CPU via `llama.cpp` and `bfloat16` Safetensors on GPU via Hugging Face `transformers`, paired with **Myers' (1999) bit-parallel exact word-level Levenshtein alignment (`tpst_align`)**, Hungarian inter-chunk permutation reconciliation, and **Locality-Gated Transcript-Preserving Speaker Transfer (`LG-TPST`)**.

> **Archival Note**: All functionality from the experimental [`diarizers-community/DiarizersLM`](https://github.com/diarizers-community/DiarizersLM) repository has been folded into and superseded by `diarizers` (`v1.0.0+`).

`diarizers` (`v1.2.0`) implements and extends the algorithms described in:
- **DiarizationLM**: Quan Wang, Yiling Huang, Guanlong Zhao, Evan Clark, Wei Xia, Hank Liao, *"DiarizationLM: Speaker Diarization Post-Processing with Large Language Models"*, **Interspeech 2024** ([arXiv:2401.03506](https://arxiv.org/abs/2401.03506)).
- **Turn-to-Diarize**: Wei Xia, Han Lu, Quan Wang, Anshuman Tripathi, Yiling Huang, Ignacio Lopez Moreno, Hasim Sak, *"Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection"*, **ICASSP 2022** ([arXiv:2109.11641](https://arxiv.org/abs/2109.11641)).
- **SpectralCluster**: Quan Wang, Carlton Downey, Li Wan, Philip Andrew Mansfield, Ignacio Lopez Moreno, *"Speaker Diarization with LSTM"*, **ICASSP 2018** ([arXiv:1710.10468](https://arxiv.org/abs/1710.10468)).

---

## Configurable Pipeline Profiles (`best-quality`, `max`, `balanced`, `fast`, `acoustic-only`)

Because speaker encoders (`ECAPA` `20.8M` vs. `FlaxSpeaker-v2-Diar` `36.3M` vs. `FlaxSpeaker-XL-Diar-Adapt` `64.4M`, **`10–15×` CPU speed difference**) and DiarizationLM sizes (`Gemma-4-E4B-v2` vs. `Gemma-4-12B-v1`, **`2×` CPU speed and `2×` RAM difference**) span a wide compute-cost spectrum, `diarizers` exposes five readable high-level profiles via `profile=...` (Python) and `--profile ...` (CLI):

| Profile (`--profile` / `profile=`) | Aliases | Stage 2 Acoustic Diarizer (`powerset_ward`) | Stage 4 DiarizationLM | 211-Test Micro / Macro WDER (%) | `SpkCntMAE` | CPU Cost (`Q4_K_M` GGUF) | Best For |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`"max"`** | `fusion`, `gpu` | **`FlaxSpeaker-XL-Diar-Adapt + ECAPA` Fusion** (`w=0.35`, `411 ms/seg` CPU) | **`DiarizationLM-Gemma-4-12B-v1`** | **3.79 / 5.14** *(**7.92%** AMI, **2.55%** Fisher)* | **0.174** | `~14.7 GiB` RAM + dual encoder | **Maximum SOTA accuracy & multi-speaker meetings** |
| **`"best-quality"`** *(default)* | `best`, `quality`, `default`, `flaxspeaker` | **`FlaxSpeaker-XL-Diar-Adapt`** (`64.4M`, segment-CMVN, `386 ms/seg` CPU) | **`DiarizationLM-Gemma-4-12B-v1`** | **3.82 / 5.16** *(**3.53%** CALLHOME, **8.04%** AMI)* | **0.177** | `~14.6 GiB` RAM, `7.5–8.0 tok/s` | **Default single-encoder champion & auto-`K` (`1–10` spk)** |
| **`"balanced"`** | `ecapa-12b`, `cpu-12b` | `ECAPA-TDNN` (`20.8M`, `25.5 ms/seg` CPU — `15×` faster) | **`DiarizationLM-Gemma-4-12B-v1`** | 3.88 / 5.25 *(Raw: 3.90 / 5.18)* | 0.239 | `~14.3 GiB` RAM, fast CPU encoder | **CPU servers (`>= 24 GiB` RAM) wanting `15×` faster encoder** |
| **`"fast"`** | `lite`, `cpu`, `e4b` | `ECAPA-TDNN` (`20.8M`, `25.5 ms/seg` CPU) | **`DiarizationLM-Gemma-4-E4B-v2`** | **4.13 / 5.49** *(20.83% `i18n`)* | 0.239 | **`~7.6 GiB` RAM, `15.4 tok/s` (`2×` faster)** | **16 GiB RAM laptops, CPU servers & Hugging Face CPU Spaces** |
| **`"acoustic-only"`** | `no-llm`, `realtime` | `ECAPA-TDNN` (`20.8M`, `25.5 ms/seg` CPU) | `None` *(disabled)* | 5.65 / 6.96 *(5.59 / 6.82 w/ `xl_diar_adapt`)* | 0.239 | **`< 2.0 GiB` RAM, no LLM latency** | **Ultra-low-resource / real-time CPU transcription** |

---

## Architecture

```text
Audio (WAV / FLAC / MP3 / 16 kHz mono NumPy array)
  ├──> Stage 1: ASR Backend (diarizers.asr)
  │      └── WhisperWordASR (openai/whisper-large-v3-turbo) or AudioCppASR (Qwen3-ASR)
  │            ──> WordSegments [start, end, word]
  │
  └──> Stage 2: Powerset Speaker Diarization Backend (diarizers.diarization, flaxspeaker_encoder & spectral)
         ├── PyanNetSegmentation (pyannote/segmentation-3.0 overlapping 10s powerset clean-speaker masks)
         ├── Speaker Encoder: SpeechBrainECAPATDNN (best-quality/fast) or FlaxSpeakerERes2Conformer (max)
         └── powerset_ward: Ward Auto-K + Cannot-Link Cosine EM + Balanced Centering + Soft Powerset Stitching
               (or legacy diarization_mode="sliding_spectral" Symmetric Normalized Laplacian SpectralClusterer)
               ──> Frame Speaker Posteriors P(t, k) & SpeakerTurns [start, end, speaker]
                     │
                     ▼
       Stage 3: Soft Posterior Alignment & Chunked Prompt Builder (diarizers.alignment)
         ├── assign_speakers_from_probs / assign_speakers (Sub-ms soft posterior integration + midpoint fallback)
         └── ChunkedPromptBuilder (4,000-char or UTF-8 byte windows, Gemma-4 turn formatting)
                     │
                     ▼
       Stage 4: DiarizationLM Post-Processing & LG-TPST (diarizers.llm & diarizers.alignment)
         ├── DiarizationLMRefiner (12B-v1 in 'best-quality'/'max' | E4B-v2 in 'fast')
         │     ├── CPU Auto Backend: llama-cpp-python (Q4_K_M GGUF + flash_attn)
         │     └── GPU Backend: Hugging Face transformers (bfloat16 model.safetensors)
         ├── Hungarian Inter-Chunk Speaker Permutation Reconciliation
         ├── Myers' (1999) Bit-Parallel Exact Word-Level Levenshtein TPST (tpst_align)
         └── Locality-Gated Speaker Transfer (LG-TPST)
                     │
                     ▼
       DiarizedTranscript (Diarized Text, NIST RTTM, SRT, WebVTT, JSON)
```

---

## Why 2024 `DiarizersLM` Had 43.99%–56.79% WDER & How `diarizers` Solves It

In the 2024 experimental `DiarizersLM` prototype (`distil-whisper/distil-large-v3` + `pyannote/speaker-diarization-3.1` + `Llama-2-7b`), Word Diarization Error Rate (WDER) degraded to **43.99% on Fisher** and **56.79% on AMI** (nearly random guessing on 2-speaker telephone calls). A systematic audit identified six compounding root causes, all resolved in `diarizers` (`v1.0.1`):

1. **Open-End Timestamp Crash (`[start, None]`)**: Whisper frequently emits `None` for the end timestamp of the final segment in a 30-second chunk; `DiarizersLM` crashed or dropped trailing words. `diarizers.alignment.assign_speakers` safely bounds open-ended timestamps (`start + 0.1s`).
2. **Silence Fallback Speaker Collapse**: In `DiarizersLM`, any word whose timestamp fell in a brief inter-turn pause with zero overlap defaulted to `turns[0].speaker` (always Speaker 1). `diarizers` assigns non-overlapping words to the **temporally nearest speaker turn** by midpoint distance.
3. **Integer-Second Timestamp Truncation**: `DiarizersLM` cast word and turn timestamps to integer seconds (`round(t)`), erasing sub-second conversational turn boundaries and backchannels (`"yeah"`, `"uh-huh"`). `diarizers` preserves full `float64` sub-millisecond precision throughout alignment and supports arbitrary integer speaker counts (`>= 10` speakers).
4. **8 kHz Telephony Acoustic Collapse, Exact HTK Mel Filterbanks & Neural Powerset Segmentation**:
   - Pretrained `speechbrain/spkrec-ecapa-voxceleb` expects **HTK Mel filterbanks** (`_build_mel_filterbank_np`) with **decibel-scaled (`10 * log10`)** 80-bin energies (`blocks.0` BatchNorm `running_mean ~ 71.85, running_var ~ 8536.5`) and global embedding mean subtraction (`mean_var_norm_emb.ckpt`). Using natural-log (`ln`) filterbanks without `glob_mean` subtraction caused all embeddings to collapse into a narrow cone (`cosine similarity > 0.90`).
   - Furthermore, 8 kHz narrowband telephone audio upsampled to 16 kHz introduces a shared channel offset vector across all windows of a call.
   - `diarizers.diarization.ECAPASpectralDiarizer` combines standalone PyTorch `pyannote/segmentation-3.0` (`PyanNetSegmentation` neural powerset segmentation, MIT License) with `SpeechBrainECAPATDNN` (exact HTK dB filterbank scaling, `glob_mean` subtraction, and **recording-level embedding centering** `embs -= embs.mean(axis=0)`) and clusters via **Symmetric Normalized Laplacian Spectral Clustering** (`L_sym = I - D^{-1/2} W D^{-1/2}`) with Gaussian temporal smoothing and calibrated eigengap speaker count estimation (`eigen_threshold=0.45`). Pure sliding-window `ECAPA-TDNN + SpectralClusterer` (without neural segmentation) also remains available via `segmentation_model="none"`.
5. **Inter-Chunk Speaker Permutation Flips**: Long transcripts segmented into 4,000-character LLM prompts previously suffered independent speaker label permutations (`1 <-> 2`) across chunks. `diarizers.alignment.ChunkedPromptBuilder` defaults to `overlap_words=0` (matching the non-overlapping chunking used during `DiarizationLM-Gemma-4-E4B-v1` SFT training) and supports configurable overlapping word windows (e.g., `overlap_words=60`) with **Hungarian (`scipy.optimize.linear_sum_assignment`) inter-chunk reconciliation**.
6. **LLM Monologue Speaker Drift & Ghost Speaker Hallucinations**: Autoregressive LLMs occasionally invent a spurious `<speaker:3>` in a 2-speaker conversation or flip a speaker label across a 200-word monologue, or fail to emit `[eod]` before repeating text. `diarizers` truncates completions at the first occurrence of `[eod]` / `<turn|>` (`truncate_completion_suffix`) and applies `diarizers.alignment.locality_gated_speaker_transfer` (**`LG-TPST`**), which absorbs ghost clusters (`<= 15` words in `>= 3`-speaker meetings) into dominant speakers, gates long-span monologue flips (`> 5` words in `>= 3`-speaker meetings and `> 15` words in `2`-speaker conversations), and uses **Myers' (1999) bit-parallel exact word-level Levenshtein Transcript-Preserving Speaker Transfer (`TPST`)** so `WER` is mathematically invariant before and after LLM post-processing.

---

## Evaluation Results

All metrics below are **micro-averaged** across the full evaluation splits (`172` Fisher conversations, `20` CALLHOME-English calls, `3` ICSI meetings, and `16` AMI `Mix-Headset` meetings = `211` test recordings) using exact Hungarian-matching dynamic programming (`diarizers.evaluation.compute_corpus_metrics` / `diarizationlm.compute_metrics_on_json_dict`).

*(Note: In accordance with dataset licensing policies, raw LDC/benchmark audio and intermediate decoded transcripts are not redistributed in this repository.)*

### What's New in `v1.2.0`: Diarization-Trained `FlaxSpeaker` Checkpoints, Per-Segment CMVN & Dual-Encoder Fusion

`diarizers` `v1.2.0` integrates the new **diarization-trained [`FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker) `SimAM-ERes2Conformer` models** (`univ_full_eres2conformer_xl_diar_adapt`, `xl_diar_pfas`, `v2_diar`, and `xl_diar`), pushing end-to-end **Micro WDER from `3.88% → 3.79%` (`5.14%` Macro WDER)** and cutting **speaker-count error (`SpkCntMAE`) by `-27%` (`0.239 → 0.174`)**:

1. **Diarization-Adapted `FlaxSpeaker` Checkpoints + Per-Segment CMVN (`flaxspeaker_cmvn_mode="segment"`, `ward_merge_threshold=0.75`)**: Unlike legacy VoxCeleb-only models (`crop_frames="140-200"`), the new `*_diar*` models were trained on variable-length `0.9s–2.2s` conversational turns (`Fisher + AMI + ICSI + CALLHOME`) with per-crop CMVN. Automatically switching `flaxspeaker_cmvn_mode="auto"` to `"segment"` for `*_diar*` checkpoints (`is_flaxspeaker_diar_model_id`) cuts `SpkCntMAE` from `0.261 → 0.177` (`-32%`), improves **unconstrained `K ∈ [1..10]` auto-clustering by `-0.59 pp` WDER (`6.72% → 6.13%`)**, and improves single-encoder `+ 12B-v1` accuracy to **`3.82%` micro / `5.16%` macro WDER** (`FLAXSPEAKER_XL_DIAR_PRESET`, now default in `profile="best-quality"`). Compact `36.3M` `univ_full_eres2conformer_v2_diar` (`FLAXSPEAKER_V2_DIAR_PRESET`, `1.57×` faster on CPU than XL) also reaches **`3.85%` micro / `5.21%` macro WDER** (`3.51%` CALLHOME).
2. **Dual-Encoder Powerset Fusion (`FLAXSPEAKER_DIAR_ECAPA_FUSION_PRESET` in `profile="max"`)**: Concatenating L2-normalized `xl_diar_adapt` (`256-D`, segment-CMVN) and `ECAPA-TDNN` (`192-D`, recording-CMVN) track embeddings ($[\sqrt{1-w}\,e_1,\; \sqrt{w}\,e_2]$, $w=0.35$, `ward_merge_threshold=0.72`) achieves a **statistically significant acoustic improvement over both `ECAPA` (`p = 0.0126` WDER, `p = 0.0020` cpWER) and legacy `FlaxSpeaker-XL` (`p = 0.0042` WDER, `p = 0.0008` cpWER)** while adding only `+6.6%` encoder time (`+25.5 ms/seg`), reaching **`3.79%` micro / `5.14%` macro WDER** and **`7.92%` WDER on AMI `Mix-Headset`** (first open-source config under `8.00%` on AMI).
3. **5-Tier Quality/Speed Profiles (`max`, `best-quality`, `balanced`, `fast`, `acoustic-only`) & Automatic Encoder Calibration (`adapt_preset_for_encoder`)**: Preserves the fast CPU `ECAPA + 12B-v1` recipe as `profile="balanced"` (`15×` faster encoder) alongside `profile="fast"` (`ECAPA + E4B-v2`, `~7.6 GiB` RAM) and `profile="acoustic-only"` (`<2 GiB` RAM), and automatically calibrates CMVN mode and Ward thresholds whenever `--encoder-model` is overridden on the CLI.

### Table 0: End-to-End `diarizers` `v1.2.0` (`powerset_ward`) Benchmark (`211` Held-Out Test Recordings, `bfloat16`)

All speaker counts are estimated **automatically without oracle `K`** (`FISHER` `K=2` telephony preset unless marked `K ∈ [1..10]`, `CALLHOME` `[2..6]`, `ICSI` `[2..10]`, `AMI` `[3..5]`). All `powerset_ward` and fusion hyperparameters were locked on a disjoint 60-recording dev set before blind held-out evaluation.

| Frontend + DiarizationLM (`Mode`) | Fisher (172) WDER / cpWER (%) | CALLHOME (20) WDER / cpWER (%) | ICSI (3) WDER / cpWER (%) | AMI (16) WDER / cpWER (%) | Micro WDER / Macro WDER (%) | SpkCntMAE (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v1.0.1` (`sliding_spectral` + `ECAPA`) Base** | 4.35 / 24.78 | 9.42 / 32.42 | 23.48 / 59.55 | 20.68 / 51.44 | 8.64 / 14.48 | 1.018 |
| `v1.0.1` + `E4B-v1` (`Raw-TPST`) | 2.64 / 21.96 | 4.84 / 25.87 | 23.11 / 59.44 | 20.09 / 50.91 | 7.10 / 12.67 | 1.018 |
| **`ECAPA` (`powerset_ward`, `acoustic-only` / `balanced` / `fast`) Base** | 4.49 / 24.58 | 6.07 / 26.81 | 7.82 / 40.14 | 9.48 / 35.68 | 5.65 / 6.96 | 0.239 |
| `ECAPA` + **`E4B-v2`** (`LG-TPST`, `profile="fast"`) | 2.86 / 22.16 | 4.05 / 24.85 | 6.59 / 39.25 | 8.45 / 34.84 | 4.13 / 5.49 | 0.239 |
| `ECAPA` + **`12B-v1`** (`LG-TPST`, `profile="balanced"`) | 2.61 / 21.85 | 3.75 / 24.30 | 6.50 / 39.19 | 8.11 / 34.44 | 3.88 / 5.25 | 0.239 |
| **Legacy `FlaxSpeaker-XL` (VoxCeleb) Base** | 4.51 / 24.59 | 5.75 / 26.55 | 7.80 / 40.14 | 9.65 / 35.69 | 5.68 / 6.93 | 0.261 |
| Legacy `FlaxSpeaker-XL` + **`12B-v1`** (`LG-TPST`) | 2.65 / 21.87 | 3.57 / 24.33 | **6.39 / 39.08** | 8.19 / 34.43 | 3.90 / 5.20 | 0.249 |
| **`FlaxSpeaker-V2-Diar` (`36.3M`, `_seg`) Base** | 4.44 / 24.50 | **5.49 / 26.38** | 7.90 / 40.23 | 9.53 / 35.76 | 5.61 / 6.84 | 0.190 |
| `FlaxSpeaker-V2-Diar` + **`E4B-v2`** (`LG-TPST`) | 2.81 / 22.08 | 4.09 / 24.84 | 6.73 / 39.44 | 8.48 / 34.87 | 4.12 / 5.53 | 0.190 |
| `FlaxSpeaker-V2-Diar` + **`12B-v1`** (`LG-TPST`) | 2.56 / 21.79 | **3.51** / 24.27 | 6.58 / 39.25 | 8.19 / 34.51 | 3.85 / 5.21 | 0.190 |
| **`FlaxSpeaker-XL-Diar-Adapt` (`64.4M`, `_seg`) Base** | 4.44 / 24.51 | 5.57 / 26.62 | **7.74 / 40.09** | 9.51 / 35.65 | 5.59 / 6.82 | 0.177 |
| `FlaxSpeaker-XL-Diar-Adapt` + **`E4B-v2`** (`LG-TPST`) | 2.84 / 22.11 | **3.88** / 24.75 | **6.58** / 39.25 | 8.41 / 34.77 | 4.10 / 5.43 | 0.177 |
| **`FlaxSpeaker-XL-Diar-Adapt` + `12B-v1` (`LG-TPST`, `profile="best-quality"`)** | 2.56 / **21.78** | 3.53 / 24.21 | 6.50 / 39.17 | 8.04 / 34.37 | **3.82 / 5.16** | 0.177 |
| **`FlaxSpeaker-XL-Diar-PFAS` (`77.1M`, `_seg`) + `12B-v1` (`LG-TPST`)** | 2.56 / **21.78** | 3.53 / 24.21 | 6.50 / 39.17 | 8.04 / 34.37 | **3.81 / 5.16** | 0.177 |
| **`XL-Diar-Adapt + ECAPA` Fusion (`w=0.35`) Base** | **4.43 / 24.47** | 5.66 / 26.57 | 7.85 / 40.17 | **9.28 / 35.42** | **5.56 / 6.81** | **0.174** |
| `XL-Diar-Adapt + ECAPA` Fusion + **`E4B-v2`** (`LG-TPST`) | **2.80 / 22.08** | 4.03 / 24.70 | 6.80 / 39.42 | **8.21 / 34.54** | **4.06 / 5.46** | **0.174** |
| **`XL-Diar-Adapt + ECAPA` Fusion + `12B-v1` (`LG-TPST`, `profile="max"`)** | **2.55 / 21.78** | 3.57 / **24.12** | 6.53 / 39.21 | **7.92 / 34.24** | **3.79 / 5.14** | **0.174** |
| *Unconstrained `K ∈ [1..10]`: `XL-Diar-Adapt` + `12B-v1` (`LG-TPST`)* | *3.05 / 22.15* | *3.53 / 24.21* | *6.50 / 39.17* | *8.04 / 34.37* | *4.17 / 5.28* | *0.216* |

*All `cpWER` values above use the **full all-speaker reference denominator** (`FISHER=323,413`, `CALLHOME=18,957`, `ICSI=33,255`, `AMI=88,826` words, counting unmapped reference speakers as full deletions). Across all 211 test utterances (`10,000`-resample paired cluster bootstrap): **`XL-Diar-Adapt + ECAPA` Fusion** significantly outperforms both `ECAPA` (`ΔWDER = -0.092 pp [-0.169, -0.020], p = 0.0126`; `ΔcpWER = -0.114 pp [-0.200, -0.038], p = 0.0020`) and legacy `FlaxSpeaker-XL` (`ΔWDER = -0.122 pp [-0.216, -0.035], p = 0.0042`; `ΔcpWER = -0.116 pp [-0.188, -0.047], p = 0.0008`), while single-encoder `XL-Diar-Adapt` (`64.4M`) matches `77.1M` `XL-Diar-PFAS` within `0.01 pp` (`3.82%` vs `3.81%`) with `16.4%` fewer parameters and cuts speaker-count `SpkCntMAE` from `0.261 → 0.177` (`-32%`).*

**Multilingual `DiarizationLM` Comparison & CPU `Q4_K_M` Cost**:

| Model (`Apache-2.0`) | `i18n_open_eval` (20 sets, 2,147 utts) Micro / Macro WDER (%) | `Q4_K_M` File Size | Peak CPU RSS (`n_ctx=6144`) | CPU Speed (`32` threads) | Recommended Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Baseline (`v1.0.1` open frontend) | 21.43 / 20.58 | — | — | — | — |
| `DiarizationLM-Gemma-4-E4B-v1` | 21.57 / 20.72 *(regresses non-EN)* | 5.30 GB | `7.59 GiB` | `15.1 tok/s` | Legacy (superseded by `E4B-v2`) |
| **`DiarizationLM-Gemma-4-E4B-v2`** | **20.87 / 20.01** *(LG: **20.83 / 19.98**)* | 5.30 GB | **`7.58 GiB`** | **`15.4 tok/s`** | **`profile="fast"` (16 GiB RAM / CPU / Spaces)** |
| **`DiarizationLM-Gemma-4-12B-v1`** | **20.55 / 19.70** *(wins 19/20 sets)* | 7.38 GB | **`14.29 GiB`** | `7.5–8.0 tok/s` | **`profile="best-quality"` (Default) & `"max"`** |

### Table 1: Open-Source Frontend (`diarizers` v1.0.1 Legacy Baseline) + `DiarizationLM-Gemma-4-E4B-v1`

Open-source pipeline configuration:
- **ASR**: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) (`WhisperWordASR`, MIT License)
- **Speaker Diarization**: [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) (`PyanNetSegmentation`, MIT License) + [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (`SpeechBrainECAPATDNN`, Apache-2.0 License) + `SpectralClusterer` (`eigen_threshold=0.45`, Apache-2.0 License)
- **DiarizationLM**: [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1) (Apache-2.0 License; evaluated in full-precision `bfloat16` on TPU with `max_new_tokens = 1.2x` input tokens and `6,000`-character chunk windows; local CPU inference uses 4-bit `Q4_K_M` GGUF via `llama.cpp` with `max_tokens = 1.3x` prompt tokens and `4,000`-character default chunk windows)

| Benchmark (Full Test Split) | Utterances | Pipeline / System | WER (%) | WDER (%) | cpWER (%) | SpkCntMAE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Fisher English** | 172 | Legacy 2024 `DiarizersLM` (`distil-large-v3` + `pyannote-3.1`) | 20.69 | 43.99 | 56.10 | — |
| **Fisher English** | 172 | `diarizers` v1.0.1 Open Frontend (`FISHER_TELEPHONY_PRESET`, Fixed Speaker Count `K=2`) | **20.80** | **4.35** *(raw 4.45)* | **24.78** *(raw 24.78, Oracle 20.46)* | **0.000** *(fixed K=2)* |
| **Fisher English** | 172 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **20.80** | **2.59** | **21.91** | **0.000** *(fixed K=2)* |
| **Fisher English** | 172 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **20.80** | **2.65** | **21.99** | **0.000** *(fixed K=2)* |
| **CALLHOME English** | 20 | `diarizers` v1.0.1 Open Frontend (`CALLHOME_TELEPHONY_PRESET`, Auto Speaker Count `[2..6]`, `eigen_threshold=0.45`) | **21.25** | **9.42** *(raw 9.47)* | **32.42** *(raw 32.67, Oracle 21.08)* | **0.050** |
| **CALLHOME English** | 20 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **21.25** | **5.98** | **27.69** | **0.050** |
| **CALLHOME English** | 20 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **21.25** | **4.89** | **25.91** | **0.050** |
| **ICSI Meeting Corpus** | 3 | `diarizers` v1.0.1 Open Frontend (`MEETING_MULTISPK_PRESET`, Auto Speaker Count `[2..10]`, `eigen_threshold=0.45`) | **35.29** | **23.48** *(raw 23.58)* | **48.97** *(raw 59.55, Oracle 34.97)* | **3.333** |
| **ICSI Meeting Corpus** | 3 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **35.29** | **23.10** | **48.62** | **3.333** |
| **ICSI Meeting Corpus** | 3 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **35.29** | **23.11** | **48.63** | **3.333** |
| **AMI (`Mix-Headset`)** | 16 | Legacy 2024 `DiarizersLM` (`distil-large-v3` + `pyannote-3.1`) | — | 56.79 | — | — |
| **AMI (`Mix-Headset`)** | 16 | `diarizers` v1.0.1 Open Frontend (`AMI_MEETING_PRESET`, Auto Speaker Count `[3..5]`, `eigen_threshold=0.45`) | **29.14** | **20.68** *(raw 20.80)* | **46.07** *(raw 51.39, Oracle 28.70)* | **0.688** |
| **AMI (`Mix-Headset`)** | 16 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **29.14** | **20.14** | **45.56** | **0.688** |
| **AMI (`Mix-Headset`)** | 16 | `diarizers` v1.0.1 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **29.14** | **20.13** | **45.53** | **0.688** |

> **Note on Pre-Filter `*(raw ...)*` vs. Ghost-Filtered Open-Frontend Baseline**: In Table 1, the primary Open-Frontend `WDER` and `cpWER` columns report the Stage 3 acoustic diarization baseline after deterministic ghost-cluster absorption (`apply_locality_gated_speaker_transfer(hyp_spk, hyp_spk, max_ghost_cluster_words=15)`), while parenthetical `*(raw ...)*` values report the un-filtered Stage 3 `assign_speakers` output prior to ghost-cluster absorption. On multi-speaker meetings (`ICSI`, `AMI`), absorbing tiny spurious acoustic clusters (`<= 15` words) reduces `cpWER` by `-10.58` percentage points on ICSI (`59.55% -> 48.97%`) and `-5.32` percentage points on AMI (`51.39% -> 46.07%`) before any LLM inference, isolating the genuine lexical turn-boundary gains of `DiarizationLM-Gemma-4-E4B-v1` (`23.48% -> 23.10%` WDER on ICSI; `20.68% -> 20.13%` WDER on AMI).

### Decomposed Stage-by-Stage Ablation, Model Reuse & Multi-Party Meeting Analysis

#### 1. Stage-by-Stage Decomposition & Continual Mixed-Frontend SFT Ablation (`DiarizationLM-diarizers-Gemma-4-E4B-v1`)

To separate the contribution of (a) the repaired Stage 1–3 acoustic frontend (`assign_speakers`), (b) deterministic Stage 3 ghost-cluster absorption (`<= 15` words), (c) Stage 4 `DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST` vs. `LG-TPST`), and (d) continual LoRA SFT on the open-source frontend (`DiarizationLM-diarizers-Gemma-4-E4B-v1`, trained on a 50/50 mixture of open-frontend and USM/Turn-to-Diarize locality-preserving SFT pairs), the table below reports all stages side-by-side across the full 211-utterance benchmark suite:

| Stage / System | Fisher (172) WDER / cpWER (%) | CALLHOME (20) WDER / cpWER (%) | ICSI (3) WDER / cpWER (%) | AMI (16) WDER / cpWER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0. Legacy 2024 `DiarizersLM` Prototype** | 43.99 / 56.10 | — / — | — / — | 56.79 / — |
| **1. Stage 3 Raw Acoustic Alignment (`assign_speakers`)** | 4.45 / 24.78 | 9.47 / 32.67 | 23.58 / 59.55 | 20.80 / 51.39 |
| **2. Stage 3 + Ghost-Cluster Absorption (`<= 15` words)** | 4.35 / 24.78 | 9.42 / 32.42 | 23.48 / 48.97 | 20.68 / 46.07 |
| **3. Stage 4 `DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | 2.65 / 21.99 | **4.89 / 25.91** | 23.11 / 48.63 | **20.13 / 45.53** |
| **4. Stage 4 `DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **2.59 / 21.91** | 5.98 / 27.69 | **23.10 / 48.62** | 20.14 / 45.56 |
| **5. Continual Mixed-Frontend SFT (`DiarizationLM-diarizers-Gemma-4-E4B-v1`, `LG-TPST`)** | 2.61 / 21.93 *(Raw: 2.64 / 21.93)* | 5.98 / 27.75 *(Raw: 4.89 / 25.96)* | 23.12 / 48.66 *(Raw: 23.12 / 48.65)* | 20.32 / 45.66 *(Raw: 20.30 / 45.64)* |
| **Hungarian TPST Oracle Ceiling (`hyp_spk_oracle`)** | *0.01 / 20.46* | *0.00 / 21.08* | *0.00 / 34.97* | *0.00 / 28.70* |

#### 2. Justification for Reusing `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (No Duplicate Model Release)

Comparing Row 4 (`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`) against Row 5 (`DiarizationLM-diarizers-Gemma-4-E4B-v1`, continual mixed-frontend SFT) shows that both checkpoints achieve **virtually identical performance across all 4 benchmarks** (`2.59%` vs. `2.61%` WDER on Fisher, `4.89%` vs. `4.89%` `Raw-TPST` and `5.98%` vs. `5.98%` `LG-TPST` on CALLHOME, `23.10%` vs. `23.12%` on ICSI, and `20.13%` vs. `20.30%` on AMI). Because `DiarizationLM-Gemma-4-E4B-v1` was already trained with Locality-Preserving Oracle Supervision across all four corpora (51,063 Fisher + 20,762 Callhome/ICSI/AMI prompt-completion pairs) and operates purely on normalized text + `<speaker:k>` tokens, it transfers zero-shot to the `diarizers` open-source frontend without requiring a separate 16 GB checkpoint release.

#### 3. Why Multi-Party Meeting (`ICSI`, `AMI`) LLM Gains Are Smaller Than Telephony (`Fisher`, `CALLHOME`)

While `DiarizationLM-Gemma-4-E4B-v1` reduces WDER by `-1.76` percentage points (`-40.5%` relative) on Fisher and `-4.53` percentage points (`-48.1%` relative) on CALLHOME, post-LLM WDER reductions on tabletop multi-party meetings are more modest (`-0.38` percentage points on ICSI, `-0.55` percentage points on AMI; and `-0.60` / `-0.79` percentage points on the USM + Turn-to-Diarize frontend in Table 2). Three structural factors explain this regime difference:
1. **Higher Far-Field ASR Word Error Rate & Overlapped Speech**: Tabletop meeting recordings exhibit higher ASR WER (`29.14%` on AMI, `35.29%` on ICSI vs. `20.80%–21.25%` on telephony) and frequent simultaneous overlapped speech, which degrades lexical turn-boundary cues.
2. **Global Speaker Count Estimation Error (`SpkCntMAE`)**: Unbounded spectral clustering on 4–9 speaker meetings exhibits `SpkCntMAE` of `3.333` on ICSI and `0.688` on AMI (compared to `0.000` on Fisher and `0.050` on CALLHOME). Because DiarizationLM is a local lexical boundary refiner rather than a global acoustic re-clusterer, global cluster merges/splits bound text-only post-processing.
3. **Multi-Party Turn-Target Ambiguity ($K > 2$)**: In a 2-speaker call, identifying a lexical turn boundary uniquely determines the target speaker (`1 -> 2`). In a 4–9 speaker meeting, even when syntax clearly indicates a speaker change, determining *which* of the other $K - 1$ participants spoke next is often lexically ambiguous without acoustic re-scoring.

### Table 2: USM + Turn-to-Diarize Frontend + `DiarizationLM-Gemma-4-E4B-v1`

For reference, below is the performance of [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1) on the exact same 211 test recordings when paired with the USM ASR and Turn-to-Diarize frontend (with 95% bootstrap confidence intervals over `B = 10,000` conversation resamples; note that `0/10,000` bootstrap resamples had `ΔWDER >= 0`, and ICSI has `n = 3` meetings):

| Benchmark | Test Split | WER (%) | System | WDER (%) [95% CI] | cpWER (%) [95% CI] | SpkCntMAE [95% CI] | Paired ΔWDER vs. Baseline |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Fisher** | TEST FULL (172 sessions) | 15.37 | Baseline (USM + Turn-to-Diarize)<br>`DiarizationLM-8b-Fisher-v2` (Llama 3 8B)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 5.32 [4.93, 5.74]<br>3.28<br>**2.99 [2.65, 3.37]** | 20.88 [19.97, 21.88]<br>18.37<br>**17.62 [16.76, 18.56]** | 0.215 [0.151, 0.291]<br>—<br>**0.093 [0.047, 0.145]** | —<br>—<br>**-2.33% [-2.51, -2.16]** (0/10,000 ≥ 0) |
| **CALLHOME** | TEST FULL (20 calls) | 15.22 | Baseline (USM + Turn-to-Diarize)<br>`DiarizationLM-8b-Fisher-v2` (Llama 3 8B)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 7.74 [6.07, 9.65]<br>6.66<br>**4.92 [3.46, 6.75]** | 24.31 [21.27, 27.43]<br>23.57<br>**20.69 [17.98, 23.66]** | 0.050 [0.000, 0.150]<br>—<br>**0.000 [0.000, 0.000]** | —<br>—<br>**-2.82% [-3.48, -2.14]** (0/10,000 ≥ 0) |
| **ICSI** | TEST FULL (3 meetings) | 29.42 | Baseline (USM + Turn-to-Diarize)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 14.70 [11.65, 20.29]<br>**14.10 [10.77, 19.94]** | 43.90 [39.10, 51.67]<br>**43.32 [38.38, 51.20]** | 0.333 [0.000, 1.000]<br>**0.333 [0.000, 1.000]** | —<br>**-0.60% [-0.88, -0.35]** (n = 3) |
| **AMI** | TEST WORD FULL (16 meetings) | 24.33 | Baseline (USM + Turn-to-Diarize)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 15.68 [10.64, 21.11]<br>**14.89 [9.80, 20.38]** | 40.32 [32.43, 48.00]<br>**39.57 [31.57, 47.29]** | 0.500 [0.188, 0.812]<br>**0.438 [0.188, 0.750]** | —<br>**-0.79% [-1.00, -0.60]** (0/10,000 ≥ 0) |

---

## Installation

### Install from PyPI

```bash
# Core package (ASR + ECAPA-TDNN + PyanNetSegmentation + SpectralClusterer + TPST/LG-TPST + Metrics)
pip install diarizers

# With optional GGUF CPU LLM inference & LoRA fine-tuning dependencies
pip install "diarizers[audio,llm]"
```

### Install from Source

```bash
git clone https://github.com/diarizers-community/diarizers.git
cd diarizers
pip install -e ".[audio,llm,dev]"
```

---

## How to Run the Pipeline

### 1. One-Liner End-to-End Python API (`DiarizersPipeline.from_pretrained`)

Run the full 4-stage pipeline on any WAV/MP3/FLAC file or 16 kHz NumPy waveform using a high-level `profile` (`"best-quality"`, `"max"`, `"balanced"`, `"fast"`, or `"acoustic-only"`):

```python
import diarizers

# 1. Default ("best-quality"): Whisper-large-v3-turbo + FlaxSpeaker-XL-Diar-Adapt + Gemma-4-12B-v1
#    On CPU, backend="auto" automatically uses the quantized Q4_K_M GGUF when llama-cpp-python is installed!
pipe = diarizers.DiarizersPipeline.from_pretrained()

# 2. Or select a quality / speed / hardware profile:
#    - profile="max":           FlaxSpeaker-XL-Diar-Adapt + ECAPA fusion + Gemma-4-12B-v1 (3.79% WDER SOTA)
#    - profile="balanced":      15x faster ECAPA encoder + Gemma-4-12B-v1 (3.88% WDER)
#    - profile="fast":          2x faster LLM & ~7.6 GiB RAM (ECAPA + Gemma-4-E4B-v2, 4.13% WDER)
#    - profile="acoustic-only": ECAPA powerset_ward without LLM (<2 GiB RAM)
pipe_fast = diarizers.DiarizersPipeline.from_pretrained(profile="fast")

# Pass num_speakers=2 if known, or num_speakers=None to auto-detect (1..10 speakers)
transcript = pipe("conversation.wav", num_speakers=None)

# Export in multiple standard formats:
print(transcript.to_diarized_text())                  # "<speaker:1> hello ... <speaker:2> hi ..."
print(transcript.to_rttm(recording_id="session_01"))  # NIST RTTM format
print(transcript.to_srt())                            # SubRip (.srt) subtitles with [Speaker k]
print(transcript.to_vtt())                            # WebVTT (.vtt) subtitles
```

To run on a **CUDA GPU** with 16-bit `bfloat16` Hugging Face `transformers` weights *(requires `transformers>=5.0.0` for `gemma4` architecture support)*:

```python
pipe_gpu = diarizers.DiarizersPipeline.from_pretrained(
    profile="max",  # FlaxSpeaker-XL-Diar-Adapt + ECAPA Fusion + DiarizationLM-Gemma-4-12B-v1
    device="cuda:0",
)
transcript = pipe_gpu("callhome_sample.wav", num_speakers=None)
```

### 2. Modular End-to-End Python API (`DiarizationPipeline`)

You can also customize each stage independently (for example, choosing among `UNIVERSAL_AUTO_PRESET`, `FLAXSPEAKER_XL_PRESET`, `FLAXSPEAKER_LARGE_PRESET`, `FISHER_TELEPHONY_PRESET`, `CALLHOME_TELEPHONY_PRESET`, `AMI_MEETING_PRESET`, `MEETING_MULTISPK_PRESET`, or setting `diarization_mode="sliding_spectral"`):

```python
from diarizers import (
    DiarizationLMRefiner,
    DiarizationPipeline,
    DiarizationPreset,
    ECAPASpectralDiarizer,
    WhisperWordASR,
)

preset = DiarizationPreset(
    name="custom_meeting",
    segmentation_model="pyannote/segmentation-3.0",
    encoder_name="speechbrain/spkrec-ecapa-voxceleb",  # Or "wq2012/univ_full_eres2conformer_xl"
    diarization_mode="powerset_ward",
    min_clusters=2,
    max_clusters=8,
)

pipeline = DiarizationPipeline(
    asr=WhisperWordASR("openai/whisper-large-v3-turbo", device="cpu"),
    diarizer=ECAPASpectralDiarizer(preset=preset, device="cpu"),
    llm_refiner=DiarizationLMRefiner(
        model_path_or_repo="diarizers-community/DiarizationLM-Gemma-4-12B-v1",
        backend="auto",  # Auto-selects Q4_K_M GGUF on CPU when llama-cpp-python is installed
    ),
    use_locality_gate=True,
)

result = pipeline("meeting.wav")
print(result.to_diarized_text())
```

### 3. Post-Processing Any Diarized Transcript with `DiarizationLM`

If you already have a diarized transcript from any ASR + diarization system (`<speaker:1> ... <speaker:2> ...`), you can refine its speaker labels directly using `DiarizationLMRefiner` (`12B-v1` or `E4B-v2`) and `locality_gated_speaker_transfer` (`LG-TPST`) or `transcript_preserving_speaker_transfer` (`TPST`):

```python
from diarizers import (
    DiarizationLMRefiner,
    create_diarized_text,
    extract_text_and_spk,
)

hyp_diarized = (
    "<speaker:1> hello how are you doing <speaker:2> today i am doing great "
    "thanks what about <speaker:1> you i am doing well too"
)

refiner = DiarizationLMRefiner(
    model_path_or_repo="diarizers-community/DiarizationLM-Gemma-4-12B-v1",  # Or E4B-v2 for fast CPU
    backend="auto",  # Uses Q4_K_M GGUF on CPU or bfloat16 transformers on GPU
)

# High-level helper returning (raw_tpst_spk, lg_tpst_spk, raw_completions):
hyp_text, hyp_spk = extract_text_and_spk(hyp_diarized)
raw_tpst_spk, lg_tpst_spk, _ = refiner.refine_utterance(hyp_text, hyp_spk)

print("Refined (LG-TPST):", create_diarized_text(hyp_text.split(), lg_tpst_spk.split()))
print("Refined (Raw-TPST):", create_diarized_text(hyp_text.split(), raw_tpst_spk.split()))
```

### 4. Command-Line Interface (`diarizers` CLI & `scripts/`)

Installing `diarizers` registers the `diarizers` console script with five subcommands (`run`/`infer`, `prepare-sft-data`/`prepare-data`, `train`, `export`, `evaluate`):

```bash
# 1a. Best-quality default (Whisper-large-v3-turbo + ECAPA powerset_ward + Gemma-4-12B-v1):
diarizers run \
  --audio conversation.wav \
  --output-json output_transcript.json \
  --output-rttm output_transcript.rttm \
  --output-srt output_transcript.srt

# 1b. Fast / 16 GiB CPU laptop profile (ECAPA powerset_ward + Gemma-4-E4B-v2, ~2x faster LLM):
diarizers run --audio conversation.wav --profile fast --output-srt output_fast.srt

# 1c. Max multi-speaker GPU profile (FlaxSpeaker-XL + Gemma-4-12B-v1):
diarizers run --audio meeting.wav --profile max --device cuda:0 --output-rttm meeting.rttm

# 1d. Acoustic-only profile (<2 GiB RAM, no LLM):
diarizers run --audio conversation.wav --profile acoustic-only --output-rttm acoustic.rttm

# 2. Prepare SFT training prompts (hyp2ora + deg2ref + Locality-Preserving Oracle):
diarizers prepare-sft-data \
  --input-json decoded_train_utterances.json \
  --output-json sft_train_prompts.json \
  --flavor both

# 3. Fine-tune DiarizationLM (Gemma-4-E4B) with PEFT LoRA:
diarizers train \
  --train-json sft_train_prompts.json \
  --model-name-or-path google/gemma-4-E4B \
  --output-dir ./lora_ckpt

# 4. Merge LoRA Safetensors and optionally quantize Q4_K_M GGUF:
diarizers export \
  --model-dir ./lora_ckpt \
  --output-dir ./merged_hf \
  --base-model google/gemma-4-E4B \
  --output-gguf model-q4_k_m.gguf

# 5. Evaluate micro WER, WDER, cpWER, and SpkCntMAE with 95% bootstrap CIs:
diarizers evaluate \
  --input-json decoded_test_utterances.json \
  --bootstrap-samples 1000 \
  --output-json eval_metrics.json
```

Standalone scripts are also provided in [`scripts/`](scripts/):
- `python scripts/inference.py --audio conversation.wav --profile best-quality --output_json out.json`
- `python scripts/inference.py --audio conversation.wav --profile fast --output_srt out_fast.srt`
- `python scripts/prepare_data.py --input_json train_utts.json --output_json sft_train.json`
- `python scripts/train.py --train_json sft_train.json --output_dir ./lora_ckpt`
- `python scripts/export_model.py --model_dir ./lora_ckpt --output_dir ./merged_hf --base_model google/gemma-4-E4B --output_gguf model-q4_k_m.gguf`
- `python scripts/evaluate.py --input_json test_utts.json --bootstrap_samples 1000`

### 5. Training & Exporting Custom DiarizationLM Models

You can fine-tune your own DiarizationLM model on custom domain data in three steps:

1. **Construct SFT Training Pairs (`diarizers.dataprep`)**:
   Given a JSON file of utterances containing `hyp_text`, `hyp_spk`, `ref_text`, and `ref_spk`, `build_sft_dataset` automatically computes:
   - `hyp_spk_oracle`: Exact Hungarian-optimal speaker labels transferred from `ref` onto `hyp` via Myers' bit-parallel Levenshtein (`TPST`).
   - `hyp_spk_locality_oracle`: Locality-preserving oracle targets that teach the LLM to fix short turn-boundary and backchannel errors (`1 <= L <= 5` words) while preserving acoustic speaker anchors on long monologues (`L >= 6` words).
   - `ref_spk_degraded`: Synthetically degraded speaker boundaries on `ref_text` (`deg2ref` flavor) for data augmentation.
2. **LoRA Supervised Fine-Tuning (`diarizers.train` / `scripts/train.py`)**:
   Fine-tunes `google/gemma-4-E4B` or `google/gemma-4-12B` using completion-only cross-entropy loss on 4,000-character prompt chunks formatted with Gemma-4 turn tags (`<|turn>user\n{prompt} --> <turn|>\n<|turn>model\n{completion} [eod]`).
3. **Merge & Quantize to GGUF (`diarizers.export` / `scripts/export_model.py`)**:
   Merges LoRA adapters into 16-bit `bfloat16` Safetensors and optionally invokes `llama.cpp` to export a 4-bit `Q4_K_M` GGUF file for fast CPU inference.

### 6. Evaluating WER, WDER, cpWER, and SpkCntMAE

```python
from diarizers import compute_corpus_metrics, compute_utterance_metrics

utterances = [
    {
        "utterance_id": "call_01",
        "hyp_text": "hello good morning how are you",
        "hyp_spk": "1 1 1 2 2 2",
        "hyp_spk_llm": "1 1 1 2 2 2",
        "ref_text": "hello good morning how are you",
        "ref_spk": "1 1 1 2 2 2",
    }
]

metrics = compute_corpus_metrics(
    utterances,
    hyp_spk_field="hyp_spk_llm",
    bootstrap_samples=1000,
)
print(f"WER: {metrics['WER'] * 100:.2f}%")
print(f"WDER: {metrics['WDER'] * 100:.2f}%")
print(f"cpWER: {metrics['cpWER'] * 100:.2f}%")
print(f"SpkCntMAE: {metrics['SpkCntMAE']:.3f}")
```

### 7. Interactive CPU Demos (`hf_space/app.py` & `hf_space/index.html`)

Two hosted interactive Spaces are available on Hugging Face:
- **End-to-End Audio Pipeline Demo**: [`https://huggingface.co/spaces/diarizers-community/diarizers`](https://huggingface.co/spaces/diarizers-community/diarizers) — Upload or select multi-speaker conversation audio, switch between **`Best Quality` (`12B-v1`)**, **`Max` (`FlaxSpeaker-XL + 12B-v1`)**, **`Fast` (`E4B-v2`)**, and **`Acoustic Only`**, and inspect the acoustic baseline, refined transcript, NIST RTTM, and SRT subtitles.
- **Text-Only DiarizationLM GGUF Demo**: [`https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF`](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF) — Paste any multilingual diarized transcript (`<speaker:1> ... <speaker:2> ...`), select `Fast (Gemma-4-E4B-v2)` or `Best Quality (Gemma-4-12B-v1)` (`Q4_K_M` GGUF), and stream speaker boundary refinement on CPU.

To launch the end-to-end audio Gradio demo locally on CPU:

```bash
pip install gradio
python hf_space/app.py
```

---

## Data Format

All training and evaluation scripts use the standard DiarizationLM JSON format (`{"utterances": [...]}`), fully compatible with [`diarizationlm`](https://github.com/google/speaker-id/tree/master/DiarizationLM):

| Field | Description |
| :--- | :--- |
| `"utterance_id"` | Unique recording or session identifier string. |
| `"hyp_text"` | Space-joined sequence of normalized ASR hypothesis words. |
| `"hyp_spk"` | Space-joined sequence of 1-indexed integer speaker IDs for each word in `hyp_text`. |
| `"ref_text"` | Space-joined sequence of normalized ground-truth reference words. |
| `"ref_spk"` | Space-joined sequence of 1-indexed integer speaker IDs for each word in `ref_text`. |
| `"hyp_spk_oracle"` | *(Optional / Auto-computed)* Oracle speaker sequence for `hyp_text` computed via `tpst_align(ref_text, ref_spk, hyp_text, hyp_spk)`. |
| `"ref_spk_degraded"` | *(Optional / Auto-computed)* Synthetically boundary-degraded speaker sequence for `ref_text` used for `deg2ref` training augmentation. |

---

## Repository Structure

```text
diarizers/
├── src/diarizers/
│   ├── __init__.py          # Public API exports (v1.2.0)
│   ├── types.py             # WordSegment, SpeakerTurn, DiarizedTranscript (RTTM/VTT/SRT/JSON)
│   ├── configs.py           # PromptConfig and acoustic DiarizationPresets (incl. FlaxSpeaker & auto)
│   ├── spectral.py          # SpectralClusterer, Ward auto-K, balanced centering & cosine EM
│   ├── flaxspeaker_encoder.py # Pure-PyTorch FlaxSpeaker SimAM-ERes2Conformer + Hybrid-PFAS encoder
│   ├── diarization.py       # PyanNetSegmentation, SpeechBrainECAPATDNN, powerset_ward & ECAPASpectralDiarizer
│   ├── asr.py               # WhisperWordASR (default openai/whisper-large-v3-turbo), AudioCppASR (Qwen3-ASR GGUF), PrecomputedASR
│   ├── alignment.py         # assign_speakers, Myers bit-parallel TPST, LG-TPST, ChunkedPromptBuilder
│   ├── dataprep.py          # Dual-flavor (hyp2ora + deg2ref) + Locality-Preserving SFT builder
│   ├── llm.py               # DiarizationLMRefiner (llama.cpp GGUF & HF Transformers)
│   ├── pipeline.py          # End-to-end DiarizationPipeline & DiarizersPipeline orchestrators
│   ├── evaluation.py        # Exact WER, WDER, cpWER, SpkCntMAE & paired bootstrap 95% CIs
│   ├── train.py             # LoRA / SFT training configuration & runner
│   ├── export.py            # Safetensors merge & llama.cpp Q4_K_M GGUF export
│   └── cli.py               # `diarizers` CLI entrypoint
├── scripts/
│   ├── prepare_data.py      # SFT dataset preparation script
│   ├── train.py             # PEFT LoRA fine-tuning script for Gemma-4-E4B
│   ├── export_model.py      # LoRA merge & GGUF quantization script
│   ├── evaluate.py          # Benchmark evaluation CLI
│   └── inference.py         # Standalone end-to-end inference CLI
├── hf_space/
│   ├── app.py               # Interactive CPU Gradio end-to-end audio demo
│   ├── index.html           # Interactive browser-executable end-to-end audio demo
│   ├── examples/            # Curated multi-speaker conversation examples & metadata
│   ├── README.md            # Hugging Face Space metadata & documentation
│   └── requirements.txt     # CPU Space dependencies
├── docs/
│   └── blog_post.md         # Technical deep-dive blog post on diarizers v1.2.0 & Gemma-4
└── tests/
    └── test_diarizers.py    # Comprehensive unit & regression test suite
```

---

## Pretrained Models on Hugging Face

- **DiarizationLM (Gemma 4, Multilingual, Safetensors + `Q4_K_M` GGUF, Apache-2.0)**:
  - **Default `best-quality` / `max` / `balanced` (`14.3 GiB` CPU RAM / GPU `bfloat16`, `3.79%–3.82%` WDER)**: [`diarizers-community/DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1)
  - **Fast CPU `fast` profile (`7.6 GiB` CPU RAM, `2×` faster, `4.13%` WDER)**: [`diarizers-community/DiarizationLM-Gemma-4-E4B-v2`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v2)
  - Legacy v1 (`en`-only): [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)
- **Speaker Encoders (Standalone PyTorch, Apache-2.0)**:
  - **Default `best-quality` & `max` diarization-adapted encoder (`64.4M`, segment-CMVN)**: [`wq2012/univ_full_eres2conformer_xl_diar_adapt`](https://huggingface.co/wq2012/univ_full_eres2conformer_xl_diar_adapt)
  - **Compact diarization encoder (`36.3M`, `1.57×` faster on CPU than XL)**: [`wq2012/univ_full_eres2conformer_v2_diar`](https://huggingface.co/wq2012/univ_full_eres2conformer_v2_diar)
  - **Additional [`FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker) `SimAM-ERes2Conformer` checkpoints**: `wq2012/univ_full_eres2conformer_xl_diar_pfas`, `wq2012/univ_full_eres2conformer_xl_diar`, `wq2012/univ_full_eres2conformer_xl`, `wq2012/univ_full_eres2conformer_v2_large`, `wq2012/univ_full_eres2conformer_large`
  - **Fast CPU encoder (`balanced`, `fast`, `acoustic-only` & `max` fusion)**: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Fine-Tuned `pyannote/segmentation-3.0` (`PyanNet`) Checkpoints (MIT)**:
  - [`diarizers-community/speaker-segmentation-fine-tuned-callhome-eng`](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-eng)
  - [`diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn`](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn)
  - [`diarizers-community/speaker-segmentation-fine-tuned-callhome-deu`](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-deu)
  - [`diarizers-community/speaker-segmentation-fine-tuned-callhome-spa`](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-spa)
  - [`diarizers-community/speaker-segmentation-fine-tuned-callhome-zho`](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-zho)

---

## Running Tests & Linter

```bash
bash run_tests.sh
```

---

## Citation

If you found this repository useful, please consider citing this work, the **DiarizationLM** paper, and the underlying segmentation and clustering papers:

```bibtex
@misc{akesbi-diarizers,
  author = {Kamil Akesbi and Quan Wang and Sanchit Gandhi},
  title = {"Diarizers: A repository for speaker diarization and DiarizationLM post-processing"},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/diarizers-community/diarizers}}
}

@inproceedings{wang24h_interspeech,
  title     = {{DiarizationLM: Speaker Diarization Post-Processing with Large Language Models}},
  author    = {Quan Wang and Yiling Huang and Guanlong Zhao and Evan Clark and Wei Xia and Hank Liao},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {3754--3758},
  doi       = {10.21437/Interspeech.2024-209},
}

@inproceedings{bredin-pyannote,
  author    = {Hervé Bredin},
  title     = {"pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe"},
  year      = {2023},
  booktitle = {Proc. INTERSPEECH 2023},
}

@inproceedings{xia2022turn,
  title     = {{Turn-to-Diarize}: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection},
  author    = {Wei Xia and Han Lu and Quan Wang and Anshuman Tripathi and Yiling Huang and Ignacio Lopez Moreno and Hasim Sak},
  booktitle = {Proc. ICASSP 2022},
  pages     = {8077--8081},
  year      = {2022},
}

@inproceedings{wang2018speaker,
  title     = {Speaker Diarization with {LSTM}},
  author    = {Quan Wang and Carlton Downey and Li Wan and Philip Andrew Mansfield and Ignacio Lopez Moreno},
  booktitle = {Proc. ICASSP 2018},
  pages     = {5239--5243},
  year      = {2018},
}
```
