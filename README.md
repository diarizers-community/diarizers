# 🤗 Diarizers (`v1.0.0`) — Unified Open-Source ASR, Speaker Diarization, and DiarizationLM Post-Processing

[![Python package](https://github.com/diarizers-community/diarizers/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/diarizers-community/diarizers/actions/workflows/pythonapp.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Version](https://img.shields.io/pypi/v/diarizers.svg)](https://pypi.org/project/diarizers/)
[![Python Versions](https://img.shields.io/pypi/pyversions/diarizers.svg)](https://pypi.org/project/diarizers/)
[![Downloads](https://static.pepy.tech/badge/diarizers)](https://www.pepy.tech/projects/diarizers)
[![arxiv](https://img.shields.io/badge/arXiv-2401.03506-blue.svg)](https://arxiv.org/abs/2401.03506)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-DiarizationLM--Gemma--4--E4B--v1-blue.svg)](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)
[![HuggingFace Space](https://img.shields.io/badge/Online-Demo-blue.svg)](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF)

## Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Why 2024 `DiarizersLM` Had 43.99%–56.79% WDER & How `diarizers` (`v1.0.0`) Solves It](#why-2024-diarizerslm-had-43995679-wder--how-diarizers-v100-solves-it)
* [Evaluation Results](#evaluation-results)
  * [Table 1: Open-Source Frontend (`diarizers` v1.0.0) + `DiarizationLM-Gemma-4-E4B-v1`](#table-1-open-source-frontend-diarizers-v100--diarizationlm-gemma-4-e4b-v1)
  * [Table 2: USM + Turn-to-Diarize Frontend + `DiarizationLM-Gemma-4-E4B-v1`](#table-2-usm--turn-to-diarize-frontend--diarizationlm-gemma-4-e4b-v1)
* [Installation](#installation)
* [How to Run the Pipeline](#how-to-run-the-pipeline)
  * [1. One-Liner End-to-End Python API (`DiarizersPipeline.from_pretrained`)](#1-one-liner-end-to-end-python-api-diarizerspipelinefrom_pretrained)
  * [2. Modular End-to-End Python API (`DiarizationPipeline`)](#2-modular-end-to-end-python-api-diarizationpipeline)
  * [3. Post-Processing Any Diarized Transcript with `DiarizationLM-Gemma-4-E4B-v1`](#3-post-processing-any-diarized-transcript-with-diarizationlm-gemma-4-e4b-v1)
  * [4. Command-Line Interface (`diarizers` CLI & `scripts/`)](#4-command-line-interface-diarizers-cli--scripts)
  * [5. Training & Exporting Custom DiarizationLM Models](#5-training--exporting-custom-diarizationlm-models)
  * [6. Evaluating WER, WDER, cpWER, and SpkCntMAE](#6-evaluating-wer-wder-cpwer-and-spkcntmae)
  * [7. Interactive CPU Gradio Demo (`hf_space/app.py`)](#7-interactive-cpu-gradio-demo-hf_spaceapppy)
* [Data Format](#data-format)
* [Repository Structure](#repository-structure)
* [Pretrained Models on Hugging Face](#pretrained-models-on-hugging-face)
* [Running Tests & Linter](#running-tests--linter)
* [Citation](#citation)

---

## Overview

**`diarizers` (`v1.0.0`)** is a unified, self-contained **Apache-2.0** Python library and CLI for end-to-end **speaker-attributed automatic speech recognition (ASR + Speaker Diarization + DiarizationLM Post-Processing)**, as well as training and evaluating custom DiarizationLM models.

It combines:
1. **Word-Timestamped Open-Source ASR (`diarizers.asr`)**: Defaulting to [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT License) with cross-attention DTW word-level timestamps (and also supporting `Qwen/Qwen3-ASR` GGUF via `audio.cpp`, Apache-2.0 License, or precomputed word segments).
2. **Neural Speaker Turn Segmentation + Speaker Embeddings + Spectral Clustering (`diarizers.diarization` & `diarizers.spectral`)**:
   - Standalone pure-PyTorch implementation of [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) (`PyanNetSegmentation`: SincNet + 4-layer BiLSTM + 7-class powerset classifier, MIT License, with automatic fallback to un-gated MIT mirrors such as [`diarizers-community/speaker-segmentation-fine-tuned-callhome-eng`](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-eng)) for local speaker-change flux boundary snapping.
   - Standalone pure-PyTorch implementation of [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (`SpeechBrainECAPATDNN`, Apache-2.0 License) with exact HTK decibel-scaled (`10 * log10`) 80-bin log-mel filterbanks, global embedding mean subtraction (`mean_var_norm_emb.ckpt`), and recording-level channel centering.
   - Clean-room **Symmetric Normalized Laplacian Spectral Clusterer** (`L_sym = I - D^{-1/2} W D^{-1/2}`, Apache-2.0 License) with Gaussian temporal smoothing, percentile thresholding, and calibrated eigengap speaker count estimation (`eigen_threshold=0.45`).
3. **Sub-Millisecond Word-to-Turn Alignment & Chunked Prompting (`diarizers.alignment`)**: Exact float64 word-to-turn overlap orchestration (`assign_speakers`), nearest-turn midpoint fallback, first-appearance speaker renumbering, and 4,000-character chunked prompt building (`ChunkedPromptBuilder`).
4. **DiarizationLM Post-Processing & Locality-Gated Transcript-Preserving Speaker Transfer (`diarizers.llm` & `diarizers.alignment`)**: Post-processing via **[`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)** (Apache-2.0 License, supporting both 4-bit `Q4_K_M` GGUF on CPU via `llama.cpp` and `bfloat16`/`float32` Safetensors on GPU/CPU via Hugging Face `transformers`), paired with **Myers' (1999) bit-parallel exact word-level Levenshtein alignment (`tpst_align`)**, Hungarian inter-chunk permutation reconciliation, and **Locality-Gated Transcript-Preserving Speaker Transfer (`LG-TPST`)**.

> **Archival Note**: All functionality from the experimental [`diarizers-community/DiarizersLM`](https://github.com/diarizers-community/DiarizersLM) repository has been folded into and superseded by `diarizers` (`v1.0.0+`).

`diarizers` (`v1.0.0`) implements and extends the algorithms described in:
- **DiarizationLM**: Quan Wang, Yiling Huang, Guanlong Zhao, Evan Clark, Wei Xia, Hank Liao, *"DiarizationLM: Speaker Diarization Post-Processing with Large Language Models"*, **Interspeech 2024** ([arXiv:2401.03506](https://arxiv.org/abs/2401.03506)).
- **Turn-to-Diarize**: Wei Xia, Han Lu, Quan Wang, Anshuman Tripathi, Yiling Huang, Ignacio Lopez Moreno, Hasim Sak, *"Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection"*, **ICASSP 2022** ([arXiv:2109.11641](https://arxiv.org/abs/2109.11641)).
- **SpectralCluster**: Quan Wang, Carlton Downey, Li Wan, Philip Andrew Mansfield, Ignacio Lopez Moreno, *"Speaker Diarization with LSTM"*, **ICASSP 2018** ([arXiv:1710.10468](https://arxiv.org/abs/1710.10468)).

---

## Architecture

```text
Audio (WAV / FLAC / MP3 / 16 kHz mono NumPy array)
  ├──> Stage 1: ASR Backend (diarizers.asr)
  │      └── WhisperWordASR (openai/whisper-large-v3-turbo) or AudioCppASR (Qwen3-ASR)
  │            ──> WordSegments [start, end, word]
  │
  └──> Stage 2: Speaker Diarization Backend (diarizers.diarization & diarizers.spectral)
         ├── PyanNetSegmentation (pyannote/segmentation-3.0 neural powerset activity curves)
         ├── SpeechBrainECAPATDNN (speechbrain/spkrec-ecapa-voxceleb 192-d embeddings)
         └── SpectralClusterer (Symmetric Normalized Laplacian L_sym + boundary flux snapping)
               ──> SpeakerTurns [start, end, speaker]
                     │
                     ▼
       Stage 3: Word-to-Turn Alignment & Chunked Prompt Builder (diarizers.alignment)
         ├── assign_speakers (Maximal overlap + nearest-turn midpoint fallback)
         └── ChunkedPromptBuilder (4,000-char windows, Gemma-4 turn formatting)
                     │
                     ▼
       Stage 4: DiarizationLM Post-Processing & LG-TPST (diarizers.llm & diarizers.alignment)
         ├── DiarizationLMRefiner (diarizers-community/DiarizationLM-Gemma-4-E4B-v1)
         │     ├── CPU Backend: llama-cpp-python (DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf)
         │     └── GPU/CPU Backend: Hugging Face transformers (model.safetensors)
         ├── Hungarian Inter-Chunk Speaker Permutation Reconciliation
         ├── Myers' (1999) Bit-Parallel Exact Word-Level Levenshtein TPST (tpst_align)
         └── Locality-Gated Speaker Transfer (LG-TPST)
                     │
                     ▼
       DiarizedTranscript (Diarized Text, NIST RTTM, SRT, WebVTT, JSON)
```

---

## Why 2024 `DiarizersLM` Had 43.99%–56.79% WDER & How `diarizers` (`v1.0.0`) Solves It

In the 2024 experimental `DiarizersLM` prototype (`distil-whisper/distil-large-v3` + `pyannote/speaker-diarization-3.1` + `Llama-2-7b`), Word Diarization Error Rate (WDER) degraded to **43.99% on Fisher** and **56.79% on AMI** (nearly random guessing on 2-speaker telephone calls). A systematic audit identified six compounding root causes, all resolved in `diarizers` (`v1.0.0`):

1. **Open-End Timestamp Crash (`[start, None]`)**: Whisper frequently emits `None` for the end timestamp of the final segment in a 30-second chunk; `DiarizersLM` crashed or dropped trailing words. `diarizers.alignment.assign_speakers` safely bounds open-ended timestamps (`start + 0.1s`).
2. **Silence Fallback Speaker Collapse**: In `DiarizersLM`, any word whose timestamp fell in a brief inter-turn pause with zero overlap defaulted to `turns[0].speaker` (always Speaker 1). `diarizers` assigns non-overlapping words to the **temporally nearest speaker turn** by midpoint distance.
3. **Integer-Second Timestamp Truncation**: `DiarizersLM` cast word and turn timestamps to integer seconds (`round(t)`), erasing sub-second conversational turn boundaries and backchannels (`"yeah"`, `"uh-huh"`). `diarizers` preserves full `float64` sub-millisecond precision throughout alignment and supports arbitrary integer speaker counts (`>= 10` speakers).
4. **8 kHz Telephony Acoustic Collapse, Exact HTK Mel Filterbanks & Neural Powerset Segmentation**:
   - Pretrained `speechbrain/spkrec-ecapa-voxceleb` expects **HTK Mel filterbanks** (`_build_mel_filterbank_np`) with **decibel-scaled (`10 * log10`)** 80-bin energies (`blocks.0` BatchNorm `running_mean ~ 71.85, running_var ~ 8536.5`) and global embedding mean subtraction (`mean_var_norm_emb.ckpt`). Using natural-log (`ln`) filterbanks without `glob_mean` subtraction caused all embeddings to collapse into a narrow cone (`cosine similarity > 0.90`).
   - Furthermore, 8 kHz narrowband telephone audio upsampled to 16 kHz introduces a shared channel offset vector across all windows of a call.
   - `diarizers.diarization.ECAPASpectralDiarizer` combines standalone PyTorch `pyannote/segmentation-3.0` (`PyanNetSegmentation` neural powerset segmentation, MIT License) with `SpeechBrainECAPATDNN` (exact HTK dB filterbank scaling, `glob_mean` subtraction, and **recording-level embedding centering** `embs -= embs.mean(axis=0)`) and clusters via **Symmetric Normalized Laplacian Spectral Clustering** (`L_sym = I - D^{-1/2} W D^{-1/2}`) with Gaussian temporal smoothing and calibrated eigengap speaker count estimation (`eigen_threshold=0.45`). Pure sliding-window `ECAPA-TDNN + SpectralClusterer` (without neural segmentation) also remains available via `segmentation_model="none"`.
5. **Inter-Chunk Speaker Permutation Flips**: Long transcripts segmented into 4,000-character LLM prompts previously suffered independent speaker label permutations (`1 <-> 2`) across chunks. `diarizers.alignment.ChunkedPromptBuilder` uses overlapping word windows and **Hungarian (`scipy.optimize.linear_sum_assignment`) inter-chunk reconciliation**.
6. **LLM Monologue Speaker Drift & Ghost Speaker Hallucinations**: Autoregressive LLMs occasionally invent a spurious `<speaker:3>` in a 2-speaker conversation or flip a speaker label across a 200-word monologue. `diarizers.alignment.locality_gated_speaker_transfer` (**`LG-TPST`**) absorbs ghost clusters (`<= 12` words) into adjacent dominant speakers, gates long-span (`> 5` words) monologue flips, and uses **Myers' (1999) bit-parallel exact word-level Levenshtein Transcript-Preserving Speaker Transfer (`TPST`)** so `WER` is mathematically invariant before and after LLM post-processing.

---

## Evaluation Results

All metrics below are **micro-averaged** across the full evaluation splits (`172` Fisher conversations, `20` CALLHOME-English calls, `3` ICSI meetings, and `16` AMI `Mix-Headset` meetings = `211` test recordings) using exact Hungarian-matching dynamic programming (`diarizers.evaluation.compute_corpus_metrics` / `diarizationlm.compute_metrics_on_json_dict`).

*(Note: In accordance with dataset licensing policies, raw LDC/benchmark audio and intermediate decoded transcripts are not redistributed in this repository.)*

### Table 1: Open-Source Frontend (`diarizers` v1.0.0) + `DiarizationLM-Gemma-4-E4B-v1`

Open-source pipeline configuration:
- **ASR**: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) (`WhisperWordASR`, MIT License)
- **Speaker Diarization**: [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) (`PyanNetSegmentation`, MIT License) + [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (`SpeechBrainECAPATDNN`, Apache-2.0 License) + `SpectralClusterer` (`eigen_threshold=0.45`, Apache-2.0 License)
- **DiarizationLM**: [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1) (Apache-2.0 License)

| Benchmark (Full Test Split) | Utterances | Pipeline / System | WER (%) | WDER (%) | cpWER (%) | SpkCntMAE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Fisher English** | 172 | Legacy 2024 `DiarizersLM` (`distil-large-v3` + `pyannote-3.1`) | 20.69 | 43.99 | 56.10 | — |
| **Fisher English** | 172 | `diarizers` v1.0.0 Open Frontend (`Whisper-large-v3-turbo` + `pyannote/seg-3.0` + `ECAPA-TDNN` + `SpectralClusterer`) | **20.80** | **4.35** *(raw 4.45)* | **24.78** *(Oracle 20.46)* | **0.000** |
| **Fisher English** | 172 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **20.80** | **2.59** | **21.91** | **0.000** |
| **Fisher English** | 172 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **20.80** | **2.65** | **21.99** | **0.000** |
| **CALLHOME English** | 20 | `diarizers` v1.0.0 Open Frontend (Auto Speaker Count `[2..6]`, `eigen_threshold=0.45`) | **21.25** | **9.42** *(raw 9.47)* | **32.42** *(Oracle 21.08)* | **0.050** |
| **CALLHOME English** | 20 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **21.25** | **5.98** | **27.69** | **0.050** |
| **CALLHOME English** | 20 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **21.25** | **4.89** | **25.91** | **0.050** |
| **ICSI Meeting Corpus** | 3 | `diarizers` v1.0.0 Open Frontend (Auto Speaker Count `[2..10]`, `eigen_threshold=0.45`) | **35.29** | **23.48** *(raw 23.58)* | **48.97** *(Oracle 34.97)* | **3.333** |
| **ICSI Meeting Corpus** | 3 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **35.29** | **23.10** | **48.62** | **3.333** |
| **ICSI Meeting Corpus** | 3 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **35.29** | **23.11** | **48.63** | **3.333** |
| **AMI (`Mix-Headset`)** | 16 | Legacy 2024 `DiarizersLM` (`distil-large-v3` + `pyannote-3.1`) | — | 56.79 | — | — |
| **AMI (`Mix-Headset`)** | 16 | `diarizers` v1.0.0 Open Frontend (Auto Speaker Count `[2..10]`, `eigen_threshold=0.45`) | **29.14** | **20.68** *(raw 20.80)* | **46.07** *(Oracle 28.70)* | **0.688** |
| **AMI (`Mix-Headset`)** | 16 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **29.14** | **20.14** | **45.56** | **0.688** |
| **AMI (`Mix-Headset`)** | 16 | `diarizers` v1.0.0 + **`DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | **29.14** | **20.13** | **45.53** | **0.688** |

**Key Takeaways:**
- On **Fisher (172 sessions)**, `diarizers` (`v1.0.0`) reduces WDER from `43.99%` (2024 prototype) to **`4.35%`** before the LLM, and **`DiarizationLM-Gemma-4-E4B-v1` reduces WDER by a further `-40.5%` relative down to `2.59%`** (`cpWER` drops from `24.78%` to **`21.91%`**, within `1.45%` of the `20.46%` Hungarian oracle cpWER ceiling).
- On **CALLHOME-English (20 calls, unknown speaker count `2–6`)**, `diarizers` (`v1.0.0`) achieves `9.42%` pre-LLM WDER (`0.050` SpkCntMAE), and **`DiarizationLM-Gemma-4-E4B-v1` reduces WDER by `-48.1%` relative down to `4.89%` (`Raw-TPST`) / `5.98%` (`LG-TPST`)**, cutting `cpWER` by `-6.51` percentage points (`32.42% -> 25.91%`).
- On **AMI (`Mix-Headset`, 16 meetings)** and **ICSI (3 meetings)**, `diarizers` (`v1.0.0`) reduces AMI WDER from `56.79%` (2024 prototype) to **`20.68%`** (`20.13%` post-LLM) and ICSI WDER to **`23.48%`** (`23.10%` post-LLM).

### Table 2: USM + Turn-to-Diarize Frontend + `DiarizationLM-Gemma-4-E4B-v1`

For reference, below is the performance of [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1) on the exact same 211 test recordings when paired with the USM ASR and Turn-to-Diarize frontend (with 95% bootstrap confidence intervals over `B = 10,000` conversation resamples):

| Benchmark | Test Split | WER (%) | System | WDER (%) [95% CI] | cpWER (%) [95% CI] | SpkCntMAE [95% CI] | Paired ΔWDER vs. Baseline (p-value) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Fisher** | TEST FULL (172 sessions) | 15.37 | Baseline (USM + Turn-to-Diarize)<br>`DiarizationLM-8b-Fisher-v2` (Llama 3 8B)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 5.32 [4.93, 5.74]<br>3.28<br>**2.99 [2.65, 3.37]** | 20.88 [19.97, 21.88]<br>18.37<br>**17.62 [16.76, 18.56]** | 0.215 [0.151, 0.291]<br>—<br>**0.093 [0.047, 0.145]** | —<br>—<br>**-2.33% [-2.51, -2.16]** (p < 0.0001) |
| **CALLHOME** | TEST FULL (20 calls) | 15.22 | Baseline (USM + Turn-to-Diarize)<br>`DiarizationLM-8b-Fisher-v2` (Llama 3 8B)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 7.74 [6.07, 9.65]<br>6.66<br>**4.92 [3.46, 6.75]** | 24.31 [21.27, 27.43]<br>23.57<br>**20.69 [17.98, 23.66]** | 0.050 [0.000, 0.150]<br>—<br>**0.000 [0.000, 0.000]** | —<br>—<br>**-2.82% [-3.48, -2.14]** (p < 0.0001) |
| **ICSI** | TEST FULL (3 meetings) | 29.42 | Baseline (USM + Turn-to-Diarize)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 14.70 [11.65, 20.29]<br>**14.10 [10.77, 19.94]** | 43.90 [39.10, 51.67]<br>**43.32 [38.38, 51.20]** | 0.333 [0.000, 1.000]<br>**0.333 [0.000, 1.000]** | —<br>**-0.60% [-0.88, -0.35]** (p < 0.0001) |
| **AMI** | TEST WORD FULL (16 meetings) | 24.33 | Baseline (USM + Turn-to-Diarize)<br>**`DiarizationLM-Gemma-4-E4B-v1` (4B)** | 15.68 [10.64, 21.11]<br>**14.89 [9.80, 20.38]** | 40.32 [32.43, 48.00]<br>**39.57 [31.57, 47.29]** | 0.500 [0.188, 0.812]<br>**0.438 [0.188, 0.750]** | —<br>**-0.79% [-1.00, -0.60]** (p < 0.0001) |

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

Run the full 4-stage pipeline (`Whisper-large-v3-turbo` ASR + `pyannote/segmentation-3.0` + `SpeechBrainECAPATDNN` + `SpectralClusterer` + `DiarizationLM-Gemma-4-E4B-v1`) on any WAV/MP3 file or 16 kHz NumPy waveform:

```python
import diarizers

# CPU mode using the 4-bit Q4_K_M GGUF checkpoint from Hugging Face Hub:
pipe = diarizers.DiarizersPipeline.from_pretrained(
    asr_model="openai/whisper-large-v3-turbo",
    diarizer_preset=diarizers.FISHER_TELEPHONY_PRESET,
    llm_model_or_path="diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
    llm_gguf_filename="DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf",
    llm_backend="gguf",
    device="cpu",
)

# Pass num_speakers=2 if known, or num_speakers=None to auto-detect via eigengap
transcript = pipe("conversation.wav", num_speakers=2)

# Export in multiple standard formats:
print(transcript.to_diarized_text())                  # "<speaker:1> hello ... <speaker:2> hi ..."
print(transcript.to_rttm(recording_id="session_01"))  # NIST RTTM format
print(transcript.to_srt())                            # SubRip (.srt) subtitles with [Speaker k]
print(transcript.to_vtt())                            # WebVTT (.vtt) subtitles
```

To run on a **CUDA GPU** with 16-bit `bfloat16` Hugging Face `transformers` weights:

```python
pipe_gpu = diarizers.DiarizersPipeline.from_pretrained(
    asr_model="openai/whisper-large-v3-turbo",
    diarizer_preset=diarizers.CALLHOME_TELEPHONY_PRESET,  # Auto speaker count [2..6]
    llm_model_or_path="diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
    llm_backend="transformers",
    device="cuda:0",
)
transcript = pipe_gpu("callhome_sample.wav", num_speakers=None)
```

### 2. Modular End-to-End Python API (`DiarizationPipeline`)

You can also customize each stage independently (for example, choosing among `FISHER_TELEPHONY_PRESET`, `CALLHOME_TELEPHONY_PRESET`, `MEETING_MULTISPK_PRESET`, or setting `segmentation_model="none"` for pure sliding-window `ECAPA-TDNN + SpectralClusterer`):

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
    segmentation_model="pyannote/segmentation-3.0",  # Or "none" for sliding-window only
    encoder_name="speechbrain/spkrec-ecapa-voxceleb",
    min_clusters=2,
    max_clusters=8,
    eigen_threshold=0.45,
)

pipeline = DiarizationPipeline(
    asr=WhisperWordASR("openai/whisper-large-v3-turbo", device="cpu"),
    diarizer=ECAPASpectralDiarizer(preset=preset, device="cpu"),
    llm_refiner=DiarizationLMRefiner(
        model_path_or_repo="diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
        gguf_filename="DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf",
        backend="gguf",
    ),
    use_locality_gate=True,
)

result = pipeline("meeting.wav")
print(result.to_diarized_text())
```

### 3. Post-Processing Any Diarized Transcript with `DiarizationLM-Gemma-4-E4B-v1`

If you already have a diarized transcript from any ASR + diarization system (`<speaker:1> ... <speaker:2> ...`), you can refine its speaker labels directly using `DiarizationLMRefiner` and `locality_gated_speaker_transfer` (`LG-TPST`) or `transcript_preserving_speaker_transfer` (`TPST`):

```python
from diarizers import (
    DiarizationLMRefiner,
    create_diarized_text,
    extract_text_and_spk,
    locality_gated_speaker_transfer,
    transcript_preserving_speaker_transfer,
)

hyp_diarized = (
    "<speaker:1> hello how are you doing <speaker:2> today i am doing great "
    "thanks what about <speaker:1> you i am doing well too"
)

refiner = DiarizationLMRefiner(
    model_path_or_repo="diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
    gguf_filename="DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf",
    backend="gguf",  # Or backend="transformers", device="cuda:0"
)

# High-level helper returning (raw_tpst_spk, lg_tpst_spk, raw_completions):
hyp_text, hyp_spk = extract_text_and_spk(hyp_diarized)
raw_tpst_spk, lg_tpst_spk, _ = refiner.refine_utterance(hyp_text, hyp_spk)

print("Refined (LG-TPST):", create_diarized_text(hyp_text.split(), lg_tpst_spk.split()))
print("Refined (Raw-TPST):", create_diarized_text(hyp_text.split(), raw_tpst_spk.split()))
```

### 4. Command-Line Interface (`diarizers` CLI & `scripts/`)

Installing `diarizers` registers the `diarizers` console script with three subcommands (`run`, `prepare-sft-data`, `evaluate`):

```bash
# 1. Transcribe, diarize, and post-process an audio file:
diarizers run \
  --audio conversation.wav \
  --asr-model openai/whisper-large-v3-turbo \
  --preset telephony \
  --num-speakers 2 \
  --llm-model diarizers-community/DiarizationLM-Gemma-4-E4B-v1 \
  --gguf-file DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf \
  --output-json output_transcript.json \
  --output-rttm output_transcript.rttm \
  --output-srt output_transcript.srt

# 2. Prepare SFT training prompts (hyp2ora + deg2ref + Locality-Preserving Oracle):
diarizers prepare-sft-data \
  --input-json decoded_train_utterances.json \
  --output-json sft_train_prompts.json \
  --flavor both

# 3. Evaluate micro WER, WDER, cpWER, and SpkCntMAE with 95% bootstrap CIs:
diarizers evaluate \
  --input-json decoded_test_utterances.json \
  --bootstrap-samples 1000 \
  --output-json eval_metrics.json
```

Standalone scripts are also provided in [`scripts/`](scripts/):
- `python scripts/inference.py --audio conversation.wav --num_speakers 2 --output_json out.json`
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
   Fine-tunes `google/gemma-4-E4B` (or `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`) using completion-only cross-entropy loss on 4,000-character prompt chunks formatted with Gemma-4 turn tags (`<|turn>user\n{prompt} --> <turn|>\n<|turn>model\n{completion} [eod]`).
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

### 7. Interactive CPU Gradio Demo (`hf_space/app.py`)

To launch the local 2-tab interactive Gradio demo on CPU:

```bash
pip install gradio
python hf_space/app.py
```

- **Tab 1 — End-to-End Audio Pipeline (CPU)**: Upload or record an audio file, select an acoustic preset (`Fisher Telephony`, `Callhome Telephony`, or `Meeting Multi-Speaker`), and generate speaker-attributed transcripts, NIST RTTM, and SRT subtitles.
- **Tab 2 — DiarizationLM + LG-TPST Text Refinement**: Paste any diarized transcript (`<speaker:1> ... <speaker:2> ...`) and refine speaker turn boundaries using `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`.

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
│   ├── __init__.py          # Public API exports (v1.0.0)
│   ├── types.py             # WordSegment, SpeakerTurn, DiarizedTranscript (RTTM/VTT/SRT/JSON)
│   ├── configs.py           # PromptConfig and acoustic DiarizationPresets
│   ├── spectral.py          # Clean-room Symmetric Normalized Laplacian SpectralClusterer
│   ├── diarization.py       # Standalone PyTorch PyanNetSegmentation, SpeechBrainECAPATDNN & ECAPASpectralDiarizer
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
│   ├── app.py               # Interactive 2-tab CPU Gradio demo
│   ├── README.md            # Hugging Face Space metadata & documentation
│   └── requirements.txt     # CPU Space dependencies
├── docs/
│   └── blog_post.md         # Technical deep-dive blog post on diarizers v1.0.0 & Gemma-4
└── tests/
    └── test_diarizers.py    # Comprehensive unit & regression test suite
```

---

## Pretrained Models on Hugging Face

- **DiarizationLM (Gemma 4 E4B, Multi-Domain, Safetensors + `Q4_K_M` GGUF, Apache-2.0)**:
  - [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)
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
