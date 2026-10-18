# Diarizers 1.0.1: An Open-Source Apache-2.0 Stack for Speaker-Attributed Transcription with DiarizationLM-Gemma-4

> `diarizers` (`v1.0.1`) is an Apache-2.0 open-source software package and end-to-end pipeline implementing the algorithms described in *DiarizationLM* ([Wang et al., Interspeech 2024](https://arxiv.org/abs/2401.03506)), *Turn-to-Diarize* ([Xia et al., ICASSP 2022](https://arxiv.org/abs/2109.11641)), and *SpectralCluster* ([Wang et al., ICASSP 2018](https://arxiv.org/abs/1710.10468)).

---

## Executive Summary

When [`DiarizationLM`](https://arxiv.org/abs/2401.03506) was introduced at Interspeech 2024, it demonstrated that fine-tuned Large Language Models (LLMs) could reduce Word Diarization Error Rate (WDER) by up to 55% on conversational speech by leveraging semantic and syntactic context (`"how are you"` followed by `"i am doing great"`). However, the original paper evaluated on proprietary ASR (`USM`) and speaker diarization (`Turn-to-Diarize`) frontends.

When the community first attempted to pair open-source ASR and diarization models with DiarizationLM in the 2024 experimental [`DiarizersLM`](https://github.com/diarizers-community/DiarizersLM) repository, end-to-end WDER degraded to **43.99% on Fisher** and **56.79% on AMI** — nearly equivalent to random coin-flipping on two-speaker telephone calls.

Today, we release **`diarizers` (`v1.0.1`)** (`pip install diarizers`), a unified **Apache-2.0 / MIT** software package and inference pipeline that pairs a repaired open-source acoustic frontend (`openai/whisper-large-v3-turbo` + `pyannote/segmentation-3.0` + `speechbrain/spkrec-ecapa-voxceleb` + `SpectralClusterer`) with **[`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)** across the **full 211-utterance evaluation suite (`172` Fisher, `20` CALLHOME, `3` ICSI, `16` AMI)**:

1. **Full-Set Fisher (`172` Calls)**: WDER drops from **`43.99%`** (2024 prototype) to **`4.35%`** *(raw `4.45%`)* (`20.80%` WER, `24.78%` cpWER, `0.000` SpkCntMAE) with the standalone `diarizers` (`v1.0.1`) open frontend, and down to **`2.59%`** (`21.91%` cpWER, `-40.5%` relative WDER reduction) with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` + `LG-TPST`.
2. **Full-Set CALLHOME-English (`20` Calls)**: WDER reaches **`9.42%`** *(raw `9.47%`)* (`21.25%` WER, `32.42%` cpWER *(raw `32.67%`)*, `0.050` SpkCntMAE) with the open frontend and drops to **`5.98%` (`LG-TPST`) / `4.89%` (`Raw-TPST`, `-48.1%` relative WDER reduction, `25.91%` cpWER)** after `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`.
3. **Full-Set AMI (`Mix-Headset`, `16` Meetings)**: WDER drops from **`56.79%`** (2024 prototype) to **`20.68%`** *(raw `20.80%`)* (`29.14%` WER, `46.07%` cpWER *(raw `51.39%`)*, `0.688` SpkCntMAE) with the open frontend and improves to **`20.14%` (`LG-TPST`) / `20.13%` (`Raw-TPST`, `45.53%` cpWER)** with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`.
4. **Full-Set ICSI (`3` Meetings)**: WDER reaches **`23.48%`** *(raw `23.58%`)* (`35.29%` WER, `48.97%` cpWER *(raw `59.55%`)*) with the open frontend and improves to **`23.10%`** (`48.62%` cpWER) with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`.
5. **100% Apache-2.0 / MIT Open-Source Stack**: Built entirely on commercially permissive open-source components (`openai/whisper-large-v3-turbo` MIT or `Qwen3-ASR` via `audio.cpp`, standalone PyTorch `pyannote/segmentation-3.0` MIT powerset segmentation, standalone PyTorch `speechbrain/spkrec-ecapa-voxceleb` Apache-2.0 with exact HTK Mel filterbanks, Symmetric Normalized Laplacian Spectral Clustering Apache-2.0, Myers' bit-parallel word-level Levenshtein alignment, and `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` via Hugging Face `transformers` or CPU `llama.cpp` GGUF).

---

## 1. Post-Mortem: Why Did the 2024 Prototype Have 43.99% WDER?

A forensic audit of the 2024 `DiarizersLM` prototype revealed six compounding bugs across feature extraction, speaker segmentation, embedding normalization, word-to-turn timestamp orchestration, and LLM post-processing:

### Bug 1: Missing Exact HTK Mel Filterbanks & Decibel Scaling (`10 / ln(10)`) in ECAPA-TDNN
Pretrained `speechbrain/spkrec-ecapa-voxceleb` (`blocks.0` BatchNorm `running_mean ~ 71.85, running_var ~ 8536.5`) was trained with SpeechBrain's `Fbank`, which uses **HTK Mel filterbank spacing** (`_build_mel_filterbank_np`) and converts linear Mel energies to **decibels (`10 * log10(E) = 4.3429448 * ln(E)`)**. Passing Slaney natural-log filterbanks (`ln(E)`) directly into `normalizer.ckpt` (`mean_var_norm`) scaled the input features off by `4.34x` relative to the network's BatchNorm statistics.

### Bug 2: Missing Global (`mean_var_norm_emb.ckpt`) & Recording-Level Embedding Centering
Pretrained `speechbrain/spkrec-ecapa-voxceleb` includes a global speaker embedding mean vector (`mean_var_norm_emb.ckpt`). Without subtracting this global mean before L2 normalization — and without subtracting the recording-level channel mean (`embs -= embs.mean(axis=0)`) on 8 kHz narrowband telephone audio upsampled to 16 kHz — sliding-window embeddings collapsed into a narrow cone (`cosine similarity > 0.90`).

### Bug 3: Neural Powerset Speaker Segmentation (`pyannote/segmentation-3.0`) + Calibrated Eigengap (`eigen_threshold=0.45`)
Fixed-length sliding windows straddle rapid conversational turn boundaries and backchannels. Integrating a standalone pure-PyTorch implementation of **`pyannote/segmentation-3.0`** (`PyanNetSegmentation`: SincNet + 4-layer BiLSTM + 7-class powerset classifier for `{silence, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3}`, MIT License) snaps sliding-window speaker transitions to local neural speaker-change flux peaks while clustering `SpeechBrainECAPATDNN` embeddings via Symmetric Normalized Laplacian (`L_sym = I - D^{-1/2} W D^{-1/2}`, `eigen_threshold=0.45`). (For lightweight environments without segmentation weights, pure sliding-window `ECAPA-TDNN + SpectralClusterer` remains available via `segmentation_model="none"`.)

### Bug 4: Timestamp Truncation, Open-Ended `[start, None]` Chunks, and Silence Collapse
The 2024 prototype rounded word and turn timestamps to integer seconds (`round(t)`), erasing sub-second turn boundaries and backchannels (`"yeah"`, `"uh-huh"`). Worse, any word falling in a brief inter-turn pause with zero overlap defaulted to `turns[0].speaker` (always Speaker 1), and speaker IDs were indexed into a 9-character string (`"123456789"`), breaking whenever a meeting had `>= 10` speaker clusters. `diarizers.alignment.assign_speakers` preserves full `float64` sub-millisecond precision, bounds `[start, None]` chunks, assigns non-overlapping words to the temporally nearest turn by midpoint distance, and supports arbitrary integer speaker counts.

### Bug 5: Approximate `difflib` vs. Exact Myers' (1999) Bit-Parallel Levenshtein
Transcript-Preserving Speaker Transfer (`TPST`) requires aligning the LLM's completion words back to the original ASR hypothesis words so that ASR word error rate (`WER`) remains strictly invariant. Replacing approximate `difflib.SequenceMatcher` with **Myers' (1999) bit-parallel exact word-level Levenshtein distance and backtrace** (`diarizers.alignment.tpst_align`) using Python arbitrary-precision bitvectors achieves **100% bit-for-bit parity with full 2D dynamic programming** while running **>220x faster** (`3.6 ms` for 2,000 x 2,000 words).

### Bug 6: Inter-Chunk Permutation Flips & Monologue Drift (`LG-TPST`)
When long conversations are split into 4,000-character LLM prompts, independent chunks can permute speaker IDs (`<speaker:1>` <-> `<speaker:2>`), and autoregressive LLMs can occasionally hallucinate a ghost `<speaker:3>` or flip speaker labels across a long monologue, or fail to emit `[eod]` before repeating text. `diarizers` solves these via:
- **`ChunkedPromptBuilder`**: Defaults to `overlap_words=0` (matching the non-overlapping chunking used during `DiarizationLM-Gemma-4-E4B-v1` SFT training) and supports configurable overlapping word windows (e.g., `overlap_words=60`) with **Hungarian (`scipy.optimize.linear_sum_assignment`) inter-chunk reconciliation**.
- **`truncate_completion_suffix` + `locality_gated_speaker_transfer` (`LG-TPST`)**: Truncates LLM completions at the first occurrence of `[eod]` or `<turn|>`, absorbs tiny ghost speaker clusters (`<= 15` words in `>= 3`-speaker meetings) into dominant speakers, and gates long-span monologue flips (`> 5` words in `>= 3`-speaker meetings and `> 15` words in `2`-speaker conversations) back to the acoustic clustering baseline while preserving local turn-boundary corrections.

---

## 2. Architecture of `diarizers` (`v1.0.1`)

```text
Audio (WAV / 16 kHz mono)
  ├──> ASR Backend (WhisperWordASR: openai/whisper-large-v3-turbo | AudioCppASR: Qwen3-ASR) ──> WordSegments [start, end, word]
  └──> Diarization Backend (PyanNetSegmentation pyannote/segmentation-3.0 + SpeechBrainECAPATDNN + SpectralClusterer) ──> SpeakerTurns [start, end, spk]
         │
         ▼
  assign_speakers (Maximal overlap + nearest-turn midpoint fallback + first-appearance renumbering)
         │
         ▼
  ChunkedPromptBuilder (4000-char windows, default overlap_words=0, configurable up to 60)
         │
         ▼
  DiarizationLMRefiner (diarizers-community/DiarizationLM-Gemma-4-E4B-v1 via llama.cpp CPU GGUF or HF Transformers)
         │
         ▼
  Hungarian Inter-Chunk Reconciliation + Myers Bit-Parallel TPST + Locality-Gated Speaker Transfer (LG-TPST)
         │
         ▼
  DiarizedTranscript (Diarized Text, NIST RTTM, SRT, WebVTT, JSON)
```

---

## 3. Verified Full-Set Benchmark Results (All 211 Test Utterances)

In accordance with data licensing and privacy policies, no benchmark datasets or intermediate hypothesis transcripts are distributed. Below are the verified metrics measured across the **complete 211-utterance evaluation suite (`172` Fisher, `20` CALLHOME, `3` ICSI, `16` AMI)**:

### Table 1: Full-Set Open-Frontend (`diarizers` v1.0.1) + `DiarizationLM-Gemma-4-E4B-v1` Evaluation (All 211 Test Utterances)

*(Post-LLM evaluations in Table 1 were run in full-precision `bfloat16` on TPU with `max_new_tokens = 1.2x` input tokens and `6,000`-character chunk windows; Fisher uses fixed speaker count `K=2`, CALLHOME uses auto speaker count `[2..6]`, ICSI uses auto speaker count `[2..10]`, and AMI uses auto speaker count `[3..5]`.)*

| Benchmark Split (Full Test) | Utterances | 2024 Prototype WDER (%) | `diarizers` v1.0.1 WER (%) | `diarizers` v1.0.1 Open-Frontend WDER (%) | `DiarizationLM-Gemma-4-E4B-v1` + `LG-TPST` WDER (%) | `DiarizationLM-Gemma-4-E4B-v1` Raw-TPST WDER (%) | `diarizers` v1.0.1 Open-Frontend cpWER (%) | `DiarizationLM-Gemma-4-E4B-v1` cpWER (%) | SpkCntMAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fisher Test (`K=2`)** | 172 | 43.99 | **20.80** | **4.35** *(raw 4.45)* | **2.59** | **2.65** | 24.78 *(raw 24.78, Oracle 20.46)* | **21.91** *(Raw: 21.99)* | **0.000** *(fixed K=2)* |
| **CALLHOME Test (`[2..6]`)** | 20 | — | **21.25** | **9.42** *(raw 9.47)* | **5.98** | **4.89** | 32.42 *(raw 32.67, Oracle 21.08)* | **27.69** *(Raw: **25.91**)* | **0.050** |
| **ICSI Test (`[2..10]`)** | 3 | — | **35.29** | **23.48** *(raw 23.58)* | **23.10** | **23.11** | 48.97 *(raw 59.55, Oracle 34.97)* | **48.62** *(Raw: 48.63)* | **3.333** |
| **AMI Test (`Mix-Headset`, `[3..5]`)** | 16 | 56.79 | **29.14** | **20.68** *(raw 20.80)* | **20.14** | **20.13** | 46.07 *(raw 51.39, Oracle 28.70)* | **45.56** *(Raw: **45.53**)* | **0.688** |

> **Note on Pre-Filter `*(raw ...)*` vs. Ghost-Filtered Open-Frontend Baseline**: In Table 1, the primary Open-Frontend `WDER` and `cpWER` columns report the Stage 3 acoustic diarization baseline after deterministic ghost-cluster absorption (`apply_locality_gated_speaker_transfer(hyp_spk, hyp_spk, max_ghost_cluster_words=15)`), while parenthetical `*(raw ...)*` values report the un-filtered Stage 3 `assign_speakers` output prior to ghost-cluster absorption. On multi-speaker meetings (`ICSI`, `AMI`), absorbing tiny spurious acoustic clusters (`<= 15` words) reduces `cpWER` by `-10.58` percentage points on ICSI (`59.55% -> 48.97%`) and `-5.32` percentage points on AMI (`51.39% -> 46.07%`) before any LLM inference, isolating the genuine lexical turn-boundary gains of `DiarizationLM-Gemma-4-E4B-v1` (`23.48% -> 23.10%` WDER on ICSI; `20.68% -> 20.13%` WDER on AMI).

### Decomposed Stage-by-Stage Ablation, Model Reuse & Multi-Party Meeting Analysis

| Stage / System | Fisher (172) WDER / cpWER (%) | CALLHOME (20) WDER / cpWER (%) | ICSI (3) WDER / cpWER (%) | AMI (16) WDER / cpWER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0. Legacy 2024 `DiarizersLM` Prototype** | 43.99 / 56.10 | — / — | — / — | 56.79 / — |
| **1. Stage 3 Raw Acoustic Alignment (`assign_speakers`)** | 4.45 / 24.78 | 9.47 / 32.67 | 23.58 / 59.55 | 20.80 / 51.39 |
| **2. Stage 3 + Ghost-Cluster Absorption (`<= 15` words)** | 4.35 / 24.78 | 9.42 / 32.42 | 23.48 / 48.97 | 20.68 / 46.07 |
| **3. Stage 4 `DiarizationLM-Gemma-4-E4B-v1` (`Raw-TPST`)** | 2.65 / 21.99 | **4.89 / 25.91** | 23.11 / 48.63 | **20.13 / 45.53** |
| **4. Stage 4 `DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)** | **2.59 / 21.91** | 5.98 / 27.69 | **23.10 / 48.62** | 20.14 / 45.56 |
| **5. Continual Mixed-Frontend SFT (`DiarizationLM-diarizers-Gemma-4-E4B-v1`, `LG-TPST`)** | 2.61 / 21.93 *(Raw: 2.64 / 21.93)* | 5.98 / 27.75 *(Raw: 4.89 / 25.96)* | 23.12 / 48.66 *(Raw: 23.12 / 48.65)* | 20.32 / 45.66 *(Raw: 20.30 / 45.64)* |
| **Hungarian TPST Oracle Ceiling (`hyp_spk_oracle`)** | *0.01 / 20.46* | *0.00 / 21.08* | *0.00 / 34.97* | *0.00 / 28.70* |

- **Why We Reuse `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` Without Releasing a Duplicate Checkpoint**: Continual LoRA SFT on a 50/50 mixture of open-frontend and USM/Turn-to-Diarize locality-preserving SFT pairs (`DiarizationLM-diarizers-Gemma-4-E4B-v1`, Row 5) achieves virtually identical WDER and cpWER to `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (Row 4) across all 4 benchmarks (`<= 0.18%` WDER difference), confirming that `DiarizationLM-Gemma-4-E4B-v1` generalizes zero-shot to the `diarizers` open-source frontend.
- **Why Multi-Party Meeting (`ICSI`, `AMI`) LLM Gains Are Smaller Than Telephony (`Fisher`, `CALLHOME`)**: Three structural factors bound text-only LLM post-processing on 4–9 speaker tabletop meetings: (1) higher far-field ASR WER (`29.14%–35.29%` vs. `20.80%–21.25%` on telephony) and overlapped speech; (2) global speaker-count estimation error (`SpkCntMAE` of `3.333` on ICSI and `0.688` on AMI vs. `0.000` and `0.050` on Fisher/CALLHOME); and (3) multi-party turn-target ambiguity when $K > 2$ (detecting a lexical turn boundary does not uniquely determine which of the other $K - 1$ speakers spoke next without acoustic re-scoring).

### Table 2: Full-Set Reference USM + Turn-to-Diarize Evaluation (`DiarizationLM-Gemma-4-E4B-v1`, All 211 Test Utterances)

| Benchmark Split (Full Test) | Utterances | USM WER (%) | USM + T2D Baseline WDER (%) | `DiarizationLM-Gemma-4-E4B-v1` WDER (%) | USM + T2D Baseline cpWER (%) | `DiarizationLM-Gemma-4-E4B-v1` cpWER (%) | SpkCntMAE (`v1` vs Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fisher (USM)** | 172 | 15.37 | 5.32 | **2.99** *(LG-TPST: 3.05)* | 20.88 | **17.62** *(LG-TPST: 17.72)* | **0.093** *(vs 0.215)* |
| **CALLHOME (USM)** | 20 | 15.22 | 7.74 | **4.92** *(LG-TPST: 5.02)* | 24.31 | **20.69** *(LG-TPST: 20.73)* | **0.000** *(vs 0.050)* |
| **ICSI (USM)** | 3 | 29.42 | 14.70 | **14.10** *(LG-TPST: 14.13)* | 43.90 | **43.32** *(LG-TPST: 43.36)* | **0.333** *(vs 0.333)* |
| **AMI (USM)** | 16 | 24.33 | 15.68 | **14.89** *(LG-TPST: 14.90)* | 40.32 | **39.57** *(LG-TPST: 39.59)* | **0.438** *(vs 0.500)* |

---

## 4. Quickstart with `diarizers` (`v1.0.1`)

### Installation

```bash
pip install "diarizers[audio,llm]"
```

Or install from source:

```bash
git clone https://github.com/diarizers-community/diarizers.git
cd diarizers
pip install -e ".[audio,llm,dev]"
```

### End-to-End Python API (CPU or GPU)

```python
from diarizers import (
    DiarizationLMRefiner,
    DiarizationPipeline,
    ECAPASpectralDiarizer,
    FISHER_TELEPHONY_PRESET,
    WhisperWordASR,
)

pipeline = DiarizationPipeline(
    asr=WhisperWordASR("openai/whisper-large-v3-turbo", device="cpu"),
    diarizer=ECAPASpectralDiarizer(preset=FISHER_TELEPHONY_PRESET, device="cpu"),
    llm_refiner=DiarizationLMRefiner(
        model_path="diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
        backend="gguf",
    ),
    use_locality_gate=True,
)

result = pipeline("conversation.wav", num_speakers=2)
print(result.to_diarized_text())
print(result.to_rttm(recording_id="conversation"))
```

### Interactive Hugging Face Spaces (`spaces/diarizers-community/diarizers` & `DiarizationLM-GGUF`)

`diarizers` provides two dedicated interactive demos that run entirely on standard CPU hardware:
- **End-to-End Audio Pipeline Demo ([`spaces/diarizers-community/diarizers`](https://huggingface.co/spaces/diarizers-community/diarizers), `hf_space/app.py` & `hf_space/index.html`)**: Upload a WAV/MP3 clip or select a built-in conversation to run `WhisperWordASR` + `ECAPASpectralDiarizer` (`pyannote/segmentation-3.0` + `speechbrain/spkrec-ecapa-voxceleb` + `SpectralClusterer`) + `DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` GGUF) + `LG-TPST`, and export diarized text, NIST RTTM, and SRT subtitles.
- **Text-Only DiarizationLM GGUF Demo ([`spaces/diarizers-community/DiarizationLM-GGUF`](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF))**: Paste any diarized transcript (`<speaker:1> ... <speaker:2> ...`), refine it with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` GGUF via `llama.cpp`) + `TPST`, and view streaming word-level speaker boundary corrections.

---

## 5. Citation

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
```
