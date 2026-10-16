# Diarizers 1.0: An Open-Source Apache-2.0 Stack for Speaker-Attributed Transcription with DiarizationLM-Gemma-4

> `diarizers` (`v1.0.0`) is an Apache-2.0 open-source software package and end-to-end pipeline implementing the algorithms described in *DiarizationLM* ([Wang et al., Interspeech 2024](https://arxiv.org/abs/2401.03506)), *Turn-to-Diarize* ([Xia et al., ICASSP 2022](https://arxiv.org/abs/2109.11641)), and *SpectralCluster* ([Wang et al., ICASSP 2018](https://arxiv.org/abs/1710.10468)).

---

## Executive Summary

When [`DiarizationLM`](https://arxiv.org/abs/2401.03506) was introduced at Interspeech 2024, it demonstrated that fine-tuned Large Language Models (LLMs) could reduce Word Diarization Error Rate (WDER) by up to 55% on conversational speech by leveraging semantic and syntactic context (`"how are you"` followed by `"i am doing great"`). However, the original paper evaluated on proprietary ASR (`USM`) and speaker diarization (`Turn-to-Diarize`) frontends.

When the community first attempted to pair open-source ASR and diarization models with DiarizationLM in the 2024 experimental [`DiarizersLM`](https://github.com/diarizers-community/DiarizersLM) repository, end-to-end WDER degraded to **43.99% on Fisher** and **56.79% on AMI** — nearly equivalent to random coin-flipping on two-speaker telephone calls.

Today, we release **`diarizers` (`v1.0.0`)** (`pip install diarizers`), a unified **Apache-2.0 / MIT** software package and inference pipeline that pairs a repaired open-source acoustic frontend (`openai/whisper-large-v3-turbo` + `pyannote/segmentation-3.0` + `speechbrain/spkrec-ecapa-voxceleb` + `SpectralClusterer`) with **[`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)** across the **full 211-utterance evaluation suite (`172` Fisher, `20` CALLHOME, `3` ICSI, `16` AMI)**:

1. **Full-Set Fisher (`172` Calls)**: WDER drops from **`43.99%`** (2024 prototype) to **`4.35%`** (`20.80%` WER, `24.78%` cpWER, `0.000` SpkCntMAE) with the standalone `diarizers` (`v1.0.0`) open frontend, and down to **`2.59%`** (`21.91%` cpWER, `-40.5%` relative WDER reduction) with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` + `LG-TPST`.
2. **Full-Set CALLHOME-English (`20` Calls)**: WDER reaches **`9.42%`** (`21.25%` WER, `32.42%` cpWER, `0.050` SpkCntMAE) with the open frontend and drops to **`5.98%` (`LG-TPST`) / `4.89%` (`Raw-TPST`, `-48.1%` relative WDER reduction, `25.91%` cpWER)** after `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`.
3. **Full-Set AMI (`Mix-Headset`, `16` Meetings)**: WDER drops from **`56.79%`** (2024 prototype) to **`20.68%`** (`29.14%` WER, `46.07%` cpWER, `0.688` SpkCntMAE) with the open frontend and improves to **`20.14%` (`LG-TPST`) / `20.13%` (`Raw-TPST`, `45.53%` cpWER)** with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`.
4. **Full-Set ICSI (`3` Meetings)**: WDER reaches **`23.48%`** (`35.29%` WER, `48.97%` cpWER) with the open frontend and improves to **`23.10%`** (`48.62%` cpWER) with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`.
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
When long conversations are split into 4,000-character LLM prompts, independent chunks can permute speaker IDs (`<speaker:1>` <-> `<speaker:2>`), and autoregressive LLMs can occasionally hallucinate a ghost `<speaker:3>` or flip speaker labels across a long monologue. `diarizers` solves both via:
- **`ChunkedPromptBuilder`**: Uses overlapping word windows (`overlap_words=60`) and **Hungarian (`scipy.optimize.linear_sum_assignment`) inter-chunk reconciliation**.
- **`locality_gated_speaker_transfer` (`LG-TPST`)**: Absorbs tiny ghost speaker clusters (`<= 12` words) into adjacent dominant speakers and gates long-span (`> 5` words) monologue flips back to the acoustic clustering baseline while preserving local turn-boundary corrections (`1 <= L <= 5` words).

---

## 2. Architecture of `diarizers` (`v1.0.0`)

```text
Audio (WAV / 16 kHz mono)
  ├──> ASR Backend (WhisperWordASR: openai/whisper-large-v3-turbo | AudioCppASR: Qwen3-ASR) ──> WordSegments [start, end, word]
  └──> Diarization Backend (PyanNetSegmentation pyannote/segmentation-3.0 + SpeechBrainECAPATDNN + SpectralClusterer) ──> SpeakerTurns [start, end, spk]
         │
         ▼
  assign_speakers (Maximal overlap + nearest-turn midpoint fallback + first-appearance renumbering)
         │
         ▼
  ChunkedPromptBuilder (4000-char windows, 60-word overlap)
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

### Table 1: Full-Set Open-Frontend (`diarizers` v1.0.0) + `DiarizationLM-Gemma-4-E4B-v1` Evaluation (All 211 Test Utterances)

| Benchmark Split (Full Test) | Utterances | 2024 Prototype WDER (%) | `diarizers` v1.0.0 WER (%) | `diarizers` v1.0.0 Open-Frontend WDER (%) | `DiarizationLM-Gemma-4-E4B-v1` + `LG-TPST` WDER (%) | `DiarizationLM-Gemma-4-E4B-v1` Raw-TPST WDER (%) | `diarizers` v1.0.0 Open-Frontend cpWER (%) | `DiarizationLM-Gemma-4-E4B-v1` cpWER (%) | SpkCntMAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fisher Test** | 172 | 43.99 | **20.80** | **4.35** *(raw 4.45)* | **2.59** | **2.65** | 24.78 *(Oracle 20.46)* | **21.91** *(Raw: 21.99)* | **0.000** |
| **CALLHOME Test** | 20 | — | **21.25** | **9.42** *(raw 9.47)* | **5.98** | **4.89** | 32.42 *(Oracle 21.08)* | **27.69** *(Raw: **25.91**)* | **0.050** |
| **ICSI Test** | 3 | — | **35.29** | **23.48** *(raw 23.58)* | **23.10** | **23.11** | 48.97 *(Oracle 34.97)* | **48.62** *(Raw: 48.63)* | **3.333** |
| **AMI Test (`Mix-Headset`)** | 16 | 56.79 | **29.14** | **20.68** *(raw 20.80)* | **20.14** | **20.13** | 46.07 *(Oracle 28.70)* | **45.56** *(Raw: **45.53**)* | **0.688** |

### Table 2: Full-Set Reference USM + Turn-to-Diarize Evaluation (`DiarizationLM-Gemma-4-E4B-v1`, All 211 Test Utterances)

| Benchmark Split (Full Test) | Utterances | USM WER (%) | USM + T2D Baseline WDER (%) | `DiarizationLM-Gemma-4-E4B-v1` WDER (%) | USM + T2D Baseline cpWER (%) | `DiarizationLM-Gemma-4-E4B-v1` cpWER (%) | SpkCntMAE (`v1` vs Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fisher (USM)** | 172 | 15.37 | 5.32 | **2.99** *(LG-TPST: 3.05)* | 20.88 | **17.62** *(LG-TPST: 17.72)* | **0.093** *(vs 0.215)* |
| **CALLHOME (USM)** | 20 | 15.22 | 7.74 | **4.92** *(LG-TPST: 5.02)* | 24.31 | **20.69** *(LG-TPST: 20.73)* | **0.000** *(vs 0.050)* |
| **ICSI (USM)** | 3 | 29.42 | 14.70 | **14.10** *(LG-TPST: 14.13)* | 43.90 | **43.32** *(LG-TPST: 43.36)* | **0.333** *(vs 0.333)* |
| **AMI (USM)** | 16 | 24.33 | 15.68 | **14.89** *(LG-TPST: 14.90)* | 40.32 | **39.57** *(LG-TPST: 39.59)* | **0.438** *(vs 0.500)* |

---

## 4. Quickstart with `diarizers` (`v1.0.0`)

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

### Interactive CPU Gradio Space (`hf_space/app.py`)

`diarizers` includes a 2-tab CPU Gradio application (`hf_space/app.py`) that runs entirely on standard CPU hardware:
- **Tab 1 (End-to-End Audio Diarization)**: Upload a WAV/MP3 clip to run `WhisperWordASR` (`openai/whisper-large-v3-turbo`) + `ECAPASpectralDiarizer` (`pyannote/segmentation-3.0` + `speechbrain/spkrec-ecapa-voxceleb` + `SpectralClusterer`) + `LG-TPST` on CPU and export diarized text, NIST RTTM, and SRT subtitles.
- **Tab 2 (Transcript Refinement)**: Paste any diarized transcript (`<speaker:1> ... <speaker:2> ...`), refine it with `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` GGUF via `llama.cpp`) + `LG-TPST`, and view word-level speaker transfer diagnostics.

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
