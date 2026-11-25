# Diarizers 1.2.0: Diarization-Trained FlaxSpeaker Checkpoints, Dual-Encoder Powerset Fusion, and Multilingual DiarizationLM-Gemma-4

> `diarizers` (`v1.2.0`) is an Apache-2.0 open-source software package and end-to-end pipeline implementing and extending the algorithms described in *DiarizationLM* ([Wang et al., Interspeech 2024](https://arxiv.org/abs/2401.03506)), *Turn-to-Diarize* ([Xia et al., ICASSP 2022](https://arxiv.org/abs/2109.11641)), and *SpectralCluster* ([Wang et al., ICASSP 2018](https://arxiv.org/abs/1710.10468)).

---

## Executive Summary

When [`DiarizationLM`](https://arxiv.org/abs/2401.03506) was introduced at Interspeech 2024, it demonstrated that fine-tuned Large Language Models (LLMs) could substantially reduce Word Diarization Error Rate (WDER) on conversational speech by leveraging semantic and syntactic turn-boundary context (`"how are you"` followed by `"i am doing great"`). However, early community prototypes ([`DiarizersLM`](https://github.com/diarizers-community/DiarizersLM)) degraded to **43.99% WDER on Fisher** and **56.79% on AMI**, and even after repairing feature scaling and alignment in `diarizers` `v1.0.1`, sliding-window spectral clustering left **multi-party tabletop meetings at `20.13%` WDER on AMI and `23.10%` on ICSI** (`12.67%` Macro WDER across the 211-utterance benchmark suite).

In **`diarizers` (`v1.2.0`)** (`pip install -U diarizers`), we combine the overlapping **powerset-masked acoustic frontend (`powerset_ward`)** with the new **diarization-trained [`FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker) `SimAM-ERes2Conformer` speaker encoders** (`univ_full_eres2conformer_xl_diar_adapt`, `xl_diar_pfas`, `v2_diar`), **per-segment CMVN (`flaxspeaker_cmvn_mode="segment"`)**, **dual-encoder powerset fusion (`xl_diar_adapt + ECAPA`)**, and multilingual **[`DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1)** / **[`E4B-v2`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v2)** post-processing across five quality/cost profiles (`max`, `best-quality`, `balanced`, `fast`, `acoustic-only`):

1. **Macro WDER Cut from `12.67% → 5.14%` (`-59.4%` relative) and Micro WDER from `7.10% → 3.79%` (`-46.6%` relative)** across all **211 held-out benchmark recordings** (`172` Fisher, `20` CALLHOME, `3` ICSI, `16` AMI) with automatic speaker count estimation (`SpkCntMAE` cut from `1.018 → 0.174`):
   - **Fisher (`172` Calls)**: **`2.55%` WDER** (`21.78%` cpWER) with `max` (`XL-Diar-Adapt + ECAPA` + `12B-v1`) and **`2.56%`** with `best-quality` (`XL-Diar-Adapt` + `12B-v1`).
   - **CALLHOME-English (`20` Calls, Auto `K`)**: **`3.51%`–`3.53%` WDER** (`LG-TPST`) with `v2_diar` / `xl_diar_adapt` + `12B-v1`.
   - **ICSI Meetings (`3` Meetings, Auto `K`)**: Drops from **`23.10% → 6.50%` WDER** (`6.33%` Raw) with `best-quality` / `max`.
   - **AMI Tabletop Meetings (`16` Meetings, Auto `K`)**: Drops from **`20.13% → 7.92%` WDER** (`34.24%` all-speaker cpWER) with `max` (`8.04%` with single-encoder `best-quality`).
2. **Per-Segment CMVN Architectural Discovery for Diarization-Trained Encoders**: Because the new `*_diar*` checkpoints were trained on variable-length `0.9s–2.2s` conversational turns (`Fisher + AMI + ICSI + CALLHOME`) with per-crop CMVN, applying per-segment CMVN (`flaxspeaker_cmvn_mode="segment"`, `ward_merge_threshold=0.75`) reduces speaker-count `SpkCntMAE` by **`-32%` (`0.261 → 0.177`)** and improves unconstrained `K ∈ [1..10]` auto-clustering by **`-0.59 pp` WDER (`6.72% → 6.13%`)**.
3. **Configurable High-Level Quality vs. Cost Profiles (`profile=...` / `--profile`)**:
   - **`max`**: `XL-Diar-Adapt + ECAPA` fusion (`w=0.35`) + `Gemma-4-12B-v1` — **`3.79%` micro / `5.14%` macro WDER** (`7.92%` AMI, `0.174` `SpkCntMAE`), overall SOTA accuracy.
   - **`best-quality`** *(default)*: `XL-Diar-Adapt` (`64.4M`, segment-CMVN) + `Gemma-4-12B-v1` — **`3.82%` micro / `5.16%` macro WDER** (`0.177` `SpkCntMAE`), single-encoder champion.
   - **`balanced`**: `ECAPA` (`20.8M`, `15×` faster CPU encoder) + `Gemma-4-12B-v1` — **`3.88%` micro / `5.25%` macro WDER**.
   - **`fast`**: `ECAPA` + `Gemma-4-E4B-v2` — **`4.13%` micro / `5.49%` macro WDER** at **`2×` faster CPU generation (`15.4 tok/s`) and half the RAM (`~7.6 GiB` GGUF)**.
   - **`acoustic-only`**: `ECAPA` without LLM — **`5.65%` micro / `6.96%` macro WDER** in `< 2 GiB` RAM.

---

## 1. What Changed in `v1.1.0` & `v1.2.0`: Overcoming the Multi-Speaker Meeting Bottleneck

In `v1.0.1`, fixed `1.2s–2.0s` sliding windows frequently straddled rapid turn transitions and overlapped speech in 4–9 speaker tabletop meetings, causing global speaker-count errors (`SpkCntMAE = 3.333` on ICSI and `0.688` on AMI) that no text-only LLM could undo downstream. `v1.1.0` and `v1.2.0` redesign Stage 2 (`powerset_ward`) and upgrade Stage 4:

1. **Powerset-Masked Clean-Speaker Track Extraction**: Overlapping `10s` (`step = 2.5s`) `pyannote/segmentation-3.0` powerset posteriors isolate strictly single-speaker active frames (`cls == s_local + 1`, `min_clean_sec = 1.0s`) before pooling speaker embeddings, preventing cross-speaker contamination during overlapped speech.
2. **Ward Auto-`K` + Per-Chunk Cannot-Link Cosine EM**: Unit-normalized track embeddings are clustered via Ward linkage + closest-centroid cosine stopping, then refined with spherical cosine EM under exact Hungarian per-chunk cannot-link constraints (two local powerset speakers active in the same `10s` window cannot collapse into the same global speaker).
3. **Balanced Speaker-Centroid Centering ($\mu_{\text{bal}} = \frac{1}{K}\sum_{k=1}^{K} c_k$)**: Subtracting the unweighted mean of cluster centroids rather than the duration-weighted frame mean prevents dominant speakers in imbalanced meetings from shifting the origin and merging minority speakers (`-0.21 pp` WDER, `p < 0.00001`).
4. **Two-Pass Uniform Soft Powerset Frame Stitching + Sub-ms Soft Word Alignment**: Local chunk powerset probabilities are mapped to global speakers via both track-embedding cosine similarity and temporal overlap continuity across adjacent chunks, smoothed (`0.25s`), and integrated directly over each word's `[start, end]` interval (`assign_speakers_from_probs`).
5. **Standalone Pure-PyTorch `FlaxSpeaker` (`SimAM-ERes2Conformer`)**: `diarizers.flaxspeaker_encoder` ports `wq2012/FlaxSpeaker` (`univ_full_eres2conformer_large`, `v2_large`, `xl`, and `Hybrid-PFAS`) to pure PyTorch with `cos = 1.0000000` float32 parity against JAX and zero JAX/Flax runtime dependency.

*(In addition, `v1.2.0` retains all six `v1.0.1` fixes over the 2024 prototype: exact HTK decibel-scaled `10 * log10` ECAPA filterbanks, global + recording-level embedding centering, sub-millisecond `float64` timestamp precision with `[start, None]` safety and midpoint gap assignment, Myers' bit-parallel exact word-level Levenshtein `tpst_align`, Hungarian inter-chunk permutation reconciliation, and `LG-TPST` ghost-cluster absorption.)*

---

## 2. Verified Full-Set Benchmark Results

### Table 1: End-to-End `diarizers` `v1.2.0` (`powerset_ward`) Matrix (`211` Held-Out Test Recordings)

| Profile / Configuration | Fisher (172) WDER / cpWER (%) | CALLHOME (20) WDER / cpWER (%) | ICSI (3) WDER / cpWER (%) | AMI (16) WDER / cpWER (%) | Micro / Macro WDER (%) | SpkCntMAE (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Legacy `v1.0.1` (`sliding_spectral` + `ECAPA`) + `E4B-v1` (`Raw`) | 2.64 / 21.96 | 4.84 / 25.87 | 23.11 / 59.44 | 20.09 / 50.91 | 7.10 / 12.67 | 1.018 |
| **`acoustic-only`** (`ECAPA` `powerset_ward`) | 4.49 / 24.58 | 6.07 / 26.81 | 7.82 / 40.14 | 9.48 / 35.68 | 5.65 / 6.96 | 0.239 |
| **`fast`** (`ECAPA` + **`E4B-v2`**, `LG-TPST`) | 2.86 / 22.16 | 4.05 / 24.85 | 6.59 / 39.25 | 8.45 / 34.84 | 4.13 / 5.49 | 0.239 |
| **`balanced`** (`ECAPA` + **`12B-v1`**, `LG-TPST`) | 2.61 / 21.85 | 3.75 / 24.30 | 6.50 / 39.19 | 8.11 / 34.44 | 3.88 / 5.25 | 0.239 |
| Legacy `v1.1.0` `FlaxSpeaker-XL` (VoxCeleb) + **`12B-v1`** (`LG-TPST`) | 2.65 / 21.87 | 3.57 / 24.33 | **6.39 / 39.08** | 8.19 / 34.43 | 3.90 / 5.20 | 0.249 |
| Compact `FlaxSpeaker-V2-Diar` (`36.3M`) + **`12B-v1`** (`LG-TPST`) | 2.56 / 21.79 | **3.51** / 24.27 | 6.58 / 39.25 | 8.19 / 34.51 | 3.85 / 5.21 | 0.190 |
| **`best-quality`** (`FlaxSpeaker-XL-Diar-Adapt` + **`12B-v1`**, `LG-TPST`) | 2.56 / **21.78** | 3.53 / 24.21 | 6.50 / 39.17 | 8.04 / 34.37 | **3.82 / 5.16** | 0.177 |
| **`max`** (`XL-Diar-Adapt + ECAPA` Fusion + **`12B-v1`**, `LG-TPST`) | **2.55 / 21.78** | 3.57 / **24.12** | 6.53 / 39.21 | **7.92 / 34.24** | **3.79 / 5.14** | **0.174** |

*All `cpWER` values above use the full all-speaker reference denominator (`323,413` / `18,957` / `33,255` / `88,826` words). Paired 10,000-resample cluster bootstrap across all 211 utterances confirms `12B-v1` significantly outperforms `E4B-v2` (`p < 0.00001`), while `XL-Diar-Adapt + ECAPA` fusion significantly outperforms both `ECAPA` (`p = 0.0126` WDER, `p = 0.0020` cpWER) and legacy `FlaxSpeaker-XL` (`p = 0.0042` WDER, `p = 0.0008` cpWER).*

### Table 2: Multilingual `i18n_open_eval` (`20` Datasets, `2,147` Utterances, `11` Languages) & CPU `Q4_K_M` Cost

| Model (`Apache-2.0`) | `i18n_open_eval` Micro / Macro WDER (%) | `Q4_K_M` Size | Peak CPU RSS (`n_ctx=6144`) | CPU Speed (`32` threads / `2` vCPUs) | Pipeline Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Acoustic Baseline (`v1.0.1` open frontend) | 21.43 / 20.58 | — | — | — | `acoustic-only` |
| `DiarizationLM-Gemma-4-E4B-v1` | 21.57 / 20.72 *(regresses non-EN)* | 5.30 GB | `7.59 GiB` | `15.1` / `5.1 tok/s` | Legacy |
| **`DiarizationLM-Gemma-4-E4B-v2`** | **20.87 / 20.01** *(LG: **20.83 / 19.98**)* | 5.30 GB | **`7.58 GiB`** | **`15.4` / `5.1 tok/s`** | **`fast`** |
| **`DiarizationLM-Gemma-4-12B-v1`** | **20.55 / 19.70** *(wins 19/20 sets)* | 7.38 GB | **`14.29 GiB`** | `7.8` / `2.6 tok/s` | **`best-quality` (Default) & `max`** |

---

## 3. Quickstart with `diarizers` (`v1.2.0`)

```bash
pip install -U "diarizers[audio,llm]"
```

```python
import diarizers

# Default: profile="best-quality" (Whisper-large-v3-turbo + FlaxSpeaker-XL-Diar-Adapt + Gemma-4-12B-v1)
# On CPU, automatically uses Q4_K_M GGUF when llama-cpp-python is installed!
pipe = diarizers.DiarizersPipeline.from_pretrained()

# Or choose "max" (XL-Diar-Adapt + ECAPA fusion + 12B-v1, 3.79% WDER), "balanced" (ECAPA + 12B-v1), "fast" (E4B-v2), or "acoustic-only":
pipe_fast = diarizers.DiarizersPipeline.from_pretrained(profile="fast")

result = pipe("conversation.wav")
print(result.to_diarized_text())
print(result.to_rttm(recording_id="conversation"))
```

Or from the command line:

```bash
diarizers run --audio conversation.wav --profile best-quality --output-srt out.srt
diarizers run --audio conversation.wav --profile fast --output-srt out_fast.srt
```

---

## 4. Interactive Hugging Face Spaces

- **End-to-End Audio Pipeline Demo ([`spaces/diarizers-community/diarizers`](https://huggingface.co/spaces/diarizers-community/diarizers))**: Compare **`Best Quality` (`12B-v1`)**, **`Max` (`FlaxSpeaker-XL + 12B-v1`)**, **`Fast` (`E4B-v2`)**, and **`Acoustic Only`** profiles side-by-side on multi-speaker audio and export diarized text, NIST RTTM, and SRT subtitles.
- **Text-Only DiarizationLM GGUF Demo ([`spaces/diarizers-community/DiarizationLM-GGUF`](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF))**: Paste any multilingual diarized transcript (`<speaker:1> ... <speaker:2> ...`), choose between `Fast (E4B-v2)` and `Best Quality (12B-v1)` (`Q4_K_M` GGUF), and stream word-level speaker boundary corrections on CPU.

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
