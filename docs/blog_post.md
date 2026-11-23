# Diarizers 1.1.0: Powerset-Masked Multi-Speaker Diarization, FlaxSpeaker, and Multilingual DiarizationLM-Gemma-4

> `diarizers` (`v1.1.0`) is an Apache-2.0 open-source software package and end-to-end pipeline implementing and extending the algorithms described in *DiarizationLM* ([Wang et al., Interspeech 2024](https://arxiv.org/abs/2401.03506)), *Turn-to-Diarize* ([Xia et al., ICASSP 2022](https://arxiv.org/abs/2109.11641)), and *SpectralCluster* ([Wang et al., ICASSP 2018](https://arxiv.org/abs/1710.10468)).

---

## Executive Summary

When [`DiarizationLM`](https://arxiv.org/abs/2401.03506) was introduced at Interspeech 2024, it demonstrated that fine-tuned Large Language Models (LLMs) could substantially reduce Word Diarization Error Rate (WDER) on conversational speech by leveraging semantic and syntactic turn-boundary context (`"how are you"` followed by `"i am doing great"`). However, early community prototypes ([`DiarizersLM`](https://github.com/diarizers-community/DiarizersLM)) degraded to **43.99% WDER on Fisher** and **56.79% on AMI**, and even after repairing feature scaling and alignment in `diarizers` `v1.0.1`, sliding-window spectral clustering left **multi-party tabletop meetings at `20.13%` WDER on AMI and `23.10%` on ICSI** (`12.67%` Macro WDER across the 211-utterance benchmark suite).

In **`diarizers` (`v1.1.0`)** (`pip install -U diarizers`), we introduce an overlapping **powerset-masked acoustic frontend (`powerset_ward`)**, pure-PyTorch **[`FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) `SimAM-ERes2Conformer`** speaker encoders, multilingual **[`DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1)** and **[`DiarizationLM-Gemma-4-E4B-v2`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v2)** post-processing, and readable **quality/cost pipeline profiles** (`best-quality`, `max`, `fast`, `acoustic-only`):

1. **Macro WDER Cut by More Than Half (`12.67% → 5.18%`, `-59.1%` relative; Micro WDER `7.10% → 3.88%`, `-45.4%` relative)** across all **211 held-out benchmark recordings** (`172` Fisher, `20` CALLHOME, `3` ICSI, `16` AMI) with automatic speaker count estimation:
   - **Fisher (`172` Calls)**: **`2.61%` WDER** (`21.85%` cpWER) with `best-quality` (`ECAPA + 12B-v1`).
   - **CALLHOME-English (`20` Calls, Auto `K`)**: **`3.49%` WDER** (`Raw-TPST` / `3.57%` `LG-TPST` with `max` `FlaxSpeaker-XL + 12B-v1`; `3.58%` / `3.75%` with `best-quality` `ECAPA + 12B-v1`).
   - **ICSI Meetings (`3` Meetings, Auto `K`)**: Drops from **`23.10% → 6.33%` WDER** (`39.05%` all-speaker cpWER) with `max` (`FlaxSpeaker-XL + 12B-v1`) and **`6.41%`** with `best-quality` (`ECAPA + 12B-v1`).
   - **AMI Tabletop Meetings (`16` Meetings, Auto `K`)**: Drops from **`20.13% → 8.04%` WDER** (`34.37%` all-speaker cpWER) with `best-quality` (`ECAPA + 12B-v1`).
2. **Multilingual Generalization Across 11 Languages (`i18n_open_eval`, `20` Datasets, `2,147` Utterances)**: While English-only `E4B-v1` regresses non-English WDER (`21.43% → 21.57%`), **`E4B-v2`** reduces WDER to **`20.83%`** (`p < 0.0001`) at the exact same `7.6 GiB` CPU RAM footprint, and **`12B-v1`** reaches **`20.55%`** (`p < 0.0001`, winning `19/20` datasets and tying `1/20`).
3. **Configurable High-Level Quality vs. Cost Profiles (`profile=...` / `--profile`)**:
   - **`best-quality`** *(default)*: `ECAPA (powerset_ward)` + `Gemma-4-12B-v1` — **`3.88%` micro / `5.25%` macro WDER** (`5.18%` Raw), combining a `13×` faster CPU speaker encoder (`30 ms/seg`) with the strongest `12B` LLM.
   - **`max`**: `FlaxSpeaker-XL (powerset_ward)` + `Gemma-4-12B-v1` — **`3.49%` CALLHOME & `6.33%` ICSI WDER**, recommended on GPU and unconstrained `1–10` speaker meetings.
   - **`fast`**: `ECAPA (powerset_ward)` + `Gemma-4-E4B-v2` — **`4.13%` micro / `5.49%` macro WDER** at **`2×` faster CPU generation (`15.4 tok/s`) and half the RAM (`~7.6 GiB` GGUF)**.
   - **`acoustic-only`**: `ECAPA (powerset_ward)` without LLM — **`5.65%` micro / `6.96%` macro WDER** in `< 2 GiB` RAM.

---

## 1. What Changed in `v1.1.0`: Overcoming the Multi-Speaker Meeting Bottleneck

In `v1.0.1`, fixed `1.2s–2.0s` sliding windows frequently straddled rapid turn transitions and overlapped speech in 4–9 speaker tabletop meetings, causing global speaker-count errors (`SpkCntMAE = 3.333` on ICSI and `0.688` on AMI) that no text-only LLM could undo downstream. `v1.1.0` redesigns Stage 2 (`powerset_ward`) and upgrades Stage 4:

1. **Powerset-Masked Clean-Speaker Track Extraction**: Overlapping `10s` (`step = 2.5s`) `pyannote/segmentation-3.0` powerset posteriors isolate strictly single-speaker active frames (`cls == s_local + 1`, `min_clean_sec = 1.0s`) before pooling speaker embeddings, preventing cross-speaker contamination during overlapped speech.
2. **Ward Auto-`K` + Per-Chunk Cannot-Link Cosine EM**: Unit-normalized track embeddings are clustered via Ward linkage + closest-centroid cosine stopping, then refined with spherical cosine EM under exact Hungarian per-chunk cannot-link constraints (two local powerset speakers active in the same `10s` window cannot collapse into the same global speaker).
3. **Balanced Speaker-Centroid Centering ($\mu_{\text{bal}} = \frac{1}{K}\sum_{k=1}^{K} c_k$)**: Subtracting the unweighted mean of cluster centroids rather than the duration-weighted frame mean prevents dominant speakers in imbalanced meetings from shifting the origin and merging minority speakers (`-0.21 pp` WDER, `p < 0.00001`).
4. **Two-Pass Uniform Soft Powerset Frame Stitching + Sub-ms Soft Word Alignment**: Local chunk powerset probabilities are mapped to global speakers via both track-embedding cosine similarity and temporal overlap continuity across adjacent chunks, smoothed (`0.25s`), and integrated directly over each word's `[start, end]` interval (`assign_speakers_from_probs`).
5. **Standalone Pure-PyTorch `FlaxSpeaker` (`SimAM-ERes2Conformer`)**: `diarizers.flaxspeaker_encoder` ports `wq2012/FlaxSpeaker` (`univ_full_eres2conformer_large`, `v2_large`, `xl`, and `Hybrid-PFAS`) to pure PyTorch with `cos = 1.0000000` float32 parity against JAX and zero JAX/Flax runtime dependency.

*(In addition, `v1.1.0` retains all six `v1.0.1` fixes over the 2024 prototype: exact HTK decibel-scaled `10 * log10` ECAPA filterbanks, global + recording-level embedding centering, sub-millisecond `float64` timestamp precision with `[start, None]` safety and midpoint gap assignment, Myers' bit-parallel exact word-level Levenshtein `tpst_align`, Hungarian inter-chunk permutation reconciliation, and `LG-TPST` ghost-cluster absorption.)*

---

## 2. Verified Full-Set Benchmark Results

### Table 1: End-to-End `diarizers` `v1.1.0` (`powerset_ward`) Matrix (`211` Held-Out Test Recordings)

| Profile / Configuration | Fisher (172) WDER / cpWER (%) | CALLHOME (20) WDER / cpWER (%) | ICSI (3) WDER / cpWER (%) | AMI (16) WDER / cpWER (%) | Micro / Macro WDER (%) | SpkCntMAE (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Legacy `v1.0.1` (`sliding_spectral` + `ECAPA`) Base | 4.35 / 24.78 | 9.42 / 32.42 | 23.48 / 59.55 | 20.68 / 51.44 | 8.64 / 14.48 | 1.018 |
| Legacy `v1.0.1` + `E4B-v1` (`Raw-TPST`) | 2.64 / 21.96 | 4.84 / 25.87 | 23.11 / 59.44 | 20.09 / 50.91 | 7.10 / 12.67 | 1.018 |
| **`acoustic-only`** (`v1.1.0` `ECAPA` `powerset_ward`) | 4.49 / 24.58 | 6.07 / 26.81 | 7.82 / 40.14 | 9.48 / 35.68 | 5.65 / 6.96 | 0.239 |
| **`fast`** (`v1.1.0` `ECAPA` + **`E4B-v2`**, `LG-TPST`) | 2.86 / 22.16 | 4.05 / 24.85 | 6.59 / 39.25 | 8.45 / 34.84 | 4.13 / 5.49 | 0.239 |
| **`best-quality`** (`v1.1.0` `ECAPA` + **`12B-v1`**, `LG-TPST`) | **2.61 / 21.85** | 3.75 / 24.30 | 6.50 / 39.19 | 8.11 / 34.44 | **3.88** / 5.25 | 0.239 |
| **`best-quality`** (`v1.1.0` `ECAPA` + **`12B-v1`**, `Raw-TPST`) | 2.68 / 21.91 | 3.58 / 24.23 | 6.41 / 39.13 | **8.04 / 34.37** | 3.90 / **5.18** | **0.214** |
| **`max`** (`v1.1.0` `FlaxSpeaker-XL` + **`12B-v1`**, `LG-TPST`) | 2.65 / 21.87 | 3.57 / 24.33 | 6.39 / 39.08 | 8.19 / 34.43 | 3.90 / 5.20 | 0.249 |
| **`max`** (`v1.1.0` `FlaxSpeaker-XL` + **`12B-v1`**, `Raw-TPST`) | 2.69 / 21.91 | **3.49** / 24.30 | **6.33 / 39.05** | 8.22 / 34.46 | 3.93 / **5.18** | 0.236 |

*All `cpWER` values above use the full all-speaker reference denominator (`323,413` / `18,957` / `33,255` / `88,826` words). Paired 10,000-resample cluster bootstrap across all 211 utterances confirms `12B-v1` significantly outperforms `E4B-v2` (`ΔWDER = -0.25 pp [-0.36, -0.15], p < 0.00001`), while `ECAPA` and `FlaxSpeaker-XL` are statistically tied overall (`p = 0.649`) with `ECAPA` running `13×` faster on CPU (`30.3 ms/seg` vs `393.6 ms/seg`).*

### Table 2: Multilingual `i18n_open_eval` (`20` Datasets, `2,147` Utterances, `11` Languages) & CPU `Q4_K_M` Cost

| Model (`Apache-2.0`) | `i18n_open_eval` Micro / Macro WDER (%) | `Q4_K_M` Size | Peak CPU RSS (`n_ctx=6144`) | CPU Speed (`32` threads / `2` vCPUs) | Pipeline Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Acoustic Baseline (`v1.0.1` open frontend) | 21.43 / 20.58 | — | — | — | `acoustic-only` |
| `DiarizationLM-Gemma-4-E4B-v1` | 21.57 / 20.72 *(regresses non-EN)* | 5.30 GB | `7.59 GiB` | `15.1` / `5.1 tok/s` | Legacy |
| **`DiarizationLM-Gemma-4-E4B-v2`** | **20.87 / 20.01** *(LG: **20.83 / 19.98**)* | 5.30 GB | **`7.58 GiB`** | **`15.4` / `5.1 tok/s`** | **`fast`** |
| **`DiarizationLM-Gemma-4-12B-v1`** | **20.55 / 19.70** *(wins 19/20 sets)* | 7.38 GB | **`14.29 GiB`** | `7.8` / `2.6 tok/s` | **`best-quality` (Default) & `max`** |

---

## 3. Quickstart with `diarizers` (`v1.1.0`)

```bash
pip install -U "diarizers[audio,llm]"
```

```python
import diarizers

# Default: profile="best-quality" (Whisper-large-v3-turbo + ECAPA powerset_ward + Gemma-4-12B-v1)
# On CPU, automatically uses Q4_K_M GGUF when llama-cpp-python is installed!
pipe = diarizers.DiarizersPipeline.from_pretrained()

# Or choose "fast" (E4B-v2, 2x faster & ~7.6 GiB RAM), "max" (FlaxSpeaker-XL + 12B-v1), or "acoustic-only":
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
