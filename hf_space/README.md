---
title: Diarizers v1.2.0 — Open-Source ASR, Diarization & DiarizationLM
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
models:
  - diarizers-community/DiarizationLM-Gemma-4-12B-v1
  - diarizers-community/DiarizationLM-Gemma-4-E4B-v2
  - wq2012/univ_full_eres2conformer_xl_diar_adapt
  - wq2012/univ_full_eres2conformer_v2_diar
  - openai/whisper-large-v3-turbo
  - pyannote/segmentation-3.0
  - speechbrain/spkrec-ecapa-voxceleb
short_description: End-to-End Open-Source ASR, Diarization & DiarizationLM Demo
---

# 🎙️ Diarizers (`v1.2.0`) — End-to-End Open-Source ASR, Speaker Diarization & DiarizationLM Demo

Interactive audio demo for the [`diarizers`](https://github.com/diarizers-community/diarizers) (`pip install -U diarizers`) `v1.2.0` end-to-end audio pipeline, featuring configurable **Quality vs. Speed Profiles**:

- **`max`**: `openai/whisper-large-v3-turbo` + [`FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) [`univ_full_eres2conformer_xl_diar_adapt`](https://huggingface.co/wq2012/univ_full_eres2conformer_xl_diar_adapt) + `ECAPA-TDNN` dual-encoder fusion (`powerset_ward`) + [`DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1) (**`3.79%` micro / `5.14%` macro WDER**, `7.92%` AMI, `0.174` `SpkCntMAE` across 211 held-out benchmark recordings)
- **`best-quality`** *(Default)*: `openai/whisper-large-v3-turbo` + [`wq2012/univ_full_eres2conformer_xl_diar_adapt`](https://huggingface.co/wq2012/univ_full_eres2conformer_xl_diar_adapt) (`powerset_ward`, segment-CMVN) + [`DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1) (**`3.82%` micro / `5.16%` macro WDER**, `0.177` `SpkCntMAE`)
- **`balanced`**: `openai/whisper-large-v3-turbo` + `ECAPA-TDNN` (`powerset_ward`, `15×` faster CPU encoder) + [`DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1) (`3.88%` micro / `5.25%` macro WDER)
- **`fast`**: `openai/whisper-large-v3-turbo` + `ECAPA-TDNN` (`powerset_ward`) + [`DiarizationLM-Gemma-4-E4B-v2`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v2) (`4.13%` micro WDER, `2×` faster CPU generation, `~7.6 GiB` RAM)
- **`acoustic-only`**: `openai/whisper-large-v3-turbo` + `ECAPA-TDNN` (`powerset_ward`, `5.65%` micro WDER, `<2 GiB` RAM, no LLM)

For the companion text-only multilingual `DiarizationLM` GGUF demo (`E4B-v2` & `12B-v1`), visit [`spaces/diarizers-community/DiarizationLM-GGUF`](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF).

## Running the Full Python Gradio App Locally

This directory includes both a zero-install browser-native static application (`index.html`) and a full Python Gradio application (`app.py`):

```bash
pip install -r requirements.txt
python app.py
```
