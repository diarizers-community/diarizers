---
title: Diarizers v1.1.0 — Open-Source ASR, Diarization & DiarizationLM
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
  - wq2012/univ_full_eres2conformer_xl
  - openai/whisper-large-v3-turbo
  - pyannote/segmentation-3.0
  - speechbrain/spkrec-ecapa-voxceleb
short_description: End-to-End Open-Source ASR, Diarization & DiarizationLM Demo
---

# 🎙️ Diarizers (`v1.1.0`) — End-to-End Open-Source ASR, Speaker Diarization & DiarizationLM Demo

Interactive audio demo for the [`diarizers`](https://github.com/diarizers-community/diarizers) (`pip install -U diarizers`) `v1.1.0` end-to-end audio pipeline, featuring configurable **Quality vs. Speed Profiles**:

- **`best-quality`** *(Default)*: `openai/whisper-large-v3-turbo` + `ECAPA-TDNN` (`powerset_ward`) + [`DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1) (`3.88%` micro / `5.25%` macro WDER across 211 held-out benchmark recordings)
- **`max`**: `openai/whisper-large-v3-turbo` + [`FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) `SimAM-ERes2Conformer-XL` (`powerset_ward`) + [`DiarizationLM-Gemma-4-12B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-12B-v1) (`3.49%` CALLHOME / `6.33%` ICSI WDER)
- **`fast`**: `openai/whisper-large-v3-turbo` + `ECAPA-TDNN` (`powerset_ward`) + [`DiarizationLM-Gemma-4-E4B-v2`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v2) (`4.13%` micro WDER, `2×` faster CPU generation, `~7.6 GiB` RAM)
- **`acoustic-only`**: `openai/whisper-large-v3-turbo` + `ECAPA-TDNN` (`powerset_ward`, `5.65%` micro WDER, `<2 GiB` RAM, no LLM)

For the companion text-only multilingual `DiarizationLM` GGUF demo (`E4B-v2` & `12B-v1`), visit [`spaces/diarizers-community/DiarizationLM-GGUF`](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF).

## Running the Full Python Gradio App Locally

This directory includes both a zero-install browser-native static application (`index.html`) and a full Python Gradio application (`app.py`):

```bash
pip install -r requirements.txt
python app.py
```
