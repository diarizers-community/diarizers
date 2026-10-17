---
title: Diarizers v1.0.1 — Open-Source ASR, Diarization & DiarizationLM
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: End-to-End Apache-2.0 ASR + Speaker Diarization + DiarizationLM on CPU
---

# Diarizers v1.0.1 — Interactive ASR, Speaker Diarization & DiarizationLM Demo

## Overview

> This interactive demo and the `diarizers` (`v1.0.1`) library provide an Apache-2.0 open-source unified inference pipeline for:
> - **DiarizationLM**: *DiarizationLM: Speaker Diarization Post-Processing with Large Language Models* ([arXiv:2401.03506](https://arxiv.org/abs/2401.03506))
> - **SpectralCluster**: *Speaker Diarization with LSTM* ([arXiv:1710.10468](https://arxiv.org/abs/1710.10468))
> - **Turn-to-Diarize**: *Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection* ([arXiv:2109.11641](https://arxiv.org/abs/2109.11641))
>
> This open-source package pairs MIT/Apache-2.0 acoustic frontends (`openai/whisper-large-v3-turbo` + `pyannote/segmentation-3.0` + `speechbrain/spkrec-ecapa-voxceleb` + `SpectralClusterer`) with [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1) and **Locality-Gated Transcript-Preserving Speaker Transfer (LG-TPST)** entirely on CPU or GPU.

## Links & Resources

- **GitHub Repository**: [`diarizers-community/diarizers`](https://github.com/diarizers-community/diarizers)
- **End-to-End Audio Pipeline Space**: [`spaces/diarizers-community/diarizers`](https://huggingface.co/spaces/diarizers-community/diarizers)
- **Text-Only DiarizationLM GGUF Space**: [`spaces/diarizers-community/DiarizationLM-GGUF`](https://huggingface.co/spaces/diarizers-community/DiarizationLM-GGUF)
- **Pretrained DiarizationLM Model**: [`diarizers-community/DiarizationLM-Gemma-4-E4B-v1`](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)
- **Legacy Repository (Archived)**: [`diarizers-community/DiarizersLM`](https://github.com/diarizers-community/DiarizersLM)

## Key Capabilities

1. **End-to-End Audio Pipeline (`spaces/diarizers-community/diarizers`)**: Runs `WhisperWordASR` (`openai/whisper-large-v3-turbo`), `ECAPASpectralDiarizer` (`pyannote/segmentation-3.0` + `speechbrain/spkrec-ecapa-voxceleb` + Symmetric Normalized Laplacian Spectral Clustering), word-to-turn overlap alignment (`assign_speakers`), and `DiarizationLM-Gemma-4-E4B-v1` + `LG-TPST` refinement, exporting formatted text, NIST RTTM, and WebVTT/SRT subtitles.
2. **Companion Text-Only DiarizationLM GGUF Demo (`spaces/diarizers-community/DiarizationLM-GGUF`)**: Refine any diarized transcript (`<speaker:1> ... <speaker:2> ...`) directly with `DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` GGUF) and inspect word-by-word speaker boundary corrections guaranteed to preserve the exact input words via `TPST`.
