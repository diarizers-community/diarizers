"""Precomputes Diarizers v1.1.0 multi-profile example outputs."""

import json
import os
import sys
import time
from typing import Dict, Optional

REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
  sys.path.insert(0, REPO_SRC)

import diarizers  # noqa: E402

LOCAL_GGUF_DIR = os.environ.get(
    "DIARIZERS_LOCAL_GGUF_DIR", "/tmp/diarizers_research/gguf"
)


def _resolve_model_source(repo_id: str) -> str:
  fname = diarizers.resolve_default_gguf_filename(repo_id)
  cand = os.path.join(LOCAL_GGUF_DIR, fname)
  if os.path.isfile(cand):
    return cand
  return repo_id


def _build_profile_entry(
    ex_id: str,
    word_segments: list,
    t_asr: float,
    ac_spks: list,
    t_diar: float,
    refiner: Optional[diarizers.DiarizationLMRefiner],
) -> dict:
  words_list = [w.word for w in word_segments]
  words_str = " ".join(words_list)
  hyp_spk_str = " ".join(ac_spks)
  before_text = diarizers.create_diarized_text(words_list, ac_spks)

  if refiner is not None:
    t2 = time.time()
    raw_llm_spk, lg_llm_spk, _ = refiner.refine_utterance(
        words_str, hyp_spk_str
    )
    t_llm = time.time() - t2
    raw_spks = raw_llm_spk.split()
    lg_spks = lg_llm_spk.split()
  else:
    t_llm = 0.0
    raw_llm_spk = hyp_spk_str
    lg_llm_spk = hyp_spk_str
    raw_spks = list(ac_spks)
    lg_spks = list(ac_spks)

  after_lg_text = diarizers.create_diarized_text(words_list, lg_spks)
  after_raw_text = diarizers.create_diarized_text(words_list, raw_spks)

  diar_words_pre = [
      diarizers.DiarizedWord(
          word=w.word,
          start=w.start,
          end=w.end,
          acoustic_speaker=s,
          speaker=s,
      )
      for w, s in zip(word_segments, ac_spks)
  ]
  dt_pre = diarizers.DiarizedTranscript(
      utterance_id=ex_id,
      words=diar_words_pre,
      hyp_text=words_str,
      hyp_spk=hyp_spk_str,
      hyp_spk_llm_raw=hyp_spk_str,
      hyp_spk_llm=hyp_spk_str,
      diarized_before_llm=before_text,
      diarized_after_llm=before_text,
  )

  diar_words_post = [
      diarizers.DiarizedWord(
          word=w.word,
          start=w.start,
          end=w.end,
          acoustic_speaker=ac_s,
          speaker=fn_s,
      )
      for w, ac_s, fn_s in zip(word_segments, ac_spks, lg_spks)
  ]
  dt_post = diarizers.DiarizedTranscript(
      utterance_id=ex_id,
      words=diar_words_post,
      hyp_text=words_str,
      hyp_spk=hyp_spk_str,
      hyp_spk_llm_raw=raw_llm_spk,
      hyp_spk_llm=lg_llm_spk,
      diarized_before_llm=before_text,
      diarized_after_llm=after_lg_text,
  )
  changed = sum(1 for a, b in zip(ac_spks, lg_spks) if a != b)
  return {
      "timings": {
          "asr_sec": round(t_asr, 2),
          "diar_sec": round(t_diar, 2),
          "llm_sec": round(t_llm, 2),
          "total_sec": round(t_asr + t_diar + t_llm, 2),
      },
      "detected_speakers": len(set(lg_spks)),
      "relabeled_words": changed,
      "words": [
          {
              "word": w.word,
              "start": round(w.start, 3),
              "end": round(w.end, 3),
              "acoustic_speaker": ac_s,
              "raw_llm_speaker": r_s,
              "llm_speaker": fn_s,
          }
          for w, ac_s, r_s, fn_s in zip(
              word_segments, ac_spks, raw_spks, lg_spks
          )
      ],
      "before_llm_text": before_text,
      "after_llm_text": after_lg_text,
      "after_raw_text": after_raw_text,
      "rttm_before": dt_pre.to_rttm(),
      "srt_before": dt_pre.to_srt(),
      "rttm_after": dt_post.to_rttm(),
      "srt_after": dt_post.to_srt(),
  }


def process_example(
    ex_id: str,
    title: str,
    description: str,
    rel_wav_path: str,
    default_profile: str,
    preset_name: str,
    num_speakers: int,
    asr_model_name: str,
    asr_engine: diarizers.WhisperWordASR,
    ecapa_diarizer: diarizers.ECAPASpectralDiarizer,
    flax_diarizer: diarizers.ECAPASpectralDiarizer,
    ref_e4b: diarizers.DiarizationLMRefiner,
    ref_12b: diarizers.DiarizationLMRefiner,
) -> dict:
  """Computes outputs across best-quality, max, fast, and acoustic-only."""
  base_dir = os.path.dirname(os.path.abspath(__file__))
  wav_path = os.path.join(base_dir, rel_wav_path)

  t0 = time.time()
  word_segments = asr_engine(wav_path)
  t_asr = time.time() - t0
  n_spk = num_speakers if num_speakers > 0 else None

  t1 = time.time()
  turns_ecapa = ecapa_diarizer(wav_path, num_speakers=n_spk)
  t_ecapa = time.time() - t1
  if ecapa_diarizer.last_frame_probs is not None:
    ac_ecapa = diarizers.assign_speakers_from_probs(
        word_segments,
        ecapa_diarizer.last_frame_probs,
        ecapa_diarizer.last_frame_dt,
    )
  else:
    ac_ecapa = diarizers.assign_speakers(word_segments, turns_ecapa)

  t2 = time.time()
  turns_flax = flax_diarizer(wav_path, num_speakers=n_spk)
  t_flax = time.time() - t2
  if flax_diarizer.last_frame_probs is not None:
    ac_flax = diarizers.assign_speakers_from_probs(
        word_segments,
        flax_diarizer.last_frame_probs,
        flax_diarizer.last_frame_dt,
    )
  else:
    ac_flax = diarizers.assign_speakers(word_segments, turns_flax)

  profiles: Dict[str, dict] = {
      "best-quality": _build_profile_entry(
          ex_id, word_segments, t_asr, ac_ecapa, t_ecapa, ref_12b
      ),
      "max": _build_profile_entry(
          ex_id, word_segments, t_asr, ac_flax, t_flax, ref_12b
      ),
      "fast": _build_profile_entry(
          ex_id, word_segments, t_asr, ac_ecapa, t_ecapa, ref_e4b
      ),
      "acoustic-only": _build_profile_entry(
          ex_id, word_segments, t_asr, ac_ecapa, t_ecapa, None
      ),
  }
  primary = profiles[default_profile]
  return {
      "id": ex_id,
      "title": title,
      "description": description,
      "audio_url": rel_wav_path,
      "default_profile": default_profile,
      "preset_name": preset_name,
      "num_speakers": num_speakers,
      "asr_model": asr_model_name,
      "word_count": len(word_segments),
      "timings": primary["timings"],
      "detected_speakers": primary["detected_speakers"],
      "relabeled_words": primary["relabeled_words"],
      "words": primary["words"],
      "before_llm_text": primary["before_llm_text"],
      "after_llm_text": primary["after_llm_text"],
      "after_raw_text": primary["after_raw_text"],
      "rttm_before": primary["rttm_before"],
      "srt_before": primary["srt_before"],
      "rttm_after": primary["rttm_after"],
      "srt_after": primary["srt_after"],
      "profiles": profiles,
  }


def main() -> None:
  base_dir = os.path.dirname(os.path.abspath(__file__))
  asr_model = "openai/whisper-base.en"
  asr_engine = diarizers.WhisperWordASR(
      model_name_or_path=asr_model, device="cpu"
  )
  ecapa_diarizer = diarizers.ECAPASpectralDiarizer(
      preset=diarizers.UNIVERSAL_AUTO_PRESET, device="cpu"
  )
  flax_diarizer = diarizers.ECAPASpectralDiarizer(
      preset=diarizers.FLAXSPEAKER_XL_PRESET, device="cpu"
  )
  ref_e4b = diarizers.DiarizationLMRefiner(
      model_path_or_repo=_resolve_model_source(diarizers.DEFAULT_LLM_MODEL_V2),
      backend="gguf",
      n_ctx=2048,
      n_threads=8,
      prompt_config=diarizers.PromptConfig(use_locality_gate=True),
  )
  ref_12b = diarizers.DiarizationLMRefiner(
      model_path_or_repo=_resolve_model_source(diarizers.DEFAULT_LLM_MODEL_12B),
      backend="gguf",
      n_ctx=2048,
      n_threads=8,
      prompt_config=diarizers.PromptConfig(use_locality_gate=True),
  )

  examples = [
      process_example(
          ex_id="example_1_telephony",
          title="Example 1: 2-Speaker Project Call (12B-v1 vs E4B-v2)",
          description=(
              "Two-speaker call where short-clip acoustic diarization merges "
              "Turns 1 & 2; Best-Quality (Gemma-4-12B-v1) recovers all 3 "
              "speaker turns cleanly."
          ),
          rel_wav_path="examples/example_conversation.wav",
          default_profile="best-quality",
          preset_name="Universal Auto (ECAPA powerset_ward)",
          num_speakers=2,
          asr_model_name=asr_model,
          asr_engine=asr_engine,
          ecapa_diarizer=ecapa_diarizer,
          flax_diarizer=flax_diarizer,
          ref_e4b=ref_e4b,
          ref_12b=ref_12b,
      ),
      process_example(
          ex_id="example_2_meeting_3spk",
          title="Example 2: 3-Speaker Engineering Sync (FlaxSpeaker-XL)",
          description=(
              "Three-speaker engineering sync meeting comparing ECAPA vs "
              "FlaxSpeaker SimAM-ERes2Conformer-XL powerset_ward and "
              "LG-TPST vs Raw-TPST."
          ),
          rel_wav_path="examples/example_2_meeting_3spk.wav",
          default_profile="max",
          preset_name="FlaxSpeaker-XL Auto (powerset_ward)",
          num_speakers=3,
          asr_model_name=asr_model,
          asr_engine=asr_engine,
          ecapa_diarizer=ecapa_diarizer,
          flax_diarizer=flax_diarizer,
          ref_e4b=ref_e4b,
          ref_12b=ref_12b,
      ),
  ]
  out_json = os.path.join(base_dir, "examples", "metadata.json")
  with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"examples": examples}, f, indent=2)
  print("Wrote", out_json)
  for ex in examples:
    print("===", ex["id"], "===")
    for p_name, p_val in ex["profiles"].items():
      rel = p_val["relabeled_words"]
      tot = p_val["timings"]["total_sec"]
      print(f"  [{p_name}] relabeled={rel}/{ex['word_count']} total={tot}s")


if __name__ == "__main__":
  main()
