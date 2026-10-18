"""Precomputes Diarizers v1.0.1 + DiarizationLM GGUF example outputs."""

import json
import os
import sys
import time

REPO_SRC = "/usr/local/google/home/quanw/Code/github/diarizers/src"
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
  sys.path.insert(0, REPO_SRC)

import diarizers  # noqa: E402

DEFAULT_GGUF_REPO = "diarizers-community/DiarizationLM-Gemma-4-E4B-v1"
DEFAULT_GGUF_FILE = "DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf"


def process_example(
    ex_id: str,
    title: str,
    description: str,
    rel_wav_path: str,
    preset_name: str,
    preset: diarizers.DiarizationPreset,
    num_speakers: int,
    asr_model_name: str = "openai/whisper-base.en",
) -> dict:
  base_dir = os.path.dirname(os.path.abspath(__file__))
  wav_path = os.path.join(base_dir, rel_wav_path)

  t0 = time.time()
  asr = diarizers.WhisperWordASR(
      model_name_or_path=asr_model_name, device="cpu"
  )
  word_segments = asr(wav_path)
  t_asr = time.time() - t0

  t1 = time.time()
  diar = diarizers.ECAPASpectralDiarizer(preset=preset, device="cpu")
  n_spk = num_speakers if num_speakers > 0 else None
  turns = diar(wav_path, num_speakers=n_spk)
  t_diar = time.time() - t1

  asr_dicts = [
      {"word": w.word, "timestamp": (w.start, w.end)} for w in word_segments
  ]
  turn_dicts = [
      {"segment": {"start": t.start, "end": t.end}, "speaker": t.speaker}
      for t in turns
  ]
  acoustic_spks = diarizers.assign_speakers(asr_dicts, turn_dicts)
  words_list = [w.word for w in word_segments]
  words_str = " ".join(words_list)
  hyp_spk_str = " ".join(acoustic_spks)
  before_llm_text = diarizers.create_diarized_text(words_list, acoustic_spks)

  diar_words_pre = [
      diarizers.DiarizedWord(
          word=w.word,
          start=w.start,
          end=w.end,
          acoustic_speaker=s,
          speaker=s,
      )
      for w, s in zip(word_segments, acoustic_spks)
  ]
  dt_pre = diarizers.DiarizedTranscript(
      utterance_id=ex_id,
      words=diar_words_pre,
      hyp_text=words_str,
      hyp_spk=hyp_spk_str,
      hyp_spk_llm_raw=hyp_spk_str,
      hyp_spk_llm=hyp_spk_str,
      diarized_before_llm=before_llm_text,
      diarized_after_llm=before_llm_text,
  )

  t2 = time.time()
  refiner = diarizers.DiarizationLMRefiner(
      model_path_or_repo=DEFAULT_GGUF_REPO,
      gguf_filename=DEFAULT_GGUF_FILE,
      backend="gguf",
      prompt_config=diarizers.PromptConfig(use_locality_gate=True),
  )
  raw_llm_spk, lg_llm_spk, _ = refiner.refine_utterance(
      words_str, hyp_spk_str
  )
  t_llm = time.time() - t2

  final_spks = lg_llm_spk.split()
  raw_spks = raw_llm_spk.split()
  after_llm_text = diarizers.create_diarized_text(words_list, final_spks)
  after_raw_text = diarizers.create_diarized_text(words_list, raw_spks)

  diar_words_post = [
      diarizers.DiarizedWord(
          word=w.word,
          start=w.start,
          end=w.end,
          acoustic_speaker=ac_s,
          speaker=fn_s,
      )
      for w, ac_s, fn_s in zip(word_segments, acoustic_spks, final_spks)
  ]
  dt_post = diarizers.DiarizedTranscript(
      utterance_id=ex_id,
      words=diar_words_post,
      hyp_text=words_str,
      hyp_spk=hyp_spk_str,
      hyp_spk_llm_raw=raw_llm_spk,
      hyp_spk_llm=lg_llm_spk,
      diarized_before_llm=before_llm_text,
      diarized_after_llm=after_llm_text,
  )

  changed = sum(1 for a, b in zip(acoustic_spks, final_spks) if a != b)
  return {
      "id": ex_id,
      "title": title,
      "description": description,
      "audio_url": rel_wav_path,
      "preset_name": preset_name,
      "num_speakers": num_speakers,
      "asr_model": asr_model_name,
      "timings": {
          "asr_sec": round(t_asr, 2),
          "diar_sec": round(t_diar, 2),
          "llm_sec": round(t_llm, 2),
          "total_sec": round(t_asr + t_diar + t_llm, 2),
      },
      "word_count": len(words_list),
      "detected_speakers": len(set(final_spks)),
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
              word_segments, acoustic_spks, raw_spks, final_spks
          )
      ],
      "before_llm_text": before_llm_text,
      "after_llm_text": after_llm_text,
      "after_raw_text": after_raw_text,
      "rttm_before": dt_pre.to_rttm(),
      "srt_before": dt_pre.to_srt(),
      "rttm_after": dt_post.to_rttm(),
      "srt_after": dt_post.to_srt(),
  }


def main() -> None:
  base_dir = os.path.dirname(os.path.abspath(__file__))
  examples = [
      process_example(
          ex_id="example_1_telephony",
          title="Example 1: 2-Speaker Project Call (Boundary Correction)",
          description=(
              "Two-speaker dialogue where acoustic diarization places the "
              "turn boundary before 'meeting'; DiarizationLM-Gemma-4-E4B-v1 "
              "restores 'meeting' to Speaker 1."
          ),
          rel_wav_path="examples/example_conversation.wav",
          preset_name="Fisher Telephony (2 speakers)",
          preset=diarizers.FISHER_TELEPHONY_PRESET,
          num_speakers=2,
      ),
      process_example(
          ex_id="example_2_meeting_3spk",
          title="Example 2: 3-Speaker Engineering Sync Meeting",
          description=(
              "Three-speaker engineering sync meeting diarized with "
              "AMI Meeting Preset + SpectralClusterer + "
              "DiarizationLM-Gemma-4-E4B-v1 (LG-TPST)."
          ),
          rel_wav_path="examples/example_2_meeting_3spk.wav",
          preset_name="AMI Meeting (3-5 speakers)",
          preset=diarizers.AMI_MEETING_PRESET,
          num_speakers=3,
      ),
  ]
  out_json = os.path.join(base_dir, "examples", "metadata.json")
  with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"examples": examples}, f, indent=2)
  print("Wrote", out_json)
  for ex in examples:
    print("===", ex["id"], "===")
    print("BEFORE:", ex["before_llm_text"])
    print("AFTER: ", ex["after_llm_text"])
    print("RELABELED:", ex["relabeled_words"], "/", ex["word_count"])


if __name__ == "__main__":
  main()
