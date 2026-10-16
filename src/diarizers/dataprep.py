"""Dual-flavor and Locality-Preserving Oracle dataset preparation."""

from collections.abc import Sequence
import random
from typing import Any, Dict, List, Set

from diarizers import alignment


def build_locality_preserving_oracle(
    hyp_spks: Sequence[str],
    ora_spks: Sequence[str],
    max_multi_spk_span: int = 5,
    max_two_spk_span: int = 20,
    min_ghost_spk_words: int = 15,
) -> List[str]:
  """Constructs a locality-preserving oracle speaker sequence."""
  hyp_int = [int(x) for x in hyp_spks]
  ora_int = [int(x) for x in ora_spks]
  n = len(hyp_int)
  if n == 0 or len(ora_int) != n:
    return list(hyp_spks)

  spk_counts: Dict[int, int] = {}
  for s in hyp_int:
    spk_counts[s] = spk_counts.get(s, 0) + 1

  num_spks = len(spk_counts)
  max_span = max_two_spk_span if num_spks <= 2 else max_multi_spk_span
  min_major = max(5, int(n * 0.005))
  valid_major_spks: Set[int] = {
      s for s, cnt in spk_counts.items() if cnt >= min_major
  }
  tiny_ghost_spks: Set[int] = {
      s for s, cnt in spk_counts.items() if cnt < min_ghost_spk_words
  }

  res = list(hyp_int)
  i = 0
  while i < n:
    if ora_int[i] == hyp_int[i]:
      i += 1
      continue
    j = i
    while j < n and ora_int[j] != hyp_int[j]:
      j += 1
    span_len = j - i
    if span_len <= max_span:
      for k in range(i, j):
        if not valid_major_spks or ora_int[k] in valid_major_spks:
          res[k] = ora_int[k]
    i = j

  if tiny_ghost_spks and num_spks >= 3:
    for idx in range(n):
      if res[idx] in tiny_ghost_spks and ora_int[idx] in valid_major_spks:
        res[idx] = ora_int[idx]

  return [str(x) for x in res]


def degrade_speakers(
    ref_spks: Sequence[str],
    flip_prob: float = 0.08,
    max_span: int = 3,
    seed: int = 42,
) -> List[str]:
  """Simulates turn-boundary diarization jitter for deg2ref training."""
  rng = random.Random(seed)
  spks = [int(x) for x in alignment.speakers_transform(ref_spks)]
  n = len(spks)
  unique_spks = sorted(set(spks))
  if n == 0 or len(unique_spks) < 2:
    return [str(x) for x in spks]

  out = list(spks)
  for i in range(1, n):
    if spks[i] != spks[i - 1] and rng.random() < 0.35:
      span = rng.randint(1, max_span)
      if rng.random() < 0.5:
        for k in range(max(0, i - span), i):
          out[k] = spks[i]
      else:
        for k in range(i, min(n, i + span)):
          out[k] = spks[i - 1]
    elif rng.random() < flip_prob * 0.15:
      others = [s for s in unique_spks if s != out[i]]
      if others:
        out[i] = rng.choice(others)

  return alignment.speakers_transform(out)


def compute_utt_oracle_and_degraded(
    utt: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
  """Populates hyp_spk_oracle and ref_spk_degraded on an utterance dict."""
  res = dict(utt)
  ref_text = alignment.normalize_text(res.get("ref_text", ""))
  hyp_text = alignment.normalize_text(res.get("hyp_text", ref_text))
  ref_spks = alignment.speakers_transform(res.get("ref_spk", "").split())
  hyp_spks = alignment.speakers_transform(res.get("hyp_spk", "").split())

  res["ref_text"] = ref_text
  res["hyp_text"] = hyp_text
  res["ref_spk"] = " ".join(ref_spks)
  res["hyp_spk"] = " ".join(hyp_spks)

  if ref_text and hyp_text and ref_spks and hyp_spks:
    ora_transferred = alignment.transcript_preserving_speaker_transfer(
        src_text=ref_text,
        src_spk=" ".join(ref_spks),
        tgt_text=hyp_text,
        tgt_spk=" ".join(hyp_spks),
    )
    ora_int = [int(x) for x in ora_transferred.split()]
    hyp_int = [int(x) for x in hyp_spks]
    if len(ora_int) == len(hyp_int):
      # Permute oracle speaker IDs into hyp_spk's coordinate space
      aligned_ora = alignment.get_oracle_speakers(ora_int, ora_int, hyp_int)
      res["hyp_spk_oracle"] = " ".join(str(x) for x in aligned_ora)
    else:
      res["hyp_spk_oracle"] = ora_transferred

  if ref_spks and not res.get("ref_spk_degraded"):
    deg = degrade_speakers(ref_spks, seed=seed)
    res["ref_spk_degraded"] = " ".join(deg)

  return res


def build_sft_dataset(
    utterances: Sequence[Dict[str, Any]],
    include_deg2ref: bool = True,
    use_locality_oracle: bool = True,
    max_chunk_chars: int = 4000,
) -> List[Dict[str, str]]:
  """Builds dual-flavor (hyp2ora + deg2ref) SFT prompt/response records."""
  chunker = alignment.UtteranceChunker(
      max_input_length=max_chunk_chars,
      max_target_length=max_chunk_chars,
  )
  examples: List[Dict[str, str]] = []

  for idx, raw_utt in enumerate(utterances):
    utt = compute_utt_oracle_and_degraded(raw_utt, seed=42 + idx)
    utt_id = str(utt.get("utterance_id", f"utt_{idx}"))

    # Flavor 1: hyp2ora
    hyp_words = utt.get("hyp_text", "").split()
    hyp_spks = utt.get("hyp_spk", "").split()
    ora_spks = utt.get("hyp_spk_oracle", "").split()
    if (
        hyp_words
        and len(hyp_words) == len(hyp_spks)
        and len(hyp_words) == len(ora_spks)
    ):
      tgt_spks = (
          build_locality_preserving_oracle(hyp_spks, ora_spks)
          if use_locality_oracle
          else ora_spks
      )
      for ch in chunker.chunk_utterance_fields(
          hyp_words, hyp_spks, tgt_spks, utt_id=f"{utt_id}_hyp2ora"
      ):
        examples.append({
            "utterance_id": utt_id,
            "flavor": "hyp2ora",
            "prompt": ch["prompt"],
            "response": ch["target"],
            "text": ch["prompt"] + ch["target"],
        })

    # Flavor 2: deg2ref
    if include_deg2ref:
      ref_words = utt.get("ref_text", "").split()
      deg_spks = utt.get("ref_spk_degraded", "").split()
      ref_spks = utt.get("ref_spk", "").split()
      if (
          ref_words
          and len(ref_words) == len(deg_spks)
          and len(ref_words) == len(ref_spks)
      ):
        tgt_spks = (
            build_locality_preserving_oracle(deg_spks, ref_spks)
            if use_locality_oracle
            else ref_spks
        )
        for ch in chunker.chunk_utterance_fields(
            ref_words, deg_spks, tgt_spks, utt_id=f"{utt_id}_deg2ref"
        ):
          examples.append({
              "utterance_id": utt_id,
              "flavor": "deg2ref",
              "prompt": ch["prompt"],
              "response": ch["target"],
              "text": ch["prompt"] + ch["target"],
          })

  return examples


def generate_synthetic_dataset(
    num_utterances: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
  """Generates a realistic synthetic multi-speaker dataset for quickstart."""
  turns_pool = [
      ("hello good morning how are you doing today", "1"),
      ("i am doing great thanks for asking how about you", "2"),
      ("pretty good let us review the agenda for the project", "1"),
      ("sure i prepared the latency and accuracy benchmarks", "2"),
      ("what did we observe on the telephone conversations", "1"),
      ("word diarization error rate dropped below nine percent", "2"),
      ("and what about the four speaker meeting recordings", "3"),
      ("locality gated speaker transfer prevented monologue drift", "2"),
  ]
  utts: List[Dict[str, Any]] = []
  for u_idx in range(num_utterances):
    words: List[str] = []
    ref_spks: List[str] = []
    for t_idx in range(len(turns_pool)):
      phrase, spk = turns_pool[(t_idx + u_idx) % len(turns_pool)]
      if u_idx % 2 == 0 and spk == "3":
        spk = "1"
      t_words = phrase.split()
      words.extend(t_words)
      ref_spks.extend([spk] * len(t_words))

    ref_spks = alignment.speakers_transform(ref_spks)
    hyp_spks = degrade_speakers(ref_spks, flip_prob=0.12, seed=seed + u_idx)
    utt = compute_utt_oracle_and_degraded(
        {
            "utterance_id": f"synth_conv_{u_idx:03d}",
            "ref_text": " ".join(words),
            "ref_spk": " ".join(ref_spks),
            "hyp_text": " ".join(words),
            "hyp_spk": " ".join(hyp_spks),
        },
        seed=seed + u_idx,
    )
    utts.append(utt)
  return {"utterances": utts}
