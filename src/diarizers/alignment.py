"""Word-speaker alignment, TPST, LG-TPST, and chunk reconciliation."""

from collections.abc import Generator, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize

from diarizers import configs
from diarizers import types

PUNCTUATIONS = [",", ".", "_", "?", "!", "-", '"', "'"]


def normalize_text(text: str) -> str:
  """Normalizes text to lowercase and strips standard punctuation."""
  text_lower = text.lower().strip()
  words = text_lower.split()
  new_words: List[str] = []
  for word in words:
    new_word = word
    for punc in PUNCTUATIONS:
      replaced = new_word.replace(punc, "")
      if len(replaced.split()) == 1:
        new_word = replaced
    if new_word:
      new_words.append(new_word)
  return " ".join(new_words)


def speakers_transform(speakers: Sequence[Union[str, int]]) -> List[str]:
  """Re-indexes speaker labels into 1, 2, ... by order of first appearance."""
  spk_map: Dict[str, int] = {}
  idx = 0
  transformed: List[str] = []
  for spk in speakers:
    key = str(spk).strip()
    if key not in spk_map:
      idx += 1
      spk_map[key] = idx
    transformed.append(str(spk_map[key]))
  return transformed


def _extract_word_bounds(
    item: Union[types.WordSegment, Dict[str, Any]],
) -> Tuple[float, float]:
  """Extracts (start, end) seconds from a WordSegment or dict."""
  if isinstance(item, types.WordSegment):
    return float(item.start), float(item.end)
  if "timestamp" in item and isinstance(item["timestamp"], (tuple, list)):
    ts = item["timestamp"]
    s = float(ts[0]) if ts[0] is not None else 0.0
    e = float(ts[1]) if len(ts) > 1 and ts[1] is not None else s + 0.1
    return s, e
  return float(item.get("start", 0.0)), float(item.get("end", 0.0))


def _extract_turn_bounds(
    item: Union[types.SpeakerTurn, Dict[str, Any]],
) -> Tuple[float, float, str]:
  """Extracts (start, end, speaker) from a SpeakerTurn or dict."""
  if isinstance(item, types.SpeakerTurn):
    return float(item.start), float(item.end), str(item.speaker)
  if "segment" in item and isinstance(item["segment"], dict):
    seg = item["segment"]
    return (
        float(seg.get("start", 0.0)),
        float(seg.get("end", 0.0)),
        str(item.get("speaker", "1")),
    )
  return (
      float(item.get("start", 0.0)),
      float(item.get("end", 0.0)),
      str(item.get("speaker", "1")),
  )


def assign_speakers(
    asr_words: Sequence[Union[types.WordSegment, Dict[str, Any]]],
    diar_segments: Sequence[Union[types.SpeakerTurn, Dict[str, Any]]],
    normalize_speaker_ids: bool = True,
) -> List[str]:
  """Assigns each ASR word to a speaker via max overlap and unsigned gap."""
  if not asr_words:
    return []
  if not diar_segments:
    return ["1"] * len(asr_words)

  parsed_turns = [_extract_turn_bounds(seg) for seg in diar_segments]
  assigned_speakers: List[str] = []

  for w in asr_words:
    ws, we = _extract_word_bounds(w)
    if we < ws:
      we = ws

    best_spk = ""
    best_overlap = 0.0
    for ss, se, spk in parsed_turns:
      overlap = min(we, se) - max(ws, ss)
      if overlap > best_overlap:
        best_overlap = overlap
        best_spk = spk

    if not best_spk:
      min_gap = float("inf")
      mid = 0.5 * (ws + we)
      for ss, se, spk in parsed_turns:
        gap = min(abs(mid - ss), abs(mid - se))
        if gap < min_gap:
          min_gap = gap
          best_spk = spk

    assigned_speakers.append(best_spk if best_spk else "1")

  if normalize_speaker_ids:
    return speakers_transform(assigned_speakers)
  return assigned_speakers


def create_diarized_text(
    word_labels: Sequence[str],
    speaker_labels: Sequence[Union[str, int]],
    speaker_prefix: str = configs.SPEAKER_PREFIX,
    speaker_suffix: str = configs.SPEAKER_SUFFIX,
    use_new_line: bool = False,
) -> str:
  """Formats words with speaker boundary tags on speaker changes."""
  output: List[str] = []
  previous_speaker: Optional[str] = None
  for word, spk in zip(word_labels, speaker_labels):
    spk_str = str(spk)
    if spk_str != previous_speaker:
      if previous_speaker is not None and use_new_line:
        output.append("\n")
      output.append(f"{speaker_prefix}{spk_str}{speaker_suffix}")
    output.append(word)
    previous_speaker = spk_str
  return " ".join(output)


def extract_text_and_spk(
    diarized_text: str,
    speaker_prefix: str = configs.SPEAKER_PREFIX,
    speaker_suffix: str = configs.SPEAKER_SUFFIX,
    max_speaker_id: int = 20,
) -> Tuple[str, str]:
  """Extracts plain word sequence and speaker sequence from diarized text."""
  spk = "1"
  previous_spk = "1"
  result_text: List[str] = []
  result_spk: List[str] = []
  for word in diarized_text.split():
    if word.startswith(speaker_prefix):
      suf_pos = word.find(speaker_suffix, len(speaker_prefix))
      if suf_pos != -1:
        extracted = word[len(speaker_prefix):suf_pos]
        trailing = word[suf_pos + len(speaker_suffix):]
      else:
        extracted = word[len(speaker_prefix):]
        trailing = ""
      try:
        spk_int = int(extracted)
        if 1 <= spk_int <= max_speaker_id:
          spk = str(spk_int)
          previous_spk = spk
        else:
          spk = previous_spk
      except ValueError:
        spk = previous_spk
      if trailing:
        result_text.append(trailing)
        result_spk.append(spk)
    else:
      result_text.append(word)
      result_spk.append(spk)
  return " ".join(result_text), " ".join(result_spk)


def _dp_levenshtein(
    words1: Sequence[str], words2: Sequence[str]
) -> Tuple[int, List[Tuple[int, int]]]:
  """Exact dynamic-programming word-level Levenshtein alignment."""
  n, m = len(words1), len(words2)
  if n == 0 and m == 0:
    return 0, []
  if n == 0:
    return m, [(-1, j) for j in range(m)]
  if m == 0:
    return n, [(i, -1) for i in range(n)]

  bp = np.zeros((n + 1, m + 1), dtype=np.uint8)
  bp[0, 1:] = 1
  bp[1:, 0] = 2

  prev_row = np.arange(m + 1, dtype=np.int32)
  curr_row = np.zeros(m + 1, dtype=np.int32)

  for i in range(1, n + 1):
    w1 = words1[i - 1]
    curr_row[0] = i
    for j in range(1, m + 1):
      cost = 0 if w1 == words2[j - 1] else 1
      diag = prev_row[j - 1] + cost
      del_c = prev_row[j] + 1
      ins_c = curr_row[j - 1] + 1
      if diag <= ins_c and diag <= del_c:
        curr_row[j] = diag
        bp[i, j] = 0
      elif ins_c <= del_c:
        curr_row[j] = ins_c
        bp[i, j] = 1
      else:
        curr_row[j] = del_c
        bp[i, j] = 2
    prev_row[:] = curr_row[:]

  dist = int(prev_row[m])
  i, j = n, m
  align: List[Tuple[int, int]] = []
  while i > 0 or j > 0:
    code = int(bp[i, j])
    if i > 0 and j > 0 and code == 0:
      align.append((i - 1, j - 1))
      i -= 1
      j -= 1
    elif j > 0 and (i == 0 or code == 1):
      align.append((-1, j - 1))
      j -= 1
    else:
      align.append((i - 1, -1))
      i -= 1

  align.reverse()
  return dist, align


def levenshtein_distance(
    words1: Sequence[str], words2: Sequence[str]
) -> int:
  """Computes exact word-level Levenshtein distance via Myers bit-parallel."""
  if not words1:
    return len(words2)
  if not words2:
    return len(words1)
  if len(words1) > len(words2):
    words1, words2 = words2, words1
  n = len(words1)
  peq: Dict[str, int] = {}
  for i, w in enumerate(words1):
    peq[w] = peq.get(w, 0) | (1 << i)
  mask = (1 << n) - 1
  pv = mask
  mv = 0
  score = n
  high_bit = 1 << (n - 1)
  for w in words2:
    eq = peq.get(w, 0)
    xv = eq | mv
    xh = (((eq & pv) + pv) ^ pv) | eq
    ph = mv | (~(xh | pv) & mask)
    mh = pv & xh
    if ph & high_bit:
      score += 1
    elif mh & high_bit:
      score -= 1
    ph = ((ph << 1) | 1) & mask
    mh = (mh << 1) & mask
    pv = mh | (~(xv | ph) & mask)
    mv = ph & xv
  return score


def levenshtein_with_edits(
    str1: str, str2: str
) -> Tuple[int, List[Tuple[int, int]]]:
  """Computes exact word-level Levenshtein distance and alignment pairs."""
  words1 = str1.split()
  words2 = str2.split()
  n, m = len(words1), len(words2)
  if n == 0 and m == 0:
    return 0, []
  if n == 0:
    return m, [(-1, j) for j in range(m)]
  if m == 0:
    return n, [(i, -1) for i in range(n)]

  peq: Dict[str, int] = {}
  for i, w in enumerate(words1):
    peq[w] = peq.get(w, 0) | (1 << i)

  mask = (1 << n) - 1
  mask_n1 = (1 << (n + 1)) - 1
  pv = mask
  mv = 0
  score = n
  high_bit = 1 << (n - 1)

  pv_hist = [pv]
  mv_hist = [mv]
  ph_hist = [0]
  mh_hist = [0]

  for w in words2:
    eq = peq.get(w, 0)
    xv = eq | mv
    xh = (((eq & pv) + pv) ^ pv) | eq
    ph = mv | (~(xh | pv) & mask)
    mh = pv & xh
    if ph & high_bit:
      score += 1
    elif mh & high_bit:
      score -= 1
    ph_shifted = ((ph << 1) | 1) & mask_n1
    mh_shifted = (mh << 1) & mask_n1
    pv = mh_shifted | (~(xv | ph_shifted) & mask)
    mv = ph_shifted & xv
    pv_hist.append(pv)
    mv_hist.append(mv)
    ph_hist.append(ph_shifted)
    mh_hist.append(mh_shifted)

  align: List[Tuple[int, int]] = []
  i, j = n, m
  while i > 0 or j > 0:
    if i > 0 and j > 0:
      dv = ((pv_hist[j] >> (i - 1)) & 1) - ((mv_hist[j] >> (i - 1)) & 1)
      dh = ((ph_hist[j] >> (i - 1)) & 1) - ((mh_hist[j] >> (i - 1)) & 1)
      cost = 0 if words1[i - 1] == words2[j - 1] else 1
      if dv + dh == cost:
        align.append((i - 1, j - 1))
        i -= 1
        j -= 1
        continue
    if j > 0 and ((ph_hist[j] >> i) & 1):
      align.append((-1, j - 1))
      j -= 1
    else:
      align.append((i - 1, -1))
      i -= 1

  align.reverse()
  return score, align


def get_oracle_speakers(
    hyp_spk: Sequence[int],
    hyp_spk_align: Sequence[int],
    ref_spk_align: Sequence[int],
) -> List[int]:
  """Aligns hypothesis speaker IDs to reference speaker IDs via Hungarian."""
  if not hyp_spk:
    return []
  if not hyp_spk_align or not ref_spk_align:
    return list(hyp_spk)

  max_spk = max(max(hyp_spk), max(hyp_spk_align), max(ref_spk_align))
  cost_matrix = np.zeros((max_spk, max_spk), dtype=np.int64)
  for h_s, r_s in zip(hyp_spk_align, ref_spk_align):
    if 1 <= h_s <= max_spk and 1 <= r_s <= max_spk:
      cost_matrix[h_s - 1, r_s - 1] += 1

  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)
  mapping = {int(r) + 1: int(c) + 1 for r, c in zip(row_ind, col_ind)}
  return [mapping.get(int(s), int(s)) for s in hyp_spk]


def transcript_preserving_speaker_transfer(
    src_text: str,
    src_spk: str,
    tgt_text: str,
    tgt_spk: str,
) -> str:
  """Transfers speakers from src onto tgt words without modifying tgt_text."""
  tgt_words = normalize_text(tgt_text).split()
  tgt_spks = tgt_spk.split()
  if len(tgt_words) != len(tgt_spks):
    tgt_words = tgt_text.split()
  if not tgt_spks:
    return ""

  src_words = normalize_text(src_text).split()
  src_spks = src_spk.split()
  if len(src_words) != len(src_spks):
    src_words = src_text.split()
  if not src_words or len(src_words) != len(src_spks):
    return " ".join(tgt_spks)

  _, align = levenshtein_with_edits(" ".join(tgt_words), " ".join(src_words))
  matched_src: List[int] = []
  matched_tgt: List[int] = []
  for t_idx, s_idx in align:
    if (
        t_idx != -1
        and s_idx != -1
        and t_idx < len(tgt_spks)
        and s_idx < len(src_spks)
    ):
      try:
        matched_src.append(int(src_spks[s_idx]))
        matched_tgt.append(int(tgt_spks[t_idx]))
      except ValueError:
        pass

  try:
    src_int = [int(x) for x in src_spks]
    if matched_src and matched_tgt:
      mapped_src = [
          str(x) for x in get_oracle_speakers(src_int, matched_src, matched_tgt)
      ]
    else:
      mapped_src = list(src_spks)
  except ValueError:
    mapped_src = list(src_spks)

  transferred = list(tgt_spks)
  last_valid_spk = mapped_src[0]

  for t_idx, s_idx in align:
    if s_idx != -1 and s_idx < len(mapped_src):
      last_valid_spk = mapped_src[s_idx]
    if t_idx != -1 and t_idx < len(transferred):
      if s_idx != -1 and s_idx < len(mapped_src):
        transferred[t_idx] = mapped_src[s_idx]
      else:
        transferred[t_idx] = (
            tgt_spks[t_idx] if tgt_spks[t_idx] else last_valid_spk
        )

  return " ".join(transferred)


def apply_locality_gated_speaker_transfer(
    hyp_spks: Sequence[str],
    llm_spks: Sequence[str],
    max_local_span_words: int = 5,
    max_ghost_cluster_words: int = 15,
    max_two_spk_span_words: int = 15,
) -> List[str]:
  """Applies Locality-Gated TPST (LG-TPST) to preserve monologue speakers."""
  n = len(hyp_spks)
  if n == 0 or len(llm_spks) != n:
    return list(hyp_spks)

  hyp_int = [int(x) for x in speakers_transform(hyp_spks)]
  llm_raw_int = [int(x) for x in speakers_transform(llm_spks)]

  # Align llm_raw_int to hyp_int coordinate space via Hungarian matching
  aligned_llm_int = get_oracle_speakers(llm_raw_int, llm_raw_int, hyp_int)

  spk_counts: Dict[int, int] = {}
  for s in hyp_int:
    spk_counts[s] = spk_counts.get(s, 0) + 1

  num_spks = len(spk_counts)
  min_major = max(5, int(n * 0.005))
  ghost_thresh = (
      min(max_ghost_cluster_words, max(2, int(n * 0.01)))
      if num_spks >= 3
      else 0
  )
  major_spks = {
      s for s, c in spk_counts.items() if c >= min_major and c > ghost_thresh
  }
  if not major_spks:
    major_spks = {s for s, c in spk_counts.items() if c > ghost_thresh}
  ghost_spks = {s for s, c in spk_counts.items() if c <= ghost_thresh}

  # Step 1: Gate contiguous modification spans (5 words for >=3 spks, 15 for 2)
  span_limit = (
      max_local_span_words
      if num_spks >= 3
      else max(max_local_span_words, max_two_spk_span_words)
  )
  result_int = list(hyp_int)
  i = 0
  while i < n:
    if aligned_llm_int[i] == hyp_int[i]:
      i += 1
      continue
    j = i
    while j < n and aligned_llm_int[j] != hyp_int[j]:
      j += 1
    span_len = j - i
    if span_len <= span_limit:
      for k in range(i, j):
        if num_spks <= 2 or not major_spks or aligned_llm_int[k] in major_spks:
          result_int[k] = aligned_llm_int[k]
    i = j

  # Step 2: Absorb tiny ghost speaker clusters in >=3 speaker conversations
  if ghost_spks and num_spks >= 3 and len(major_spks) >= 2:
    for idx in range(n):
      if result_int[idx] in ghost_spks:
        if aligned_llm_int[idx] in major_spks:
          result_int[idx] = aligned_llm_int[idx]
        else:
          for offset in range(1, n):
            if idx - offset >= 0 and result_int[idx - offset] in major_spks:
              result_int[idx] = result_int[idx - offset]
              break
            if idx + offset < n and result_int[idx + offset] in major_spks:
              result_int[idx] = result_int[idx + offset]
              break

  return [str(x) for x in result_int]


def locality_gated_speaker_transfer(
    hyp_text: Union[str, Sequence[str]] = "",
    hyp_spk: Union[str, Sequence[str]] = "",
    llm_text: Optional[str] = None,
    llm_spk: Optional[str] = None,
    max_local_span_words: int = 5,
    max_ghost_cluster_words: int = 15,
    max_two_spk_span_words: int = 15,
) -> str:
  """Transfers speakers from LLM output onto hyp_text with LG-TPST gating."""
  if llm_text is None and llm_spk is None:
    # Called as locality_gated_speaker_transfer(hyp_spks, llm_spks)
    h_seq = (
        hyp_text.split() if isinstance(hyp_text, str) else list(hyp_text)
    )
    l_seq = (
        hyp_spk.split() if isinstance(hyp_spk, str) else list(hyp_spk)
    )
    gated = apply_locality_gated_speaker_transfer(
        h_seq,
        l_seq,
        max_local_span_words=max_local_span_words,
        max_ghost_cluster_words=max_ghost_cluster_words,
        max_two_spk_span_words=max_two_spk_span_words,
    )
    return " ".join(gated)

  h_text_str = (
      hyp_text if isinstance(hyp_text, str) else " ".join(hyp_text)
  )
  h_spk_str = hyp_spk if isinstance(hyp_spk, str) else " ".join(hyp_spk)
  l_text_str = llm_text or h_text_str
  l_spk_str = llm_spk or h_spk_str

  tpst_spk_str = transcript_preserving_speaker_transfer(
      src_text=l_text_str,
      src_spk=l_spk_str,
      tgt_text=h_text_str,
      tgt_spk=h_spk_str,
  )
  gated = apply_locality_gated_speaker_transfer(
      h_spk_str.split(),
      tpst_spk_str.split(),
      max_local_span_words=max_local_span_words,
      max_ghost_cluster_words=max_ghost_cluster_words,
      max_two_spk_span_words=max_two_spk_span_words,
  )
  return " ".join(gated)


tpst_align = transcript_preserving_speaker_transfer


class UtteranceChunker:
  """Splits utterances into length-constrained chunks in linear O(N) time."""

  def __init__(
      self,
      max_input_length: int = configs.EMIT_INPUT_LENGTH,
      max_target_length: int = configs.EMIT_TARGET_LENGTH,
      prompt_prefix: str = configs.PROMPT_PREFIX,
      prompt_suffix: str = configs.PROMPT_SUFFIX,
      completion_suffix: str = configs.COMPLETION_SUFFIX,
      speaker_prefix: str = configs.SPEAKER_PREFIX,
      speaker_suffix: str = configs.SPEAKER_SUFFIX,
      overlap_words: int = 0,
  ):
    self.max_input_length = max_input_length
    self.max_target_length = max_target_length
    self.prompt_prefix = prompt_prefix
    self.prompt_suffix = prompt_suffix
    self.completion_suffix = completion_suffix
    self.speaker_prefix = speaker_prefix
    self.speaker_suffix = speaker_suffix
    self.overlap_words = max(0, int(overlap_words))

  def chunk_utterance_fields(
      self,
      words: Sequence[str],
      input_spks: Sequence[str],
      target_spks: Optional[Sequence[str]] = None,
      utt_id: str = "utt",
  ) -> Generator[Dict[str, Any], None, None]:
    """Yields prompt/target chunk dicts for word and speaker sequences."""
    total_words = len(words)
    if total_words == 0 or len(input_spks) != total_words:
      return

    has_target = target_spks is not None and len(target_spks) == total_words
    prompt_overhead = len(self.prompt_prefix) + len(self.prompt_suffix)
    completion_overhead = len(self.completion_suffix)

    start_idx = 0
    seg_idx = 0
    while start_idx < total_words:
      end_idx = start_idx
      running_in_len = prompt_overhead
      running_tgt_len = completion_overhead
      prev_in_spk: Optional[str] = None
      prev_tgt_spk: Optional[str] = None

      while end_idx < total_words:
        w_len = len(words[end_idx]) + (1 if end_idx > start_idx else 0)
        curr_in_spk = str(input_spks[end_idx])
        in_spk_cost = (
            len(self.speaker_prefix)
            + len(curr_in_spk)
            + len(self.speaker_suffix)
            + (1 if end_idx > start_idx else 0)
            if curr_in_spk != prev_in_spk
            else 0
        )
        new_in_len = running_in_len + w_len + in_spk_cost

        if has_target and target_spks is not None:
          curr_tgt_spk = str(target_spks[end_idx])
          tgt_spk_cost = (
              len(self.speaker_prefix)
              + len(curr_tgt_spk)
              + len(self.speaker_suffix)
              + (1 if end_idx > start_idx else 0)
              if curr_tgt_spk != prev_tgt_spk
              else 0
          )
          new_tgt_len = running_tgt_len + w_len + tgt_spk_cost
        else:
          new_tgt_len = 0

        if end_idx > start_idx and (
            new_in_len > self.max_input_length
            or (has_target and new_tgt_len > self.max_target_length)
        ):
          break

        running_in_len = new_in_len
        running_tgt_len = new_tgt_len
        prev_in_spk = curr_in_spk
        if has_target and target_spks is not None:
          prev_tgt_spk = str(target_spks[end_idx])
        end_idx += 1

      if end_idx == start_idx:
        end_idx = start_idx + 1

      chunk_words = list(words[start_idx:end_idx])
      chunk_in_spks = [str(s) for s in input_spks[start_idx:end_idx]]
      diarized_in = create_diarized_text(
          chunk_words,
          chunk_in_spks,
          self.speaker_prefix,
          self.speaker_suffix,
      )
      curr_prompt = self.prompt_prefix + diarized_in + self.prompt_suffix

      if has_target and target_spks is not None:
        chunk_tgt_spks = [str(s) for s in target_spks[start_idx:end_idx]]
        diarized_tgt = create_diarized_text(
            chunk_words,
            chunk_tgt_spks,
            self.speaker_prefix,
            self.speaker_suffix,
        )
        curr_target = diarized_tgt + self.completion_suffix
        tgt_spk_str = " ".join(chunk_tgt_spks)
      else:
        curr_target = ""
        tgt_spk_str = ""

      yield {
          "chunk_id": f"{utt_id}_seg{seg_idx}",
          "utterance_id": utt_id,
          "start_idx": start_idx,
          "end_idx": end_idx,
          "prompt": curr_prompt,
          "target": curr_target,
          "words": " ".join(chunk_words),
          "input_spk": " ".join(chunk_in_spks),
          "target_spk": tgt_spk_str,
      }
      if end_idx >= total_words:
        break
      if self.overlap_words > 0:
        start_idx = max(start_idx + 1, end_idx - self.overlap_words)
      else:
        start_idx = end_idx
      seg_idx += 1


def reconcile_chunk_speakers_hungarian(
    utt_words: Sequence[str],
    utt_hyp_spks: Sequence[str],
    chunk_completions: Sequence[str],
    chunker: Optional[UtteranceChunker] = None,
    use_locality_gate: bool = True,
    max_local_span_words: int = 5,
    max_ghost_cluster_words: int = 15,
    max_two_spk_span_words: int = 15,
) -> Tuple[str, str]:
  """Reconciles chunk-level LLM completions via TPST and Hungarian matching."""
  if not utt_words or not utt_hyp_spks:
    return "", ""
  if chunker is None:
    chunker = UtteranceChunker()

  chunks = list(chunker.chunk_utterance_fields(utt_words, utt_hyp_spks))
  reconciled_raw: List[str] = []

  for idx, chunk in enumerate(chunks):
    s_i = chunk["start_idx"]
    e_i = chunk["end_idx"]
    c_words = list(utt_words[s_i:e_i])
    c_hyp_spks = [int(x) for x in utt_hyp_spks[s_i:e_i]]
    c_ref_spks = [
        int(reconciled_raw[k])
        if k < len(reconciled_raw)
        else int(utt_hyp_spks[k])
        for k in range(s_i, e_i)
    ]
    already = max(0, len(reconciled_raw) - s_i)

    if idx < len(chunk_completions) and chunk_completions[idx].strip():
      comp = chunk_completions[idx].strip()
      if comp.endswith(chunker.completion_suffix.strip()):
        comp = comp[:-len(chunker.completion_suffix.strip())].strip()
      llm_text, llm_spk = extract_text_and_spk(
          comp,
          speaker_prefix=chunker.speaker_prefix,
          speaker_suffix=chunker.speaker_suffix,
      )
      transferred_str = transcript_preserving_speaker_transfer(
          src_text=llm_text,
          src_spk=llm_spk,
          tgt_text=" ".join(c_words),
          tgt_spk=" ".join(str(x) for x in c_hyp_spks),
      )
      c_llm_spks = [int(x) for x in transferred_str.split()]
      if len(c_llm_spks) == len(c_hyp_spks):
        aligned_c = get_oracle_speakers(c_llm_spks, c_llm_spks, c_ref_spks)
        reconciled_raw.extend(str(x) for x in aligned_c[already:])
      else:
        reconciled_raw.extend(str(x) for x in c_hyp_spks[already:])
    else:
      reconciled_raw.extend(str(x) for x in c_hyp_spks[already:])

  raw_str = " ".join(reconciled_raw)
  if use_locality_gate:
    gated = apply_locality_gated_speaker_transfer(
        utt_hyp_spks,
        reconciled_raw,
        max_local_span_words=max_local_span_words,
        max_ghost_cluster_words=max_ghost_cluster_words,
        max_two_spk_span_words=max_two_spk_span_words,
    )
    return raw_str, " ".join(gated)
  return raw_str, raw_str


ChunkedPromptBuilder = UtteranceChunker
