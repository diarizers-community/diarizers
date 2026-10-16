"""Hungarian WER, WDER, cpWER, SpkCntMAE, and paired bootstrap CIs."""

from collections.abc import Sequence
import json
from typing import Any, Dict, List

import numpy as np
from scipy import optimize

from diarizers import alignment


def evaluate_utterance(
    ref_text: str,
    hyp_text: str,
    ref_spk: str,
    hyp_spk: str,
) -> Dict[str, Any]:
  """Computes exact Hungarian WER, WDER, cpWER, and SpkCntMAE."""
  ref_norm = alignment.normalize_text(ref_text)
  hyp_norm = alignment.normalize_text(hyp_text)
  ref_words = ref_norm.split()
  hyp_words = hyp_norm.split()
  ref_spks = [int(x) for x in alignment.speakers_transform(ref_spk.split())]
  hyp_spks = [int(x) for x in alignment.speakers_transform(hyp_spk.split())]

  # 1. WER and WDER via Levenshtein word alignment + Hungarian matching
  wer_dist, align = alignment.levenshtein_with_edits(ref_norm, hyp_norm)
  wer_den = len(ref_words)
  wer_num = int(wer_dist)

  aligned_pairs = [
      (i, j)
      for i, j in align
      if i != -1 and j != -1 and i < len(ref_spks) and j < len(hyp_spks)
  ]
  wder_den = len(aligned_pairs)
  if wder_den == 0:
    wder_num = 0
  else:
    max_spk = max(max(ref_spks, default=1), max(hyp_spks, default=1))
    cost_mat = np.zeros((max_spk, max_spk), dtype=np.int64)
    for i, j in aligned_pairs:
      cost_mat[ref_spks[i] - 1, hyp_spks[j] - 1] += 1
    r_ind, c_ind = optimize.linear_sum_assignment(cost_mat, maximize=True)
    correct_spk = int(cost_mat[r_ind, c_ind].sum())
    wder_num = wder_den - correct_spk

  # 2. cpWER via pairwise speaker WER matrix + Hungarian matching
  cpwer_den = len(ref_words)
  r_spk_map: Dict[int, List[str]] = {}
  for w, s in zip(ref_words, ref_spks):
    r_spk_map.setdefault(s, []).append(w)
  h_spk_map: Dict[int, List[str]] = {}
  for w, s in zip(hyp_words, hyp_spks):
    h_spk_map.setdefault(s, []).append(w)

  k_max = max(
      max(r_spk_map.keys(), default=1), max(h_spk_map.keys(), default=1)
  )
  cp_cost = np.zeros((k_max, k_max), dtype=np.int64)
  for r_s in range(1, k_max + 1):
    r_words_s = r_spk_map.get(r_s, [])
    r_len = len(r_words_s)
    for h_s in range(1, k_max + 1):
      h_words_s = h_spk_map.get(h_s, [])
      h_len = len(h_words_s)
      if r_len == 0:
        cp_cost[r_s - 1, h_s - 1] = h_len
      elif h_len == 0:
        cp_cost[r_s - 1, h_s - 1] = r_len
      else:
        cp_cost[r_s - 1, h_s - 1] = alignment.levenshtein_distance(
            r_words_s, h_words_s
        )

  r_ind, c_ind = optimize.linear_sum_assignment(cp_cost, maximize=False)
  cpwer_num = int(cp_cost[r_ind, c_ind].sum())

  # 3. Speaker count absolute error
  ref_spk_cnt = len(set(ref_spks)) if ref_spks else 0
  hyp_spk_cnt = len(set(hyp_spks)) if hyp_spks else 0
  spk_cnt_err = float(abs(ref_spk_cnt - hyp_spk_cnt))

  return {
      "wer_num": wer_num,
      "wer_den": wer_den,
      "wder_num": wder_num,
      "wder_den": wder_den,
      "cpwer_num": cpwer_num,
      "cpwer_den": cpwer_den,
      "spk_cnt_err": spk_cnt_err,
      "WER": float(wer_num) / max(1, wer_den),
      "WDER": float(wder_num) / max(1, wder_den),
      "cpWER": float(cpwer_num) / max(1, cpwer_den),
      "SpkCntMAE": spk_cnt_err,
  }


def evaluate_json_dict(
    data_dict: Dict[str, Any],
    ref_text_field: str = "ref_text",
    hyp_text_field: str = "hyp_text",
    ref_spk_field: str = "ref_spk",
    hyp_spk_field: str = "hyp_spk_llm",
) -> Dict[str, Any]:
  """Computes micro WER, WDER, cpWER, and macro SpkCntMAE across utterances."""
  utts = data_dict.get("utterances", [])
  if not utts:
    return {
        "WER": 0.0,
        "WDER": 0.0,
        "cpWER": 0.0,
        "SpkCntMAE": 0.0,
        "WER(%)": 0.0,
        "WDER(%)": 0.0,
        "cpWER(%)": 0.0,
        "NumUtterances": 0,
        "utterances": [],
    }

  tot_wer_num = 0
  tot_wer_den = 0
  tot_wder_num = 0
  tot_wder_den = 0
  tot_cpwer_num = 0
  tot_cpwer_den = 0
  tot_spk_err = 0.0
  utt_stats: List[Dict[str, Any]] = []

  for utt in utts:
    ref_t = utt.get(ref_text_field, "")
    hyp_t = utt.get(hyp_text_field, ref_t)
    ref_s = utt.get(ref_spk_field, "")
    hyp_s = utt.get(hyp_spk_field, utt.get("hyp_spk", ""))
    st = evaluate_utterance(ref_t, hyp_t, ref_s, hyp_s)
    st["utterance_id"] = utt.get("utterance_id", "")
    utt_stats.append(st)

    tot_wer_num += st["wer_num"]
    tot_wer_den += st["wer_den"]
    tot_wder_num += st["wder_num"]
    tot_wder_den += st["wder_den"]
    tot_cpwer_num += st["cpwer_num"]
    tot_cpwer_den += st["cpwer_den"]
    tot_spk_err += st["spk_cnt_err"]

  wer = float(tot_wer_num) / max(1, tot_wer_den)
  wder = float(tot_wder_num) / max(1, tot_wder_den)
  cpwer = float(tot_cpwer_num) / max(1, tot_cpwer_den)
  mae = float(tot_spk_err) / max(1, len(utts))

  return {
      "WER": wer,
      "WDER": wder,
      "cpWER": cpwer,
      "SpkCntMAE": round(mae, 4),
      "WER(%)": round(wer * 100.0, 2),
      "WDER(%)": round(wder * 100.0, 2),
      "cpWER(%)": round(cpwer * 100.0, 2),
      "NumUtterances": len(utts),
      "utterances": utt_stats,
  }


def evaluate_json_file(
    json_path: str,
    ref_text_field: str = "ref_text",
    hyp_text_field: str = "hyp_text",
    ref_spk_field: str = "ref_spk",
    hyp_spk_field: str = "hyp_spk_llm",
) -> Dict[str, Any]:
  """Loads a JSON dataset file and computes official Hungarian metrics."""
  with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  return evaluate_json_dict(
      data,
      ref_text_field=ref_text_field,
      hyp_text_field=hyp_text_field,
      ref_spk_field=ref_spk_field,
      hyp_spk_field=hyp_spk_field,
  )


def bootstrap_paired_ci(
    utt_stats_a: Sequence[Dict[str, Any]],
    utt_stats_b: Sequence[Dict[str, Any]],
    num_resamples: int = 10000,
    seed: int = 42,
    num_samples: int = 0,
) -> Dict[str, Any]:
  """Computes paired bootstrap 95% CIs and two-sided p-values (A - B)."""
  if num_samples > 0:
    num_resamples = num_samples
  n_utts = min(len(utt_stats_a), len(utt_stats_b))
  if n_utts == 0:
    return {}

  mat_a = np.array(
      [
          [
              u["wder_num"],
              u["wder_den"],
              u["cpwer_num"],
              u["cpwer_den"],
              u["spk_cnt_err"],
          ]
          for u in utt_stats_a[:n_utts]
      ],
      dtype=np.float64,
  )
  mat_b = np.array(
      [
          [
              u["wder_num"],
              u["wder_den"],
              u["cpwer_num"],
              u["cpwer_den"],
              u["spk_cnt_err"],
          ]
          for u in utt_stats_b[:n_utts]
      ],
      dtype=np.float64,
  )

  rng = np.random.default_rng(seed)
  indices = rng.integers(0, n_utts, size=(num_resamples, n_utts))
  sum_a = mat_a[indices].sum(axis=1)
  sum_b = mat_b[indices].sum(axis=1)

  wder_a = 100.0 * sum_a[:, 0] / np.maximum(sum_a[:, 1], 1.0)
  wder_b = 100.0 * sum_b[:, 0] / np.maximum(sum_b[:, 1], 1.0)
  cpwer_a = 100.0 * sum_a[:, 2] / np.maximum(sum_a[:, 3], 1.0)
  cpwer_b = 100.0 * sum_b[:, 2] / np.maximum(sum_b[:, 3], 1.0)

  diff_wder = wder_a - wder_b
  diff_cpwer = cpwer_a - cpwer_b

  def _summarize(diff: np.ndarray, pt_a: float, pt_b: float) -> Dict[str, Any]:
    ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
    p_pos = float(np.mean(diff >= 0.0))
    p_neg = float(np.mean(diff <= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "point_a": round(float(pt_a), 2),
        "point_b": round(float(pt_b), 2),
        "point_diff": round(float(pt_a - pt_b), 2),
        "ci_95_low": round(float(ci_low), 2),
        "ci_95_high": round(float(ci_high), 2),
        "p_value_two_sided": round(float(p_two), 5),
    }

  tot_a = mat_a.sum(axis=0)
  tot_b = mat_b.sum(axis=0)
  pt_wder_a = float(100.0 * tot_a[0] / max(tot_a[1], 1.0))
  pt_wder_b = float(100.0 * tot_b[0] / max(tot_b[1], 1.0))
  pt_cpwer_a = float(100.0 * tot_a[2] / max(tot_a[3], 1.0))
  pt_cpwer_b = float(100.0 * tot_b[2] / max(tot_b[3], 1.0))

  return {
      "WDER": _summarize(diff_wder, pt_wder_a, pt_wder_b),
      "cpWER": _summarize(diff_cpwer, pt_cpwer_a, pt_cpwer_b),
  }
