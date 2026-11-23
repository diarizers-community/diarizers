"""Clean-room Apache-2.0 Spectral Clustering engine for Speaker Diarization."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.cluster import hierarchy
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance


def compute_cosine_affinity(embeddings: np.ndarray) -> np.ndarray:
  """Computes pairwise cosine similarity matrix in [0, 1]."""
  norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
  norms = np.maximum(norms, 1e-12)
  normed = embeddings / norms
  sim = np.matmul(normed, normed.T)
  return np.clip(0.5 * (sim + 1.0), 0.0, 1.0)


def refine_affinity_matrix(
    affinity: np.ndarray,
    gaussian_blur_sigma: float = 1.0,
    p_percentile: float = 0.50,
    multiplier: float = 0.01,
) -> np.ndarray:
  """Applies Gaussian smoothing, percentile soft thresholding, and symmetry."""
  mat = np.copy(affinity).astype(np.float64)
  n = mat.shape[0]
  if n <= 1:
    return mat

  np.fill_diagonal(mat, 0.0)
  di = np.diag_indices(n)
  mat[di] = mat.max(axis=1)

  if gaussian_blur_sigma > 0:
    mat = gaussian_filter(mat, sigma=gaussian_blur_sigma)

  if p_percentile > 0:
    eff_p = min(p_percentile, 0.70) if n < 100 else p_percentile
    thresh = np.percentile(mat, eff_p * 100.0, axis=1, keepdims=True)
    mat = np.where(mat >= thresh, mat, mat * multiplier)

  mat = 0.5 * (mat + mat.T)
  np.fill_diagonal(mat, 0.0)
  return mat


def _cosine_kmeans(
    spectral_embeddings: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> np.ndarray:
  """K-Means++ initialization followed by cosine-distance Lloyd iterations."""
  n = spectral_embeddings.shape[0]
  if n_clusters <= 1 or n <= 1:
    return np.zeros(n, dtype=np.int32)
  n_clusters = min(n_clusters, n)

  try:
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=1,
        n_init=10,
        random_state=0,
    )
    km.fit(spectral_embeddings)
    centroids = np.copy(km.cluster_centers_)
  except Exception:
    centroids = np.zeros(
        (n_clusters, spectral_embeddings.shape[1]), dtype=np.float64
    )
    centroids[0] = spectral_embeddings[0]
    for k in range(1, n_clusters):
      euclid = distance.cdist(spectral_embeddings, centroids[:k], "euclidean")
      min_d = np.min(euclid, axis=1)
      centroids[k] = spectral_embeddings[int(np.argmax(min_d))]

  sample_ids = np.arange(n)
  prev_mean_dist = 0.0
  labels = np.zeros(n, dtype=np.int32)
  for iter_idx in range(max_iter + 1):
    dists = distance.cdist(spectral_embeddings, centroids, metric="cosine")
    dists = np.nan_to_num(dists, nan=1.0)
    labels = dists.argmin(axis=1).astype(np.int32)
    mean_dist = float(np.mean(dists[sample_ids, labels]))
    if (
        mean_dist <= prev_mean_dist
        and mean_dist >= (1.0 - tol) * prev_mean_dist
    ) or iter_idx == max_iter:
      break
    prev_mean_dist = mean_dist
    for k in range(n_clusters):
      mask = np.where(labels == k)[0]
      if mask.size > 0:
        centroids[k] = np.mean(spectral_embeddings[mask], axis=0)
  return labels


class SpectralClusterer:
  """Spectral clusterer using symmetric normalized Laplacian and eigengap."""

  def __init__(
      self,
      min_clusters: int = 2,
      max_clusters: int = 7,
      p_percentile: float = 0.50,
      gaussian_blur_sigma: float = 1.0,
      eigen_threshold: float = 0.45,
  ):
    self.min_clusters = max(1, int(min_clusters))
    self.max_clusters = max(self.min_clusters, int(max_clusters))
    self.p_percentile = float(p_percentile)
    self.gaussian_blur_sigma = float(gaussian_blur_sigma)
    self.eigen_threshold = float(eigen_threshold)

  def estimate_num_clusters(
      self,
      eigenvalues: np.ndarray,
      min_k: Optional[int] = None,
      max_k: Optional[int] = None,
  ) -> int:
    """Estimates speaker count k via maximum eigengap bounded by threshold."""
    n = len(eigenvalues)
    lo = min(max(1, min_k if min_k is not None else self.min_clusters), n)
    hi = min(max(lo, max_k if max_k is not None else self.max_clusters), n)
    if lo == hi:
      return lo
    best_gap = -1.0
    k_opt = lo
    for k in range(lo, hi + 1):
      if k >= n:
        break
      gap = float(eigenvalues[k]) - float(eigenvalues[k - 1])
      if float(eigenvalues[k - 1]) > self.eigen_threshold and k > lo:
        break
      if gap > best_gap:
        best_gap = gap
        k_opt = k
    return k_opt

  def predict(
      self,
      embeddings: np.ndarray,
      num_clusters: Optional[int] = None,
  ) -> np.ndarray:
    """Predicts 0-indexed cluster labels for each row in embeddings."""
    n = embeddings.shape[0]
    if n == 0:
      return np.array([], dtype=np.int32)
    if n == 1:
      return np.array([0], dtype=np.int32)

    min_k = num_clusters if num_clusters is not None else self.min_clusters
    max_k = num_clusters if num_clusters is not None else self.max_clusters
    min_k = min(max(1, min_k), n)
    max_k = min(max(min_k, max_k), n)

    affinity = compute_cosine_affinity(embeddings)
    w_mat = refine_affinity_matrix(
        affinity,
        gaussian_blur_sigma=self.gaussian_blur_sigma,
        p_percentile=self.p_percentile,
    )

    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.sum(w_mat, axis=1), 1e-12))
    laplacian = np.eye(n) - (d_inv_sqrt[:, None] * w_mat * d_inv_sqrt[None, :])
    laplacian = 0.5 * (laplacian + laplacian.T)

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    k_opt = self.estimate_num_clusters(eigenvalues, min_k=min_k, max_k=max_k)

    spectral_embs = eigenvectors[:, :k_opt]
    norms = np.maximum(
        np.linalg.norm(spectral_embs, axis=1, keepdims=True), 1e-12
    )
    spectral_embs = spectral_embs / norms
    return _cosine_kmeans(spectral_embs, n_clusters=k_opt)


def l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
  """Returns row-wise L2-normalized float32 embeddings."""
  arr = np.asarray(embeddings, dtype=np.float32)
  if arr.size == 0:
    return arr
  norms = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
  return (arr / norms).astype(np.float32)


def compute_weighted_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[int]]:
  """Computes L2-normalized weighted speaker centroids for unique labels."""
  arr = np.asarray(embeddings, dtype=np.float32)
  lbl = np.asarray(labels, dtype=np.int32)
  uniq = sorted(set(lbl.tolist()))
  if not uniq:
    return np.zeros((0, arr.shape[1]), dtype=np.float32), []
  w_arr = (
      np.asarray(weights, dtype=np.float32)
      if weights is not None
      else None
  )
  cents: List[np.ndarray] = []
  for k in uniq:
    mask = lbl == k
    if w_arr is not None:
      w = w_arr[mask][:, None]
      c = np.sum(arr[mask] * w, axis=0) / max(1e-8, float(w.sum()))
    else:
      c = np.mean(arr[mask], axis=0)
    nrm = float(np.linalg.norm(c))
    cents.append(c / max(1e-12, nrm))
  return np.stack(cents, axis=0).astype(np.float32), uniq


def balance_center_embeddings(
    all_embeddings: np.ndarray,
    cluster_embeddings: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    beta: float = 1.0,
) -> np.ndarray:
  """Removes the equal-weighted mean of speaker centroids from embeddings.

  Unlike talk-time-weighted utterance centering (which drags the dominant
  speaker toward the origin on imbalanced meetings), taking the unweighted
  mean across the ``K`` discovered speaker centroids estimates the shared
  room/channel acoustic offset cleanly without hub collapse.
  """
  normed_all = l2_normalize_rows(all_embeddings)
  if beta <= 0.0 or len(cluster_embeddings) == 0:
    return normed_all
  cents, uniq = compute_weighted_centroids(
      l2_normalize_rows(cluster_embeddings), labels, weights=weights
  )
  if len(uniq) < 2:
    return normed_all
  mu_bal = np.mean(cents, axis=0, keepdims=True)
  return l2_normalize_rows(normed_all - float(beta) * mu_bal)


def ward_cluster_auto_k(
    embeddings: np.ndarray,
    min_clusters: int = 1,
    max_clusters: int = 10,
    merge_threshold: float = 0.60,
    min_cluster_weight: float = 8.0,
    weights: Optional[np.ndarray] = None,
    num_clusters: Optional[int] = None,
) -> np.ndarray:
  """Clusters unit-norm embeddings via Ward linkage + closest-centroid cosine.

  Selects the largest ``k`` in ``[min_clusters, max_clusters]`` for which
  every cluster meets ``min_cluster_weight`` (e.g., clean active seconds)
  and the highest pairwise cosine similarity between surviving speaker
  centroids stays strictly below ``merge_threshold``.
  """
  embs_n = l2_normalize_rows(embeddings)
  n = embs_n.shape[0]
  if n == 0:
    return np.zeros((0,), dtype=np.int32)
  if n == 1:
    return np.zeros((1,), dtype=np.int32)

  if num_clusters is not None:
    lo = hi = min(max(1, int(num_clusters)), n)
  else:
    lo = min(max(1, int(min_clusters)), n)
    hi = min(max(lo, int(max_clusters)), n)

  if lo == 1 and hi == 1:
    return np.zeros((n,), dtype=np.int32)

  w_arr = (
      np.asarray(weights, dtype=np.float32)
      if weights is not None
      else np.ones((n,), dtype=np.float32)
  )
  z_link = hierarchy.linkage(
      embs_n.astype(np.float64), method="ward", metric="euclidean"
  )
  if lo == hi:
    raw = hierarchy.fcluster(z_link, t=lo, criterion="maxclust") - 1
    _, inv = np.unique(raw, return_inverse=True)
    return inv.astype(np.int32)

  k_sel = lo
  for k in range(hi, lo, -1):
    lbl_k = hierarchy.fcluster(z_link, t=k, criterion="maxclust") - 1
    cents: List[np.ndarray] = []
    for c_id in np.unique(lbl_k):
      mask = lbl_k == c_id
      mass = float(w_arr[mask].sum())
      if min_cluster_weight > 0.0 and mass < min_cluster_weight:
        continue
      vec = np.sum(embs_n[mask] * w_arr[mask][:, None], axis=0)
      nrm = float(np.linalg.norm(vec))
      if nrm > 1e-12:
        cents.append(vec / nrm)
    if len(cents) < k:
      continue
    c_mat = np.stack(cents, axis=0)
    cos_m = c_mat @ c_mat.T
    np.fill_diagonal(cos_m, -2.0)
    if float(cos_m.max()) < float(merge_threshold):
      k_sel = k
      break

  raw = hierarchy.fcluster(z_link, t=k_sel, criterion="maxclust") - 1
  _, inv = np.unique(raw, return_inverse=True)
  return inv.astype(np.int32)


def refine_cosine_em(
    embeddings: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    chunk_ids: Optional[np.ndarray] = None,
    n_iters: int = 5,
    enforce_cannot_link: bool = True,
) -> np.ndarray:
  """Spherical k-means EM with per-chunk Hungarian cannot-link constraints."""
  embs_n = l2_normalize_rows(embeddings)
  _, out = np.unique(np.asarray(labels, dtype=np.int32), return_inverse=True)
  out = out.astype(np.int32)
  if embs_n.shape[0] <= 1:
    return out
  k_count = int(out.max()) + 1
  if k_count <= 1 or n_iters <= 0:
    return out

  w_arr = (
      np.asarray(weights, dtype=np.float32)
      if weights is not None
      else None
  )
  groups: Dict[int, List[int]] = {}
  if chunk_ids is not None and enforce_cannot_link:
    for i, cid in enumerate(np.asarray(chunk_ids, dtype=np.int32).tolist()):
      groups.setdefault(cid, []).append(i)

  for _ in range(int(n_iters)):
    cents = np.zeros((k_count, embs_n.shape[1]), dtype=np.float32)
    for k in range(k_count):
      mask = out == k
      if not np.any(mask):
        continue
      if w_arr is not None:
        vec = np.sum(embs_n[mask] * w_arr[mask][:, None], axis=0)
      else:
        vec = np.sum(embs_n[mask], axis=0)
      nrm = float(np.linalg.norm(vec))
      if nrm > 1e-12:
        cents[k] = vec / nrm
    sim = embs_n @ cents.T
    new_out = out.copy()
    if groups:
      for idxs in groups.values():
        if len(idxs) == 1:
          new_out[idxs[0]] = int(np.argmax(sim[idxs[0]]))
        elif len(idxs) <= k_count:
          ri, ci = optimize.linear_sum_assignment(sim[idxs], maximize=True)
          for r_pos, c_pos in zip(ri, ci):
            new_out[idxs[int(r_pos)]] = int(c_pos)
        else:
          for i in idxs:
            new_out[i] = int(np.argmax(sim[i]))
    else:
      new_out = np.argmax(sim, axis=1).astype(np.int32)
    if np.array_equal(new_out, out):
      break
    out = new_out

  _, compact = np.unique(out, return_inverse=True)
  return compact.astype(np.int32)
