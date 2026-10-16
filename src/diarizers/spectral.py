"""Clean-room Apache-2.0 Spectral Clustering engine for Speaker Diarization."""

from typing import Optional

import numpy as np
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
