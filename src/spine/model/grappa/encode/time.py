"""Charge-light matching (t0) features for GrapPA nodes and edges."""

import numpy as np
import torch

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch

__all__ = ["get_cluster_t0_batch", "get_edge_t0"]


def _to_numpy(x):
    """Convert a TensorBatch, torch tensor or array to a numpy array."""
    if isinstance(x, TensorBatch):
        x = x.tensor
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_cluster_t0_batch(data, t0: TensorBatch, clusts: IndexBatch) -> np.ndarray:
    """Charge-light matched t0 of each cluster.

    The cluster t0 is the energy-weighted mode of its voxel t0 values
    (unmatched voxels, t0 < 0, are ignored); the cluster confidence is the
    energy-weighted mean confidence of the voxels carrying that t0 (voxels with
    unavailable confidence, cl < 0, are ignored).

    Parameters
    ----------
    data : TensorBatch or ClusterLabelBatch
        Voxel data, used for the voxel values (weights)
    t0 : TensorBatch
        (N, 2) Voxel-aligned [t0 (ns), confidence]
    clusts : IndexBatch
        (C) Voxel indexes that make up each cluster

    Returns
    -------
    np.ndarray
        (C, 3) [t0 (us, 0 if none), has_t0 (0/1), confidence (-1 if none)]
    """
    values, t0_np = _to_numpy(data.values), _to_numpy(t0)
    if len(t0_np) != len(values):
        raise ValueError(
            f"`t0` has {len(t0_np)} rows but `data` has {len(values)}; they "
            "must be parsed on the same voxel set."
        )

    clusts = clusts.to_numpy()
    num_clusts = int(np.sum(clusts.counts))
    out = np.zeros((num_clusts, 3), dtype=np.float32)
    out[:, 2] = -1.0
    if num_clusts == 0:
        return out

    index = clusts.full_index.astype(np.int64)
    clust_ids = clusts.index_ids.astype(np.int64)
    voxel_t0, voxel_cl = t0_np[index, 0], t0_np[index, 1]
    matched = voxel_t0 >= 0
    if not matched.any():
        return out

    # Energy per (cluster, t0) pair, and energy-weighted confidence
    clust_ids, voxel_t0, voxel_cl = clust_ids[matched], voxel_t0[matched], voxel_cl[matched]
    weights = np.maximum(values[index][matched], 0.0) + 1e-9
    has_cl = voxel_cl >= 0
    pairs, inv = np.unique(
        np.stack([clust_ids.astype(np.float64), voxel_t0.astype(np.float64)], 1),
        axis=0, return_inverse=True,
    )
    inv = inv.ravel()
    wsum = np.bincount(inv, weights=weights)
    wcsum = np.bincount(inv, weights=weights * has_cl)
    csum = np.bincount(inv, weights=weights * np.where(has_cl, voxel_cl, 0.0))

    # Keep the heaviest t0 in each cluster (ties -> smaller t0)
    order = np.lexsort((pairs[:, 1], -wsum, pairs[:, 0]))
    best = order[np.unique(pairs[order, 0], return_index=True)[1]]
    clust = pairs[best, 0].astype(np.int64)
    out[clust, 0] = pairs[best, 1] / 1000.0
    out[clust, 1] = 1.0
    out[clust, 2] = np.where(
        wcsum[best] > 0, csum[best] / np.maximum(wcsum[best], 1e-30), -1.0
    )

    return out


def get_edge_t0(clust_t0: np.ndarray, edge_index: EdgeIndexBatch) -> np.ndarray:
    """|t0 difference| between the two clusters of each (directed) edge.

    Returns
    -------
    np.ndarray
        (E, 2) [|dt0| (us, 0 unless both have t0), both_have_t0 (0/1)]
    """
    index = edge_index.index if edge_index.directed else edge_index.directed_index
    src, dst = _to_numpy(index)
    both = (clust_t0[src, 1] > 0) & (clust_t0[dst, 1] > 0)
    dt0 = np.where(both, np.abs(clust_t0[src, 0] - clust_t0[dst, 0]), 0.0)

    return np.stack([dt0, both], axis=1).astype(np.float32)
