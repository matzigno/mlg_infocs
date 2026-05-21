from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array


@dataclass
class SemiSupervisedTargets:
    y: np.ndarray
    labeled_mask: np.ndarray
    labeled_indices: np.ndarray
    labeled_count: int
    classes: np.ndarray


@dataclass
class SemiSupervisedClassificationData:
    X: np.ndarray
    y: np.ndarray
    labeled_mask: np.ndarray
    labeled_indices: np.ndarray
    labeled_count: int
    classes: np.ndarray


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-scores))
        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T
    return stable_softmax(scores)


def to_adjacency_matrix(
    graph_or_adjacency,
    *,
    symmetrize: bool = True,
    remove_self_loops: bool = True,
    allow_negative: bool = False,
) -> np.ndarray:
    if isinstance(graph_or_adjacency, nx.Graph):
        adjacency = nx.to_numpy_array(graph_or_adjacency, dtype=float)
    else:
        adjacency = np.asarray(graph_or_adjacency, dtype=float)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "Input must be a NetworkX graph or a square adjacency matrix."
        )
    if not np.isfinite(adjacency).all():
        raise ValueError("Adjacency contains NaN or infinite values.")
    if not allow_negative and np.any(adjacency < 0.0):
        raise ValueError("Adjacency weights must be non-negative.")

    adjacency = adjacency.copy()
    if symmetrize:
        adjacency = 0.5 * (adjacency + adjacency.T)
    if remove_self_loops:
        np.fill_diagonal(adjacency, 0.0)
    return adjacency


def laplacian_matrix(adjacency: np.ndarray, normalized: bool = False, eps: float = 1e-10) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    if normalized:
        inv_sqrt = np.zeros_like(degrees)
        positive = degrees > eps
        inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
        d_inv_sqrt = np.diag(inv_sqrt)
        return np.eye(adjacency.shape[0], dtype=float) - d_inv_sqrt @ adjacency @ d_inv_sqrt
    return np.diag(degrees) - adjacency


def row_normalized_adjacency(adjacency: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    inv = np.zeros_like(degrees)
    positive = degrees > eps
    inv[positive] = 1.0 / degrees[positive]
    return np.diag(inv) @ adjacency


def symmetric_normalized_adjacency(adjacency: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > eps
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    d_inv_sqrt = np.diag(inv_sqrt)
    return d_inv_sqrt @ adjacency @ d_inv_sqrt


def prepare_semisupervised_targets(
    y,
    *,
    n_samples: int,
    unlabeled_value: int = -1,
) -> SemiSupervisedTargets:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D label array.")
    if y.shape[0] != n_samples:
        raise ValueError("y must have the same number of elements as graph nodes.")

    labeled_mask = y != unlabeled_value
    labeled_indices = np.flatnonzero(labeled_mask)
    if labeled_indices.size < 2:
        raise ValueError("At least two labeled samples are required.")

    y_labeled = y[labeled_mask]
    check_classification_targets(y_labeled)
    classes = np.unique(y_labeled)
    if classes.shape[0] < 2:
        raise ValueError("At least two labeled classes are required.")

    return SemiSupervisedTargets(
        y=y,
        labeled_mask=labeled_mask,
        labeled_indices=labeled_indices,
        labeled_count=int(labeled_indices.size),
        classes=classes,
    )


def prepare_semisupervised_classification_data(
    X,
    y,
    *,
    unlabeled_value: int = -1,
) -> SemiSupervisedClassificationData:
    X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=float)
    targets = prepare_semisupervised_targets(
        y=y,
        n_samples=X.shape[0],
        unlabeled_value=unlabeled_value,
    )
    return SemiSupervisedClassificationData(
        X=X,
        y=targets.y,
        labeled_mask=targets.labeled_mask,
        labeled_indices=targets.labeled_indices,
        labeled_count=targets.labeled_count,
        classes=targets.classes,
    )


def build_seed_label_matrix(
    y: np.ndarray,
    classes: np.ndarray,
    *,
    unlabeled_value: int = -1,
) -> np.ndarray:
    del unlabeled_value
    Y0 = np.zeros((y.shape[0], classes.shape[0]), dtype=float)
    for class_idx, cls in enumerate(classes):
        Y0[y == cls, class_idx] = 1.0
    return Y0


def take_node_slice(values: np.ndarray, nodes=None) -> np.ndarray:
    if nodes is None:
        return values

    idx = np.asarray(nodes)
    if idx.ndim == 0:
        idx = idx.reshape(1)

    if np.issubdtype(idx.dtype, np.bool_):
        if idx.shape[0] != values.shape[0]:
            raise ValueError(
                "Boolean node mask must have the same length as the number of nodes."
            )
        return values[idx]

    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("Node selection must be an integer index array or a boolean mask.")
    return values[idx]
