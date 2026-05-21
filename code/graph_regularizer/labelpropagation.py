from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._utils import (
    build_seed_label_matrix,
    prepare_semisupervised_targets,
    row_normalized_adjacency,
    symmetric_normalized_adjacency,
    take_node_slice,
    to_adjacency_matrix,
)


class _BaseGraphPropagationClassifier(BaseEstimator, ClassifierMixin):
    """
    Common iterative diffusion loop for graph-based semi-supervised classifiers.
    """

    def __init__(
        self,
        max_iter: int = 30,
        tol: float = 1e-3,
        unlabeled_value: int = -1,
        eps: float = 1e-12,
        verbose: bool = False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.unlabeled_value = unlabeled_value
        self.eps = eps
        self.verbose = verbose

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for semi-supervised fitting.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if self.tol < 0:
            raise ValueError("tol must be >= 0.")
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

        adjacency = to_adjacency_matrix(X)
        targets = prepare_semisupervised_targets(
            y,
            n_samples=adjacency.shape[0],
            unlabeled_value=self.unlabeled_value,
        )
        propagation_matrix = self._build_propagation_matrix(adjacency)
        Y0 = build_seed_label_matrix(
            y=targets.y,
            classes=targets.classes,
            unlabeled_value=self.unlabeled_value,
        )

        Y_prev = Y0.copy()
        delta = np.inf
        n_iter = 0
        while n_iter < self.max_iter and delta > self.tol:
            Y = self._update_labels(
                propagation_matrix=propagation_matrix,
                Y_prev=Y_prev,
                Y0=Y0,
                labeled_indices=targets.labeled_indices,
            )
            n_iter += 1
            delta = float(np.sum(np.abs(Y - Y_prev)))
            Y_prev = Y
            if self.verbose:
                print(f"Iteration {n_iter}, convergence tolerance: {delta:.5f}")

        self.adjacency_ = adjacency
        self.propagation_matrix_ = propagation_matrix
        self.label_distributions_ = Y_prev
        self.classes_ = targets.classes
        self.n_nodes_in_ = adjacency.shape[0]
        self.n_iter_ = n_iter
        self.converged_delta_ = delta
        return self

    def predict_proba(self, X=None):
        check_is_fitted(self, attributes=["label_distributions_", "classes_"])
        return take_node_slice(self.label_distributions_, X)

    def decision_function(self, X=None):
        check_is_fitted(self, attributes=["label_distributions_", "classes_"])
        # A natural score for propagation-based methods is the class distribution itself.
        return take_node_slice(self.label_distributions_, X)

    def predict(self, X=None):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)].ravel()

    # Interface for subclasses.
    def _build_propagation_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _update_labels(
        self,
        *,
        propagation_matrix: np.ndarray,
        Y_prev: np.ndarray,
        Y0: np.ndarray,
        labeled_indices: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class GRFGraphRegularizer(_BaseGraphPropagationClassifier):
    """
    Classic label propagation with hard clamping (GRF-style update):
        Y_{t+1} = P Y_t, with labeled rows fixed to Y0 at each iteration.
    """

    def _build_propagation_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        return row_normalized_adjacency(adjacency, eps=self.eps)

    def _update_labels(
        self,
        *,
        propagation_matrix: np.ndarray,
        Y_prev: np.ndarray,
        Y0: np.ndarray,
        labeled_indices: np.ndarray,
    ) -> np.ndarray:
        Y = propagation_matrix @ Y_prev
        Y[labeled_indices, :] = Y0[labeled_indices, :]
        return Y


class LabelSpreading(_BaseGraphPropagationClassifier):
    """
    Label spreading with normalized affinity and soft clamping:
        Y_{t+1} = alpha * S Y_t + (1 - alpha) * Y0
    """

    def __init__(
        self,
        alpha: float = 0.5,
        max_iter: int = 30,
        tol: float = 1e-3,
        unlabeled_value: int = -1,
        eps: float = 1e-12,
        verbose: bool = False,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            unlabeled_value=unlabeled_value,
            eps=eps,
            verbose=verbose,
        )
        self.alpha = alpha

    def fit(self, X, y=None):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        return super().fit(X=X, y=y)

    def _build_propagation_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        return symmetric_normalized_adjacency(adjacency, eps=self.eps)

    def _update_labels(
        self,
        *,
        propagation_matrix: np.ndarray,
        Y_prev: np.ndarray,
        Y0: np.ndarray,
        labeled_indices: np.ndarray,
    ) -> np.ndarray:
        del labeled_indices
        return self.alpha * propagation_matrix @ Y_prev + (1.0 - self.alpha) * Y0
