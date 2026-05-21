from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class LocalLinearEmbedding(TransformerMixin, BaseEstimator):
    """
    LLE for graphs with explicit adjacency matrix passed at construction time.

    Notes
    -----
    - The graph adjacency is used only to define the neighborhood of each node.
    - `fit` expects node features `X` with shape (num_nodes, num_features).
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        n_components: int = 2,
        reg: float = 1e-3,
        eps: float = 1e-12,
    ) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.n_components = n_components
        self.reg = reg
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LocalLinearEmbedding":
        del y
        adjacency = self._validate_adjacency()
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=float)
        num_nodes = adjacency.shape[0]
        if X.shape[0] != num_nodes:
            raise ValueError(
                "X must be a 2D array with shape (num_nodes, num_features)."
            )

        W = np.zeros((num_nodes, num_nodes), dtype=float)

        for node_id in range(num_nodes):
            neighbors = np.flatnonzero(adjacency[node_id] != 0.0)
            if neighbors.size == 0:
                continue
            if neighbors.size == 1:
                W[node_id, neighbors[0]] = 1.0
                continue

            Z = X[neighbors] - X[node_id]
            C = Z @ Z.T
            trace_C = float(np.trace(C))
            regularizer = self.reg * (trace_C if trace_C > self.eps else 1.0)
            C = C + regularizer * np.eye(neighbors.size)

            ones = np.ones(neighbors.size, dtype=float)
            weights = np.linalg.solve(C, ones)
            denom = float(weights.sum())
            if abs(denom) < self.eps:
                weights = np.full_like(weights, 1.0 / neighbors.size)
            else:
                weights = weights / denom

            W[node_id, neighbors] = weights

        I = np.eye(num_nodes, dtype=float)
        M = (I - W).T @ (I - W)
        M = 0.5 * (M + M.T)

        eigenvalues, eigenvectors = np.linalg.eigh(M)
        order = np.argsort(eigenvalues)
        embedding = eigenvectors[:, order[1 : self.n_components + 1]]

        self.adjacency_matrix_ = adjacency
        self.weights_ = W
        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues[order]
        self.X_fit_ = X.copy()
        self.n_features_in_ = X.shape[1]
        self.n_samples_in_ = X.shape[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=[
                "embedding_",
                "X_fit_",
                "n_features_in_",
                "n_samples_in_",
            ],
        )
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than in fit.")

        if X.shape[0] != self.n_samples_in_ or not np.allclose(
            X, self.X_fit_, atol=max(self.eps, 1e-8), rtol=1e-7
        ):
            raise ValueError(
                "This LocalLinearEmbedding implementation does not support out-of-sample "
                "transform for unseen nodes when adjacency is fixed. "
                "Call fit_transform on the full node-feature matrix."
            )

        return self.embedding_.copy()

    def _validate_adjacency(self) -> np.ndarray:
        adjacency = np.asarray(self.adjacency_matrix, dtype=float)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency_matrix must be a square 2D array.")
        if not np.isfinite(adjacency).all():
            raise ValueError("adjacency_matrix contains NaN or infinite values.")
        if self.n_components < 1 or self.n_components >= adjacency.shape[0]:
            raise ValueError(
                "n_components must be >= 1 and strictly smaller than num_nodes."
            )
        if self.reg <= 0:
            raise ValueError("reg must be > 0.")

        adjacency = adjacency.copy()
        np.fill_diagonal(adjacency, 0.0)
        return adjacency


LocallyLinearEmbedding = LocalLinearEmbedding
