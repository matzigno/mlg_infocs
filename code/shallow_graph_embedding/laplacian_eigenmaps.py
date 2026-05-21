from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class LaplacianEigenmaps(TransformerMixin, BaseEstimator):
    """
    Laplacian Eigenmaps with explicit graph adjacency.

    A custom `weight_function` can optionally be used to reweight edges
    from endpoint feature vectors.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        n_components: int = 2,
        weight_function: Callable[[np.ndarray, np.ndarray], float] | None = None,
        zero_tol: float = 1e-10,
    ) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.n_components = n_components
        self.weight_function = weight_function
        self.zero_tol = zero_tol

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LaplacianEigenmaps":
        del y
        adjacency = self._validate_adjacency()
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=float)
        num_nodes = adjacency.shape[0]
        if X.shape[0] != num_nodes:
            raise ValueError(
                "X must be a 2D array with shape (num_nodes, num_features)."
            )

        W = self._build_weighted_adjacency(adjacency, X)
        # Standard Laplacian Eigenmaps assumes an undirected weighted graph.
        W = 0.5 * (W + W.T)

        degrees = W.sum(axis=1)
        isolated = np.where(degrees <= self.zero_tol)[0]
        if isolated.size > 0:
            raise ValueError(
                "The weighted graph contains isolated nodes; "
                "Laplacian Eigenmaps is undefined for zero-degree nodes. "
                f"Isolated node ids (first 10): {isolated[:10].tolist()}"
            )

        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        L_sym = np.eye(num_nodes) - D_inv_sqrt @ W @ D_inv_sqrt
        L_sym = 0.5 * (L_sym + L_sym.T)

        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        nontrivial = np.where(eigenvalues > self.zero_tol)[0]
        if nontrivial.size < self.n_components:
            raise ValueError(
                "Not enough non-trivial eigenvectors to build the embedding. "
                "Try reducing n_components."
            )

        selected = nontrivial[: self.n_components]
        embedding = D_inv_sqrt @ eigenvectors[:, selected]

        self.weighted_adjacency_ = W
        self.degrees_ = degrees
        self.eigenvalues_ = eigenvalues
        self.embedding_ = embedding
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
            X, self.X_fit_, atol=max(self.zero_tol, 1e-8), rtol=1e-7
        ):
            raise ValueError(
                "This LaplacianEigenmaps implementation does not support out-of-sample "
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

        adjacency = adjacency.copy()
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    def _build_weighted_adjacency(self, adjacency: np.ndarray, X: np.ndarray) -> np.ndarray:
        W = adjacency.copy()
        if self.weight_function is None:
            return W

        is_undirected = np.allclose(W, W.T, atol=self.zero_tol)
        if is_undirected:
            rows, cols = np.where(np.triu(W, k=1) != 0.0)
            for i, j in zip(rows, cols, strict=False):
                weight = float(self.weight_function(X[i], X[j]))
                if not np.isfinite(weight) or weight < 0.0:
                    raise ValueError(
                        "weight_function must return finite non-negative values."
                    )
                W[i, j] = weight
                W[j, i] = weight
            return W

        rows, cols = np.where(W != 0.0)
        for i, j in zip(rows, cols, strict=False):
            if i == j:
                continue
            weight = float(self.weight_function(X[i], X[j]))
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    "weight_function must return finite non-negative values."
                )
            W[i, j] = weight
        return W
