from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted

from ._utils import (
    laplacian_matrix,
    prepare_semisupervised_classification_data,
    scores_to_probabilities,
    to_adjacency_matrix,
)


class LapRLSClassifier(BaseEstimator, ClassifierMixin):
    """
    Laplacian Regularized Least Squares classifier (linear, inductive).

    Objective (one-vs-rest):
        min_W (1/l) ||X_l W - Y||_F^2
              + lambda_ambient ||W||_F^2
              + (lambda_intrinsic / n^2) Tr(W^T X^T L X W)
    """

    def __init__(
        self,
        graph_or_adjacency,
        lambda_ambient: float = 1.0,
        lambda_intrinsic: float = 1.0,
        fit_intercept: bool = True,
        normalized_laplacian: bool = False,
        unlabeled_value: int = -1,
        eps: float = 1e-10,
    ):
        self.graph_or_adjacency = graph_or_adjacency
        self.lambda_ambient = lambda_ambient
        self.lambda_intrinsic = lambda_intrinsic
        self.fit_intercept = fit_intercept
        self.normalized_laplacian = normalized_laplacian
        self.unlabeled_value = unlabeled_value
        self.eps = eps

    def fit(self, X, y):
        if self.lambda_ambient < 0.0:
            raise ValueError("lambda_ambient must be >= 0.")
        if self.lambda_intrinsic < 0.0:
            raise ValueError("lambda_intrinsic must be >= 0.")

        fit_data = prepare_semisupervised_classification_data(
            X=X,
            y=y,
            unlabeled_value=self.unlabeled_value,
        )

        adjacency = to_adjacency_matrix(self.graph_or_adjacency)
        if adjacency.shape[0] != fit_data.X.shape[0]:
            raise ValueError(
                "Adjacency size must match the number of rows in X."
            )
        laplacian = laplacian_matrix(
            adjacency=adjacency,
            normalized=self.normalized_laplacian,
            eps=self.eps,
        )

        X_aug = self._augment_features(fit_data.X)
        X_l = X_aug[fit_data.labeled_mask]
        y_l = fit_data.y[fit_data.labeled_mask]

        n_samples = X_aug.shape[0]
        n_labeled = fit_data.labeled_count
        n_features = X_aug.shape[1]
        n_classes = fit_data.classes.shape[0]

        Y = -np.ones((n_labeled, n_classes), dtype=float)
        for idx, cls in enumerate(fit_data.classes):
            Y[y_l == cls, idx] = 1.0

        empirical_term = (X_l.T @ X_l) / n_labeled
        intrinsic_term = (X_aug.T @ laplacian @ X_aug) / (n_samples * n_samples)

        ambient_reg = self.lambda_ambient * np.eye(n_features, dtype=float)
        if self.fit_intercept:
            ambient_reg[-1, -1] = 0.0

        system = empirical_term + ambient_reg + self.lambda_intrinsic * intrinsic_term
        rhs = (X_l.T @ Y) / n_labeled

        try:
            weights = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(system) @ rhs

        if self.fit_intercept:
            self.coef_ = weights[:-1, :].T
            self.intercept_ = weights[-1, :]
        else:
            self.coef_ = weights.T
            self.intercept_ = np.zeros(n_classes, dtype=float)

        self.classes_ = fit_data.classes
        self.n_features_in_ = fit_data.X.shape[1]
        self.laplacian_ = laplacian
        return self

    def decision_function(self, X):
        check_is_fitted(self, attributes=["coef_", "intercept_", "classes_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than in fit.")

        scores = X @ self.coef_.T + self.intercept_
        if scores.shape[1] == 1:
            return scores.ravel()
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        return scores_to_probabilities(scores)

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([X, ones])


class LapSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear Laplacian SVM with graph regularization in the primal.

    We optimize a standard linear SVM on transformed features:
        X_tilde = X @ M^{-1/2}
    where:
        M = I + (lambda_intrinsic / n^2) * X^T L X
    so the SVM regularizer becomes:
        1/2 * w^T M w
    """

    def __init__(
        self,
        graph_or_adjacency,
        C: float = 1.0,
        lambda_intrinsic: float = 1.0,
        fit_intercept: bool = True,
        normalized_laplacian: bool = False,
        unlabeled_value: int = -1,
        max_iter: int = 4000,
        tol: float = 1e-4,
        random_state: int | None = 42,
        eps: float = 1e-10,
    ):
        self.graph_or_adjacency = graph_or_adjacency
        self.C = C
        self.lambda_intrinsic = lambda_intrinsic
        self.fit_intercept = fit_intercept
        self.normalized_laplacian = normalized_laplacian
        self.unlabeled_value = unlabeled_value
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eps = eps

    def fit(self, X, y):
        if self.C <= 0:
            raise ValueError("C must be > 0.")
        if self.lambda_intrinsic < 0.0:
            raise ValueError("lambda_intrinsic must be >= 0.")

        fit_data = prepare_semisupervised_classification_data(
            X=X,
            y=y,
            unlabeled_value=self.unlabeled_value,
        )

        adjacency = to_adjacency_matrix(self.graph_or_adjacency)
        if adjacency.shape[0] != fit_data.X.shape[0]:
            raise ValueError(
                "Adjacency size must match the number of rows in X."
            )
        laplacian = laplacian_matrix(
            adjacency=adjacency,
            normalized=self.normalized_laplacian,
            eps=self.eps,
        )

        n_samples, n_features = fit_data.X.shape
        regularizer_matrix = np.eye(n_features, dtype=float) + (
            self.lambda_intrinsic / (n_samples * n_samples)
        ) * (fit_data.X.T @ laplacian @ fit_data.X)

        eigvals, eigvecs = np.linalg.eigh(0.5 * (regularizer_matrix + regularizer_matrix.T))
        eigvals = np.clip(eigvals, self.eps, None)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        transformed_X = fit_data.X @ inv_sqrt

        X_l = transformed_X[fit_data.labeled_mask]
        y_l = fit_data.y[fit_data.labeled_mask]

        svm = LinearSVC(
            C=self.C,
            fit_intercept=self.fit_intercept,
            tol=self.tol,
            max_iter=self.max_iter,
            dual="auto",
            random_state=self.random_state,
        )
        svm.fit(X_l, y_l)

        self.svm_ = svm
        self.classes_ = svm.classes_
        self.n_features_in_ = n_features
        self.transform_matrix_ = inv_sqrt
        self.laplacian_ = laplacian
        return self

    def decision_function(self, X):
        check_is_fitted(self, attributes=["svm_", "transform_matrix_", "classes_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than in fit.")

        transformed_X = X @ self.transform_matrix_
        return self.svm_.decision_function(transformed_X)

    def predict(self, X):
        check_is_fitted(self, attributes=["svm_", "transform_matrix_", "classes_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than in fit.")

        transformed_X = X @ self.transform_matrix_
        return self.svm_.predict(transformed_X)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        return scores_to_probabilities(scores)
