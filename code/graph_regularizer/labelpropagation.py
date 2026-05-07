from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
import numpy as np
import networkx as nx


class GRFGraphRegularizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=30, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)
        self.X_ = X
        D = [X.degree(n) for n in X.nodes()]
        D = np.diag(D)  # goal is to compute D^-1 * A
        P = np.dot(
            np.linalg.inv(D), nx.to_numpy_array(self.X_)
        )  # compute the propagation matrix
        # U and L construction
        labeled_index = np.where(y != -1)[0]
        self.classes_ = np.unique(y[labeled_index])

        # Label propagation algorithm
        labeled_indexes = {c: np.where(y == c)[0] for c in self.classes_}
        indicators = np.eye(len(self.classes_))
        Y0 = np.zeros((len(y), len(self.classes_)))
        for c, class_index in labeled_indexes.items():
            Y0[class_index, :] = indicators[c]
        Y_prev, it, c_tol = Y0, 0, 10
        while it < self.max_iter and c_tol > self.tol:
            Y = np.dot(P, Y_prev)
            Y[labeled_index, :] = Y0[labeled_index, :]  # force labeled nodes
            it += 1
            c_tol = np.sum(np.abs(Y - Y_prev))
            Y_prev = Y
            print(f"Iteration {it}, convergence tolerance: {c_tol:.5f}")
        self.label_distributions_ = Y
        return self

    def predict_proba(self, X=None):
        return (
            self.label_distributions_[X] if X is not None else self.label_distributions_
        )

    def predict(self, X=None):
        return self.classes_[np.argmax(self.predict_proba(X=X), axis=1)].ravel()

    def _validate_data(self, X, y):
        if not isinstance(X, nx.Graph):
            raise ValueError("Input should be a networkX graph")
        if not len(y) == len(X.nodes()):
            raise ValueError(
                "Label data input shape should be equal to the number of nodes in the graph"
            )
        return X, y


class LabelSpreading(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5, max_iter=30, tol=1e-3):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)
        self.X_ = X
        D = [X.degree(n) for n in X.nodes()]
        D_inv_square = np.diag(
            1.0 / np.sqrt(D)
        )  # goal is to compute D^-1/2 * A * D^-1/2
        S = (
            D_inv_square @ nx.to_numpy_array(self.X_) @ D_inv_square
        )  # compute the propagation matrix
        # U and L construction
        labeled_index = np.where(y != -1)[0]
        self.classes_ = np.unique(y[labeled_index])

        # Label propagation algorithm
        labeled_indexes = {c: np.where(y == c)[0] for c in self.classes_}
        indicators = np.eye(len(self.classes_))
        Y0 = np.zeros((len(y), len(self.classes_)))
        for c, class_index in labeled_indexes.items():
            Y0[class_index, :] = indicators[c]
        Y_prev, it, c_tol = Y0, 0, 10
        while it < self.max_iter and c_tol > self.tol:
            Y = self.alpha * S @ Y_prev + (1 - self.alpha) * Y0
            it += 1
            c_tol = np.sum(np.abs(Y - Y_prev))
            Y_prev = Y
            print(f"Iteration {it}, convergence tolerance: {c_tol:.5f}")
        self.label_distributions_ = Y
        return self

    def predict_proba(self, X=None):
        return (
            self.label_distributions_[X] if X is not None else self.label_distributions_
        )

    def predict(self, X=None):
        return self.classes_[np.argmax(self.predict_proba(X=X), axis=1)].ravel()

    def _validate_data(self, X, y):
        if not isinstance(X, nx.Graph):
            raise ValueError("Input should be a networkX graph")
        if not len(y) == len(X.nodes()):
            raise ValueError(
                "Label data input shape should be equal to the number of nodes in the graph"
            )
        return X, y
