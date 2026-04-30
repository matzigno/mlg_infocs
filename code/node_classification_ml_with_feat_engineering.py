import warnings
import numpy as np
import networkx as nx
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ------------------------------------------------------------
# 1. Load Cora from PyTorch Geometric
# ------------------------------------------------------------

def load_cora(root: str = "./data/planetoid"):
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    return dataset, data


# ------------------------------------------------------------
# 2. Convert PyG Data object into a NetworkX graph
# ------------------------------------------------------------

def pyg_cora_to_networkx(data, to_undirected: bool = True):
    """
    If to_undirected=True, returns nx.Graph.
    If to_undirected=False, returns nx.DiGraph.

    For standard Cora node-classification experiments, the undirected
    interpretation is commonly used.
    """
    G = to_networkx(
        data,
        to_undirected=to_undirected,
        remove_self_loops=True,
    )

    # Ensure node ids are standard Python integers.
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


# ------------------------------------------------------------
# 3. Define train/test masks
# ------------------------------------------------------------

def define_train_test_masks(
    data,
    mode: str = "planetoid",
    train_ratio: float = 0.6,
    test_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    mode='planetoid':
        Uses the canonical masks already stored in the PyG Cora object.

    mode='stratified':
        Creates new stratified train/test masks from the node labels.
    """
    num_nodes = data.num_nodes

    if mode == "planetoid":
        train_mask = data.train_mask.clone()
        test_mask = data.test_mask.clone()
        return train_mask, test_mask

    if mode != "stratified":
        raise ValueError("mode must be either 'planetoid' or 'stratified'.")

    y = data.y.cpu().numpy()
    node_ids = np.arange(num_nodes)

    train_ids, remaining_ids = train_test_split(
        node_ids,
        train_size=train_ratio,
        stratify=y,
        random_state=random_state,
    )

    relative_test_ratio = test_ratio / (1.0 - train_ratio)

    _, test_ids = train_test_split(
        remaining_ids,
        test_size=relative_test_ratio,
        stratify=y[remaining_ids],
        random_state=random_state,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(train_ids, dtype=torch.long)] = True
    test_mask[torch.tensor(test_ids, dtype=torch.long)] = True

    return train_mask, test_mask


# ------------------------------------------------------------
# 4. Sklearn transformer for NetworkX feature engineering
# ------------------------------------------------------------

class NetworkXNodeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts one feature vector per node from a NetworkX graph.

    Features include:
    - raw degree information;
    - degree centrality;
    - in/out centrality for directed graphs;
    - closeness centrality;
    - betweenness centrality;
    - eigenvector centrality when available;
    - PageRank;
    - HITS hub/authority scores for directed graphs;
    - clustering coefficient.

    The transformer expects X to contain node ids.
    """

    def __init__(
        self,
        graph,
        approximate_betweenness: bool = True,
        betweenness_k: int = 256,
        max_iter: int = 1000,
        tol: float = 1e-06,
        random_state: int = 42,
    ):
        self.graph = graph
        self.approximate_betweenness = approximate_betweenness
        self.betweenness_k = betweenness_k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X=None, y=None):
        G = self.graph
        n = G.number_of_nodes()
        self.feature_maps_ = []
        self.feature_names_ = []

        def add_feature(name, values):
            self.feature_names_.append(name)
            self.feature_maps_.append(values)

        def safe_add_feature(name, func):
            try:
                values = func()
                add_feature(name, values)
            except Exception as exc:
                warnings.warn(f"Skipping feature '{name}': {exc}")

        # Degree counts
        if G.is_directed():
            add_feature("in_degree", dict(G.in_degree()))
            add_feature("out_degree", dict(G.out_degree()))
            add_feature("total_degree", dict(G.degree()))
        else:
            add_feature("degree", dict(G.degree()))

        # Degree centralities
        if G.is_directed():
            safe_add_feature("in_degree_centrality", lambda: nx.in_degree_centrality(G))
            safe_add_feature("out_degree_centrality", lambda: nx.out_degree_centrality(G))
            safe_add_feature("total_degree_centrality", lambda: nx.degree_centrality(G))
        else:
            safe_add_feature("degree_centrality", lambda: nx.degree_centrality(G))

        # Closeness centrality
        if G.is_directed():
            safe_add_feature("closeness_in", lambda: nx.closeness_centrality(G))
            safe_add_feature("closeness_out", lambda: nx.closeness_centrality(G.reverse(copy=True)))
        else:
            safe_add_feature("closeness", lambda: nx.closeness_centrality(G))

        # Betweenness centrality: approximate by default for speed.
        def compute_betweenness():
            if self.approximate_betweenness and self.betweenness_k is not None:
                k = min(self.betweenness_k, n)
                return nx.betweenness_centrality(
                    G,
                    k=k,
                    normalized=True,
                    seed=self.random_state,
                )
            return nx.betweenness_centrality(G, normalized=True)

        safe_add_feature("betweenness", compute_betweenness)

        # Eigenvector centrality
        safe_add_feature(
            "eigenvector_centrality",
            lambda: nx.eigenvector_centrality(
                G,
                max_iter=self.max_iter,
                tol=self.tol,
            ),
        )

        # PageRank
        safe_add_feature(
            "pagerank",
            lambda: nx.pagerank(
                G,
                max_iter=self.max_iter,
                tol=self.tol,
            ),
        )

        # HITS is meaningful for directed graphs.
        if G.is_directed():
            def compute_hubs():
                hubs, _ = nx.hits(
                    G,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    normalized=True,
                )
                return hubs

            def compute_authorities():
                _, authorities = nx.hits(
                    G,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    normalized=True,
                )
                return authorities

            safe_add_feature("hits_hub", compute_hubs)
            safe_add_feature("hits_authority", compute_authorities)

        # Clustering coefficient
        safe_add_feature("clustering", lambda: nx.clustering(G))

        return self

    def transform(self, X):
        if not hasattr(self, "feature_maps_"):
            raise RuntimeError("The transformer must be fitted before calling transform().")

        node_ids = np.asarray(X).reshape(-1).astype(int)

        X_out = np.zeros((len(node_ids), len(self.feature_maps_)), dtype=float)

        for j, feature_map in enumerate(self.feature_maps_):
            for i, node in enumerate(node_ids):
                X_out[i, j] = feature_map.get(node, 0.0)

        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


# ------------------------------------------------------------
# 5. Train and evaluate models
# ------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1:   {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Micro-F1:   {f1_score(y_test, y_pred, average='micro'):.4f}")
    print()
    print(classification_report(y_test, y_pred, digits=4))


def main():
    RANDOM_STATE = 70
    # Set to False if you want to preserve the directed citation graph.
    TO_UNDIRECTED = True
    # Choose 'planetoid' for the canonical PyG split, or 'stratified'
    # for a newly generated train/test split.
    MASK_MODE = "stratified"
    dataset, data = load_cora()
    G = pyg_cora_to_networkx(data, to_undirected=TO_UNDIRECTED)

    train_mask, test_mask = define_train_test_masks(
        data,
        mode=MASK_MODE,
        train_ratio=0.6,
        test_ratio=0.2,
        random_state=RANDOM_STATE,
    )

    y = data.y.cpu().numpy()
    node_ids = np.arange(data.num_nodes).reshape(-1, 1)

    train_mask_np = train_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()

    # Feature engineering from the NetworkX graph.
    nx_feature_extractor = NetworkXNodeFeatureTransformer(
        graph=G,
        approximate_betweenness=True,
        betweenness_k=256,
        random_state=RANDOM_STATE,
    )

    X_structural = nx_feature_extractor.fit_transform(node_ids)

    print("Extracted structural features:")
    print(nx_feature_extractor.get_feature_names_out())

    X_train = X_structural[train_mask_np]
    X_test = X_structural[test_mask_np]
    y_train = y[train_mask_np]
    y_test = y[test_mask_np]

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Stacking Classifier
    # --------------------------------------------------------

    base_estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "svc",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            C=10.0,
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]

    stacking_classifier = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )

    # Train
    logistic_regression.fit(X_train, y_train)
    stacking_classifier.fit(X_train, y_train)

    # Evaluate on the test mask
    evaluate_model("Logistic Regression", logistic_regression, X_test, y_test)
    evaluate_model("Stacking Classifier", stacking_classifier, X_test, y_test)


if __name__ == "__main__":
    main()