import warnings
import numpy as np
import networkx as nx
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
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
            safe_add_feature(
                "out_degree_centrality", lambda: nx.out_degree_centrality(G)
            )
            safe_add_feature("total_degree_centrality", lambda: nx.degree_centrality(G))
        else:
            safe_add_feature("degree_centrality", lambda: nx.degree_centrality(G))

        # Closeness centrality
        if G.is_directed():
            safe_add_feature("closeness_in", lambda: nx.closeness_centrality(G))
            safe_add_feature(
                "closeness_out", lambda: nx.closeness_centrality(G.reverse(copy=True))
            )
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
        safe_add_feature("clustering", lambda: nx.clustering(G.to_undirected()))

        return self

    def transform(self, X):
        if not hasattr(self, "feature_maps_"):
            raise RuntimeError(
                "The transformer must be fitted before calling transform()."
            )

        node_ids = np.asarray(X).reshape(-1).astype(int)

        X_out = np.zeros((len(node_ids), len(self.feature_maps_)), dtype=float)

        for j, feature_map in enumerate(self.feature_maps_):
            for i, node in enumerate(node_ids):
                X_out[i, j] = feature_map.get(node, 0.0)

        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


# ------------------------------------------------------------
# 5. Full GDV computation with orbit-count / ORCA
# ------------------------------------------------------------


def make_orca_compatible_graph(G, num_nodes: int):
    """
    ORCA-style orbit counting assumes a simple undirected graph.

    This function:
    - converts the graph to undirected;
    - removes self-loops;
    - removes parallel edges if present;
    - makes sure all nodes 0, ..., num_nodes - 1 are present.
    """
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    H.add_nodes_from(range(num_nodes))
    return H


def compute_full_gdv_features(
    G,
    num_nodes: int,
    graphlet_size: int = 5,
    log_transform: bool = True,
):
    """
    Computes node Graphlet Degree Vectors using orbit-count.

    graphlet_size=4 returns orbit counts up to 4-node graphlets.
    graphlet_size=5 returns the standard full node GDV up to 5-node graphlets.

    Returns
    -------
    X_gdv : np.ndarray
        Matrix of shape (num_nodes, num_orbits).

    gdv_feature_names : list[str]
        Feature names gdv_o0, gdv_o1, ...
    """
    try:
        import orbit_count
    except ImportError as exc:
        raise ImportError(
            "The package 'orbit-count' is required. Install it with:\n\n"
            "    uv add orbit-count\n\n"
            "or:\n\n"
            "    pip install orbit-count\n"
        ) from exc

    if graphlet_size not in {4, 5}:
        raise ValueError("orbit-count supports graphlet_size=4 or graphlet_size=5.")

    H = make_orca_compatible_graph(G, num_nodes=num_nodes)

    node_list = list(range(num_nodes))

    X_gdv = orbit_count.node_orbit_counts(
        H,
        graphlet_size=graphlet_size,
        node_list=node_list,
    ).astype(float)

    if log_transform:
        X_gdv = np.log1p(X_gdv)

    gdv_feature_names = [f"gdv_o{i}" for i in range(X_gdv.shape[1])]

    return X_gdv, gdv_feature_names


# ------------------------------------------------------------
# 6. Dimensionality reduction of Cora node attributes
# ------------------------------------------------------------


def compute_reduced_cora_node_features(
    data,
    train_mask_np,
    n_components: int = 7,
    random_state: int = 42,
):
    """
    Reduces original Cora node attributes to n_components.

    Cora node attributes are high-dimensional sparse bag-of-words-like
    vectors. TruncatedSVD is appropriate because it does not require
    centering the feature matrix.

    To avoid supervised evaluation leakage, the reducer is fitted only
    on training nodes and then applied to all nodes.
    """
    X_raw = data.x.cpu().numpy().astype(float)

    reducer = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    )

    reducer.fit(X_raw[train_mask_np])

    X_reduced = reducer.transform(X_raw)

    feature_names = [f"cora_svd_{i}" for i in range(n_components)]

    return X_reduced, feature_names, reducer


# ------------------------------------------------------------
# 7. Train and evaluate models
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

    # For ORCA / GDV computation, the graph must be simple and undirected.
    TO_UNDIRECTED = True

    # Choose 'planetoid' for the canonical PyG split, or 'stratified'
    # for a newly generated train/test split.
    MASK_MODE = "stratified"

    # Feature switches
    USE_NETWORKX_STRUCTURAL_FEATURES = True
    USE_GDV_FEATURES = False
    USE_REDUCED_CORA_NODE_FEATURES = True

    # Full GDV: graphlets up to 5 nodes.
    GDV_GRAPHLET_SIZE = 5

    # Cora node attribute reduction.
    CORA_NODE_FEATURE_COMPONENTS = 40

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

    feature_blocks = []
    feature_names = []

    # --------------------------------------------------------
    # NetworkX structural features
    # --------------------------------------------------------

    if USE_NETWORKX_STRUCTURAL_FEATURES:
        nx_feature_extractor = NetworkXNodeFeatureTransformer(
            graph=G,
            approximate_betweenness=True,
            betweenness_k=256,
            random_state=RANDOM_STATE,
        )

        X_structural = nx_feature_extractor.fit_transform(node_ids)

        structural_feature_names = list(nx_feature_extractor.get_feature_names_out())

        feature_blocks.append(X_structural)
        feature_names.extend(structural_feature_names)

        print("\nNetworkX structural features:")
        print(structural_feature_names)

    # --------------------------------------------------------
    # Full Graphlet Degree Vector features
    # --------------------------------------------------------

    if USE_GDV_FEATURES:
        X_gdv, gdv_feature_names = compute_full_gdv_features(
            G,
            num_nodes=data.num_nodes,
            graphlet_size=GDV_GRAPHLET_SIZE,
            log_transform=True,
        )

        feature_blocks.append(X_gdv)
        feature_names.extend(gdv_feature_names)

        print("\nGDV features:")
        print(f"Number of GDV features: {X_gdv.shape[1]}")
        print(gdv_feature_names)

    # --------------------------------------------------------
    # Reduced original Cora node attributes
    # --------------------------------------------------------

    if USE_REDUCED_CORA_NODE_FEATURES:
        X_cora_reduced, cora_feature_names, reducer = (
            compute_reduced_cora_node_features(
                data,
                train_mask_np=train_mask_np,
                n_components=CORA_NODE_FEATURE_COMPONENTS,
                random_state=RANDOM_STATE,
            )
        )

        feature_blocks.append(X_cora_reduced)
        feature_names.extend(cora_feature_names)

        print("\nReduced Cora node-attribute features:")
        print(cora_feature_names)
        print(
            "Explained variance ratio sum:",
            reducer.explained_variance_ratio_.sum(),
        )

    # --------------------------------------------------------
    # Final feature matrix
    # --------------------------------------------------------

    X_all = np.hstack(feature_blocks)

    print("\nFinal feature matrix")
    print("--------------------")
    print("Shape:", X_all.shape)
    print("Number of features:", len(feature_names))

    X_train = X_all[train_mask_np]
    X_test = X_all[test_mask_np]
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
