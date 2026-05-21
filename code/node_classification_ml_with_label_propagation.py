import numpy as np
import networkx as nx
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from graph_regularizer import (
    GRFGraphRegularizer,
    LabelSpreading,
    LapRLSClassifier,
    LapSVMClassifier,
)


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


def define_masks(
    data,
    mode: str = "planetoid",
    train_ratio: float = 0.8,
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
        val_mask = data.val_mask.clone()
        test_mask = data.test_mask.clone()
        return train_mask, val_mask, test_mask

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

    valid_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=relative_test_ratio,
        stratify=y[remaining_ids],
        random_state=random_state,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(train_ids, dtype=torch.long)] = True
    val_mask[torch.tensor(valid_ids, dtype=torch.long)] = True
    test_mask[torch.tensor(test_ids, dtype=torch.long)] = True
    return train_mask, val_mask, test_mask


# ------------------------------------------------------------
# 4. Train and evaluate models
# ------------------------------------------------------------


def evaluate_predictions(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1:   {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Micro-F1:   {f1_score(y_true, y_pred, average='micro'):.4f}")
    print()
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def demonstrate_inductive_behavior(
    grf_regularizer,
    ls_regularizer,
    laprls,
    lapsvm,
    X,
    labels,
    A,
    train_mask_np,
    val_mask_np,
    test_mask_np,
    random_state=42,
):
    print("\n" + "=" * 72)
    print("Inductive behavior checks")
    print("=" * 72)

    # --------------------------------------------------------
    # 1) Out-of-sample predictions for synthetic unseen nodes.
    # --------------------------------------------------------
    rng = np.random.default_rng(random_state)
    train_idx = np.where(train_mask_np)[0]
    seed_idx = rng.choice(train_idx, size=15, replace=False)
    X_new = X[seed_idx] + 0.03 * rng.normal(size=(15, X.shape[1]))

    pred_new_laprls = laprls.predict(X_new)
    pred_new_lapsvm = lapsvm.predict(X_new)
    print("\nOut-of-sample synthetic nodes (not in training graph):")
    print("LapRLS predictions (first 10):", pred_new_laprls[:10])
    print("LapSVM predictions (first 10):", pred_new_lapsvm[:10])

    try:
        _ = grf_regularizer.predict(X_new)
    except Exception as exc:
        print(
            "GRF Label Propagation is not out-of-sample by design:",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        _ = ls_regularizer.predict(X_new)
    except Exception as exc:
        print(
            "Label Spreading is not out-of-sample by design:",
            f"{type(exc).__name__}: {exc}",
        )

    # ------------------------------------------------------------------
    # 2) Train LapRLS / LapSVM without test nodes in graph, then predict.
    # ------------------------------------------------------------------
    trainval_mask_np = train_mask_np | val_mask_np
    trainval_idx = np.where(trainval_mask_np)[0]
    test_idx = np.where(test_mask_np)[0]

    X_trainval = X[trainval_idx]
    A_trainval = A[np.ix_(trainval_idx, trainval_idx)]

    y_trainval_semi = np.full(trainval_idx.shape[0], -1, dtype=labels.dtype)
    local_train_nodes = np.where(train_mask_np[trainval_idx])[0]
    y_trainval_semi[local_train_nodes] = labels[trainval_idx][local_train_nodes]

    laprls_no_test_nodes = LapRLSClassifier(
        graph_or_adjacency=A_trainval,
        lambda_ambient=1.0,
        lambda_intrinsic=1.0,
        fit_intercept=True,
    ).fit(X_trainval, y_trainval_semi)

    lapsvm_no_test_nodes = LapSVMClassifier(
        graph_or_adjacency=A_trainval,
        C=1.0,
        lambda_intrinsic=1.0,
        fit_intercept=True,
        max_iter=5000,
    ).fit(X_trainval, y_trainval_semi)

    y_test = labels[test_idx]
    pred_laprls_no_test = laprls_no_test_nodes.predict(X[test_idx])
    pred_lapsvm_no_test = lapsvm_no_test_nodes.predict(X[test_idx])

    print(
        "\nPredicting test nodes that were excluded from the graph used in training "
        f"({test_idx.shape[0]} nodes):"
    )
    evaluate_predictions("LapRLS (trained without test nodes)", y_test, pred_laprls_no_test)
    evaluate_predictions("LapSVM (trained without test nodes)", y_test, pred_lapsvm_no_test)


def main():
    RANDOM_STATE = 70
    # Choose 'planetoid' for the canonical PyG split, or 'stratified' for a newly generated train/test split.
    MASK_MODE = "stratified"

    _, data = load_cora()
    G = pyg_cora_to_networkx(data, to_undirected=True)
    A = nx.to_numpy_array(G, dtype=float)
    X = data.x.cpu().numpy()

    train_mask, val_mask, test_mask = define_masks(
        data,
        mode=MASK_MODE,
        train_ratio=0.8,
        test_ratio=0.15,
        random_state=RANDOM_STATE,
    )
    train_mask_np = train_mask.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()
    labels = data.y.cpu().numpy()
    y = labels.copy()
    y[~train_mask_np] = -1  # Semi-supervised: only train labels are visible

    # Train
    grf_regularizer = GRFGraphRegularizer(max_iter=70, tol=1e-4)
    grf_regularizer.fit(G, y)
    ls_regularizer = LabelSpreading(alpha=0.2, max_iter=70, tol=1e-4)
    ls_regularizer.fit(G, y)

    laprls = LapRLSClassifier(
        graph_or_adjacency=A,
        lambda_ambient=1.0,
        lambda_intrinsic=1.0,
        fit_intercept=True,
    ).fit(X, y)
    lapsvm = LapSVMClassifier(
        graph_or_adjacency=A,
        C=1.0,
        lambda_intrinsic=1.0,
        fit_intercept=True,
        max_iter=5000,
    ).fit(X, y)

    # Evaluate on test nodes
    test_idx = np.where(test_mask_np)[0]
    y_test = labels[test_idx]
    pred_grf = grf_regularizer.predict(test_idx)
    pred_ls = ls_regularizer.predict(test_idx)
    pred_laprls = laprls.predict(X[test_idx])
    pred_lapsvm = lapsvm.predict(X[test_idx])

    evaluate_predictions(
        "Graph Regularization with GRF Label Propagation",
        y_test,
        pred_grf,
    )
    evaluate_predictions(
        "Graph Regularization with Label Spreading",
        y_test,
        pred_ls,
    )
    evaluate_predictions(
        "Graph Regularization with LapRLS",
        y_test,
        pred_laprls,
    )
    evaluate_predictions(
        "Graph Regularization with LapSVM",
        y_test,
        pred_lapsvm,
    )

    demonstrate_inductive_behavior(
        grf_regularizer=grf_regularizer,
        ls_regularizer=ls_regularizer,
        laprls=laprls,
        lapsvm=lapsvm,
        X=X,
        labels=labels,
        A=A,
        train_mask_np=train_mask_np,
        val_mask_np=val_mask_np,
        test_mask_np=test_mask_np,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
