import numpy as np
import networkx as nx
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from graph_regularizer.labelpropagation import GRFGraphRegularizer, LabelSpreading


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


def evaluate_model(name, model, labels, test_mask):
    y_pred = model.predict(test_mask)
    y_test = labels[test_mask]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1:   {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Micro-F1:   {f1_score(y_test, y_pred, average='micro'):.4f}")
    print()
    print(classification_report(y_test, y_pred, digits=4))


def main():
    RANDOM_STATE = 70
    # Choose 'planetoid' for the canonical PyG split, or 'stratified' for a newly generated train/test split.
    MASK_MODE = "planetoid"

    dataset, data = load_cora()
    G = pyg_cora_to_networkx(data, to_undirected=True)

    train_mask, val_mask, test_mask = define_masks(
        data,
        mode=MASK_MODE,
        train_ratio=0.6,
        test_ratio=0.2,
        random_state=RANDOM_STATE,
    )
    train_mask_np = train_mask.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()
    labels = data.y.cpu().numpy()
    y = labels.copy()  # Mask test labels for semi-supervised learning
    y[test_mask_np] = -1  # Use -1 to indicate unlabeled nodes

    # Train
    grf_regularizer = GRFGraphRegularizer(max_iter=70, tol=1e-4)
    grf_regularizer.fit(G, y)
    ls_regularizer = LabelSpreading(alpha=0.5, max_iter=70, tol=1e-4)
    ls_regularizer.fit(G, y)

    # Evaluate on the test mask
    evaluate_model(
        "Graph Regularization with GRF Label Propagation",
        grf_regularizer,
        labels,
        test_mask_np,
    )
    evaluate_model(
        "Graph Regularization with Label Spreading",
        ls_regularizer,
        labels,
        test_mask_np,
    )


if __name__ == "__main__":
    main()
