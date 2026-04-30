import torch
import numpy as np
import networkx as nx

from collections import defaultdict
from torch_geometric.data import HeteroData


def networkx_to_heterodata(
    G,
    node_type_attr="type",
    edge_type_attr="relation",
    node_feat_attr="x",
    node_label_attr="y",
    add_reverse_for_undirected=True,
    reverse_relation_prefix="rev_",
    store_multigraph_edge_key=True,
):
    """
    Convert a NetworkX graph into a PyG HeteroData object.

    Supports:
    - nx.Graph
    - nx.DiGraph
    - nx.MultiGraph
    - nx.MultiDiGraph

    For undirected NetworkX graphs, reciprocal PyG edges are added explicitly.
    For heterogeneous edges, reciprocal edges are stored under a reverse edge type,
    e.g. ('paper', 'rev_writes', 'author').

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    node_type_attr : str
        Node attribute containing the node type.
    edge_type_attr : str
        Edge attribute containing the relation type.
    node_feat_attr : str
        Node attribute containing node features.
    node_label_attr : str
        Node attribute containing node labels.
    add_reverse_for_undirected : bool
        If True, add reciprocal edges when G is undirected.
    reverse_relation_prefix : str
        Prefix used for reciprocal heterogeneous relations.
    store_multigraph_edge_key : bool
        If True, stores NetworkX multigraph edge keys as edge attributes.

    Returns
    -------
    data : torch_geometric.data.HeteroData
    """

    data = HeteroData()

    # ------------------------------------------------------------------
    # 1. Group nodes by node type.
    # ------------------------------------------------------------------
    nodes_by_type = defaultdict(list)

    for node, attrs in G.nodes(data=True):
        if node_type_attr not in attrs:
            raise ValueError(
                f"Node {node!r} has no attribute {node_type_attr!r}."
            )

        node_type = attrs[node_type_attr]
        nodes_by_type[node_type].append(node)

    # ------------------------------------------------------------------
    # 2. Create local node ids for each node type.
    # ------------------------------------------------------------------
    local_id = {}

    for node_type, nodes in nodes_by_type.items():
        local_id[node_type] = {
            node: i for i, node in enumerate(nodes)
        }

        # Node features.
        if all(node_feat_attr in G.nodes[n] for n in nodes):
            x = np.stack([
                G.nodes[n][node_feat_attr]
                for n in nodes
            ])
            data[node_type].x = torch.tensor(x, dtype=torch.float32)
        else:
            data[node_type].num_nodes = len(nodes)

        # Node labels, if available for all nodes of this type.
        if all(node_label_attr in G.nodes[n] for n in nodes):
            y = [
                G.nodes[n][node_label_attr]
                for n in nodes
            ]
            data[node_type].y = torch.tensor(y, dtype=torch.long)

    # ------------------------------------------------------------------
    # 3. Collect edges by heterogeneous edge type.
    # ------------------------------------------------------------------
    edges_by_type = defaultdict(list)
    edge_keys_by_type = defaultdict(list)

    is_directed = nx.is_directed(G)
    is_multigraph = G.is_multigraph()

    if is_multigraph:
        edge_iterator = G.edges(keys=True, data=True)
    else:
        edge_iterator = (
            (u, v, None, attrs)
            for u, v, attrs in G.edges(data=True)
        )

    def add_edge(u, v, attrs, key=None, reverse=False):
        src_type = G.nodes[u][node_type_attr]
        dst_type = G.nodes[v][node_type_attr]

        if edge_type_attr not in attrs:
            raise ValueError(
                f"Edge ({u!r}, {v!r}) has no attribute {edge_type_attr!r}."
            )

        relation = attrs[edge_type_attr]

        if reverse:
            relation = f"{reverse_relation_prefix}{relation}"

        edge_type = (src_type, relation, dst_type)

        src_local = local_id[src_type][u]
        dst_local = local_id[dst_type][v]

        edges_by_type[edge_type].append((src_local, dst_local))

        if store_multigraph_edge_key and key is not None:
            edge_keys_by_type[edge_type].append(key)

    for u, v, key, attrs in edge_iterator:
        # Original direction.
        add_edge(u, v, attrs, key=key, reverse=False)

        # Explicit reciprocal direction for undirected NetworkX graphs.
        if add_reverse_for_undirected and not is_directed and u != v:
            add_edge(v, u, attrs, key=key, reverse=True)

    # ------------------------------------------------------------------
    # 4. Create PyG edge_index tensors.
    # ------------------------------------------------------------------
    for edge_type, edges in edges_by_type.items():
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data[edge_type].edge_index = edge_index

        if store_multigraph_edge_key and edge_type in edge_keys_by_type:
            keys = edge_keys_by_type[edge_type]

            # Only store numeric keys as tensors.
            if all(isinstance(k, (int, np.integer)) for k in keys):
                data[edge_type].edge_key = torch.tensor(keys, dtype=torch.long)

    return data