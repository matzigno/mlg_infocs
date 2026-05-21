from __future__ import annotations

import numpy as np
import torch


class GraphFactorization:
    """
    Graph Factorization implemented in PyTorch.

    Objective (edge reconstruction with regularization):
        min_Z mean_{(i,j) in E} (A_ij - <z_i, z_j>)^2
             + negative_ratio * mean_{(i,j) in N} (<z_i, z_j>)^2
             + reg * mean_i ||z_i||^2
    where N is a sampled set of non-edges.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_dim: int = 2,
        epochs: int = 1500,
        lr: float = 0.03,
        reg: float = 1e-4,
        negative_ratio: float = 1.0,
        seed: int = 42,
        device: str | None = None,
        verbose: bool = False,
        log_every: int = 200,
    ) -> None:
        adjacency = np.asarray(adjacency_matrix, dtype=float)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency_matrix must be a square 2D array.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1.")
        if epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if lr <= 0:
            raise ValueError("lr must be > 0.")
        if reg < 0:
            raise ValueError("reg must be >= 0.")
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be >= 0.")

        self.adjacency_matrix = adjacency.copy()
        np.fill_diagonal(self.adjacency_matrix, 0.0)

        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.negative_ratio = negative_ratio
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.log_every = log_every

    def fit(self) -> "GraphFactorization":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        num_nodes = self.adjacency_matrix.shape[0]
        use_device = (
            torch.device(self.device)
            if self.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        is_undirected = np.allclose(
            self.adjacency_matrix,
            self.adjacency_matrix.T,
            atol=1e-12,
        )

        if is_undirected:
            rows, cols = np.where(np.triu(self.adjacency_matrix, k=1) > 0.0)
        else:
            rows, cols = np.where(self.adjacency_matrix > 0.0)
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]

        if rows.size == 0:
            raise ValueError("The graph has no edges with positive weight.")

        edge_weights = self.adjacency_matrix[rows, cols].astype(np.float32)
        pos_src = torch.tensor(rows, dtype=torch.long, device=use_device)
        pos_dst = torch.tensor(cols, dtype=torch.long, device=use_device)
        pos_w = torch.tensor(edge_weights, dtype=torch.float32, device=use_device)

        adjacency_mask = torch.tensor(
            self.adjacency_matrix > 0.0, dtype=torch.bool, device=use_device
        )
        adjacency_mask.fill_diagonal_(True)

        if is_undirected:
            max_pairs = num_nodes * (num_nodes - 1) // 2
        else:
            max_pairs = num_nodes * (num_nodes - 1)
        max_non_edges = max_pairs - int(rows.size)
        neg_per_epoch = int(round(self.negative_ratio * rows.size))
        neg_per_epoch = max(0, min(neg_per_epoch, max_non_edges))

        embedding_param = torch.nn.Embedding(num_nodes, self.embedding_dim).to(
            use_device
        )
        torch.nn.init.normal_(embedding_param.weight, mean=0.0, std=0.1)

        optimizer = torch.optim.Adam(embedding_param.parameters(), lr=self.lr)
        losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            Z = embedding_param.weight
            pos_scores = (Z[pos_src] * Z[pos_dst]).sum(dim=1)
            pos_loss = torch.mean((pos_w - pos_scores) ** 2)

            if neg_per_epoch > 0:
                neg_src, neg_dst = self._sample_negative_edges(
                    num_nodes=num_nodes,
                    num_samples=neg_per_epoch,
                    adjacency_mask=adjacency_mask,
                    is_undirected=is_undirected,
                    device=use_device,
                )
                neg_scores = (Z[neg_src] * Z[neg_dst]).sum(dim=1)
                neg_loss = torch.mean(neg_scores**2)
            else:
                neg_loss = torch.tensor(0.0, device=use_device)

            reg_loss = torch.mean(Z.pow(2))
            loss = pos_loss + neg_loss + self.reg * reg_loss

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            if self.verbose and (epoch == 1 or epoch % self.log_every == 0):
                print(f"[GraphFactorization] epoch={epoch:04d} loss={loss_value:.6f}")

        self.device_ = str(use_device)
        self.is_undirected_ = is_undirected
        self.loss_history_ = losses
        self.embedding_torch_ = embedding_param.weight.detach().cpu()
        self.embedding_ = self.embedding_torch_.numpy()
        return self

    def fit_transform(self) -> np.ndarray:
        self.fit()
        return self.embedding_

    @staticmethod
    def _sample_negative_edges(
        num_nodes: int,
        num_samples: int,
        adjacency_mask: torch.Tensor,
        is_undirected: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        src_chunks: list[torch.Tensor] = []
        dst_chunks: list[torch.Tensor] = []
        collected = 0
        attempts = 0
        max_attempts = 30

        while collected < num_samples and attempts < max_attempts:
            attempts += 1
            needed = num_samples - collected
            trial_size = max(needed * 4, 1024)

            src = torch.randint(0, num_nodes, (trial_size,), device=device)
            dst = torch.randint(0, num_nodes, (trial_size,), device=device)

            non_diag = src != dst
            src = src[non_diag]
            dst = dst[non_diag]

            if is_undirected:
                src_new = torch.minimum(src, dst)
                dst_new = torch.maximum(src, dst)
                src, dst = src_new, dst_new

            not_edges = ~adjacency_mask[src, dst]
            src = src[not_edges]
            dst = dst[not_edges]

            if src.numel() == 0:
                continue

            take = min(needed, int(src.numel()))
            src_chunks.append(src[:take])
            dst_chunks.append(dst[:take])
            collected += take

        if collected < num_samples:
            raise RuntimeError(
                "Unable to sample enough negative edges. "
                "The graph may be too dense for the requested negative_ratio."
            )

        neg_src = torch.cat(src_chunks, dim=0)
        neg_dst = torch.cat(dst_chunks, dim=0)
        return neg_src, neg_dst
