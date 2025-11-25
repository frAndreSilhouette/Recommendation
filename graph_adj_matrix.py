import torch
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

def load_sequences_from_file(path):
    """
    读取训练/验证/测试文件，每行格式：
    user seq target
    seq 形如 4,5,6,7
    """
    user2seqs = defaultdict(list)
    user_set = set()
    item_set = set()

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            user, seq_str, target = int(parts[0]), parts[1], int(parts[2])
            user_set.add(user)
            seq = list(map(int, seq_str.split(","))) if "," in seq_str else [int(seq_str)]
            user2seqs[user].append(seq)
            item_set.update(seq + [target])

    return user2seqs, user_set, item_set


def compute_user_item_graph_scipy(user2seqs, n_users, n_items):
    """
    构造用户-商品二值图（bipartite adjacency matrix）
    每个用户-商品组合只生成一条边，权重为1
    """
    row_idx, col_idx = [], []

    for user, seq_list in user2seqs.items():
        all_items = set(item for seq in seq_list for item in seq)  # 合并所有历史序列去重
        for item in all_items:
            row_idx.extend([user, n_users + item])
            col_idx.extend([n_users + item, user])

    data = np.ones(len(row_idx), dtype=np.float32)
    adj = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n_users + n_items, n_users + n_items))
    return adj.tocsr()


def normalize_adj_scipy(adj):
    """稀疏矩阵归一化 D^{-1/2} A D^{-1/2}"""
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum>0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()


def scipy_to_torch_sparse(adj):
    """Scipy 稀疏矩阵转换为 torch.sparse_coo_tensor"""
    adj = adj.tocoo()
    indices = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
    values = torch.tensor(adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=adj.shape).coalesce()


def build_user_item_graph(train_samples, n_users, n_items):
    """
    构建用户-商品图，使用所有历史序列，不提取最长序列
    train_samples: list of (user_id, seq, target_item)
    """
    user2seqs = defaultdict(list)
    for user, seq, _ in train_samples:
        user2seqs[user].append(seq)

    adj = compute_user_item_graph_scipy(user2seqs, n_users, n_items)
    # adj = normalize_adj_scipy(adj)  # 如果需要归一化可以开启
    return scipy_to_torch_sparse(adj)
