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
    item_set = set()
    user_set = set()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            user = int(parts[0])
            seq_str = parts[1]
            target = int(parts[2])

            user_set.add(user)

            # 解析 seq
            if "," in seq_str:
                seq = list(map(int, seq_str.split(",")))
            else:
                seq = [int(seq_str)]

            # 记录所有商品
            for i in seq:
                item_set.add(i)
            item_set.add(target)

            # 存储：用户 → 多条训练序列
            user2seqs[user].append(seq)

    return user2seqs, user_set, item_set


def extract_last_sequences(user2seqs):
    """
    对每个用户，提取最长的序列（即购买次数最多的那条）
    """
    last_seqs = []

    for user, seq_list in user2seqs.items():
        longest = max(seq_list, key=lambda x: len(x))
        last_seqs.append((user, longest))  # 保留 user_id

    return last_seqs


def compute_user_item_graph_scipy(seqs_with_user, n_users, n_items):
    """
    使用 scipy 构造用户-物品 bipartite 邻接矩阵
    返回 scipy CSR 矩阵
    """
    row_idx = []
    col_idx = []
    data = []

    for user, seq in seqs_with_user:
        for item in seq:
            # user -> item
            row_idx.append(user)
            col_idx.append(n_users + item)
            data.append(1.0)
            # item -> user
            row_idx.append(n_users + item)
            col_idx.append(user)
            data.append(1.0)

    adj = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n_users + n_items, n_users + n_items))
    return adj.tocsr()


def normalize_adj_scipy(adj):
    """
    使用 scipy 对稀疏矩阵归一化
    对方阵 A: D^{-1/2} A D^{-1/2}
    """
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1)).flatten()

    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return norm_adj.tocsr()


def scipy_to_torch_sparse(adj):
    """
    将 scipy csr/coo 转换为 torch.sparse_coo_tensor
    """
    adj = adj.tocoo()

    # indices = torch.tensor([adj.row, adj.col], dtype=torch.long)
    rows = np.array(adj.row)
    cols = np.array(adj.col)
    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

    values = torch.tensor(adj.data, dtype=torch.float32)
    shape = adj.shape
    return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()


def build_user_item_graph(train_samples, n_users, n_items):
    """
    从训练数据构造用户-商品邻接矩阵（只考虑 user-item 边）
    train_samples: list of (user_id, seq, target_item)
    """
    user2seqs = defaultdict(list)
    for user, seq, _ in train_samples:
        user2seqs[user].append(seq)

    final_seqs = extract_last_sequences(user2seqs)  # list of (user_id, seq)
    adj = compute_user_item_graph_scipy(final_seqs, n_users, n_items)
    # adj = normalize_adj_scipy(adj)
    adj = scipy_to_torch_sparse(adj)
    return adj
