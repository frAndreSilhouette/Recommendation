import os
import torch
import numpy as np
from collections import defaultdict
from numba import njit

def load_sequences_from_file(path):
    """
    读取训练/验证/测试文件，每行格式：
    user seq target
    seq 形如 4,5,6,7
    """
    user2seqs = defaultdict(list)
    item_set = set()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            user = int(parts[0])
            seq_str = parts[1]
            target = int(parts[2])

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

    return user2seqs, item_set


def extract_last_sequences(user2seqs):
    """
    对每个用户，提取最长的序列（即购买次数最多的那条）
    """
    last_seqs = []

    for user, seq_list in user2seqs.items():
        longest = max(seq_list, key=lambda x: len(x))
        last_seqs.append(longest)

    return last_seqs


# def compute_item_graph(seqs, n_items):
#     """
#     seqs: 所有用户的最终序列 (list of list)
#     n_items: 商品数量
#     构造 item-item 邻接矩阵，权重为 (平均倒数距离)，未共现为0
#     """

#     # 用字典累积距离倒数
#     dist_sum = defaultdict(float)
#     dist_cnt = defaultdict(int)

#     for seq in seqs:
#         # print("[DEBUG]:seq", seq)
#         length = len(seq)
#         for i in range(length):
#             for j in range(i+1, length):
#                 item_i = seq[i]
#                 item_j = seq[j]
#                 dist = j - i

#                 weight = 1.0 / dist

#                 # print("[DEBUG]:item_i, item_j, weight", item_i, item_j, weight)

#                 dist_sum[(item_i, item_j)] += weight
#                 dist_sum[(item_j, item_i)] += weight
#                 dist_cnt[(item_i, item_j)] += 1
#                 dist_cnt[(item_j, item_i)] += 1

#     # 构造稀疏邻接矩阵
#     rows, cols, vals = [], [], []

#     for (i, j), s in dist_sum.items():
#         avg_weight = s / dist_cnt[(i, j)]
#         rows.append(i)
#         cols.append(j)
#         vals.append(avg_weight)

#     # 转换为 PyTorch sparse tensor
#     indices = torch.tensor([rows, cols], dtype=torch.long)
#     values = torch.tensor(vals, dtype=torch.float32)
#     adj = torch.sparse_coo_tensor(indices, values, (n_items, n_items))

#     return adj

# def compute_item_graph(seqs, n_items, max_dist=None):
#     """
#     seqs: list of sequences
#     n_items: 商品总数
#     max_dist: 只考虑 j - i <= max_dist 的共现（None 表示不限）
#     """

#     dist_sum = defaultdict(float)
#     dist_cnt = defaultdict(int)

#     for seq in seqs:
#         L = len(seq)
#         ds = dist_sum
#         dc = dist_cnt

#         for i in range(L):
#             item_i = seq[i]
#             base_i = item_i * n_items

#             # ----------- 新增：限制 j 的最大范围 ----------------
#             if max_dist is None:
#                 end_j = L
#             else:
#                 end_j = min(L, i + 1 + max_dist)
#             # ---------------------------------------------------

#             for j in range(i+1, end_j):
#                 item_j = seq[j]
#                 dist = j - i  # dist <= max_dist 保证成立

#                 w = 1.0 / dist

#                 key1 = base_i + item_j
#                 key2 = item_j * n_items + item_i

#                 ds[key1] += w
#                 dc[key1] += 1
#                 ds[key2] += w
#                 dc[key2] += 1

#     # 预分配
#     size = len(dist_sum)
#     rows = [0] * size
#     cols = [0] * size
#     vals = [0.0] * size

#     idx = 0
#     for key, s in dist_sum.items():
#         i = key // n_items
#         j = key % n_items
#         avg = s / dist_cnt[key]

#         rows[idx] = i
#         cols[idx] = j
#         vals[idx] = avg
#         idx += 1

#     # 转为稀疏矩阵
#     indices = torch.tensor([rows, cols], dtype=torch.long)
#     values = torch.tensor(vals, dtype=torch.float32)
#     adj = torch.sparse_coo_tensor(indices, values, (n_items, n_items))

#     return adj

def compute_item_graph(seqs, n_items, z=3):
    """
    构造 item-item 邻接矩阵（简化逻辑）
    seqs: list of sequences
    n_items: 商品总数
    z: 最大距离阈值，如果两个 item 在序列中距离 <= z，则边权为1
    """
    edges = set()  # 保存所有出现的边

    for seq in seqs:
        L = len(seq)
        for i in range(L):
            item_i = seq[i]
            # 只考虑 item_j 与 item_i 的距离 <= z
            end_j = min(L, i + 1 + z)
            for j in range(i+1, end_j):
                item_j = seq[j]
                # 添加无向边
                edges.add((item_i, item_j))
                edges.add((item_j, item_i))

    if len(edges) == 0:
        # 如果没有边，返回全零稀疏矩阵
        return torch.sparse_coo_tensor(
            indices=torch.empty((2,0), dtype=torch.long),
            values=torch.empty((0,), dtype=torch.float32),
            size=(n_items, n_items)
        )

    rows, cols = zip(*edges)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)  # 所有边权为1

    adj = torch.sparse_coo_tensor(indices, values, (n_items, n_items))
    return adj


def build_item_graph(train_samples, n_items):
    """
    直接从训练数据构造图，而不从文件读取
    train_samples: list of (user_id, seq, target_item)
    n_items: 商品总数
    """
    # 将训练样本转成 user2seqs
    user2seqs = defaultdict(list)
    for user, seq, target in train_samples:
        user2seqs[user].append(seq)

    final_seqs = extract_last_sequences(user2seqs)
    adj = compute_item_graph(final_seqs, n_items)
    return adj


if __name__ == "__main__":
    # Step1: 读取训练集
    train_path = "dataset/train.txt"
    train_samples, _ = load_sequences_from_file(train_path)

    # 因为 load_sequences_from_file 返回的是 dict，需要转换为列表形式
    train_samples_list = []
    for u, seq_list in train_samples.items():
        for seq in seq_list:
            target = seq[-1]  # 假设最后一个 item 是目标
            train_samples_list.append((u, seq, target))

    # Step2: 计算 item 数量
    all_items = set()
    for _, seq, tgt in train_samples_list:
        all_items.update(seq)
        all_items.add(tgt)
    num_items = max(all_items) + 1

    # Step3: 构建图
    adj = build_item_graph(train_samples_list, num_items)

    print("Graph size:", adj.shape)
    print("Number of edges:", adj._nnz())

    # Step4: 保存邻接矩阵
    save_path = "./log/item_graph.pt"
    torch.save(adj, save_path)
    print(f">>> Saved adjacency matrix to {save_path}")

    # 转为稠密矩阵查看
    dense_adj = adj.to_dense()
    print(">>> Adjacency matrix (first 10x10 items):")
    print(dense_adj[:10, :10])
