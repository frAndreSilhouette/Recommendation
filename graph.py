import os
import torch
import numpy as np
from collections import defaultdict


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


def compute_item_graph(seqs, n_items):
    """
    seqs: 所有用户的最终序列 (list of list)
    n_items: 商品数量
    构造 item-item 邻接矩阵，权重为 (平均倒数距离)，未共现为0
    """

    # 用字典累积距离倒数
    dist_sum = defaultdict(float)
    dist_cnt = defaultdict(int)

    for seq in seqs:
        length = len(seq)
        for i in range(length):
            for j in range(i+1, length):
                item_i = seq[i]
                item_j = seq[j]
                dist = j - i

                weight = 1.0 / dist

                dist_sum[(item_i, item_j)] += weight
                dist_sum[(item_j, item_i)] += weight
                dist_cnt[(item_i, item_j)] += 1
                dist_cnt[(item_j, item_i)] += 1

    # 构造稀疏邻接矩阵
    rows, cols, vals = [], [], []

    for (i, j), s in dist_sum.items():
        avg_weight = s / dist_cnt[(i, j)]
        rows.append(i)
        cols.append(j)
        vals.append(avg_weight)

    # 转换为 PyTorch sparse tensor
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (n_items, n_items))

    return adj


def build_and_save_graph(train_path, save_path="item_graph.pt"):
    print(">>> Loading datasets...")

    u2s_train, items1 = load_sequences_from_file(train_path)

    # 所有商品
    all_items = sorted(list(items1))
    n_items = max(all_items) + 1

    print(f">>> Total item count = {n_items}")

    print(">>> Extracting longest sequences per user...")
    final_seqs = extract_last_sequences(u2s_train)

    print(">>> Constructing adjacency matrix...")
    adj = compute_item_graph(final_seqs, n_items)

    print(f">>> Saving graph to {save_path}")
    torch.save(adj, save_path)

    print(">>> Done.")
    return adj


if __name__ == "__main__":
    train_path = "dataset/train.txt"

    adj = build_and_save_graph(train_path)
    print("Graph size:", adj.shape)
    print("Number of edges:", adj._nnz())

    # 转为稠密矩阵
    dense_adj = adj.to_dense()

    # 显示前 10*10 的子矩阵
    print(">>> Adjacency matrix (first 10x10 items):")
    print(dense_adj[:10, :10])