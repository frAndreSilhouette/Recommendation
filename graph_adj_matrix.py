import torch
from collections import defaultdict

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


def compute_user_item_graph(seqs_with_user, n_users, n_items):
    """
    构造用户-商品 bipartite 邻接矩阵（稀疏）
    seqs_with_user: list of (user_id, seq)
    n_users: 用户总数
    n_items: 商品总数
    输出：
        sparse adjacency matrix [n_users+n_items, n_users+n_items]
    """
    edges = set()

    for user, seq in seqs_with_user:
        # 用户-序列商品连边（双向）
        for item in seq:
            edges.add((user, n_users + item))      # user -> item
            edges.add((n_users + item, user))      # item -> user

    rows, cols = zip(*edges)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, size=(n_users + n_items, n_users + n_items))
    return adj

def build_user_item_graph(train_samples, n_users, n_items):
    """
    从训练数据构造用户-商品邻接矩阵（只考虑 user-item 边）
    train_samples: list of (user_id, seq, target_item)
    """
    user2seqs = defaultdict(list)
    for user, seq, _ in train_samples:
        user2seqs[user].append(seq)

    final_seqs = extract_last_sequences(user2seqs)  # list of (user_id, seq)
    adj = compute_user_item_graph(final_seqs, n_users, n_items)
    return adj