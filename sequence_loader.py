import pandas as pd
import os
import random
import copy
import torch

def load_and_preprocess(file_path="交易数据.feather", campus_id=143):
    print(">>> Loading data...")
    df = pd.read_feather(file_path)
    # df = df.iloc[:1000, :]
    # print(df["campus_id"].unique())
    df = df[df['campus_id']==campus_id]
    df = df[df['order_date']<='2024-06-30']
    print(f">>> Total interactions: {len(df)}")
    df = df[["user_id", "spu_id", "order_date"]]

    # 排序
    df = df.sort_values(["user_id", "order_date"]).reset_index(drop=True)

    # ====== Step 1：过滤购买次数不足 threshold 的用户和商品 ======
    filter_threshold = 10
    print(f">>> Filtering users with < {filter_threshold} interactions...")
    user_counts = df.groupby("user_id")["spu_id"].count()
    valid_users = user_counts[user_counts >= filter_threshold].index
    df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

    print(f">>> Filtering items purchased < {filter_threshold} times...")
    item_counts = df.groupby("spu_id")["user_id"].count()
    valid_items = item_counts[item_counts >= filter_threshold].index
    df = df[df["spu_id"].isin(valid_items)].reset_index(drop=True)

    # ====== Step 2：重新编码 user_id ======
    print(">>> Encoding user_id...")
    user2newid = {u: i for i, u in enumerate(df["user_id"].unique())}
    df["user_id_new"] = df["user_id"].map(user2newid)

    # ====== Step 3：按照用户顺序编码 item ======
    print(">>> Encoding items by user order...")
    df_sorted = df.sort_values(["user_id_new", "order_date"]).reset_index(drop=True)
    user_groups = df_sorted.groupby("user_id_new")["spu_id"].apply(list)

    item2newid = dict()
    next_id = 0

    # 遍历每个用户，训练集只取倒数前 len(seq)-2 项
    for user_id, seq in user_groups.items():
        train_seq = seq[:-2]
        for item in train_seq:
            if item not in item2newid:
                item2newid[item] = next_id
                next_id += 1

    # 编码函数，训练集已有 item 沿用，验证/测试集新 item 顺序追加
    def encode_item(x):
        nonlocal next_id
        if x in item2newid:
            return item2newid[x]
        else:
            item2newid[x] = next_id
            next_id += 1
            return item2newid[x]

    df["spu_id_new"] = df["spu_id"].apply(encode_item)

    print(f">>> Total users after preprocessing: {df['user_id_new'].nunique()}")
    print(f">>> Total items after preprocessing: {df['spu_id_new'].nunique()}")

    return df, item2newid


def build_sequences(df):
    print(">>> Building sequences...")
    user_seqs = (
        df.groupby("user_id_new")["spu_id_new"]
        .apply(list)
        .reset_index(name="sequence")
    )
    return user_seqs


def disturb_sequence(seq, max_item_id, seed=None):
    """
    对序列进行随机扰动
    - 序列长度 <= 5 不扰动
    - 可选操作：
        1. 删除 (cropping)
        2. 交换相邻两个 item (swap)
        3. 随机替换 (replace)
    """
    if seed is not None:
        random.seed(seed)

    is_tensor = isinstance(seq, torch.Tensor)
    if is_tensor:
        seq = seq.tolist()  # 转成 list 方便操作

    n = len(seq)
    if n <= 5:
        return seq  # 太短不扰动

    seq = copy.deepcopy(seq)
    op = random.choice(['delete', 'swap', 'replace'])

    if op == 'delete':
        idx = random.randint(0, n - 1)
        seq.pop(idx)
    elif op == 'swap':
        if n >= 2:
            idx = random.randint(0, n - 2)
            seq[idx], seq[idx + 1] = seq[idx + 1], seq[idx]
    elif op == 'replace':
        idx = random.randint(0, n - 1)
        new_item = random.randint(0, max_item_id)
        seq[idx] = new_item

    return seq

def generate_sequence(user_seqs, disturb=None, max_item_id=None):
    train_sequences, valid_sequences, test_sequences = [], [], []

    for _, row in user_seqs.iterrows():
        u = row["user_id_new"]
        seq = row["sequence"]

        if len(seq) < 5:
            continue

        # ---------- 训练集 ----------
        # for i in range(1, len(seq) - 2):
        #     input_seq = seq[:i]
        #     target = seq[i]
        #     if disturb is not None and max_item_id is not None:
        #         input_seq = disturb_sequence(input_seq, max_item_id, seed=disturb)
        #     train_sequences.append((u, input_seq, target))

        # 新的训练集方案：为每个用户只生成一条
        train_input = seq[:-3]
        train_target = seq[-3]
        train_sequences.append((u, train_input, train_target))

        # ---------- 验证集 ----------
        valid_input = seq[:-2]
        valid_target = seq[-2]
        valid_sequences.append((u, valid_input, valid_target))

        # ---------- 测试集 ----------
        test_input = seq[:-1]
        test_target = seq[-1]
        test_sequences.append((u, test_input, test_target))

    return train_sequences, valid_sequences, test_sequences


def write_file(samples, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for u, seq, tgt in samples:
            seq_str = ",".join(str(i) for i in seq)
            f.write(f"{u} {seq_str} {tgt}\n")
