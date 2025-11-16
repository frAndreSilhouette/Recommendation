import pandas as pd
import os

def load_and_preprocess(file_path="交易数据.feather"):
    print(">>> Loading data...")
    df = pd.read_feather(file_path)
    df = df.iloc[:10000, :]
    df = df[["user_id", "spu_id", "order_date"]]

    # 排序
    df = df.sort_values(["user_id", "order_date"]).reset_index(drop=True)

    # ====== Step 1：过滤购买次数不足 4 的用户 ======
    print(">>> Filtering users with < 4 interactions...")
    user_counts = df.groupby("user_id")["spu_id"].count()
    valid_users = user_counts[user_counts >= 4].index
    df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

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

    return df, item2newid


def build_sequences(df):
    print(">>> Building sequences...")
    user_seqs = (
        df.groupby("user_id_new")["spu_id_new"]
        .apply(list)
        .reset_index(name="sequence")
    )
    return user_seqs


def generate_samples(user_seqs):
    train_samples, valid_samples, test_samples = [], [], []

    for _, row in user_seqs.iterrows():
        u = row["user_id_new"]
        seq = row["sequence"]

        if len(seq) < 4:
            continue

        # ---------- 训练集 ----------
        for i in range(1, len(seq) - 2):
            input_seq = seq[:i]
            target = seq[i]
            train_samples.append((u, input_seq, target))

        # ---------- 验证集 ----------
        valid_input = seq[:-2]
        valid_target = seq[-2]
        valid_samples.append((u, valid_input, valid_target))

        # ---------- 测试集 ----------
        test_input = seq[:-1]
        test_target = seq[-1]
        test_samples.append((u, test_input, test_target))

    return train_samples, valid_samples, test_samples


def write_file(samples, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for u, seq, tgt in samples:
            seq_str = ",".join(str(i) for i in seq)
            f.write(f"{u} {seq_str} {tgt}\n")


def main():
    os.makedirs("./dataset", exist_ok=True)

    # Step1: load + encode
    df, item2newid = load_and_preprocess("交易数据.feather")
    print(f">>> Total items after encoding: {len(item2newid)}")

    # Step2: build sequences
    user_seqs = build_sequences(df)

    # Step3: generate train/valid/test
    train_samples, valid_samples, test_samples = generate_samples(user_seqs)

    # Step4: write files
    write_file(train_samples, "./dataset/train.txt")
    write_file(valid_samples, "./dataset/valid.txt")
    write_file(test_samples, "./dataset/test.txt")

    # Step5: 展示部分结果
    print("\n=== Train sample example ===")
    print(train_samples[:3])
    print("\n=== Valid sample example ===")
    print(valid_samples[:3])
    print("\n=== Test sample example ===")
    print(test_samples[:3])


if __name__ == "__main__":
    main()
