import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
from tqdm import tqdm
import datetime
import os
from time import time

from sequence_loader import generate_sequence, build_sequences, load_and_preprocess, disturb_sequence
from graph_adj_matrix import build_user_item_graph
from graph_encoder import GraphConvolutionalEncoder
from sequence_encoder import SequenceEncoder
from evaluate import evaluate_model, hit_ndcg
from dataset import SequenceDataset, collate_fn
from model import MultiViewRecommender

# ---------------- 训练函数 ----------------
def train_model(model, train_samples, valid_samples, num_epochs=10, batch_size=1024, lr=1e-3, device='cuda', early_stop_patience=5):
    """
    训练多视图推荐模型
    1. 每个 epoch 训练完成后，计算验证集 HR@10 和 NDCG@10
    2. 若连续 early_stop_patience 个 epoch 验证集效果未提升，则早停
    """
    train_dataset = SequenceDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_ndcg10 = 0.0
    no_improve_count = 0

    # 提取训练集输入
    train_users = [s[0] for s in train_samples]
    train_seqs = [s[1] for s in train_samples]
    train_targets = torch.tensor([s[2] for s in train_samples], device=device)

    # 提取验证集输入
    valid_users = [s[0] for s in valid_samples]
    valid_seqs = [s[1] for s in valid_samples]
    valid_targets = torch.tensor([s[2] for s in valid_samples], device=device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_CL_loss = 0.0  # 总对比损失
        total_pred_loss = 0.0  # 总预测损失
        for seq_batch, user_batch, target_batch in tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} Training"
        ):
            seq_batch = [seq.tolist() for seq in seq_batch]
            seq_batch = list(seq_batch)

            user = list(user_batch)

            # 两个随机增强序列
            seq_aug1, seq_aug2 = [], []
            for seq in seq_batch:
                seq1 = disturb_sequence(seq, max_item_id=model.num_items-1)
                seq2 = disturb_sequence(seq, max_item_id=model.num_items-1)
                seq_aug1.append(seq1)
                seq_aug2.append(seq2)

            optimizer.zero_grad()
            user_emb, item_emb, CL_loss = model(seq_batch, user, seq_aug1, seq_aug2)

            pred_scores = model.predict(user_emb, item_emb)

            target_tensor = target_batch.to(device)
            pred_loss = F.cross_entropy(pred_scores, target_tensor)

            loss = model.CL_loss_weight * CL_loss + pred_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_CL_loss += CL_loss.item()
            total_pred_loss += pred_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_CL_loss = total_CL_loss / len(train_loader)
        avg_pred_loss = total_pred_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Train CL loss={avg_CL_loss:.4f}, prediction loss={avg_pred_loss:.4f}, total Loss={avg_loss:.4f}")

        # ---------- 验证集评估 ----------
        model.eval()
        with torch.no_grad():
            user_emb, item_emb, _ = model(valid_seqs, valid_users, valid_seqs, valid_seqs)
            valid_scores = model.predict(user_emb, item_emb)
            metrics = hit_ndcg(valid_scores, valid_targets, k_list=[10])
            valid_hr10 = metrics['HR@10']
            valid_ndcg10 = metrics['NDCG@10']

        print(f"Epoch {epoch+1}/{num_epochs} - Valid HR@10={valid_hr10:.4f}, NDCG@10={valid_ndcg10:.4f}")

        # Early stopping
        if valid_ndcg10 > best_ndcg10:
            best_ndcg10 = valid_ndcg10
            no_improve_count = 0
            torch.save(model.state_dict(), "./log/best_model.pt")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs")
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load("./log/best_model.pt"))
    return model


# ---------------- Main ----------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # campus_list = ['13', '18', '38', '107', '151']
    campus_list = ['13']
    for campus_id in campus_list:

        print(f"Campus Zone ID: {campus_id}")

        df, item2newid = load_and_preprocess("交易数据.feather", campus_id)
        user_seqs = build_sequences(df)
        max_item_id = max(item2newid.values())
        train_samples, valid_samples, test_samples = generate_sequence(user_seqs, disturb=None, max_item_id=max_item_id)

        num_items = max(
            max(s[2] for s in train_samples),
            max(s[2] for s in valid_samples),
            max(s[2] for s in test_samples)
        ) + 1

        num_users = max(
            max(s[0] for s in train_samples),
            max(s[0] for s in valid_samples),
            max(s[0] for s in test_samples)
        ) + 1

        model = MultiViewRecommender(num_users=num_users, num_items=num_items, embed_dim=64, device=device)

        model = train_model(model, train_samples, valid_samples, num_epochs=200, batch_size=1024, lr=1e-3, device=device, early_stop_patience=200)

        test_metrics = evaluate_model(model, test_samples, k_list=[5,10,20], device=device)
        print("=== Test Set Metrics ===")
        for k,v in test_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # 生成时间戳
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存路径（可自定义）
        save_dir = "./results"
        os.makedirs(save_dir, exist_ok=True)

        # 文件名加入模型名或数据集名也可以，这里简单用 test_results
        save_path = os.path.join(save_dir, f"test_results_campus{campus_id}_{ts}.txt")

        # 写入文件
        with open(save_path, "w") as f:
            f.write("=== Test Set Metrics ===\n")
            for k, v in test_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        print(f"测试结果已保存到: {save_path}")
