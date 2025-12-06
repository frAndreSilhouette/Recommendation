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

def bpr_loss_from_full_matrix(pred_scores, target_tensor):
    """
    pred_scores: [N_user, N_item] (same device as model output)
    target_tensor: [N_user]        (same device)
    """

    device = pred_scores.device

    N_user, N_item = pred_scores.shape
    user_idx = torch.arange(N_user, device=device)

    # 正样本得分
    pos_scores = pred_scores[user_idx, target_tensor]

    # 负采样
    neg_items = torch.randint(0, N_item, (N_user,), device=device)
    mask = (neg_items == target_tensor)
    while mask.any():
        neg_items[mask] = torch.randint(0, N_item, (mask.sum(),), device=device)
        mask = (neg_items == target_tensor)

    # 负样本得分
    neg_scores = pred_scores[user_idx, neg_items]

    # BPR Loss
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    return loss

# ---------------- 训练函数 ----------------
def train_model(model, train_samples, valid_samples, num_users, num_items, num_epochs=10, batch_size=1024, lr=1e-3, device='cuda', early_stop_patience=5):
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
    print("Train samples:", len(train_samples))

    model.adj = build_user_item_graph(list(zip(train_users, train_seqs, train_targets)), num_users, num_items, weight=True) # 一次性根据所有训练数据构建图
    print("Graph built.")

    # 提取验证集输入
    valid_users = [s[0] for s in valid_samples]
    valid_seqs = [s[1] for s in valid_samples]
    valid_targets = torch.tensor([s[2] for s in valid_samples], device=device)
    print("Valid samples:", len(valid_samples))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_CL_loss = 0.0  # 总对比损失
        total_pred_loss = 0.0  # 总预测损失
        total_l2_reg_loss = 0.0 # L2 正则化损失
        for seq_batch, user_batch, target_batch in tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} Training"
        ):
            seq_batch = [seq.tolist() for seq in seq_batch]
            seq_batch = list(seq_batch)

            user = list(user_batch)

            # 两个随机增强序列
            # seq_aug1, seq_aug2 = [], []
            # for seq in seq_batch:
            #     seq1 = disturb_sequence(seq, max_item_id=model.num_items-1)
            #     seq2 = disturb_sequence(seq, max_item_id=model.num_items-1)
            #     seq_aug1.append(seq1)
            #     seq_aug2.append(seq2)

            optimizer.zero_grad()
            # user_emb, item_emb, CL_loss = model(seq_batch, user, seq_aug1, seq_aug2)
            user_emb, item_emb, CL_loss = model(seq_batch, user) # 这里的CL_loss已经乘了model.CL_loss_weight

            pred_scores = model.predict(user_emb, item_emb)

            # print("pred_scores grad:", pred_scores.requires_grad)
            # print("user_emb grad:", user_emb.requires_grad)
            # print("item_emb grad:", item_emb.requires_grad)
            # print("pred_scores min/max:", pred_scores.min().item(), pred_scores.max().item())
            # print('graph encoder grad:', model.gcn_encoder.user_embedding.weight.requires_grad, model.gcn_encoder.item_embedding.weight.requires_grad)
            # print('sequence encoder grad:', model.seq_encoder.item_emb.requires_grad)

            target_tensor = target_batch.to(device)
            if model.has_graph_encoder and (not model.has_sequence_encoder) :
                pred_loss = bpr_loss_from_full_matrix(pred_scores, target_tensor)
            else :
                pred_loss = F.cross_entropy(pred_scores, target_tensor)
            l2_reg_loss = model.l2_reg_loss_weight * (
                torch.sum(user_emb**2)/user_emb.shape[0] + torch.sum(item_emb**2)/item_emb.shape[0]
            )

            loss = CL_loss + pred_loss + l2_reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_CL_loss += CL_loss.item()
            total_pred_loss += pred_loss.item()
            total_l2_reg_loss += l2_reg_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_CL_loss = total_CL_loss / len(train_loader)
        avg_pred_loss = total_pred_loss / len(train_loader)
        avg_l2_reg_loss = total_l2_reg_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Train CL loss={avg_CL_loss:.4f}, prediction loss={avg_pred_loss:.4f}, l2 reg loss={avg_l2_reg_loss:.4f}, total Loss={avg_loss:.4f}")

        # ---------- 验证集评估 ----------
        model.eval()
        with torch.no_grad():
            user_emb, item_emb, _ = model(valid_seqs, valid_users)
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
    
def save_samples_to_txt(samples, path):
    """
    samples: list of (user_id, seq_list, target)
    path: 输出文件路径
    """
    with open(path, "w") as f:
        for user, seq, target in samples:
            seq_str = ",".join(map(str, seq))  # 序列转成 item1,item2,item3
            f.write(f"{user} {seq_str} {target}\n")


# ---------------- Main ----------------
def main(has_graph_encoder=True, has_sequence_encoder=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    campus_list = ['13', '18', '38', '107', '151']
    for campus_id in campus_list:

        print(f"Campus Zone ID: {campus_id}")

        df, item2newid = load_and_preprocess("交易数据.feather", campus_id)
        user_seqs = build_sequences(df)
        max_item_id = max(item2newid.values())
        train_samples, valid_samples, test_samples = generate_sequence(user_seqs, disturb=None, max_item_id=max_item_id)
        
        os.makedirs("./dataset", exist_ok=True)
        save_samples_to_txt(train_samples, f"./dataset/campus{campus_id}_train.txt")
        save_samples_to_txt(valid_samples, f"./dataset/campus{campus_id}_valid.txt")
        save_samples_to_txt(test_samples, f"./dataset/campus{campus_id}_test.txt")
        print("Data Files saved: train.txt / valid.txt / test.txt")

        all_item_ids = set()
        for s in train_samples + valid_samples + test_samples:
            all_item_ids.update(s[1])  # seq
            all_item_ids.add(s[2])     # target
        num_items = max(all_item_ids) + 1
        num_users = max(s[0] for s in train_samples + valid_samples + test_samples) + 1

        model = MultiViewRecommender(num_users=num_users, num_items=num_items, embed_dim=64, device=device,
                                    has_graph_encoder=has_graph_encoder, has_sequence_encoder=has_sequence_encoder)

        model = train_model(model, train_samples, valid_samples, num_users=num_users, num_items=num_items,
                            num_epochs=500, batch_size=4096, lr=1e-3, device=device, early_stop_patience=10)

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
        save_path = os.path.join(save_dir, f"graph_{has_graph_encoder}_seq_{has_sequence_encoder}_campus{campus_id}_{ts}.txt")

        # 写入文件
        with open(save_path, "w") as f:
            f.write("=== Test Set Metrics ===\n")
            for k, v in test_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        print(f"测试结果已保存到: {save_path}")


if __name__ == "__main__":
    main(has_graph_encoder=True, has_sequence_encoder=False)
    main(has_graph_encoder=False, has_sequence_encoder=True)
    main(has_graph_encoder=True, has_sequence_encoder=True)