import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math

from sequence_loader import generate_sequence, build_sequences, load_and_preprocess, disturb_sequence
from graph_adj_matrix import build_item_graph
from graph_encoder import GraphConvolutionalEncoder
from sequence_encoder import SequenceEncoder
from evaluate import evaluate_model, hit_ndcg
from tqdm import tqdm

# ---------------- Dataset ----------------
class SequenceDataset(Dataset):
    """
    PyTorch Dataset，用于存储训练样本
    每条样本为 (seq, user_id, target_item)
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, user, target = self.samples[idx][1], self.samples[idx][0], self.samples[idx][2]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(user, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# ---------------- Multi-View Model ----------------
class MultiViewRecommender(nn.Module):
    """
    多视图推荐模型：
    - GraphConvolutionalEncoder 生成图嵌入
    - SequenceEncoder 生成序列嵌入
    - Attention 融合 Graph+Seq embedding
    - Transformer 对用户历史序列进行编码
    """
    def __init__(self, num_items, embed_dim=64, seq_hidden_dim=64, gcn_layers=2, device='cuda'):
        super().__init__()
        self.device = device
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.cross_weight = 1.0
        self.cl_weight = 1.0

        # Graph encoder 占位，后续 forward 时用不同 adj_matrix 重新构造
        self.gcn_layers = gcn_layers

        # Sequence encoder
        self.seq_encoder = SequenceEncoder(embed_dim=embed_dim,
                                           hidden_dim=seq_hidden_dim,
                                           device=device)

        # Attention fusion参数
        self.att_w = nn.Parameter(torch.randn(embed_dim * 2))

        # 用户 Transformer encoder
        self.user_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=1, batch_first=True),
            num_layers=1
        )

        self.to(device)

    # ---------------- InfoNCE 对比学习 ----------------
    @staticmethod
    def info_nce_loss(e1, e2, temperature=0.1):
        num_items = e1.size(0)
        losses = []
        for i in range(num_items):
            pos_sim = F.cosine_similarity(e1[i:i+1], e2[i:i+1])
            neg_idx = random.randint(0, num_items - 1)
            while neg_idx == i:
                neg_idx = random.randint(0, num_items - 1)
            neg_sim = F.cosine_similarity(e1[i:i+1], e2[neg_idx:neg_idx+1])
            loss = -torch.log(torch.exp(pos_sim/temperature) / (torch.exp(pos_sim/temperature) + torch.exp(neg_sim/temperature)))
            losses.append(loss)
        return torch.stack(losses).mean()

    # ---------------- Forward ----------------
    def forward(self, seq_aug1, seq_aug2):
        """
        输入:
            seq_aug1/seq_aug2: list of list，每个子列表为用户增强序列
        输出:
            item_embeddings: [num_items, embed_dim]
            L_CL: contrastive learning loss
        """
        # ---------------- Graph Embedding ----------------
        # 基于每条增强序列分别构造邻接矩阵
        adj1 = build_item_graph([[user, seq, seq[-1]] for user, seq in enumerate(seq_aug1)], self.num_items)
        adj2 = build_item_graph([[user, seq, seq[-1]] for user, seq in enumerate(seq_aug2)], self.num_items)

        gcn_encoder1 = GraphConvolutionalEncoder(adj_matrix=adj1, embed_dim=self.embed_dim,
                                                 num_layers=self.gcn_layers, device=self.device)
        gcn_encoder2 = GraphConvolutionalEncoder(adj_matrix=adj2, embed_dim=self.embed_dim,
                                                 num_layers=self.gcn_layers, device=self.device)

        g_emb_1 = gcn_encoder1()
        g_emb_2 = gcn_encoder2()

        # ---------------- Sequence Embedding ----------------
        s_emb_1 = self.seq_encoder(seq_aug1, g_emb_1, self.num_items)
        s_emb_2 = self.seq_encoder(seq_aug2, g_emb_2, self.num_items)

        # ---------------- Contrastive Loss ----------------
        L_graph = self.info_nce_loss(g_emb_1, g_emb_2)
        L_seq   = self.info_nce_loss(s_emb_1, s_emb_2)

        g_avg = 0.5 * (g_emb_1 + g_emb_2)
        s_avg = 0.5 * (s_emb_1 + s_emb_2)
        L_cross = self.info_nce_loss(g_avg, s_avg)

        L_CL = L_graph + L_seq + self.cross_weight * L_cross

        # ---------------- Attention Fusion ----------------
        combined = torch.cat([g_avg, s_avg], dim=1)
        alpha = torch.sigmoid(torch.matmul(combined, self.att_w)).unsqueeze(1)
        beta = 1.0 - alpha
        item_embeddings = alpha * g_avg + beta * s_avg

        return item_embeddings, L_CL

    # ---------------- 用户编码 + 预测 ----------------
    def predict_next(self, user_seqs, item_embeddings):
        all_user_embeds = []
        for seq in user_seqs:
            seq_embeds = item_embeddings[torch.tensor(seq, device=self.device)].unsqueeze(0)
            h = self.user_transformer(seq_embeds)
            u_emb = h.mean(dim=1)
            all_user_embeds.append(u_emb)
        all_user_embeds = torch.cat(all_user_embeds, dim=0)
        scores = torch.matmul(all_user_embeds, item_embeddings.t())
        return F.softmax(scores, dim=1)

def collate_fn(batch):
    """
    batch: list of samples, 每个 sample=(seq, user, target)
    返回：
        seqs: list of list
        users: [batch_size] tensor
        targets: [batch_size] tensor
    """
    seqs, users, targets = zip(*batch)
    return list(seqs), torch.tensor(users, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

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
    train_seqs = [s[1] for s in train_samples]
    train_targets = torch.tensor([s[2] for s in train_samples], device=device)

    # 提取验证集输入
    valid_seqs = [s[1] for s in valid_samples]
    valid_targets = torch.tensor([s[2] for s in valid_samples], device=device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_LCL = 0.0  # 总对比损失
        total_Lpred = 0.0  # 总预测损失
        for seq_batch, user_batch, target_batch in tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} Training"
        ):

            # seq_batch = [seq.tolist() for seq in seq_batch]
            seq_batch = list(seq_batch)

            # 两个随机增强序列
            seq_aug1, seq_aug2 = [], []
            for seq in seq_batch:
                seq1 = disturb_sequence(seq, max_item_id=model.num_items-1)
                seq2 = disturb_sequence(seq, max_item_id=model.num_items-1)
                seq_aug1.append(seq1)
                seq_aug2.append(seq2)

            optimizer.zero_grad()
            item_embeds, L_CL = model(seq_aug1, seq_aug2)
            pred_scores = model.predict_next(seq_batch, item_embeds)
            target_tensor = torch.tensor([s[2] for s in train_samples[:len(seq_batch)]], device=device)
            L_pred = F.cross_entropy(pred_scores, target_tensor)
            L_total = model.cl_weight * L_CL + L_pred
            L_total.backward()
            optimizer.step()

            total_loss += L_total.item()
            total_LCL += L_CL.item()
            total_Lpred += L_pred.item()

        avg_loss = total_loss / len(train_loader)
        avg_LCL = total_LCL / len(train_loader)
        avg_Lpred = total_Lpred / len(train_loader)

        # ---------- 训练集指标 ----------
        with torch.no_grad():
            item_embeds, _ = model([s for s in train_seqs], [s for s in train_seqs])
            train_scores = model.predict_next([s[1] for s in train_samples], item_embeds)
            train_metrics = hit_ndcg(train_scores, train_targets, k_list=[10])
            train_hr10 = train_metrics['HR@10']
            train_ndcg10 = train_metrics['NDCG@10']

        print(f"Epoch {epoch+1}/{num_epochs} - Train L_CL={avg_LCL:.4f}, L_pred={avg_Lpred:.4f}, Total Loss={avg_loss:.4f}, HR@10={train_hr10:.4f}, NDCG@10={train_ndcg10:.4f}")

        # ---------- 验证集评估 ----------
        model.eval()
        with torch.no_grad():
            item_embeds, _ = model([s for s in valid_seqs], [s for s in valid_seqs])
            valid_scores = model.predict_next(valid_seqs, item_embeds)
            metrics = hit_ndcg(valid_scores, valid_targets, k_list=[10])
            valid_hr10 = metrics['HR@10']
            valid_ndcg10 = metrics['NDCG@10']

        print(f"Epoch {epoch+1}/{num_epochs} - Valid HR@10={valid_hr10:.4f}, NDCG@10={valid_ndcg10:.4f}")

        # Early stopping
        if valid_ndcg10 > best_ndcg10:
            best_ndcg10 = valid_ndcg10
            no_improve_count = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pt"))
    return model


# ---------------- Main ----------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    df, item2newid = load_and_preprocess("交易数据.feather")
    user_seqs = build_sequences(df)
    max_item_id = max(item2newid.values())
    train_samples, valid_samples, test_samples = generate_sequence(user_seqs, disturb=None, max_item_id=max_item_id)

    num_items = max([s[2] for s in train_samples]) + 1
    model = MultiViewRecommender(num_items=num_items, embed_dim=64, seq_hidden_dim=64, device=device)

    model = train_model(model, train_samples, valid_samples, num_epochs=50, batch_size=1024, lr=1e-3, device=device, early_stop_patience=5)

    test_metrics = evaluate_model(model, test_samples, k_list=[5,10,20], device=device)
    print("=== Test Set Metrics ===")
    for k,v in test_metrics.items():
        print(f"{k}: {v:.4f}")
