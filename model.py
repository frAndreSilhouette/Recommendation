import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from time import time
from tqdm import tqdm

from graph_adj_matrix import build_item_graph
from graph_encoder import GraphConvolutionalEncoder
from sequence_encoder import SequenceEncoder


# ---------------- Multi-View Model ----------------
class MultiViewRecommender(nn.Module):
    """
    多视图推荐模型：
    - GraphConvolutionalEncoder 生成图嵌入
    - SequenceEncoder 生成序列嵌入
    - Attention 融合 Graph+Seq embedding
    - Transformer 对用户历史序列进行编码
    """
    def __init__(self, num_items, embed_dim=64, seq_hidden_dim=64, gcn_layers=2, seq_layer=1, device='cuda'):
        super().__init__()
        self.device = device
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.cross_weight = 1.0
        self.cl_weight = 1.0

        # Graph encoder
        self.gcn_layers = gcn_layers
        self.gcn_encoder = GraphConvolutionalEncoder(num_items=self.num_items,
                                                      embed_dim=self.embed_dim,
                                                      num_layers=self.gcn_layers,
                                                      device=self.device)

        # Sequence encoder
        self.seq_layer = seq_layer
        self.seq_hidden_dim = seq_hidden_dim
        self.seq_encoder = SequenceEncoder(num_items = self.num_items,
                                           embed_dim=self.embed_dim,
                                           hidden_dim=self.seq_hidden_dim,
                                           num_layers=self.seq_layer,
                                           device=self.device)

        # 将 sequence 输出的 hidden_dim 投影到 embed_dim 以便与 GCN 融合
        # 如果seq_hidden_dim == embed_dim，则不需要self.seq_project
        self.seq_project = nn.Linear(seq_hidden_dim, embed_dim).to(device)

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
    def get_active_items(seq_aug1, seq_aug2):
        active_items = set()
        for seq in seq_aug1 + seq_aug2:
            active_items.update(seq)
        return list(active_items)
    @staticmethod
    def info_nce_loss(e1, e2, active_idx, temperature=1):
        num_items = e1.size(0)
        num_active_items = len(active_idx)

        losses = []
        for i in active_idx:
        # for i in tqdm(active_idx, desc=f"Calculating CL loss for {num_active_items} items among {num_items}", ncols=80):
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
        # time1 = time()
        adj1 = build_item_graph([[user, seq, seq[-1]] for user, seq in enumerate(seq_aug1)], self.num_items)
        adj2 = build_item_graph([[user, seq, seq[-1]] for user, seq in enumerate(seq_aug2)], self.num_items)
        # time1_1 = time()
        # print(f"----构建图邻接矩阵耗时: {time1_1-time1:.2f}s")

        g_emb_1 = self.gcn_encoder(adj1)
        g_emb_2 = self.gcn_encoder(adj2)
        # time2 = time()
        # print(f"----构建图嵌入耗时: {time2-time1_1:.2f}s")

        # ---------------- Sequence Embedding ----------------
        s_emb_1 = self.seq_encoder(seq_aug1)
        s_emb_2 = self.seq_encoder(seq_aug2)
        # time3 = time()
        # print(f"----构建序列嵌入耗时: {time3-time2:.2f}s")

        # ---------------- Contrastive Loss ----------------
        active_items = self.get_active_items(seq_aug1, seq_aug2)

        L_graph = self.info_nce_loss(g_emb_1, g_emb_2, active_items)
        L_seq   = self.info_nce_loss(s_emb_1, s_emb_2, active_items)

        g_avg = 0.5 * (g_emb_1 + g_emb_2)
        s_avg = 0.5 * (s_emb_1 + s_emb_2)
        L_cross = self.info_nce_loss(g_avg, s_avg, active_items)

        L_CL = L_graph + L_seq + self.cross_weight * L_cross
        # time4 = time()
        # print(f"----构建对比损失耗时: {time4-time3:.2f}s")

        # ---------------- Attention Fusion ----------------
        combined = torch.cat([g_avg, s_avg], dim=1)
        alpha = torch.sigmoid(torch.matmul(combined, self.att_w)).unsqueeze(1)
        beta = 1.0 - alpha
        item_embeddings = alpha * g_avg + beta * s_avg
        # time5 = time()
        # print(f"----构建融合嵌入耗时: {time5-time4:.2f}s")

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
        # return F.softmax(scores, dim=1)
        return scores # 不进行softmax归一化