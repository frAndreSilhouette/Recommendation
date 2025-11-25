import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from tqdm import tqdm

from graph_adj_matrix import build_user_item_graph
from graph_encoder import GraphConvolutionalEncoder
from sequence_encoder import SequenceEncoder

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.attn_weights = nn.Parameter(torch.zeros(2, embed_dim))  # 两个来源的权重

    def forward(self, emb1, emb2):
        """
        emb1, emb2: [batch_size, embed_dim]，两个来源的 embedding
        返回加权后的融合 embedding
        """
        # 计算两个 embedding 的加权和
        weight1 = torch.sigmoid(self.attn_weights[0])  # [embed_dim]，sigmoid 让它在 0 到 1 之间
        weight2 = torch.sigmoid(self.attn_weights[1])
        return weight1 * emb1 + weight2 * emb2  # 通过加权和融合

# ---------------- Multi-View Model ----------------
class MultiViewRecommender(nn.Module):
    """
    多视图推荐模型：
    - GraphConvolutionalEncoder 生成图嵌入
    - SequenceEncoder 生成序列嵌入
    - Attention 融合 Graph+Seq embedding
    - Transformer 对用户历史序列进行编码
    """
    def __init__(self, num_users, num_items, embed_dim=64, device='cuda', has_graph_encoder=True, has_sequence_encoder=True):
        super().__init__()
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.CL_loss_graph_weight = 1.0
        self.CL_loss_sequence_weight = 1.0
        self.CL_loss_weight = 0.05
        self.l2_reg_loss_weight = 0.0001

        self.has_graph_encoder = has_graph_encoder
        self.has_sequence_encoder = has_sequence_encoder

        if not self.has_graph_encoder and not self.has_sequence_encoder:
            raise ValueError("At least one of graph_encoder or sequence_encoder must be True")

        # Graph encoder
        if self.has_graph_encoder:
            print(">>> Initializing Graph encoder")
            self.gcn_encoder = GraphConvolutionalEncoder(num_users=self.num_users,
                                                      num_items=self.num_items,
                                                      embed_dim=self.embed_dim,
                                                      device=self.device)

        # Sequence encoder
        self.seq_max_length = 50
        if self.has_sequence_encoder:
            print(">>> Initializing Sequence encoder")
            self.seq_encoder = SequenceEncoder(num_items = self.num_items,
                                            embed_dim=self.embed_dim,
                                            max_len=self.seq_max_length,
                                            device=self.device)

        # Attention fusion参数
        self.user_attention = AttentionFusion(self.embed_dim)
        self.item_attention = AttentionFusion(self.embed_dim)

        self.to(device)

    # ---------------- InfoNCE 对比学习 ----------------
    @staticmethod
    def info_nce_loss(e1, e2, temperature=1):
        num_elements = e1.size(0) # 不局限于item，也可以是user，也可以是sequence

        losses = []
        for i in range(num_elements):
            pos_sim = F.cosine_similarity(e1[i:i+1], e2[i:i+1])
            neg_idx = random.randint(0, num_elements - 1)
            while neg_idx == i:
                neg_idx = random.randint(0, num_elements - 1)
            neg_sim = F.cosine_similarity(e1[i:i+1], e2[neg_idx:neg_idx+1])
            loss = -torch.log(torch.exp(pos_sim/temperature) / (torch.exp(pos_sim/temperature) + torch.exp(neg_sim/temperature)))
            losses.append(loss)
        return torch.stack(losses).mean()

    # ---------------- Forward ----------------
    def forward(self, seq, user, seq_aug1=None, seq_aug2=None):
        """
        输入:
            seq: 原始的用户购买序列
            user: 对应的用户id列表
            seq_aug1/seq_aug2: list of list，每个子列表为用户增强序列
        输出:
            item_embeddings: [num_items, embed_dim]
            L_CL: contrastive learning loss
        """

        # 【方案1】embedding扰动
        if seq_aug1 is None or seq_aug2 is None:
            seq_lengths = [len(s) for s in seq]
            # ---------------- Graph Embedding ----------------
            if self.has_graph_encoder:
                adj = build_user_item_graph(list(zip(user, seq, [None] * len(user))), self.num_users, self.num_items) # 后面那个None没有实际意义，只是为了占位
                graph_user_emb, graph_item_emb, graph_user_emb_cl, graph_item_emb_cl = self.gcn_encoder(adj, perturbed=True)

            # ---------------- Sequence Embedding ----------------
            if self.has_sequence_encoder:
                sequence_emb, sequence_emb_cl = self.seq_encoder(seq, perturbed=True)
                # 注意，这里返回的是序列中的单品的embedding，而不是单品的embedding
                # 维度是[batch_size, seq_len, embed_dim]
                # 所以接下来以填充前的最后一个item的embedding作为每条序列的embedding（原论文做法）
                last_indices = (torch.tensor(seq_lengths) - 1).to(self.device)
                last_indices = torch.clamp(last_indices, min=0, max=self.seq_max_length - 1)
                sequence_last_emb = sequence_emb[torch.arange(len(seq)).to(self.device), last_indices]
                sequence_last_emb_cl = sequence_emb_cl[torch.arange(len(seq)).to(self.device), last_indices]

             # ---------------- Contrastive Loss ----------------
            if self.has_graph_encoder:
                CL_loss_graph_user = self.info_nce_loss(graph_user_emb, graph_user_emb_cl)
                CL_loss_graph_item = self.info_nce_loss(graph_item_emb, graph_item_emb_cl)
                CL_loss_graph = CL_loss_graph_user + CL_loss_graph_item
            else:
                CL_loss_graph = 0

            if self.has_sequence_encoder:
                CL_loss_sequence = self.info_nce_loss(sequence_last_emb, sequence_last_emb_cl)
            else:
                CL_loss_sequence = 0

            CL_loss = self.CL_loss_graph_weight * CL_loss_graph + self.CL_loss_sequence_weight * CL_loss_sequence
            CL_loss = self.CL_loss_weight * CL_loss
            
            # 未考虑跨图-序列的对比损失

            # ---------------- Attention Fusion ----------------

            if self.has_graph_encoder and self.has_sequence_encoder:
                graph_user_avg_emb = graph_user_emb[user]
                graph_item_avg_emb = graph_item_emb
                sequence_user_avg_emb = sequence_last_emb
                sequence_item_avg_emb = self.seq_encoder.item_emb[:self.num_items]               
                # 注意，此时得到的user embedding只有batch_size个用户，而非全部的num_users个用户
                # 而且是按照batch里面的出现顺序排列的，不是按照id顺序排列的
                # 但无所谓，反正我也不预测该batch没有出现的那些用户的行为
                user_emb = self.user_attention(graph_user_avg_emb, sequence_user_avg_emb)
                item_emb = self.item_attention(graph_item_avg_emb, sequence_item_avg_emb)

            elif self.has_graph_encoder:
                user_emb = graph_user_emb[user]
                item_emb = graph_item_emb

            elif self.has_sequence_encoder:
                user_emb = sequence_last_emb
                item_emb = self.seq_encoder.item_emb[:self.num_items] 

            return user_emb, item_emb, CL_loss


        # 【方案2】序列扰动
        else:
            seq_aug1_lengths = [len(s) for s in seq_aug1]
            seq_aug2_lengths = [len(s) for s in seq_aug2]

            # ---------------- Graph Embedding ----------------
            # 基于每条增强序列分别构造邻接矩阵
            if self.has_graph_encoder:
                adj1 = build_user_item_graph(list(zip(user, seq_aug1, [None] * len(user))), self.num_users, self.num_items) # 后面那个user没有实际意义，只是为了占位
                adj2 = build_user_item_graph(list(zip(user, seq_aug2, [None] * len(user))), self.num_users, self.num_items) # 后面那个user没有实际意义，只是为了占位
                
                graph_user_emb1, graph_item_emb1 = self.gcn_encoder(adj1, perturbed=False)
                graph_user_emb2, graph_item_emb2 = self.gcn_encoder(adj2, perturbed=False)

            # ---------------- Sequence Embedding ----------------
            if self.has_sequence_encoder:
                sequence_emb1 = self.seq_encoder(seq_aug1, perturbed=False)
                sequence_emb2 = self.seq_encoder(seq_aug2, perturbed=False)
                # 注意，这里返回的是序列中的单品的embedding，而不是单品的embedding
                # 维度是[batch_size, seq_len, embed_dim]
                # 所以接下来以填充前的最后一个item的embedding作为每条序列的embedding（原论文做法）
                last_indices1 = (torch.tensor(seq_aug1_lengths) - 1).to(self.device)
                last_indices2 = (torch.tensor(seq_aug2_lengths) - 1).to(self.device)
                last_indices1 = torch.clamp(last_indices1, min=0, max=self.seq_max_length - 1)
                last_indices2 = torch.clamp(last_indices2, min=0, max=self.seq_max_length - 1)
                sequence_last_emb1 = sequence_emb1[torch.arange(len(seq_aug1)).to(self.device), last_indices1]
                sequence_last_emb2 = sequence_emb2[torch.arange(len(seq_aug2)).to(self.device), last_indices2]
    
            # 到现在，已有
            # graph：user和item的embedding
            # sequence：序列的embedding（也可以有item的embedding，但是还没提取出来，也产生不了两个视图）
            
            # ---------------- Contrastive Loss ----------------
            if self.has_graph_encoder:
                CL_loss_graph_user = self.info_nce_loss(graph_user_emb1, graph_user_emb2)
                CL_loss_graph_item = self.info_nce_loss(graph_item_emb1, graph_item_emb2)
                CL_loss_graph = CL_loss_graph_user + CL_loss_graph_item
            else:
                CL_loss_graph = 0

            if self.has_sequence_encoder:
                CL_loss_sequence = self.info_nce_loss(sequence_last_emb1, sequence_last_emb2)
            else:
                CL_loss_sequence = 0

            CL_loss = self.CL_loss_graph_weight * CL_loss_graph + self.CL_loss_sequence_weight * CL_loss_sequence
            CL_loss = self.CL_loss_weight * CL_loss

            # 未考虑跨图-序列的对比损失

            # ---------------- Attention Fusion ----------------

            if self.has_graph_encoder and self.has_sequence_encoder:
                graph_user_avg_emb = 0.5 * (graph_user_emb1 + graph_user_emb2)[user]
                graph_item_avg_emb = 0.5 * (graph_item_emb1 + graph_item_emb2)
                sequence_user_avg_emb = 0.5 * (sequence_last_emb1 + sequence_last_emb2)
                sequence_item_avg_emb = self.seq_encoder.item_emb[:self.num_items]               
                # 注意，此时得到的user embedding只有batch_size个用户，而非全部的num_users个用户
                # 而且是按照batch里面的出现顺序排列的，不是按照id顺序排列的
                # 但无所谓，反正我也不预测该batch没有出现的那些用户的行为
                user_emb = self.user_attention(graph_user_avg_emb, sequence_user_avg_emb)
                item_emb = self.item_attention(graph_item_avg_emb, sequence_item_avg_emb)

            elif self.has_graph_encoder:
                user_emb = 0.5 * (graph_user_emb1 + graph_user_emb2)[user]
                item_emb = 0.5 * (graph_item_emb1 + graph_item_emb2)

            elif self.has_sequence_encoder:
                user_emb = 0.5 * (sequence_last_emb1 + sequence_last_emb2)
                item_emb = self.seq_encoder.item_emb[:self.num_items] 

            return user_emb, item_emb, CL_loss

    # ---------------- 预测 ----------------
    def predict(self, user_emb, item_emb):
        """
        预测用户对物品的偏好分数（或概率）
        输入:
            user_emb: [batch_size, embed_dim] 用户的最终嵌入（不是全部的num_users个用户）
            item_emb: [num_items, embed_dim] 物品的最终嵌入
        输出:
            scores: [batch_size, num_items] 用户与物品的相似度分数
        """
        # 计算用户与物品的相似度，通过矩阵乘法（点积）
        scores = torch.matmul(user_emb, item_emb.t())  # [batch_size, num_items]
        
        # 使用 softmax 对每个用户的所有物品进行归一化（按行归一化）
        # scores = F.softmax(scores, dim=1)  # [batch_size, num_items]
        return scores # F.cross_entropy自带softmax
