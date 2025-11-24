import torch
import torch.nn as nn

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, activation='relu'):
        super(PointWiseFeedForward, self).__init__()
        act = torch.nn.ReLU() if activation=='relu' else torch.nn.GELU()
        self.pwff = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            act,
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_rate)
        )
    def forward(self, x):
        out = self.pwff(x)
        out += x
        return out


class SequenceEncoder(nn.Module):
    """
    SASRec 风格 sequence encoder
    - 返回所有 item embedding (num_items, D)
    - 可选对比学习扰动
    """
    def __init__(self, num_items, embed_dim=64, max_len=50,
                 block_num=2, head_num=2, drop_rate=0.1, eps=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.block_num = block_num
        self.head_num = head_num
        self.drop_rate = drop_rate
        self.eps = eps

        initializer = nn.init.xavier_uniform_
        self.item_emb = nn.Parameter(initializer(torch.empty(self.num_items+1, self.embed_dim)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len+1, self.embed_dim)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.embed_dim, eps=1e-8))
            new_attn_layer =  torch.nn.MultiheadAttention(self.embed_dim, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.embed_dim, eps=1e-8))
            new_fwd_layer = PointWiseFeedForward(self.embed_dim, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, perturbed=False):
        """
        seqs: List[List[int]]，batch 序列（每条序列长度不同）
        返回:
            item_embeddings (num_items, D)
            (可选) item_embeddings_perturbed (num_items, D)
        """
        # 1. 处理输入 seqs，确保每条序列的长度都为 self.max_len
        batch_size = len(seqs)
        max_len = self.max_len  # 获取统一的序列长度
        device = self.device

        # 填充或裁剪每条序列
        padded_seqs = []
        seq_lens = []  # 用于保存每条序列的有效长度
        for seq in seqs:
            seq_len = len(seq)
            seq_lens.append(seq_len)
            
            if seq_len > max_len:  # 如果序列太长，裁剪前面部分
                padded_seq = seq[-max_len:]  # 保留后 max_len 个元素
            else:  # 如果序列太短，填充到 max_len 长度
                padded_seq = seq + [self.num_items] * (max_len - seq_len)  # 用 self.num_items 填充

            padded_seqs.append(padded_seq)

        seqs = torch.tensor(padded_seqs, dtype=torch.long, device=device)  # (batch_size, max_len)

        # 2. 获取 item embedding + position embedding
        seq_emb = self.item_emb[seqs]  # (B, L, D)
        seq_emb *= self.embed_dim ** 0.5
        pos_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb[pos_ids]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)

        # 3. 构建 padding mask
        pad_mask = (seqs == self.num_items)  # 填充部分为 self.num_items
        seq_emb = seq_emb * (~pad_mask).unsqueeze(-1)  # 将 padding 置零

        # 4. 构建注意力 mask (防止未来信息泄漏)
        attn_mask = ~torch.tril(torch.ones((max_len, max_len), dtype=torch.bool, device=device))  # (L, L)

        # 5. 经过多层 Self-Attention + FeedForward
        for i in range(self.block_num):
            # MultiheadAttention 需要 (L, B, D)
            seq_emb_T = seq_emb.transpose(0, 1)  # (L, B, D)
            norm_emb = self.attention_layer_norms[i](seq_emb_T)
            attn_out, _ = self.attention_layers[i](norm_emb, seq_emb_T, seq_emb_T, attn_mask=attn_mask)
            seq_emb_T = norm_emb + attn_out
            seq_emb_T = seq_emb_T.transpose(0, 1)  # (B, L, D)
            seq_emb_T = self.forward_layer_norms[i](seq_emb_T)
            seq_emb_T = self.forward_layers[i](seq_emb_T)
            seq_emb_T = seq_emb_T * (~pad_mask).unsqueeze(-1)
            seq_emb = seq_emb_T

        seq_emb = self.last_layer_norm(seq_emb)

        # 6. 可选扰动 (用于对比学习)
        if perturbed:
            noise = torch.randn_like(seq_emb) * self.eps
            seq_emb_perturbed = seq_emb + noise
            return seq_emb, seq_emb_perturbed

        return seq_emb # 维度是[batch_size, seq_len, embed_dim]
        