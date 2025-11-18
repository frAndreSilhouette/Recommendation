import torch
import torch.nn as nn
from collections import defaultdict

class SequenceEncoder(nn.Module):
    """
    基于序列的 item embedding（完全独立于 graph）
    - LSTM 输入为独立 Xavier 初始化的 item embedding
    - 输出每个 item 的序列感知 embedding (seq_embed)
    - 未出现的 item embedding 保持不变
    """
    def __init__(self, num_items, embed_dim=64, hidden_dim=64, num_layers=1, device='cuda'):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # 独立的 item embedding
        self.item_embeddings = nn.Embedding(num_items, embed_dim).to(device)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        # LSTM 序列编码
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(device)

    def forward(self, user_sequences, prev_seq_embed=None):
        """
        user_sequences: list[list]，每个序列是 item_id 列表
        prev_seq_embed: [num_items, hidden_dim] tensor，上一轮 seq_embed（可选）
        """
        if prev_seq_embed is None:
            # 初始化 seq_embed 为零
            seq_embed = torch.zeros(self.num_items, self.hidden_dim, device=self.device)
        else:
            # 保留上一轮 seq_embed
            seq_embed = prev_seq_embed.clone()

        # 暂存每个 item 的 hidden embedding
        hidden_embed_dict = defaultdict(list)

        for seq in user_sequences:
            seq = torch.tensor(seq, device=self.device)
            hidden_input = self.item_embeddings(seq).unsqueeze(0)  # [1, seq_len, embed_dim]

            hidden_output, _ = self.lstm(hidden_input)  # [1, seq_len, hidden_dim]
            hidden_output = hidden_output.squeeze(0)    # [seq_len, hidden_dim]

            for item_id, hidden_embed in zip(seq, hidden_output):
                hidden_embed_dict[int(item_id)].append(hidden_embed)

        # 聚合 hidden_embed 得到 seq_embed（只更新出现过的 item）
        for item_id, hidden_list in hidden_embed_dict.items():
            seq_embed[item_id] = torch.stack(hidden_list).mean(dim=0)

        return seq_embed
