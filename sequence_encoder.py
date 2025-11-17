import torch
import torch.nn as nn
from collections import defaultdict

class SequenceEncoder(nn.Module):
    """
    基于序列的 item embedding
    输入：用户序列、graph-based item embeddings
    输出：每个 item 的序列感知 embedding (e_i^seq)
    """
    def __init__(self, embed_dim=64, hidden_dim=64, num_layers=1, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        # LSTM 用于序列编码
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(device)

    def forward(self, user_sequences, graph_embeddings, num_items):
        """
        user_sequences: list of sequences, 每个元素是 item_id 列表
        graph_embeddings: [num_items, embed_dim] tensor
        num_items: 总 item 数量
        """
        # 用于存储每个 item 的 hidden state
        item_hidden_states = defaultdict(list)

        for seq in user_sequences:
            # 获取序列中每个 item 的 graph embedding
            seq_embeds = graph_embeddings[torch.tensor(seq, device=self.device)]  # [seq_len, embed_dim]
            seq_embeds = seq_embeds.unsqueeze(0)  # batch=1

            # LSTM 前向传播
            outputs, _ = self.lstm(seq_embeds)  # outputs: [1, seq_len, hidden_dim]
            outputs = outputs.squeeze(0)  # [seq_len, hidden_dim]

            # 遍历序列，将 hidden state 收集到对应 item
            for item_id, h_t in zip(seq, outputs):
                item_hidden_states[item_id].append(h_t)

        # 聚合每个 item 的 hidden state（这里用 mean pooling）
        seq_item_embeddings = torch.zeros(num_items, self.hidden_dim, device=self.device)
        for item_id, h_list in item_hidden_states.items():
            h_stack = torch.stack(h_list, dim=0)
            seq_item_embeddings[item_id] = h_stack.mean(dim=0)

        return seq_item_embeddings  # [num_items, hidden_dim]


if __name__ == "__main__":
    # 假设已有 graph embedding
    item_embeddings = torch.load("item_graph_embeddings.pt")  # [num_items, embed_dim]
    num_items = item_embeddings.shape[0]
    device = 'cuda'

    # 假设从 train.txt 读取用户序列
    user_sequences = []
    with open("./dataset/train.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            seq = parts[1].split(",")  # 序列部分
            seq = [int(i) for i in seq]
            user_sequences.append(seq)

    # 构建 SequenceEncoder
    seq_encoder = SequenceEncoder(embed_dim=item_embeddings.shape[1], hidden_dim=64, device=device)
    seq_encoder.to(device)

    # 前向传播得到 sequence-based item embedding
    seq_item_embeddings = seq_encoder(user_sequences, item_embeddings.to(device), num_items)

    print("Sequence-based item embeddings shape:", seq_item_embeddings.shape)
    print("示例 item 0 的 embedding:", seq_item_embeddings[0])