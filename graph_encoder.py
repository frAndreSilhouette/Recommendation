import torch
import torch.nn as nn

class GraphConvolutionalEncoder(nn.Module):
    """
    简化版 GCN encoder (LightGCN-style) 用于 item
    输入：稀疏邻接矩阵 (item_graph.pt)
    输出：经过 L 层传播后的 item embedding
    """
    def __init__(self, num_items, embed_dim=64, num_layers=2, device='cuda'):
        super().__init__()
        self.device = device

        # 自动获取节点数量
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 初始化 item embedding
        self.embedding = nn.Embedding(self.num_items, embed_dim).to(device)
        nn.init.xavier_uniform_(self.embedding.weight)  # Xavier 初始化

    def forward(self, adj_matrix):
        """
        L 层传播
        """

        adj = adj_matrix.to(self.device)

        # 保存每层 embedding，用于最终平均
        all_embeddings = [self.embedding.weight]

        e = self.embedding.weight
        for layer in range(self.num_layers):
            # 传播公式: e^(l+1) = e^(l) + sum_{neighbors} w(i,i') * e_{i'}^(l)
            e = e + torch.sparse.mm(adj, e)
            all_embeddings.append(e)

        # 最终 embedding: 对所有层的 embedding 求平均
        final_embedding = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final_embedding


if __name__ == "__main__":
    device = 'cuda'

    # 1. 读取稀疏邻接矩阵，并放到 GPU
    adj = torch.load('./log/item_graph.pt').to(device)  # 强制放到 cuda

    # 2. 构建 GCN encoder
    gcn_encoder = GraphConvolutionalEncoder(adj_matrix=adj.shape[0], embed_dim=64, num_layers=2, device=device)
    gcn_encoder.to(device)  # embedding 也放到 cuda

    # 3. 前向传播，得到 item embedding
    item_embeddings = gcn_encoder()  # shape: [num_items, embed_dim]

    # 4. 保存到 cpu
    torch.save(item_embeddings.cpu(), "./log/item_graph_embeddings.pt")

    print("Item embeddings shape:", item_embeddings.shape)
    print("Example embedding for item 0:", item_embeddings[0])

