import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionalEncoder(nn.Module):
    """
    XSimGCL-style LightGCN encoder for users and items
    - L layers propagation on user-item bipartite graph
    - Optional perturbation for contrastive learning
    输入：
        num_users: 用户数量
        num_items: 物品数量
        embed_dim: embedding 维度
        num_layers: GCN 层数
        eps: 对比学习扰动幅度
        layer_cl: 对比学习使用的层
    输出：
        user_embeddings, item_embeddings
        (可选) user_embeddings_cl, item_embeddings_cl
    """
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=2, eps=0.1, layer_cl=1, device='cuda'):
        super().__init__()
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.eps = eps
        self.layer_cl = layer_cl

        # 初始化 user 和 item embedding
        self.user_embedding = nn.Embedding(num_users, embed_dim).to(device)
        self.item_embedding = nn.Embedding(num_items, embed_dim).to(device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix, perturbed=True):
        """
        adj_matrix: 稀疏用户-物品邻接矩阵 [num_users+num_items, num_users+num_items]
        perturbed: 是否添加对比学习扰动
        返回：
            user_embeddings, item_embeddings
            (可选) user_embeddings_cl, item_embeddings_cl
        """
        adj = adj_matrix.to(self.device)

        # concat user+item embedding
        e = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = []
        embeddings_cl = e.clone()

        for layer in range(self.num_layers):
            e = torch.sparse.mm(adj, e)
            # perturbation 用于 CL
            if perturbed and layer == self.layer_cl:
                noise = torch.rand_like(e).to(self.device)
                embeddings_cl = e + torch.sign(e) * F.normalize(noise, dim=-1) * self.eps
            all_embeddings.append(e)

        # 最终 embedding: 平均所有层
        final_embedding = torch.stack(all_embeddings, dim=0).mean(dim=0)

        # 拆分 user 和 item embedding
        user_emb, item_emb = torch.split(final_embedding, [self.num_users, self.num_items], dim=0)

        if perturbed:
            user_emb_cl, item_emb_cl = torch.split(embeddings_cl, [self.num_users, self.num_items], dim=0)
            return user_emb, item_emb, user_emb_cl, item_emb_cl
        return user_emb, item_emb
