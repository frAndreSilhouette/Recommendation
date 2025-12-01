import torch
import math

# ---------------- 评估指标 ----------------
def hit_ndcg(scores, targets, k_list=[5,10,20]):
    """
    计算 HR@k 和 NDCG@k
    inputs:
        scores: [num_users, num_items] 预测分数
        targets: [num_users] ground truth item id
        k_list: list of cutoff
    outputs:
        dict: {'HR@5':..., 'NDCG@5':..., ...}
    """
    num_users = scores.size(0)
    results = {}
    _, indices = scores.topk(max(k_list), dim=1)
    indices = indices.cpu().numpy()
    targets = targets.cpu().numpy()
    for k in k_list:
        hits, ndcgs = 0.0, 0.0
        for i in range(num_users):
            topk = indices[i,:k]
            if targets[i] in topk:
                hits += 1
                rank = list(topk).index(targets[i])
                ndcgs += 1 / math.log2(rank + 2)
        results[f'HR@{k}'] = hits / num_users
        results[f'NDCG@{k}'] = ndcgs / num_users
    return results

# ---------------- 测试集评估 ----------------
def evaluate_model(model, test_samples, k_list=[5,10,20], device='cuda'):
    test_users = [s[0] for s in test_samples]
    test_seqs = [s[1] for s in test_samples]
    test_targets = torch.tensor([s[2] for s in test_samples], device=device)
    model.eval()
    with torch.no_grad():
        user_emb, item_emb, _ = model(test_seqs, test_users)
        scores = model.predict(user_emb, item_emb)
        metrics = hit_ndcg(scores, test_targets, k_list=k_list)
    return metrics

def recommendation(model, samples, N=10,device='cuda') :
    users = [s[0] for s in samples]
    seqs = [s[1] for s in samples]
    model.eval()
    with torch.no_grad():
        user_emb, item_emb, _ = model(seqs, users)
        scores = model.predict(user_emb, item_emb)
        _, indices = scores.topk(N, dim=1)
        indices = indices.cpu().numpy()
    return users, indices