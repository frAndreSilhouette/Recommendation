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
    test_seqs = [s[1] for s in test_samples]
    test_targets = torch.tensor([s[2] for s in test_samples], device=device)
    model.eval()
    with torch.no_grad():
        item_embeds, _ = model([s for s in test_seqs], [s for s in test_seqs])
        scores = model.predict_next(test_seqs, item_embeds)
        metrics = hit_ndcg(scores, test_targets, k_list=k_list)
    return metrics

# def evaluate_model(model, test_samples, k_list=[5,10,20], device='cuda'):
#     test_seqs = [s[1] for s in test_samples]
#     test_targets = torch.tensor([s[2] for s in test_samples], device='cpu')  # 回 CPU

#     batch_size = 128

#     model.eval()
#     with torch.no_grad():
#         # ---------- 逐条计算 item_embeds，存 CPU ----------
#         item_embeds_list = []
#         for seq in tqdm(test_seqs, desc="Test Embeds"):
#             emb, _ = model([seq], [seq])
#             item_embeds_list.append(emb.cpu())
#         item_embeds = torch.cat(item_embeds_list, dim=0)

#         # ---------- 逐条预测，每条 seq 分 batch 处理 embedding ----------
#         scores_list = []
#         for seq in tqdm(test_seqs, desc="Test Predict"):
#             seq_scores_batches = []
#             for i in range(0, item_embeds.size(0), batch_size):
#                 emb_batch = item_embeds[i:i+batch_size].to(device)
#                 score_batch = model.predict_next([seq], emb_batch)
#                 seq_scores_batches.append(score_batch.cpu())  # 预测结果回 CPU
#             seq_scores = torch.cat(seq_scores_batches, dim=1)
#             scores_list.append(seq_scores)

#         scores = torch.cat(scores_list, dim=0)
#         scores = F.softmax(scores, dim=1) # softmax归一化

#         metrics = hit_ndcg(scores, test_targets, k_list=k_list)
#     return metrics
