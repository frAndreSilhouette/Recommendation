import torch
from torch.utils.data import Dataset, DataLoader

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