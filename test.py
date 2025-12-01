import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
from tqdm import tqdm
import datetime
import os
from time import time
import json

from sequence_loader import generate_sequence, build_sequences, load_and_preprocess, disturb_sequence
from graph_adj_matrix import build_user_item_graph
from graph_encoder import GraphConvolutionalEncoder
from sequence_encoder import SequenceEncoder
from evaluate import evaluate_model, hit_ndcg, recommendation
from dataset import SequenceDataset, collate_fn
from model import MultiViewRecommender
from run import train_model
"""

t0 = time()
#campus_list = ['13', '18', '38', '107', '151']
campus_list = ['13','18']
final_result_item={}
final_result_store={}
for campus_id in campus_list:
    has_graph_encoder=False
    has_sequence_encoder=True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    print(f"Campus Zone ID: {campus_id}")

    df, item2newid, user_new2raw, item_new2raw, spuid2storeid = load_and_preprocess("交易数据.feather", campus_id)
    user_seqs = build_sequences(df)
    max_item_id = max(item2newid.values())
    train_samples, valid_samples, test_samples = generate_sequence(user_seqs, disturb=None, max_item_id=max_item_id)

    all_item_ids = set()
    for s in train_samples + valid_samples + test_samples:
        all_item_ids.update(s[1])  # seq
        all_item_ids.add(s[2])     # target
    num_items = max(all_item_ids) + 1
    num_users = max(s[0] for s in train_samples + valid_samples + test_samples) + 1

    model = MultiViewRecommender(num_users=num_users, num_items=num_items, embed_dim=64, device=device,
                                has_graph_encoder=has_graph_encoder, has_sequence_encoder=has_sequence_encoder)

    model = train_model(model, train_samples, valid_samples, num_users=num_users, num_items=num_items,
                        num_epochs=1, batch_size=4096, lr=1e-3, device=device, early_stop_patience=10)
    rec_users, indices_item = recommendation(model, test_samples, N=10,device=device)

    result_one_campus_item = {user_new2raw[k]:[(item_new2raw[v] + '@' + spuid2storeid[item_new2raw[v]]) for v in rec_ls_temp] for k,rec_ls_temp in zip(rec_users, indices_item)}
    final_result_item[campus_id] = result_one_campus_item

    rec_users, indices_store = recommendation(model, test_samples, N=100,device=device)
    rec_stores = [list(set([spuid2storeid[item_new2raw[v]] for v in rec_ls_temp])) for rec_ls_temp in indices_store]
    result_one_campus_store = {user_new2raw[k]: stores[:10] for k, stores in zip(rec_users, rec_stores)}
    final_result_store[campus_id] = result_one_campus_store
    print(f"耗时{time()-t0}")
    

rec_res_folder = "rec_result"
item_result_filepath = os.path.join(rec_res_folder, "item_final_result.json")
store_result_filepath = os.path.join(rec_res_folder, "store_final_result.json")

os.makedirs(rec_res_folder, exist_ok=True)

with open(item_result_filepath,"w",encoding="utf-8") as f:
    json.dump(final_result_item,f,ensure_ascii=False,indent=2)

with open(store_result_filepath,"w",encoding="utf-8") as f:
    json.dump(final_result_store,f,ensure_ascii=False,indent=2)

print("结果保存完毕")
"""

import json

with open("./rec_result/store_final_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

campus_list = ['13', '18', '38', '107', '151']
for campus_id in campus_list:
    print(f"Campus Zone ID: {campus_id}")
    print(sum([(len(data[campus_id][k])<=0) for k in data[campus_id]]))