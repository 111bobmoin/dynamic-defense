import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from scipy.sparse import lil_matrix, save_npz
import networkx as nx
from tqdm import tqdm
import os
import pickle

def load_bgl_logs(log_path):
    messages, labels = [], []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            label = 0 if parts[0] == '-' else 1
            message = ' '.join(parts[9:])
            messages.append(message)
            labels.append(label)
    return messages, labels

def tokenize(msg):
    return re.sub(r'\d+', '<*>', msg)

def extract_templates(messages):
    template_map = {}
    template_list = []
    index = 0
    for msg in messages:
        temp = tokenize(msg)
        if temp not in template_map:
            template_map[temp] = index
            index += 1
        template_list.append(template_map[temp])
    return template_list, template_map

def build_cooccurrence_graph(template_ids, window_size=10):
    cooccurrence = defaultdict(Counter)
    for i in range(len(template_ids) - window_size):
        window = template_ids[i:i+window_size]
        for u in window:
            for v in window:
                if u != v:
                    cooccurrence[u][v] += 1
    G = nx.Graph()
    for u in cooccurrence:
        for v in cooccurrence[u]:
            G.add_edge(u, v, weight=cooccurrence[u][v])
    return G

def generate_feature_matrix_sparse(template_ids, labels, num_templates, window_size=10, stride=5):
    n_windows = ((len(template_ids) - window_size) // stride) + 1
    X = lil_matrix((n_windows, num_templates), dtype=np.uint8)
    y = np.zeros(n_windows, dtype=np.uint8)

    idx = 0
    for i in tqdm(range(0, len(template_ids) - window_size, stride), desc="生成特征窗口"):
        window = template_ids[i:i+window_size]
        label_window = labels[i:i+window_size]
        for tid in window:
            X[idx, tid] += 1
        y[idx] = 1 if 1 in label_window else 0
        idx += 1
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preprocess_bgl_for_gcn(log_path, window_size=10, stride=5, output_dir="processed_bgl"):
    os.makedirs(output_dir, exist_ok=True)

    print("[1] 读取日志...")
    messages, labels = load_bgl_logs(log_path)

    print("[2] 提取模板...")
    template_ids, template_map = extract_templates(messages)
    num_templates = len(template_map)

    print(f"[3] 构造模板共现图...（节点数: {num_templates}）")
    G = build_cooccurrence_graph(template_ids, window_size=window_size)

    print("[4] 构造稀疏窗口特征矩阵...")
    X, y = generate_feature_matrix_sparse(template_ids, labels, num_templates, window_size, stride)

    print("[5] 划分训练集与验证集...")
    X_train, X_val, y_train, y_val = split_dataset(X, y)

    print(f"[6] 保存训练/验证集到 {output_dir}...")
    save_npz(os.path.join(output_dir, "X_train.npz"), X_train.tocsr())
    save_npz(os.path.join(output_dir, "X_val.npz"), X_val.tocsr())
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)

    with open(os.path.join(output_dir, "cooccurrence_graph.gpickle"), "wb") as f:
        pickle.dump(G, f)

    print("[✅] 数据预处理完成，已保存")
    return os.path.join(output_dir, "X_train.npz")

# 运行主流程
log_path = "BGL.log"
preprocess_bgl_for_gcn(log_path, window_size=10, stride=5)
