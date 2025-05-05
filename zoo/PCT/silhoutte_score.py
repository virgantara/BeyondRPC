from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import RPC  # or your chosen model
from data import ScanObjectNN
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='model path')
parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset and model
test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points),
                         batch_size=args.test_batch_size, shuffle=False)

model = RPC(args=args, output_channels=15).to(device)
state_dict = torch.load(args.model_path)
model_state_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
model_state_dict.update(pretrained_dict)
model.load_state_dict(model_state_dict)
model.eval()

# Extract features
all_feats = []
all_labels = []

with torch.no_grad():
    for data, labels in tqdm(test_loader):
        data, labels = data.to(device), labels.to(device)
        data = data.permute(0, 2, 1)  # (B, 3, N)
        feats = model.extract_features(data)  # (B, 1024)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

X = np.concatenate(all_feats, axis=0)
y = np.concatenate(all_labels, axis=0).flatten()

# ---------- Evaluate Feature Embedding Quality ----------
# Dimensionality reduction before clustering metrics
print("Computing clustering quality metrics...")
X_pca = PCA(n_components=50).fit_transform(X)

sil_score = silhouette_score(X_pca, y)
db_score = davies_bouldin_score(X_pca, y)

print(f"Silhouette Score: {sil_score:.4f} (higher is better)")
print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")
