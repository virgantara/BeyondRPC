from sklearn.manifold import TSNE
from umap import UMAP
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

file_name = 'RPC'

with torch.no_grad():
    for data, labels in tqdm(test_loader):
        data, labels = data.to(device), labels.to(device)
        data = data.permute(0, 2, 1)  # (B, 3, N)
        feats = model.extract_features(data)  # (B, 1024)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

X = np.concatenate(all_feats, axis=0)
y = np.concatenate(all_labels, axis=0).flatten()

# t-SNE and UMAP
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
X_umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)

axis_label_fontsize = 30
tick_fontsize = 28  # or any value you prefer


# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10', s=40, ax=axes[0])
axes[0].set_title("t-SNE "+file_name, fontsize=axis_label_fontsize)
axes[0].set_xlabel("Component 1", fontsize=axis_label_fontsize)
axes[0].set_ylabel("Component 2", fontsize=axis_label_fontsize)
axes[0].legend(loc='best', title='Class')

sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='tab10', s=40, ax=axes[1])
axes[1].set_title("UMAP "+file_name, fontsize=axis_label_fontsize)
axes[1].set_xlabel("Component 1", fontsize=axis_label_fontsize)
axes[1].set_ylabel("Component 2", fontsize=axis_label_fontsize)
axes[1].legend(loc='best', title='Class')

axes[0].tick_params(axis='both', labelsize=tick_fontsize)
axes[1].tick_params(axis='both', labelsize=tick_fontsize)

plt.tight_layout()
plt.savefig("tsne_vs_umap "+file_name+".png")
plt.show()
