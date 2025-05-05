from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import RPC  # or your chosen model
from data import ScanObjectNN
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset and model
test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024),
                         batch_size=32, shuffle=False)

model = RPC(args=None, output_channels=15).to(device)
model.load_state_dict(torch.load('checkpoints/your_experiment/models/model.t7'))
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
y = np.concatenate(all_labels, axis=0)

# Dimensionality reduction
X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette='tab10', s=40)
plt.title("t-SNE of BeyondRPC Feature Embeddings (ScanObjectNN)")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("tsne_beyondrpc.png")
plt.show()
