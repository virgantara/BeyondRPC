import subprocess
import numpy as np
from scipy.stats import ttest_rel

seeds = [42, 43, 44, 45, 46]
rpc_scores = []
beyondrpc_scores = []

# Run RPC baseline
print("Running RPC model...")
for seed in seeds:
    result = subprocess.check_output([
        "python", "zoo/PCT/main.py",
        "--exp_name", f"RPC_seed{seed}",
        "--model", "RPC",
        "--dataset", "modelnet40",
        "--seed", str(seed),
        "--epochs", "100",
        "--batch_size","64",
        "--test_batch_size","33",
        "--use_initweight"
    ])
    acc = float(result.decode().split("acc:")[-1].split(",")[0].strip())
    rpc_scores.append(acc)

# Run BeyondRPC
print("\nRunning BeyondRPC model...")
for seed in seeds:
    result = subprocess.check_output([
        "python", "zoo/PCT/main.py",
        "--exp_name", f"BeyondRPC_seed{seed}",
        "--model", "RPC",
        "--dataset", "modelnet40",
        "--seed", str(seed),
        "--pretrain_path", "ssl_models/adacrossnet_best.pth",
        "--epochs", "100",
        "--batch_size","64",
        "--test_batch_size","33",
        "--pw"
    ])
    acc = float(result.decode().split("acc:")[-1].split(",")[0].strip())
    beyondrpc_scores.append(acc)

# Convert to NumPy arrays
rpc_scores = np.array(rpc_scores)
beyondrpc_scores = np.array(beyondrpc_scores)

# Run paired t-test
t_stat, p_val = ttest_rel(rpc_scores, beyondrpc_scores)

print("\n=== Paired t-test ===")
print(f"RPC mean acc: {rpc_scores.mean():.4f}")
print(f"BeyondRPC mean acc: {beyondrpc_scores.mean():.4f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")

if p_val < 0.05:
    print(" Statistically significant difference (p < 0.05)")
else:
    print(" No statistically significant difference")