import numpy as np
from scipy.stats import ttest_rel

# mCE results across 5 seeds
rpc_mce =       [0.940, 0.881, 0.850, 0.845, 0.863]
beyondrpc_mce = [0.648, 0.631, 0.655, 0.658, 0.659]

# Paired t-test
t_stat, p_value = ttest_rel(rpc_mce, beyondrpc_mce)

# Mean and std
rpc_mean = np.mean(rpc_mce)
rpc_std = np.std(rpc_mce, ddof=1)
beyondrpc_mean = np.mean(beyondrpc_mce)
beyondrpc_std = np.std(beyondrpc_mce, ddof=1)

# Print results
print("=== Paired t-test (mCE: lower is better) ===")
print(f"RPC mean mCE:       {rpc_mean:.4f} ± {rpc_std:.4f}")
print(f"BeyondRPC mean mCE: {beyondrpc_mean:.4f} ± {beyondrpc_std:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Statistically significant (p < 0.05)")
else:
    print("Not statistically significant (p ≥ 0.05)")
