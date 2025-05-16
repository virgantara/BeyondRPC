import numpy as np
from scipy.stats import ttest_rel

# Replace these with your actual repeated experimental results (e.g., mCE or accuracy)
rpc_scores =     [0.863, 0.860, 0.866, 0.862, 0.861]  # baseline RPC scores
beyondrpc_scores = [0.455, 0.458, 0.452, 0.454, 0.456]  # BeyondRPC results (e.g., mCE, lower is better)

# Perform paired t-test
t_stat, p_value = ttest_rel(rpc_scores, beyondrpc_scores)

# Display results
print(f"Paired t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("✅ The difference is statistically significant (p < 0.05).")
else:
    print("⚠️ The difference is NOT statistically significant.")
