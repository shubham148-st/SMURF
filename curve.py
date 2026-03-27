import matplotlib.pyplot as plt
import numpy as np

# Experimental means and standard deviations from 3 independent chronological runs
# Replace these hardcoded arrays with the exact F1 outputs from your local terminal
epsilons = [0.1, 0.5, 1.0, 5.0]
f1_means = [0.344, 0.333, 0.341, 0.335] 
f1_stds = [0.012, 0.015, 0.009, 0.011]  

# The baseline performance of the model when Differential Privacy is disabled
no_dp_mean = 0.383 
no_dp_std = 0.008  

plt.figure(figsize=(8, 5))

# Plot the Privacy-Utility curve with error bars demonstrating run-to-run variance
plt.errorbar(epsilons, f1_means, yerr=f1_stds, fmt='-o', color='b', 
             capsize=5, label='PrivateSmurf ($\epsilon$-DP)')

# Plot the 'No DP' performance ceiling with a shaded standard deviation band
plt.axhline(y=no_dp_mean, color='r', linestyle='--', label='No DP Ceiling ($\epsilon=\infty$)')
plt.fill_between(epsilons, no_dp_mean - no_dp_std, no_dp_mean + no_dp_std, color='r', alpha=0.1)

# Formatting
plt.title('Privacy-Utility Tradeoff in Temporal Fraud Detection\n(Mean $\pm$ Std over 3 Runs)')
plt.xlabel('Privacy Budget ($\epsilon$)')
plt.ylabel('F1-Score (Detection Utility)')
plt.xscale('log') # Epsilon is traditionally plotted on a logarithmic scale
plt.xticks(epsilons, labels=[str(e) for e in epsilons])
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# Export for publication
plt.savefig('privacy_utility_curve_errorbars.png', dpi=300, bbox_inches='tight')
print("Saved privacy_utility_curve_errorbars.png successfully.")