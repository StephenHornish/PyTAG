import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
#Class that calculates teh chi squared of the observed counts of resources 
# Labels for each fuel category in the order we track them
categories = [
    "Coal",
    "Gas",
    "Oil",
    "Hybrid",
    "Uranium",
    "Renewable",
]

# Observed counts from agent behavior this is where data in inputed from a evlauation 
observed = np.array([
    12405,   # Coal
    42942,  # Gas
    18255,  # Oil
    11725,   # Hybrid
    23834,   # Uranium
    49659,   # Renwable
], dtype=float)

# Expected proportions based on deck composition.
expected_proportions = np.array([
    12/42,  # Coal
    8/42,  # Gas
    7/42,   # Oil
    3/42,   # Hybrid
    5/42,   # uranium
    7/42,   # Renewable
], dtype=float)


total = observed.sum()

# Expected counts for each category if agent were random
expected = expected_proportions * total

# Chi-square goodness-of-fit
chi2, p = stats.chisquare(observed, expected)

# Percentages
observed_pct = observed / total * 100.0
expected_pct = expected / total * 100.0 

# Standardized residuals
std_residuals = (observed - expected) / np.sqrt(expected)

# print report

print("=== Chi-Square Goodness-of-Fit ===")
print(f"Total samples: {int(total)}")
print(f"Chi² = {chi2:.2f}")
print(f"p-value = {p:.6f}")
print()

print("=== Category Breakdown ===")
print(f"{'Category':<12} {'Obs':>8} {'Exp':>8} {'Obs%':>8} {'Exp%':>8} {'StdResid':>10}")
for cat, o, e, op, ep, r in zip(categories, observed, expected, observed_pct, expected_pct, std_residuals):
    print(f"{cat:<12} {int(o):>8} {e:>8.1f} {op:>7.2f}% {ep:>7.2f}% {r:>10.2f}")

# Interpretation hint
print("\nNote:")
print("  StdResid >> 0  → Agent OVER-uses this resource vs expectation.")
print("  StdResid << 0  → Agent UNDER-uses this resource vs expectation.")
print("  |StdResid| > ~2 is usually considered large.")

# Plot Observed vs Expected 
x = np.arange(len(categories))

plt.figure(figsize=(10, 5))

bar_width = 0.35

plt.bar(x - bar_width/2, observed_pct, width=bar_width, label="Observed %")
plt.bar(x + bar_width/2, expected_pct, width=bar_width, label="Expected %")

plt.xticks(x, categories, rotation=30, ha="right")
plt.ylabel("Share of Usage (%)")
plt.title("Observed vs Expected Resource Usage")
plt.legend()
plt.tight_layout()

plt.savefig("resource_usage_observed_vs_expected.png", dpi=200, bbox_inches="tight")
print("\nSaved plot: resource_usage_observed_vs_expected.png")

