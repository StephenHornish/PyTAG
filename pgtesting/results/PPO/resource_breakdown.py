import pandas as pd
import numpy as np
import scipy.stats as stats
import re
from pathlib import Path

# -------------------------------------------------
# CONFIG & INPUT
# -------------------------------------------------

csv_dir = input("Enter the directory path containing the run data CSV: ").strip().strip('"')
csv_path = Path(csv_dir) / "run_plant_aggregate.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Could not find {csv_path}. Make sure it's named run_plant_aggregate.csv")

print(f"\nLoading data from: {csv_path}\n")


RESOURCE_ORDER = ["Coal", "Gas", "Oil", "Hybrid", "Uranium", "Renewable"]

# Expected proportions under the null hypothesis (edit if needed!)
expected_proportions = np.array([
    12/42,  # Coal
    8/42,   # Gas
    7/42,   # Oil
    3/42,   # Hybrid
    5/42,   # Uranium
    7/42,   # Renewable
], dtype=float)

# sanity check normalization
expected_proportions = expected_proportions / expected_proportions.sum()


def extract_plant_number(label: str) -> int:
    """Extract numeric plant id (e.g. 'run_22_GAS3' → 22)."""
    m = re.match(r"run_(\d+)_", label)
    return int(m.group(1)) if m else -1

def classify_resource(label: str, plant_number: int) -> str:
    """
    Classify plant into resource type:
    - Hybrid if plant_number in {8, 22, 35}
    - else Coal/Gas/Oil/Uranium by substring
    - else Renewable
    """
    if plant_number in {8, 22, 35}:
        return "Hybrid"
    if "COAL" in label:
        return "Coal"
    if "GAS" in label:
        return "Gas"
    if "OIL" in label:
        return "Oil"
    if "URANIUM" in label:
        return "Uranium"
    return "Renewable"

def chi_square_for_df(df_scope: pd.DataFrame, scope_name: str, save_csv_path: Path | None = None):
    """
    Compute + print summary & chi-square for a subset of the data (e.g. one player or all players).
    If save_csv_path is provided, also write out a CSV with the breakdown.
    """
    # Totals by resource
    summary = (
        df_scope.groupby("resource_type")["count"]
        .sum()
        .reindex(RESOURCE_ORDER)
        .fillna(0)
        .astype(int)
    )

    total = summary.sum()

    print(f"=== {scope_name}: Total Plant Run Frequency by Resource Type ===")
    print(f"{'Resource Type':<12} {'Total Count':>12} {'Percent of Total':>18}")
    print("-" * 44)
    for res, count in summary.items():
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"{res:<12} {count:>12,} {pct:>17.2f}%")
    print("-" * 44)
    print(f"{'TOTAL':<12} {total:>12,} {100.00:>17.2f}%\n")

    # Chi-square goodness of fit
    observed = summary.values.astype(float)
    expected = expected_proportions * total

    # Avoid all-zero or degenerate case
    if total == 0 or np.all(expected == 0):
        print(f"=== {scope_name}: Chi-square skipped (no data) ===\n")
        return

    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)

    observed_pct = observed / total * 100.0
    expected_pct = expected / total * 100.0
    std_residuals = (observed - expected) / np.sqrt(expected)

    print(f"=== {scope_name}: Chi-Square Goodness-of-Fit ===")
    print(f"Total samples: {int(total)}")
    print(f"Chi² = {chi2:.2f}")
    print(f"p-value = {p:.6e}\n")

    print("Category Breakdown:")
    print(f"{'Category':<12} {'Obs':>10} {'Exp':>10} {'Obs%':>8} {'Exp%':>8} {'StdResid':>10}")
    for cat, o, e, op, ep, r in zip(
        RESOURCE_ORDER,
        observed,
        expected,
        observed_pct,
        expected_pct,
        std_residuals
    ):
        print(f"{cat:<12} {int(o):>10} {e:>10.1f} {op:>7.2f}% {ep:>7.2f}% {r:>10.2f}")
    print()

    # Optional CSV output (used for ALL PLAYERS combined)
    if save_csv_path is not None:
        summary_df = pd.DataFrame({
            "Resource": RESOURCE_ORDER,
            "Observed_Count": observed.astype(int),
            "Observed_%": np.round(observed_pct, 2),
            "Expected_Count": np.round(expected, 1),
            "Expected_%": np.round(expected_pct, 2),
            "StdResid": np.round(std_residuals, 2),
        })
        summary_df.to_csv(save_csv_path, index=False)
        print(f" Saved summary CSV with chi-square details → {save_csv_path}\n")


df = pd.read_csv(csv_path)

if "player_id" not in df.columns:
    raise ValueError("Expected a 'player_id' column in run_plant_aggregate.csv")

# Derive numeric plant ID and resource type
df["plant_number"] = df["label"].apply(extract_plant_number)
df["resource_type"] = df.apply(
    lambda r: classify_resource(r["label"], r["plant_number"]), axis=1
)


players = sorted(df["player_id"].unique())

for pid in players:
    df_p = df[df["player_id"] == pid]
    chi_square_for_df(df_p, scope_name=f"Player {pid}")


out_path = csv_path.parent / "resource_breakdown.csv"

chi_square_for_df(df, scope_name="ALL PLAYERS COMBINED", save_csv_path=out_path)

print("Note:")
print("  StdResid >> 0  → Agent OVER-uses this resource vs expectation.")
print("  StdResid << 0  → Agent UNDER-uses this resource vs expectation.")
print("  |StdResid| > ~2 is usually considered large.\n")
print(f"Data source: {csv_path.parent}\n")
