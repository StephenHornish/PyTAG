import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np


def extract_plant_number(label: str) -> int:
    """Extract numeric ID from labels like 'run_22_GAS3' → 22"""
    m = re.match(r"run_(\d+)_", label)
    return int(m.group(1)) if m else -1

def classify_resource(label: str, plant_number: int) -> str:
    """
    Classification rules:
    - Hybrid: plant number in {8, 22, 35}
    - Else if "COAL" in label -> Coal
    - Else if "GAS" in label -> Gas
    - Else if "OIL" in label -> Oil
    - Else if "URANIUM" in label -> Uranium
    - Else -> Renewable
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

# consistent plotting order
RESOURCE_ORDER = ["Coal", "Gas", "Oil", "Hybrid", "Uranium", "Renewable"]


csv_dir = input("Enter the directory path containing the run_plant_aggregate.csv file: ").strip().strip('"')
csv_path = Path(csv_dir) / "run_plant_aggregate.csv"

if not csv_path.exists():
    raise FileNotFoundError(f" Could not find {csv_path}. Make sure it's named run_plant_aggregate.csv")

print(f"\nLoading data from: {csv_path}\n")


OUT_PNG_PLANT = csv_path.parent / "plant_run_frequency_per_label.png"
OUT_PNG_RESOURCE = csv_path.parent / "plant_run_frequency_by_resource.png"

df = pd.read_csv(csv_path)

# derive numeric plant number and resource class
df["plant_number"] = df["label"].apply(extract_plant_number)
df["resource_type"] = df.apply(lambda r: classify_resource(r["label"], r["plant_number"]), axis=1)

# -------------------------------------------------
# PLOT 1: per-label 
# -------------------------------------------------

group_label = (
    df.groupby(["label", "plant_number", "player_id"], as_index=False)["count"]
      .sum()
      .sort_values(["plant_number", "label"])
)

pivot_label = group_label.pivot(index="label", columns="player_id", values="count").fillna(0)
pivot_label = pivot_label.reindex(group_label["label"].unique())

plt.figure(figsize=(18, 7))
pivot_label.plot(kind="bar", stacked=True, colormap="tab20", width=0.85, figsize=(18, 7))
plt.ylim(0, 16000)  
plt.title("Plant Run Frequency by Action (Stacked by Player Count)")
plt.xlabel("Plant Action (run_XX_...)")
plt.ylabel("Total Runs ")
plt.legend(title="Player Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(OUT_PNG_PLANT, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved per-label plot → {OUT_PNG_PLANT}")

# -------------------------------------------------
# PLOT 2: per resourcetype 
# -------------------------------------------------

group_resource = df.groupby(["resource_type", "player_id"], as_index=False)["count"].sum()
group_resource["resource_type"] = pd.Categorical(group_resource["resource_type"],
                                                 categories=RESOURCE_ORDER, ordered=True)
group_resource = group_resource.sort_values(["resource_type", "player_id"])

pivot_resource = group_resource.pivot(index="resource_type", columns="player_id", values="count").fillna(0)
pivot_resource = pivot_resource.reindex(RESOURCE_ORDER).dropna(how="all")

plt.figure(figsize=(10, 6))
pivot_resource.plot(kind="bar", stacked=True, colormap="tab20", width=0.75, figsize=(10, 6))
plt.ylim(0, 80000)  
plt.title("Total Plant Run Frequency by Resource Type (Stacked by Player Count)")
plt.xlabel("Resource Type")
plt.ylabel("Total Runs")
plt.legend(title="Player Count", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_PNG_RESOURCE, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved per-resource plot → {OUT_PNG_RESOURCE}")


#Console summary
total_by_resource = df.groupby("resource_type")["count"].sum().reindex(RESOURCE_ORDER).fillna(0)
grand_total = total_by_resource.sum()

print("\n=== Total Plant Run Frequency by Resource Type ===")
print(f"{'Resource Type':<12} {'Total Count':>12} {'Percent of Total':>18}")
print("-" * 44)
for rtype, count in total_by_resource.items():
    pct = (count / grand_total * 100) if grand_total > 0 else 0
    print(f"{rtype:<12} {int(count):>12,} {pct:>17.2f}%")
print("-" * 44)
print(f"{'TOTAL':<12} {int(grand_total):>12,} {100.00:>17.2f}%")

print("\nDone! Plots and summary saved in:")
print(f"  {csv_path.parent}\n")
