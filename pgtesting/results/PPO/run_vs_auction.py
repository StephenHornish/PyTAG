import pandas as pd
import matplotlib.pyplot as plt
import re
import os
def extract_opponent_type(path):
    """
    Looks for known opponent tags in the path:
    random, mcts, osla, python, etc.
    Returns the first match or None.
    """
    opponents = ["random", "mcts", "osla", "python"]
    lowered = path.lower()
    for opp in opponents:
        if opp in lowered:
            return opp.upper()  
    return None

def extract_player_count(path):
    """
    Looks for substrings like np3, np4, np6 in the file path.
    """
    m = re.search(r"np(\d+)", path)
    return int(m.group(1)) if m else None

def clean_path(p):
    p = p.strip().strip('"').strip("'")
    if p.startswith("\\") or p.startswith("/"):
        return os.path.join(os.getcwd(), p.lstrip("\\/"))
    return p

def extract_plant_id(label):
    """
    Pulls the first integer from the label string.
    e.g. 'run_22_GAS1OIL2' -> 22
         'auction_plant_37' -> 37
    """
    nums = re.findall(r"\d+", str(label))
    return int(nums[0]) if nums else None


raw_auction_path = input("Enter path to auction_plant CSV: ")
raw_run_path = input("Enter path to run_plant CSV: ")

auction_path = clean_path(raw_auction_path)
run_path = clean_path(raw_run_path)
num_players = extract_player_count(auction_path) or extract_player_count(run_path)
opp_type = extract_opponent_type(auction_path) or extract_opponent_type(run_path)


if not os.path.exists(auction_path):
    raise FileNotFoundError(f"Auction file not found: {auction_path}")
if not os.path.exists(run_path):
    raise FileNotFoundError(f"Run file not found: {run_path}")

# ---- Load data ----
auction = pd.read_csv(auction_path)
run = pd.read_csv(run_path)

# ---- Extract numeric plant IDs ----
auction["plant_id"] = auction["label"].apply(extract_plant_id)
run["plant_id"] = run["label"].apply(extract_plant_id)

# ----  collapse multi-fuel Hybrid plant variants in run ----
run_grouped = (
    run.groupby("plant_id", as_index=False)["count"]
       .sum()
)

# ---- Normalize counts WITHIN each table ----
auction["norm"] = auction["count"] / auction["count"].sum()

# after grouping, we now normalize run by the summed counts
total_run_count = run_grouped["count"].sum()
run_grouped["norm"] = run_grouped["count"] / total_run_count

# ---- We also need auction grouped by plant_id in case auction has dupes (it shouldn't, but safe) ----
auction_grouped = (
    auction.groupby("plant_id", as_index=False)[["count","norm"]]
           .first()
)

# ---- Merge auction vs run on plant_id ----
merged = pd.merge(
    auction_grouped,
    run_grouped,
    on="plant_id",
    how="inner",
    suffixes=("_auction", "_run")
)

# ---- Compute difference metrics ----
merged["diff_norm"] = (merged["norm_run"] - merged["norm_auction"]).abs()
merged_sorted = merged.sort_values("diff_norm", ascending=False)

# ---- Print summary ----
print("\n=== Largest magnitude differences (relative share of phase activity) ===")
print(
    merged_sorted[["plant_id", "norm_auction", "norm_run", "diff_norm"]]
    .head(10)
    .to_string(index=False)
)

# ---- Scatter plot of relative magnitude ----
plt.figure(figsize=(8,7))
plt.scatter(merged["norm_auction"], merged["norm_run"], color="steelblue", s=60, alpha=0.7)

merged["plant_id"] = merged["plant_id"].astype(int)
for _, row in merged.iterrows():
    plt.text(
        row["norm_auction"] + 0.0005,
        row["norm_run"],
        str(int(row["plant_id"])),  
        fontsize=9,
        alpha=0.8
    )

plt.xlabel("Auction relative frequency")
plt.ylabel("Run relative frequency")
title = "Auction vs Run Action Magnitude Comparison (by Plant)"

if num_players:
    title += f" — {num_players} Players"
if opp_type:
    title += f" vs {opp_type}"

plt.title(title)


# 1:1 line (equal relative magnitude)
plt.axline((0, 0), slope=1, linestyle="--", color="red", label="Equal magnitude")
plt.xlim(0, 0.13)
plt.ylim(0, 0.13)
plt.legend()
plt.tight_layout()
plt.show()


#console print out summary
print("\n=== Full merged table (by plant_id) ===")
print(
    merged.sort_values("plant_id")[
        ["plant_id", "count_auction", "count_run", "norm_auction", "norm_run", "diff_norm"]
    ].to_string(index=False)
)
