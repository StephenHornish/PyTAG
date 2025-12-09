import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

# -------------------------------------------------
# Region map (city_number -> region label)
# -------------------------------------------------

region_map = {
    # Region 1: North East
    0:"North East", 1:"North East", 2:"North East", 3:"North East",
    4:"North East", 5:"North East",

    # Region 2: Mid Atlantic
    6:"Mid Atlantic", 7:"Mid Atlantic", 8:"Mid Atlantic",
    9:"Mid Atlantic", 10:"Mid Atlantic", 11:"Mid Atlantic", 12:"Mid Atlantic",

    # Region 3: South US
    13:"South US", 14:"South US", 15:"South US",
    16:"South US", 17:"South US", 18:"South US", 19:"South US",

    # Region 4: Central US
    20:"Central US", 21:"Central US", 22:"Central US",
    23:"Central US", 24:"Central US", 25:"Central US", 26:"Central US",

    # Region 5: Northwest
    27:"Northwest", 28:"Northwest", 29:"Northwest",
    30:"Northwest", 31:"Northwest", 32:"Northwest", 33:"Northwest",

    # Region 6: West
    34:"West", 35:"West", 36:"West", 37:"West",
    38:"West", 39:"West", 40:"West",

    # Region 7: Mexico
    41:"Mexico", 42:"Mexico", 43:"Mexico",
    44:"Mexico", 45:"Mexico", 46:"Mexico",
}

REGION_ORDER = [
    "North East",
    "Mid Atlantic",
    "South US",
    "Central US",
    "Northwest",
    "West",
    "Mexico",
]

# -------------------------------------------------
# City name map (city_number -> city name)
# -------------------------------------------------

city_name_map = {
    # Region 1: North East
    0:  "Quebec",
    1:  "Montreal",
    2:  "Boston",
    3:  "New_York",
    4:  "Philadelphia",
    5:  "Ottawa",

    # Region 2: Mid Atlantic
    6:  "Toronto",
    7:  "Detroit",
    8:  "Pittsburgh",
    9:  "Columbus",
    10: "Washington",
    11: "Charlotte",
    12: "Nashville",

    # Region 3: South US
    13: "Atlanta",
    14: "Jacksonville",
    15: "Miami",
    16: "New_Orleans",
    17: "Houston",
    18: "DallasFort_Worth",
    19: "San_Antonio",

    # Region 4: Central US
    20: "Memphis",
    21: "Oklahoma_City",
    22: "StLouis",
    23: "Indianapolis",
    24: "Chicago",
    25: "Milwaukee",
    26: "Kansas_City",

    # Region 5: Northwest
    27: "Minneapolis",
    28: "Winnipeg",
    29: "Regina",
    30: "Edmonton",
    31: "Calgary",
    32: "Vancouver",
    33: "Seattle",

    # Region 6: West
    34: "Portland",
    35: "Salt_Lake_City",
    36: "Denver",
    37: "Las_Vegas",
    38: "San_Francisco",
    39: "Los_Angeles",
    40: "San_Diego",

    # Region 7: Mexico
    41: "Albuquerque",
    42: "Juarez",
    43: "Chihuahua",
    44: "Monterrey",
    45: "Guadalajara",
    46: "Mexico_City",
}


csv_dir = input("Enter the directory path containing the build_city_aggregate.csv file: ").strip().strip('"')
csv_path = Path(csv_dir) / "build_city_aggregate.csv"

if not csv_path.exists():
    raise FileNotFoundError(f" Could not find {csv_path}. Make sure it's named build_city_aggregate.csv")

print(f"\nLoading data from: {csv_path}\n")

OUT_PNG_REGION = csv_path.parent / "city_build_frequency_by_region.png"
OUT_PNG_CITY   = csv_path.parent / "city_build_frequency_by_city.png"


df = pd.read_csv(csv_path)
# expected columns: player_id, label, count

def extract_city_number(label: str) -> int:
    m = re.search(r"build_city_(\d+)", str(label))
    return int(m.group(1)) if m else None

df["city_number"] = df["label"].apply(extract_city_number)
df["region"] = df["city_number"].map(region_map).fillna("UNKNOWN")

# city name: map from city_number, then clean underscores
df["city_name_raw"] = df["city_number"].map(city_name_map)
df["city_name"] = df["city_name_raw"].fillna(df["city_number"].astype(str))
df["city_name"] = df["city_name"].apply(lambda s: s.replace("_", " "))

# -------------------------------------------------
# PLOT 1: Region 
# -------------------------------------------------

group_region = (
    df.groupby(["region", "player_id"], as_index=False)["count"]
      .sum()
)

group_region["region"] = pd.Categorical(
    group_region["region"],
    categories=REGION_ORDER,
    ordered=True
)
group_region = group_region.sort_values(["region", "player_id"])

pivot_region = group_region.pivot(index="region", columns="player_id", values="count").fillna(0)
pivot_region = pivot_region.reindex(REGION_ORDER).dropna(how="all")

plt.figure(figsize=(10, 6))
pivot_region.plot(
    kind="bar",
    stacked=True,
    colormap="tab20",
    width=0.75,
    figsize=(10, 6),
)
plt.title("City Build Frequency by Region (Stacked by Player Count)")
plt.xlabel("Region")
plt.ylabel("Total Builds")
plt.ylim(0, 15000)
plt.legend(title="Player Count", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_PNG_REGION, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved region-by-player plot → {OUT_PNG_REGION}")

# -------------------------------------------------
# PLOT 2: City
# -------------------------------------------------

# aggregate by city_number & city_name 
group_city = (
    df.groupby(["city_number", "city_name", "player_id"], as_index=False)["count"]
      .sum()
      .sort_values(["city_number", "player_id"])
)

pivot_city = group_city.pivot(index="city_name", columns="player_id", values="count").fillna(0)


city_order = group_city.sort_values("city_number")["city_name"].unique()
pivot_city = pivot_city.reindex(city_order)

plt.figure(figsize=(16, 7))
pivot_city.plot(
    kind="bar",
    stacked=True,
    colormap="tab20",
    width=0.85,
    figsize=(16, 7),
)
plt.title("City Build Frequency by City (Stacked by Player Count)")
plt.xlabel("City")
plt.ylabel("Total Builds")
plt.ylim(0, 4000)  
plt.xticks(rotation=90, fontsize=8)
plt.legend(title="Player Count", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_PNG_CITY, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved city-by-player plot → {OUT_PNG_CITY}")

print("\nDone! Plots saved in:")
print(f"  {csv_path.parent}\n")
