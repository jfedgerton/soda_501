###############################################################################
# Deterministic and Probabilistic Matching Tutorial + Assignment (Python)
# Author: Jared Edgerton
# Date: date.today()
#
# This script demonstrates:
#   1) Generating two "messy" datasets that represent the same people
#   2) Deterministic (exact) matching using pandas merge()
#   3) Probabilistic matching using the recordlinkage package (Fellegi–Sunter style)
#   4) How match thresholds affect match counts and match quality
#   5) Using Levenshtein distance to evaluate match quality
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are repeated explicitly so students can follow and modify each piece.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# If you do not have these installed, run (in Terminal / Anaconda Prompt):
#   pip install pandas numpy matplotlib recordlinkage rapidfuzz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import recordlinkage as rl
from rapidfuzz.distance import Levenshtein

from datetime import date


# -----------------------------------------------------------------------------
# Part 0: Instructor Prep (Generate Two Datasets)
# -----------------------------------------------------------------------------
# Goal:
# - Create df_a (clean-ish) and df_b (noisier copy of df_a)
# - df_b will have:
#   * typos in some first names
#   * typos in some last names
#   * small random shifts in some birthyears
#
# Teaching note:
# - These are "toy" datasets designed to make matching concepts visible.

np.random.seed(123)   # Reproducibility (use the same seed each time)
n = 10000             # Target size (before drop_duplicates)

# -----------------------------------------------------------------------------
# Step 1: Create Dataset A (df_a)
# -----------------------------------------------------------------------------
first_names = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "William", "Emma", "James", "Olivia"]
last_names  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

df_a = pd.DataFrame({
    "id": np.arange(1, n + 1),
    "firstname": np.random.choice(first_names, size=n, replace=True),
    "lastname": np.random.choice(last_names, size=n, replace=True),
    "birthyear": np.random.randint(1970, 2001, size=n),
    "zipcode": np.random.randint(10000, 20001, size=n)
})

# Mimic R's distinct(): drop duplicates across all columns
df_a = df_a.drop_duplicates().reset_index(drop=True)

# -----------------------------------------------------------------------------
# Step 2: Create Dataset B (df_b) as a copy of df_a, then add "noise"
# -----------------------------------------------------------------------------
df_b = df_a.copy()

# Decide which rows will be modified (25% probability each type of noise)
mod_firstname = np.random.rand(len(df_b)) < 0.25
mod_lastname  = np.random.rand(len(df_b)) < 0.25
mod_birthyear = np.random.rand(len(df_b)) < 0.25

# ---- 2A: Add typos to FIRST NAMES (only for rows selected by mod_firstname) ----
idx_firstname = np.where(mod_firstname)[0]

for i in idx_firstname:
    firstname = df_b.loc[i, "firstname"]
    chars = list(firstname)

    num_replace = np.random.randint(1, len(chars) + 1)
    positions = np.random.choice(np.arange(len(chars)), size=num_replace, replace=False)

    for pos in positions:
        chars[pos] = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"))

    df_b.loc[i, "firstname"] = "".join(chars)

# ---- 2B: Add typos to LAST NAMES (only for rows selected by mod_lastname) ----
idx_lastname = np.where(mod_lastname)[0]

for i in idx_lastname:
    lastname = df_b.loc[i, "lastname"]
    chars = list(lastname)

    num_replace = np.random.randint(1, len(chars) + 1)
    positions = np.random.choice(np.arange(len(chars)), size=num_replace, replace=False)

    for pos in positions:
        chars[pos] = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"))

    df_b.loc[i, "lastname"] = "".join(chars)

# ---- 2C: Shift BIRTH YEAR slightly (only for rows selected by mod_birthyear) ----
idx_birthyear = np.where(mod_birthyear)[0]
birthyear_shift = np.random.choice(np.arange(-2, 3), size=len(idx_birthyear), replace=True)
df_b.loc[idx_birthyear, "birthyear"] = df_b.loc[idx_birthyear, "birthyear"].to_numpy() + birthyear_shift

# -----------------------------------------------------------------------------
# Step 3: Save datasets to CSV (so students can load them like "real" files)
# -----------------------------------------------------------------------------
df_a.to_csv("dataset_a.csv", index=False)
df_b.to_csv("dataset_b.csv", index=False)


# -----------------------------------------------------------------------------
# Part 1: Student Assignment Starts Here
# -----------------------------------------------------------------------------
# Your task is to perform both deterministic and probabilistic matching on these
# two datasets. Follow the steps below and answer the questions at the end.

# -----------------------------------------------------------------------------
# Step 1: Load the datasets
# -----------------------------------------------------------------------------
df_a = pd.read_csv("dataset_a.csv")
df_b = pd.read_csv("dataset_b.csv")

# Set id as the index (helps later when we work with record linkage pairs)
df_a = df_a.set_index("id")
df_b = df_b.set_index("id")

# -----------------------------------------------------------------------------
# Step 2: Examine the datasets
# -----------------------------------------------------------------------------
# describe() gives distributions and missingness
# head() shows the first few rows

print(df_a.describe(include="all"))
print(df_a.head())

print(df_b.describe(include="all"))
print(df_b.head())

# -----------------------------------------------------------------------------
# Step 3: Deterministic matching (exact matching)
# -----------------------------------------------------------------------------
# Deterministic matching means:
# - A record in df_a matches a record in df_b ONLY IF all chosen fields match exactly.
#
# In pandas, merge(..., on=[...]) is a standard exact-match join.

det_matches = (
    df_a.reset_index()
      .merge(
          df_b.reset_index(),
          on=["firstname", "lastname", "birthyear", "zipcode"],
          how="inner",
          suffixes=(".a", ".b")
      )
)

print("Number of deterministic matches:", det_matches.shape[0])


# -----------------------------------------------------------------------------
# Step 4: Probabilistic matching using recordlinkage
# -----------------------------------------------------------------------------
# Probabilistic matching means:
# - We do NOT require exact matches on all fields.
# - Instead, we compute similarity features and estimate a probabilistic model.
#
# recordlinkage workflow:
#   1) Create candidate pairs (indexing / blocking)
#   2) Compare fields to create similarity features
#   3) Fit an unsupervised probabilistic model (ECMClassifier)
#   4) Get match probabilities ("posterior") for candidate pairs

# -----------------------------------------------------------------------------
# Step 4A: Create candidate pairs (blocking)
# -----------------------------------------------------------------------------
# Blocking reduces the number of candidate pairs we consider.
# Here we block on zipcode (exact), because zipcode has no noise in our toy data.

indexer = rl.Index()
indexer.block("zipcode")
candidate_pairs = indexer.index(df_a, df_b)

print("Number of candidate pairs after blocking:", len(candidate_pairs))

# -----------------------------------------------------------------------------
# Step 4B: Create similarity features (comparisons)
# -----------------------------------------------------------------------------
# We'll create a feature matrix with one row per candidate pair.
# Features are BINARIZED (0/1) because ECMClassifier expects binary agree/disagree
# vectors (this is a core assumption of the Fellegi-Sunter model).
# The threshold parameter converts continuous similarity to binary:
#   1 if similarity >= threshold, 0 otherwise.
# - firstname: Jaro-Winkler >= 0.85 -> 1 (agree)
# - lastname:  Jaro-Winkler >= 0.85 -> 1 (agree)
# - birthyear: Gaussian similarity >= 0.5 -> 1 (agree) [binarized after compute]
# - zipcode:   exact match (already binary)

compare = rl.Compare()
compare.string("firstname", "firstname", method="jarowinkler", threshold=0.85, label="firstname_sim")
compare.string("lastname",  "lastname",  method="jarowinkler", threshold=0.85, label="lastname_sim")
compare.numeric("birthyear", "birthyear", method="gauss", offset=0, scale=2, label="birthyear_sim")
compare.exact("zipcode", "zipcode", label="zipcode_exact")

features = compare.compute(candidate_pairs, df_a, df_b)

# Binarize birthyear_sim (numeric comparison doesn't support threshold parameter)
# Values >= 0.5 are treated as "agree"
features["birthyear_sim"] = (features["birthyear_sim"] >= 0.5).astype(int)

print(features.head())

# -----------------------------------------------------------------------------
# Step 4C: Fit an unsupervised probabilistic linkage model (ECM)
# -----------------------------------------------------------------------------
# ECMClassifier is an unsupervised mixture-model approach (Fellegi–Sunter style).
# After fitting, we can get match probabilities for each candidate pair.

ecm = rl.ECMClassifier()
ecm.fit(features)

posterior = ecm.prob(features)  # pandas Series indexed by (id_a, id_b)

# Put probabilities into a DataFrame for easier inspection/merging
posterior_df = posterior.reset_index()
posterior_df.columns = ["id_a", "id_b", "posterior"]

print(posterior_df.head())


# -----------------------------------------------------------------------------
# Step 5: Analyze how the match threshold affects the number of matches
# -----------------------------------------------------------------------------
# recordlinkage gives posterior match probabilities (0 to 1).
# We choose a threshold:
# - Higher threshold -> fewer matches, but likely higher quality
# - Lower threshold  -> more matches, but likely lower quality

threshold_grid = np.arange(0, 1.01, 0.01)

match_counts = []
for th in threshold_grid:
    match_counts.append((posterior >= th).sum())

count_of_matches = pd.DataFrame({
    "threshold": threshold_grid,
    "matches": match_counts
})

plt.figure()
plt.plot(count_of_matches["threshold"], count_of_matches["matches"])
plt.xlabel("Threshold level (posterior probability cutoff)")
plt.ylabel("Count of matches returned")
plt.title("Match count vs threshold")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Step 6: Match quality vs posterior probability (string distance diagnostics)
# -----------------------------------------------------------------------------
# Idea:
# - Keep a very low threshold so we get "almost everything"
# - Bin matches into posterior-probability bins:
#     (0.0, 0.1], (0.1, 0.2], ..., (0.9, 1.0]
# - For each bin, compute:
#     * Levenshtein distance for first names
#     * Levenshtein distance for last names
#     * absolute difference in birth year
#
# Interpretation:
# - If posterior is doing what we want, then higher posterior bins should have:
#     * smaller string distances
#     * smaller birth-year differences

matches_low = posterior_df[posterior_df["posterior"] > 0.000001].copy()

bin_edges = np.arange(0.0, 1.0000001, 0.1)
bin_labels = [
    "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
    "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"
]

matches_low["threshold_bin"] = pd.cut(
    matches_low["posterior"],
    bins=bin_edges,
    labels=bin_labels,
    right=True
)

# Join pair rows back to df_a and df_b so we can compare fields
pairs_plus_a = matches_low.merge(df_a.reset_index(), left_on="id_a", right_on="id", how="left")
pairs_plus_a = pairs_plus_a.drop(columns=["id"]).rename(columns={
    "firstname": "firstname_a",
    "lastname": "lastname_a",
    "birthyear": "birthyear_a",
    "zipcode": "zipcode_a"
})

pairs_plus_ab = pairs_plus_a.merge(df_b.reset_index(), left_on="id_b", right_on="id", how="left")
pairs_plus_ab = pairs_plus_ab.drop(columns=["id"]).rename(columns={
    "firstname": "firstname_b",
    "lastname": "lastname_b",
    "birthyear": "birthyear_b",
    "zipcode": "zipcode_b"
})

# Compute string distances (Levenshtein edit distance)
pairs_plus_ab["first_name_distance"] = [
    Levenshtein.distance(a, b) for a, b in zip(pairs_plus_ab["firstname_a"], pairs_plus_ab["firstname_b"])
]
pairs_plus_ab["last_name_distance"] = [
    Levenshtein.distance(a, b) for a, b in zip(pairs_plus_ab["lastname_a"], pairs_plus_ab["lastname_b"])
]

# Compute birthyear distance: absolute difference
pairs_plus_ab["birth_year_distance"] = (pairs_plus_ab["birthyear_a"] - pairs_plus_ab["birthyear_b"]).abs()

# Average distance by posterior bin (mean distances + counts)
avg_dist_by_bin = (
    pairs_plus_ab
    .groupby("threshold_bin", observed=False)
    .agg(
        mean_first=("first_name_distance", "mean"),
        mean_last=("last_name_distance", "mean"),
        mean_birth=("birth_year_distance", "mean"),
        n=("posterior", "size")
    )
    .reset_index()
)

print(avg_dist_by_bin)

# Plot: mean first-name distance by posterior bin
plt.figure()
plt.plot(avg_dist_by_bin["threshold_bin"].astype(str), avg_dist_by_bin["mean_first"], marker="o")
plt.xlabel("Posterior probability bin")
plt.ylabel("Mean Levenshtein distance (first name)")
plt.title("Match quality vs posterior probability (first name distance)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot: mean last-name distance by posterior bin
plt.figure()
plt.plot(avg_dist_by_bin["threshold_bin"].astype(str), avg_dist_by_bin["mean_last"], marker="o")
plt.xlabel("Posterior probability bin")
plt.ylabel("Mean Levenshtein distance (last name)")
plt.title("Match quality vs posterior probability (last name distance)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot: mean birthyear distance by posterior bin
plt.figure()
plt.plot(avg_dist_by_bin["threshold_bin"].astype(str), avg_dist_by_bin["mean_birth"], marker="o")
plt.xlabel("Posterior probability bin")
plt.ylabel("Mean absolute difference (birth year)")
plt.title("Match quality vs posterior probability (birth year difference)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
