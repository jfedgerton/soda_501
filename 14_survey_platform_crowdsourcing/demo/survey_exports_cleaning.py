###############################################################################
# Survey Exports + Cleaning + Codebooks Tutorial: Python (Platform Data Example)
# Author: Jared Edgerton
# Date: (use your local date/time)
#
# This script demonstrates:
#   1) Loading survey exports (CSV / Qualtrics-style structure)
#   2) Cleaning common survey fields (missingness, types, recodes)
#   3) Building a simple codebook (variable definitions + value meanings)
#   4) Adding labels (variable labels + value labels)
#   5) Running basic quality checks (speeding, attention checks, straightlining)
#   6) Saving a cleaned dataset for analysis
#
# Week  context:
# - Surveys, Platforms, and Crowdsourcing
# - Coding lab: survey exports; cleaning; codebooks; labeling; quality checks.
# - Pre-class video: Data quality, missingness, and measurement in platform data.
# - Application papers: Berinsky et al. (2012) Political Analysis; Bisbee (2024) Political Analysis
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are written explicitly so students can follow how the code unfolds.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# Recommended pip installs (run once in terminal):
#   pip install pandas numpy matplotlib

import os
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create common project folders (safe to run repeatedly)
os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("src", exist_ok=True)

# -----------------------------------------------------------------------------
# Part 1: Survey Export (Qualtrics-style) --- Loading data
# -----------------------------------------------------------------------------
# In real work, you'll start from a platform export:
#   - Qualtrics (CSV; sometimes with extra header rows)
#   - Prolific / MTurk exports (CSV)
#   - Department pool exports (CSV)
#
# Qualtrics CSV note:
# - Qualtrics exports often include "extra header rows" (question text, etc.).
# - A common fix is: pd.read_csv(..., skiprows=2)
#
# Examples (uncomment + edit file names once you have them):
#
# survey_raw = pd.read_csv("data_raw/survey_export.csv")
# survey_raw = pd.read_csv("data_raw/qualtrics_export.csv", skiprows=2)
#
# For this tutorial, we create a small "toy export" that mimics typical fields.
# Replace this with your real export by overwriting survey_raw above.

survey_raw = pd.DataFrame(
    {
        "ResponseId": [f"R_{i:03d}" for i in range(1, 13)],
        "StartDate": [
            "2026-02-05 10:00:05", "2026-02-05 10:01:10", "2026-02-05 10:02:02",
            "2026-02-05 10:03:41", "2026-02-05 10:05:00", "2026-02-05 10:05:45",
            "2026-02-05 10:06:10", "2026-02-05 10:07:33", "2026-02-05 10:08:09",
            "2026-02-05 10:09:18", "2026-02-05 10:10:07", "2026-02-05 10:11:55"
        ],
        "EndDate": [
            "2026-02-05 10:04:20", "2026-02-05 10:02:05", "2026-02-05 10:10:44",
            "2026-02-05 10:06:10", "2026-02-05 10:07:12", "2026-02-05 10:08:03",
            "2026-02-05 10:07:05", "2026-02-05 10:08:20", "2026-02-05 10:14:45",
            "2026-02-05 10:10:55", "2026-02-05 10:12:50", "2026-02-05 10:19:40"
        ],
        "Duration_seconds": [255, 55, 522, 149, 132, 138, 55, 47, 396, 97, 163, 465],
        "IPAddress": [
            "73.10.20.1", "73.10.20.2", "73.10.20.3", "73.10.20.4",
            "73.10.20.5", "73.10.20.6", "73.10.20.7", "73.10.20.8",
            "73.10.20.9", "73.10.20.10", "73.10.20.11", "73.10.20.12"
        ],
        "Platform": [
            "Prolific", "Prolific", "Prolific", "Prolific",
            "MTurk", "MTurk", "MTurk", "MTurk",
            "DepartmentPool", "DepartmentPool", "DepartmentPool", "DepartmentPool"
        ],
        "Prolific_ID": ["5f1aA", "5f1aB", "5f1aC", "5f1aD", "", "", "", "", "", "", "", ""],
        "MTurk_WorkerId": ["", "", "", "", "A11X", "A11Y", "A11Z", "A11W", "", "", "", ""],
        "Consent": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
        "Age": ["34", "29", "41", "", "22", "27", "18", "19", "45", "38", "33", "Prefer not to say"],
        "Gender": ["Man", "Woman", "Woman", "Man", "Nonbinary", "Woman", "Man", "", "Woman", "Man", "Prefer not to say", "Woman"],
        "PartyID": ["Democrat", "Independent", "Republican", "Democrat", "Independent", "Democrat", "", "Republican", "Democrat", "Independent", "Republican", "Independent"],
        "AttentionCheck": [
            "Strongly disagree", "Strongly disagree", "Strongly disagree", "Agree",
            "Strongly disagree", "Strongly disagree", "Agree", "Strongly disagree",
            "Strongly disagree", "Strongly disagree", "Strongly disagree", "Strongly disagree"
        ],
        "Q1_policy": ["Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree"],
        "Q2_policy": ["Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree"],
        "Q3_policy": ["Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree"],
        "Q4_policy": ["Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree"],
    }
)

# Inspect the raw export (students should do this every time)
print(survey_raw)
print(survey_raw.dtypes)
print(survey_raw.head())

# -----------------------------------------------------------------------------
# Part 2: Cleaning --- names, types, and missing values
# -----------------------------------------------------------------------------
# Step 1: Standardize column names (consistent snake_case)
# (Python version of janitor::clean_names())

clean_cols = [
    re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", c)).strip("_").lower()
    for c in survey_raw.columns
]
survey_clean = survey_raw.copy()
survey_clean.columns = clean_cols

# Inspect cleaned names
print(list(survey_clean.columns))

# Step 2: Parse timestamps into datetime (platform metadata)
# Set timezone to America/New_York for class consistency.
survey_clean["start_date"] = pd.to_datetime(survey_clean["startdate"], errors="coerce").dt.tz_localize(ZoneInfo("America/New_York"))
survey_clean["end_date"] = pd.to_datetime(survey_clean["enddate"], errors="coerce").dt.tz_localize(ZoneInfo("America/New_York"))

# Step 3: Convert common "missing" strings into NA
survey_clean["age"] = survey_clean["age"].replace({"": np.nan, "Prefer not to say": np.nan})
survey_clean["gender"] = survey_clean["gender"].replace({"": np.nan, "Prefer not to say": np.nan})
survey_clean["partyid"] = survey_clean["partyid"].replace({"": np.nan})

# Step 4: Convert age to numeric (after cleaning)
survey_clean["age_num"] = pd.to_numeric(survey_clean["age"], errors="coerce")

# Step 5: Create duration in minutes (often easier to interpret)
survey_clean["duration_min"] = survey_clean["duration_seconds"] / 60.0

# Print a quick summary of key fields
print(
    survey_clean[
        ["responseid", "platform", "consent", "duration_seconds", "duration_min", "age", "age_num", "gender", "partyid"]
    ].head(12)
)

# -----------------------------------------------------------------------------
# Part 3: Codebooks --- documenting variables and values
# -----------------------------------------------------------------------------
codebook = pd.DataFrame(
    {
        "variable": [
            "responseid", "start_date", "end_date", "duration_seconds",
            "ipaddress", "platform", "prolific_id", "mturk_workerid",
            "consent", "age_num", "gender", "partyid",
            "attentioncheck", "q1_policy", "q2_policy", "q3_policy", "q4_policy"
        ],
        "description": [
            "Unique respondent identifier (from survey platform)",
            "Survey start time (parsed)",
            "Survey end time (parsed)",
            "Time spent in survey (seconds, from platform metadata)",
            "IP address (platform metadata; treat as sensitive)",
            "Recruitment platform indicator (Prolific / MTurk / Department pool)",
            "Prolific participant ID (if applicable)",
            "MTurk worker ID (if applicable)",
            "Consent to participate (Yes/No)",
            "Age in years (numeric; missing if refused/blank)",
            "Self-reported gender (cleaned; missing if refused/blank)",
            "Self-reported party identification (blank -> NA)",
            "Attention check response (used for quality screening)",
            "Policy attitude item 1 (Likert text)",
            "Policy attitude item 2 (Likert text)",
            "Policy attitude item 3 (Likert text)",
            "Policy attitude item 4 (Likert text)",
        ],
        "notes": [
            "Do not share raw IDs publicly in a replication package",
            "Timezone set to America/New_York for class consistency",
            "Timezone set to America/New_York for class consistency",
            "Can be noisy if respondent leaves tab open",
            "Often should not be released publicly; check IRB / policy",
            "Platform differences can affect data quality",
            "Blank for non-Prolific respondents",
            "Blank for non-MTurk respondents",
            "Use to filter out non-consenting cases",
            "Converted from string; 'Prefer not to say' -> NA",
            "Converted 'Prefer not to say' and blank -> NA",
            "Blank -> NA",
            "Incorrect response indicates low attention",
            "Consider mapping to numeric 1--5 for analysis",
            "Consider mapping to numeric 1--5 for analysis",
            "Consider mapping to numeric 1--5 for analysis",
            "Consider mapping to numeric 1--5 for analysis",
        ],
    }
)

print(codebook)
codebook.to_csv("outputs/codebook.csv", index=False)

# -----------------------------------------------------------------------------
# Part 4: Labeling --- variable labels and value labels
# -----------------------------------------------------------------------------
# pandas doesn't have built-in "labelled" variables like haven/labelled in R.
# We store labels as explicit metadata dictionaries.

var_labels = {
    "responseid": "Unique respondent identifier",
    "duration_seconds": "Survey duration (seconds)",
    "age_num": "Age in years (numeric)",
    "partyid": "Party identification",
    "attentioncheck": "Attention check response",
}
survey_clean.attrs["var_labels"] = var_labels

party_map = {"Democrat": 1, "Independent": 2, "Republican": 3}
survey_clean["party_id_num"] = survey_clean["partyid"].map(party_map)

value_labels_party = {1: "Democrat", 2: "Independent", 3: "Republican"}
survey_clean.attrs["value_labels_party_id_num"] = value_labels_party

likert_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

survey_clean["q1_num"] = survey_clean["q1_policy"].map(likert_map)
survey_clean["q2_num"] = survey_clean["q2_policy"].map(likert_map)
survey_clean["q3_num"] = survey_clean["q3_policy"].map(likert_map)
survey_clean["q4_num"] = survey_clean["q4_policy"].map(likert_map)

print(survey_clean[["responseid", "partyid", "party_id_num", "q1_policy", "q1_num"]].head(12))
print("Variable labels:", survey_clean.attrs["var_labels"])
print("Value labels (party_id_num):", survey_clean.attrs["value_labels_party_id_num"])

# -----------------------------------------------------------------------------
# Part 5: Quality checks --- speeding, attention, missingness, straightlining
# -----------------------------------------------------------------------------
survey_clean["flag_fast"] = survey_clean["duration_seconds"] < 120
survey_clean["flag_attention_fail"] = survey_clean["attentioncheck"] != "Strongly disagree"

key_vars = survey_clean[["age_num", "gender", "party_id_num", "q1_num", "q2_num", "q3_num", "q4_num"]]
survey_clean["missing_share"] = key_vars.isna().mean(axis=1)
survey_clean["flag_missing_high"] = survey_clean["missing_share"] > 0.30

survey_clean["flag_straightline"] = (
    (survey_clean["q1_num"] == survey_clean["q2_num"]) &
    (survey_clean["q2_num"] == survey_clean["q3_num"]) &
    (survey_clean["q3_num"] == survey_clean["q4_num"])
)

survey_clean["flag_no_consent"] = survey_clean["consent"] != "Yes"

dup_prolific = (
    survey_clean.loc[survey_clean["prolific_id"] != "", ["prolific_id"]]
    .value_counts()
    .reset_index(name="n")
)
dup_prolific = dup_prolific.loc[dup_prolific["n"] > 1]
print(dup_prolific)

flag_summary = pd.DataFrame(
    {
        "n_total": [len(survey_clean)],
        "n_fast": [survey_clean["flag_fast"].sum(skipna=True)],
        "n_attention_fail": [survey_clean["flag_attention_fail"].sum(skipna=True)],
        "n_missing_high": [survey_clean["flag_missing_high"].sum(skipna=True)],
        "n_straightline": [survey_clean["flag_straightline"].sum(skipna=True)],
        "n_no_consent": [survey_clean["flag_no_consent"].sum(skipna=True)],
    }
)
print(flag_summary)

flagged_view = survey_clean[
    [
        "responseid", "platform", "duration_seconds", "attentioncheck", "missing_share",
        "flag_fast", "flag_attention_fail", "flag_missing_high", "flag_straightline", "flag_no_consent",
    ]
].sort_values(by=["flag_no_consent", "flag_fast", "flag_attention_fail"], ascending=[False, False, False])

print(flagged_view.head(12))

# -----------------------------------------------------------------------------
# Part 6: Create an analysis-ready dataset (filter out low-quality cases)
# -----------------------------------------------------------------------------
survey_final = survey_clean.loc[
    (~survey_clean["flag_no_consent"]) &
    (~survey_clean["flag_fast"]) &
    (~survey_clean["flag_attention_fail"]) &
    (~survey_clean["flag_missing_high"])
].copy()

print("\n------------------------------")
print("Rows before/after filtering")
print("------------------------------")
print("Before:", len(survey_clean))
print("After: ", len(survey_final), "\n")

survey_final.to_csv("data_processed/survey_clean.csv", index=False)

# -----------------------------------------------------------------------------
# Part 7: Quick visual checks (plots)
# -----------------------------------------------------------------------------
plt.figure()
plt.hist(survey_clean["duration_seconds"].dropna(), bins=12)
plt.title("Survey Duration Distribution")
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/duration_hist.png", dpi=200)
plt.close()

plt.figure()
plt.hist(survey_clean["missing_share"].dropna(), bins=10)
plt.title("Missingness Share Across Key Variables")
plt.xlabel("Share missing (per respondent)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/missingness_hist.png", dpi=200)
plt.close()

platform_flags = (
    survey_clean.groupby("platform", dropna=False)
    .agg(
        n=("responseid", "size"),
        share_fast=("flag_fast", "mean"),
        share_attention_fail=("flag_attention_fail", "mean"),
    )
    .reset_index()
)
print(platform_flags)

platform_flags_long = platform_flags.melt(
    id_vars=["platform"],
    value_vars=["share_fast", "share_attention_fail"],
    var_name="flag_type",
    value_name="share_flagged",
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)

plot_fast = platform_flags_long.loc[platform_flags_long["flag_type"] == "share_fast"]
axes[0].bar(plot_fast["platform"], plot_fast["share_flagged"])
axes[0].set_title("share_fast")
axes[0].set_xlabel("Platform")
axes[0].set_ylabel("Share flagged")
axes[0].tick_params(axis="x", rotation=20)

plot_attn = platform_flags_long.loc[platform_flags_long["flag_type"] == "share_attention_fail"]
axes[1].bar(plot_attn["platform"], plot_attn["share_flagged"])
axes[1].set_title("share_attention_fail")
axes[1].set_xlabel("Platform")
axes[1].tick_params(axis="x", rotation=20)

fig.suptitle("Share Flagged by Platform (Toy Data)")
fig.tight_layout()
fig.savefig("figures/flags_by_platform.png", dpi=200)
plt.close(fig)

# -----------------------------------------------------------------------------
# Part 8: One simple analysis example (optional in class)
# -----------------------------------------------------------------------------
survey_final["policy_scale"] = survey_final[["q1_num", "q2_num", "q3_num", "q4_num"]].mean(axis=1, skipna=True)

scale_summary = (
    survey_final.groupby("partyid", dropna=False)
    .agg(
        n=("responseid", "size"),
        mean_scale=("policy_scale", "mean"),
        median_scale=("policy_scale", "median"),
    )
    .reset_index()
)

print(scale_summary)
scale_summary.to_csv("outputs/policy_scale_by_party.csv", index=False)
