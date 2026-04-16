###############################################################################
# Deterministic and Probabilistic Matching Tutorial + Assignment (R)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates:
#   1) Generating two "messy" datasets that represent the same people
#   2) Deterministic (exact) matching using merge()
#   3) Probabilistic matching using the fastLink package
#   4) How match thresholds affect match counts and match quality
#   5) Using string distance (Levenshtein) to evaluate match quality
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
# Install (if needed) and load the necessary libraries.
# NOTE: For teaching, we keep installation lines commented out.
#
# install.packages(c("fastLink", "dplyr", "ggplot2", "stringdist"))
library(fastLink)
library(dplyr)
library(ggplot2)
library(stringdist)

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

set.seed(123)      # Reproducibility (use the same seed each time)
n <- 10000         # Target size (before distinct())

# -----------------------------------------------------------------------------
# Step 1: Create Dataset A (df_a)
# -----------------------------------------------------------------------------
df_a <- data.frame(
  id = 1:n,
  firstname = sample(
    c("John", "Jane", "Michael", "Emily", "David", "Sarah", "William", "Emma", "James", "Olivia"),
    n, replace = TRUE
  ),
  lastname = sample(
    c("Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"),
    n, replace = TRUE
  ),
  birthyear = sample(1970:2000, n, replace = TRUE),
  zipcode = sample(10000:20000, n, replace = TRUE)
) %>%
  distinct()

# -----------------------------------------------------------------------------
# Step 2: Create Dataset B (df_b) as a copy of df_a, then add "noise"
# -----------------------------------------------------------------------------
df_b <- df_a

# Decide which rows will be modified (25% probability each type of noise)
mod_firstname <- runif(nrow(df_b)) < 0.25
mod_lastname  <- runif(nrow(df_b)) < 0.25
mod_birthyear <- runif(nrow(df_b)) < 0.25

# ---- 2A: Add typos to FIRST NAMES (only for rows selected by mod_firstname) ----
idx_firstname <- which(mod_firstname)

for (i in idx_firstname) {

  firstname <- df_b$firstname[i]
  chars <- strsplit(firstname, "")[[1]]

  # Choose how many characters to replace (sample 1..length(chars))
  num_replace <- sample(1:length(chars), 1)

  # Choose positions to replace
  positions <- sample(1:length(chars), num_replace)

  # Replace selected positions with random letters
  for (pos in positions) {
    chars[pos] <- sample(letters, 1)
  }

  df_b$firstname[i] <- paste0(chars, collapse = "")
}

# ---- 2B: Add typos to LAST NAMES (only for rows selected by mod_lastname) ----
idx_lastname <- which(mod_lastname)

for (i in idx_lastname) {

  lastname <- df_b$lastname[i]
  chars <- strsplit(lastname, "")[[1]]

  # Choose how many characters to replace (sample 1..length(chars))
  num_replace <- sample(1:length(chars), 1)

  # Choose positions to replace
  positions <- sample(1:length(chars), num_replace)

  # Replace selected positions with random letters
  for (pos in positions) {
    chars[pos] <- sample(letters, 1)
  }

  df_b$lastname[i] <- paste0(chars, collapse = "")
}

# ---- 2C: Shift BIRTH YEAR slightly (only for rows selected by mod_birthyear) ----
idx_birthyear <- which(mod_birthyear)

# Add a small random shift: -2, -1, 0, 1, 2
birthyear_shift <- sample(-2:2, length(idx_birthyear), replace = TRUE)
df_b$birthyear[idx_birthyear] <- df_b$birthyear[idx_birthyear] + birthyear_shift

# -----------------------------------------------------------------------------
# Step 3: Save datasets to CSV (so students can load them like "real" files)
# -----------------------------------------------------------------------------
write.csv(df_a, "dataset_a.csv", row.names = FALSE)
write.csv(df_b, "dataset_b.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# Part 1: Student Assignment Starts Here
# -----------------------------------------------------------------------------
# Your task is to perform both deterministic and probabilistic matching on these
# two datasets. Follow the steps below and answer the questions at the end.

# -----------------------------------------------------------------------------
# Step 1: Load the datasets
# -----------------------------------------------------------------------------
df_a <- read.csv("dataset_a.csv")
df_b <- read.csv("dataset_b.csv")

# -----------------------------------------------------------------------------
# Step 2: Examine the datasets
# -----------------------------------------------------------------------------
# summary() gives distributions and missingness
# head() shows the first few rows

summary(df_a)
head(df_a)

summary(df_b)
head(df_b)

# -----------------------------------------------------------------------------
# Step 3: Deterministic matching (exact matching)
# -----------------------------------------------------------------------------
# Deterministic matching means:
# - A record in df_a matches a record in df_b ONLY IF all chosen fields match exactly.
#
# In R, merge(..., by = c(...)) is the standard exact-match join.

det_matches <- merge(
  df_a, df_b,
  by = c("firstname", "lastname", "birthyear", "zipcode")
)

print(paste("Number of deterministic matches:", nrow(det_matches)))

# -----------------------------------------------------------------------------
# Step 4: Probabilistic matching using fastLink
# -----------------------------------------------------------------------------
# Probabilistic matching means:
# - We do NOT require exact matches on all fields.
# - Instead, we estimate the probability that a pair of records is a "true match"
#   even if some fields disagree (e.g., typos).
#
# fastLink() fits a probabilistic linkage model and returns posterior match probs.

fl_out <- fastLink(
  dfA = df_a,
  dfB = df_b,
  varnames = c("firstname", "lastname", "birthyear", "zipcode"),
  return.all = TRUE
)

# -----------------------------------------------------------------------------
# Step 5: Analyze how the match threshold affects the number of matches
# -----------------------------------------------------------------------------
# fastLink produces posterior match probabilities (0 to 1).
# We choose a threshold:
# - Higher threshold -> fewer matches, but likely higher quality
# - Lower threshold  -> more matches, but likely lower quality

threshold_grid <- seq(0, 1, 0.01)

match_counts <- c()

for (th in threshold_grid) {

  matches_th <- getMatches(
    dfA = df_a,
    dfB = df_b,
    fl.out = fl_out,
    threshold.match = th
  )

  match_counts <- c(match_counts, nrow(matches_th))
}

count_of_matches <- data.frame(
  threshold = threshold_grid,
  matches = match_counts
)

ggplot(count_of_matches, aes(x = threshold, y = matches)) +
  geom_line() +
  labs(
    x = "Threshold level (posterior probability cutoff)",
    y = "Count of matches returned"
  ) +
  theme_bw()

# -----------------------------------------------------------------------------
# Step 6: Match quality vs posterior probability (string distance diagnostics)
# -----------------------------------------------------------------------------
# Idea:
# - Take a very low threshold so we get "almost everything"
# - Then split matches into posterior-probability bins:
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

matches_low <- getMatches(
  dfA = df_a,
  dfB = df_b,
  fl.out = fl_out,
  threshold.match = 0.000001
)

comp_data_by_match <- data.frame()

for (i in seq(0.1, 1, 0.1)) {

  temp_data <- matches_low %>%
    filter(
      posterior > i - 0.1,
      posterior <= i
    )

  # Skip empty bins — not all posterior probability ranges will have matches
  if (nrow(temp_data) == 0) next

  df_a_temp <- df_a %>% filter(id %in% temp_data$id)
  df_b_temp <- df_b %>% filter(id %in% temp_data$id)

  compare_data <- df_a_temp %>%
    inner_join(df_b_temp, by = "id", suffix = c(".a", ".b"))

  if (nrow(compare_data) == 0) next

  # Levenshtein distance (lv) measures how many edits it takes to transform one string into another
  string_dist_fname <- stringdist(compare_data$firstname.a, compare_data$firstname.b, method = "lv")
  string_dist_lname <- stringdist(compare_data$lastname.a, compare_data$lastname.b, method = "lv")

  # Birthyear distance: absolute difference
  byear_dist <- abs(compare_data$birthyear.a - compare_data$birthyear.b)

  comp_data_by_match <- bind_rows(
    comp_data_by_match,
    data.frame(
      first_name_distance = string_dist_fname,
      last_name_distance = string_dist_lname,
      birth_year_distance = byear_dist,
      threshold_bin = paste0(i - 0.1, "-", i),
      count_in_bin = nrow(compare_data)
    )
  )
}

# Optional visualization: average distance by posterior bin
avg_dist_by_bin <- comp_data_by_match %>%
  group_by(threshold_bin) %>%
  summarize(
    mean_first = mean(first_name_distance),
    mean_last  = mean(last_name_distance),
    mean_birth = mean(birth_year_distance),
    n = max(count_in_bin),
    .groups = "drop"
  )

print(avg_dist_by_bin)

# ggplot(avg_dist_by_bin, aes(x = threshold_bin, y = mean_first, group = 1)) +
#   geom_line() +
#   geom_point() +
#   labs(
#     x = "Posterior probability bin",
#     y = "Mean Levenshtein distance (first name)",
#     title = "Match quality vs posterior probability (first name distance)"
#   ) +
#   theme_bw()
# 
# ggplot(avg_dist_by_bin, aes(x = threshold_bin, y = mean_last, group = 1)) +
#   geom_line() +
#   geom_point() +
#   labs(
#     x = "Posterior probability bin",
#     y = "Mean Levenshtein distance (last name)",
#     title = "Match quality vs posterior probability (last name distance)"
#   ) +
#   theme_bw()
# 
# ggplot(avg_dist_by_bin, aes(x = threshold_bin, y = mean_birth, group = 1)) +
#   geom_line() +
#   geom_point() +
#   labs(
#     x = "Posterior probability bin",
#     y = "Mean absolute difference (birth year)",
#     title = "Match quality vs posterior probability (birth year difference)"
#   ) +
#   theme_bw()
