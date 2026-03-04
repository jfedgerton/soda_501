###############################################################################
# Network Data & Graph Workflows
# Graph Analysis Tutorial with Synthetic Data (R)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# Coding lab focus:
#   1) Graph construction (from edge lists)
#   2) Centrality (what it is + how to compute it)
#   3) Community detection (Louvain) + evaluation vs "ground truth"
#   4) Node classification baseline (logistic regression on features)
#   5) Scalability tips (sparse matrices)
#
# Pre-class video: Network representation choices and pitfalls
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - Steps are explicit so students can follow and modify each piece.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed):
#   install.packages(c("igraph", "tidyverse", "Matrix", "nnet"))

library(igraph)
library(tidyverse)
library(Matrix)
library(nnet)

# Reproducibility
set.seed(123)

# -----------------------------------------------------------------------------
# Part 0: Representation Choices and Pitfalls (Conceptual comments)
# -----------------------------------------------------------------------------
# Before coding, students should decide:
# - Are edges directed or undirected?
# - Are there self-loops?
# - Is the graph weighted? (and what do weights mean?)
# - Are there multiple edges between the same nodes?
# - What is the unit of analysis: node, edge, dyad, subgraph, time-slice?
#
# Common pitfalls:
# - Mixing directed/undirected assumptions
# - Dropping isolates without realizing it changes inference/coverage
# - Confusing "absence of edge" with "missing data"
# - Using dense adjacency matrices for large graphs (memory blow-up)

# -----------------------------------------------------------------------------
# Part 1: Generate a Synthetic Network (Stochastic Block Model)
# -----------------------------------------------------------------------------
# Three communities (blocks)
block_sizes <- c(400, 350, 250)
num_blocks  <- length(block_sizes)
num_nodes   <- sum(block_sizes)

# SBM probability matrix
P <- matrix(c(
  0.06, 0.01, 0.005,
  0.01, 0.05, 0.008,
  0.005, 0.008, 0.04
), nrow = 3, byrow = TRUE)

# Generate graph using igraph's SBM
G <- sample_sbm(num_nodes, pref.matrix = P, block.sizes = block_sizes, directed = FALSE)

cat("Synthetic graph summary:\n")
cat("  Nodes:", vcount(G), "\n")
cat("  Edges:", ecount(G), "\n")

# Ground-truth community labels
true_labels <- rep(0:(num_blocks - 1), times = block_sizes)

# -----------------------------------------------------------------------------
# Part 1b: Create synthetic node features correlated with communities
# -----------------------------------------------------------------------------
num_features <- 8

centers <- matrix(c(
  2, 0, 0, 1, 0, 0, 1, 0,   # community 0 center
  0, 2, 0, 0, 1, 0, 0, 1,   # community 1 center
  0, 0, 2, 0, 0, 1, 1, 0    # community 2 center
), nrow = 3, byrow = TRUE)

noise_sd <- 0.8

X <- matrix(0, nrow = num_nodes, ncol = num_features)
X[true_labels == 0, ] <- sweep(matrix(rnorm(sum(true_labels == 0) * num_features,
                                             sd = noise_sd),
                                       ncol = num_features),
                                2, centers[1, ], "+")
X[true_labels == 1, ] <- sweep(matrix(rnorm(sum(true_labels == 1) * num_features,
                                             sd = noise_sd),
                                       ncol = num_features),
                                2, centers[2, ], "+")
X[true_labels == 2, ] <- sweep(matrix(rnorm(sum(true_labels == 2) * num_features,
                                             sd = noise_sd),
                                       ncol = num_features),
                                2, centers[3, ], "+")

# -----------------------------------------------------------------------------
# Part 2: Graph Construction Workflows (Edge list + adjacency)
# -----------------------------------------------------------------------------
# Step 1: Create an edge list from the graph
edge_list <- as_data_frame(G, what = "edges")
colnames(edge_list) <- c("u", "v")

cat("\nEdge list (first 10 rows):\n")
print(head(edge_list, 10))

# Step 2: Reconstruct graph from edge list
G2 <- graph_from_data_frame(edge_list, directed = FALSE)

cat("\nReconstructed graph summary:\n")
cat("  Nodes:", vcount(G2), "\n")
cat("  Edges:", ecount(G2), "\n")

# Step 3: Sparse adjacency matrix
A <- as_adjacency_matrix(G2, sparse = TRUE)

cat("\nAdjacency matrix:\n")
cat("  Shape:", nrow(A), "x", ncol(A), "\n")
cat("  Nonzeros:", nnzero(A), "\n")

# -----------------------------------------------------------------------------
# Part 3: Centrality (Examples + Interpretations)
# -----------------------------------------------------------------------------
# Step 1: Degree centrality (fast)
deg <- degree(G2)
cat("\nDegree summary:\n")
print(summary(deg))

hist(deg, breaks = 40,
     main = "Degree distribution (synthetic SBM)",
     xlab = "Degree", ylab = "Count of nodes")

# Step 2: Betweenness centrality
# Note: exact betweenness can be expensive for large graphs
betw <- betweenness(G2, directed = FALSE)

hist(betw, breaks = 40,
     main = "Betweenness centrality distribution",
     xlab = "Betweenness", ylab = "Count of nodes")

# Step 3: Eigenvector centrality
eig <- eigen_centrality(G2, directed = FALSE)$vector

hist(eig, breaks = 40,
     main = "Eigenvector centrality distribution",
     xlab = "Eigenvector centrality", ylab = "Count of nodes")

# -----------------------------------------------------------------------------
# Part 4: Community Detection (Louvain) + Evaluation
# -----------------------------------------------------------------------------
# Step 1: Louvain community detection
louvain_result <- cluster_louvain(G2)
louvain_labels <- membership(louvain_result) - 1  # 0-indexed to match true_labels

cat("\nCommunity detection summary:\n")
cat("  Number of Louvain communities:", max(louvain_labels) + 1, "\n")
cat("  Louvain community sizes:", table(louvain_labels), "\n")

# Step 2: Compare detected vs true communities (Adjusted Rand Index)
# igraph provides compare() for comparing community structures
ari <- compare(true_labels, louvain_labels, method = "adjusted.rand")
cat("\nAdjusted Rand Index (Louvain vs truth):", ari, "\n")

# -----------------------------------------------------------------------------
# Part 5: Baseline Model (Features Only) — Logistic Regression
# -----------------------------------------------------------------------------
# Step 1: Train/val/test splits
perm <- sample(seq_len(num_nodes))

train_size <- floor(0.60 * num_nodes)
val_size   <- floor(0.20 * num_nodes)
test_size  <- num_nodes - train_size - val_size

train_idx <- perm[1:train_size]
val_idx   <- perm[(train_size + 1):(train_size + val_size)]
test_idx  <- perm[(train_size + val_size + 1):num_nodes]

y <- factor(true_labels)

# Step 2: Fit multinomial logistic regression on features only
df_features <- data.frame(X)
df_features$y <- y

lr_fit <- multinom(y ~ ., data = df_features[train_idx, ], trace = FALSE, maxit = 200)

yhat_val  <- predict(lr_fit, newdata = df_features[val_idx, ])
yhat_test <- predict(lr_fit, newdata = df_features[test_idx, ])

val_acc  <- mean(yhat_val == y[val_idx])
test_acc <- mean(yhat_test == y[test_idx])

cat("\nBaseline (features-only) logistic regression:\n")
cat("  Validation accuracy:", val_acc, "\n")
cat("  Test accuracy:", test_acc, "\n")

# -----------------------------------------------------------------------------
# Part 6: Network Visualization
# -----------------------------------------------------------------------------
# Color nodes by true community
colors <- c("tomato", "steelblue", "forestgreen")
V(G)$color <- colors[true_labels + 1]

# Layout (use Fruchterman-Reingold for medium graphs)
cat("\nComputing layout (this may take a moment)...\n")
layout_fr <- layout_with_fr(G)

plot(G, vertex.size = 2, vertex.label = NA,
     edge.color = rgb(0, 0, 0, 0.1),
     layout = layout_fr,
     main = "Synthetic SBM graph (colored by true community)")
legend("bottomright", legend = paste("Community", 0:2),
       fill = colors, bty = "n")

# -----------------------------------------------------------------------------
# Part 7: Post-Analysis Summary
# -----------------------------------------------------------------------------
# Confusion table for logistic regression (test set)
confusion <- table(True = y[test_idx], Predicted = yhat_test)

cat("\nLogistic regression confusion table (test set):\n")
print(confusion)

cat("\nModel comparison:\n")
cat("  Baseline LR test accuracy:", test_acc, "\n")
cat("  (GCN implementation is in the Python version of this demo)\n")
