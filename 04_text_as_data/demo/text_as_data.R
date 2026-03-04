###############################################################################
# Text as Data Pipelines Tutorial: R
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates (lightweight workflow):
#   1) Tokenization + basic text preprocessing (document-term matrix)
#   2) A classic topic model (LDA)
#   3) Word-embedding regression (word2vec -> document vectors -> Ridge regression)
#
# Week context:
# - Text as Data Pipelines
# - Coding lab: tokenization; embeddings; topic models.
# - Pre-class video: practical text pipeline architecture.
#
# Teaching note (important):
# - This file is intentionally written as a sequential workflow so students can
#   see how the pipeline unfolds.
# - No user-defined functions.
# - Minimal "magic": explicit steps and prints.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed):
#   install.packages(c("tidyverse", "tidytext", "topicmodels", "tm",
#                       "word2vec", "glmnet", "SnowballC", "wordcloud",
#                       "LDAvis", "jsonlite"))

library(tidyverse)
library(tidytext)
library(topicmodels)
library(tm)
library(word2vec)
library(glmnet)
library(SnowballC)
library(wordcloud)
library(jsonlite)

# Reproducibility
set.seed(123)

# Create project folders (safe to run repeatedly)
dir.create("04_text_as_data/demo/data_raw", recursive = TRUE, showWarnings = FALSE)
dir.create("04_text_as_data/demo/data_processed", recursive = TRUE, showWarnings = FALSE)
dir.create("04_text_as_data/demo/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("04_text_as_data/demo/outputs", recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Part 0: Load the CMU Movie Summary Corpus
# -----------------------------------------------------------------------------
# Expected files after extraction:
#   04_text_as_data/demo/MovieSummaries_extracted/MovieSummaries/plot_summaries.txt
#   04_text_as_data/demo/MovieSummaries_extracted/MovieSummaries/movie.metadata.tsv
#
# You can download here:
#   http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
#
# Extract manually or with:
#   untar("04_text_as_data/demo/MovieSummaries.tar.gz",
#         exdir = "04_text_as_data/demo/MovieSummaries_extracted")

# 0.1 Paths
archive_path <- "04_text_as_data/demo/MovieSummaries.tar.gz"
extract_dir  <- "04_text_as_data/demo/MovieSummaries_extracted"

dir.create(extract_dir, recursive = TRUE, showWarnings = FALSE)

# 0.2 Extract (if needed)
plots_path <- list.files(extract_dir, pattern = "plot_summaries\\.txt$",
                         recursive = TRUE, full.names = TRUE)

if (length(plots_path) == 0) {
  cat("\n--- Extracting MovieSummaries.tar.gz ---\n")
  untar(archive_path, exdir = extract_dir)
  plots_path <- list.files(extract_dir, pattern = "plot_summaries\\.txt$",
                           recursive = TRUE, full.names = TRUE)
}

meta_path <- list.files(extract_dir, pattern = "movie\\.metadata\\.tsv$",
                        recursive = TRUE, full.names = TRUE)

cat("\n--- Located extracted files ---\n")
cat("plots_path:", plots_path, "\n")
cat("meta_path: ", meta_path, "\n")

# 0.3 Load plot summaries (tab-separated: wikipedia_movie_id \t plot_summary)
plots <- read_tsv(plots_path[1], col_names = c("wikipedia_movie_id", "text"),
                  col_types = cols(.default = col_character()))
plots$wikipedia_movie_id <- as.integer(plots$wikipedia_movie_id)

cat("\n--- Plots loaded ---\n")
cat("plots shape:", nrow(plots), "x", ncol(plots), "\n")
print(head(plots))

# 0.4 Load metadata
meta <- read_tsv(meta_path[1], col_names = FALSE,
                 col_types = cols(.default = col_character()))

colnames(meta)[1] <- "wikipedia_movie_id"
colnames(meta)[3] <- "movie_name"
colnames(meta)[4] <- "release_date"
colnames(meta)[9] <- "genres_raw"

meta$wikipedia_movie_id <- as.integer(meta$wikipedia_movie_id)
meta <- meta %>% select(wikipedia_movie_id, movie_name, release_date, genres_raw)

cat("\n--- Metadata loaded ---\n")
cat("meta shape:", nrow(meta), "x", ncol(meta), "\n")

# 0.5 Merge plots + metadata
df <- plots %>% left_join(meta, by = "wikipedia_movie_id")

cat("\n--- Merged movie df ---\n")
cat("df shape:", nrow(df), "x", ncol(df), "\n")

# 0.6 Clean text (drop missing/very short summaries)
df$text_len <- nchar(df$text)
df <- df %>% filter(!is.na(text), text_len >= 200)

cat("\n--- After filtering short plots ---\n")
cat("df shape:", nrow(df), "x", ncol(df), "\n")
print(summary(df$text_len))

# 0.7 Create an outcome: y_outcome = 1 if "action" in genres, else 0
# Also create a rough topic label for exploration
df$y_outcome <- 0L
df$true_topic <- "other"

for (i in seq_len(nrow(df))) {
  g <- df$genres_raw[i]
  if (!is.na(g) && nchar(g) > 0) {
    g_lower <- tolower(g)
    if (grepl("action", g_lower)) {
      df$y_outcome[i] <- 1L
      df$true_topic[i] <- "action"
    } else if (grepl("comedy", g_lower)) {
      df$true_topic[i] <- "comedy"
    } else if (grepl("drama", g_lower)) {
      df$true_topic[i] <- "drama"
    } else if (grepl("horror", g_lower)) {
      df$true_topic[i] <- "horror"
    }
  }
}

cat("\n--- Outcome + rough topic labels ---\n")
print(table(df$y_outcome))
print(head(sort(table(df$true_topic), decreasing = TRUE), 10))

# 0.8 Downsample for classroom runtime
max_docs <- 800
if (nrow(df) > max_docs) {
  df <- df %>% slice_sample(n = max_docs)
}

# 0.9 Add doc_id and keep clean columns
df <- df %>%
  mutate(doc_id = row_number()) %>%
  select(doc_id, wikipedia_movie_id, movie_name, release_date, text, true_topic, y_outcome)

cat("\n--- Final movie corpus preview ---\n")
print(head(df))
cat("\n--- Corpus size ---\n")
cat(nrow(df), "x", ncol(df), "\n")

write_csv(df, "04_text_as_data/demo/data_raw/week_movie_corpus.csv")

# -----------------------------------------------------------------------------
# Part 1: Tokenization + basic preprocessing
# -----------------------------------------------------------------------------
# Build a document-term matrix (DTM) using the tm package.
# This mirrors the Python CountVectorizer approach.

df <- read_csv("04_text_as_data/demo/data_raw/week_movie_corpus.csv",
               show_col_types = FALSE)

corpus <- VCorpus(VectorSource(df$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, stripWhitespace)

dtm <- DocumentTermMatrix(corpus,
                          control = list(wordLengths = c(2, Inf),
                                         bounds = list(global = c(5, Inf))))

cat("\n--- Document-term matrix (counts) ---\n")
cat("Shape:", nrow(dtm), "x", ncol(dtm), "\n")
cat("Vocabulary size:", ncol(dtm), "\n")

# Top terms by total count
term_totals <- col_sums(dtm)
top_terms <- sort(term_totals, decreasing = TRUE)[1:15]
top_terms_df <- data.frame(term = names(top_terms), total_count = unname(top_terms))
cat("\n--- Top terms by total count ---\n")
print(top_terms_df)

write_csv(top_terms_df, "04_text_as_data/demo/outputs/week_top_terms.csv")

# Word cloud
png("04_text_as_data/demo/figures/wordcloud.png", width = 800, height = 400)
wordcloud(names(term_totals), freq = term_totals,
          max.words = 200, random.order = FALSE, colors = brewer.pal(8, "Dark2"))
dev.off()

# -----------------------------------------------------------------------------
# Part 2: Classic topic model (LDA)
# -----------------------------------------------------------------------------
n_topics <- 6

lda_model <- LDA(dtm, k = n_topics, method = "Gibbs",
                 control = list(seed = 123, burnin = 500, iter = 1000))

# Topic-word distributions
lda_terms <- terms(lda_model, 10)
cat("\n--- LDA topics: top words ---\n")
print(lda_terms)

# Document-topic proportions
doc_topic <- posterior(lda_model)$topics
df_lda <- df
df_lda$lda_topic <- apply(doc_topic, 1, which.max)
df_lda$lda_topic_prob <- apply(doc_topic, 1, max)

cat("\n--- LDA: dominant topic counts ---\n")
print(table(df_lda$lda_topic))

write_csv(df_lda, "04_text_as_data/demo/data_processed/week_with_lda_topics.csv")

# Bar plot of topic counts
png("04_text_as_data/demo/figures/week_lda_dominant_topic_counts.png", width = 800, height = 400)
barplot(table(df_lda$lda_topic),
        main = "LDA: Dominant Topic Counts (Movie Plots)",
        xlab = "Dominant topic", ylab = "Number of documents")
dev.off()

# -----------------------------------------------------------------------------
# Part 3: Word embedding regression (word2vec -> doc vectors -> Ridge regression)
# -----------------------------------------------------------------------------
# Tokenize text into one-token-per-line format for word2vec
tokenized <- df %>%
  mutate(text_clean = tolower(text)) %>%
  mutate(text_clean = str_replace_all(text_clean, "[^a-z ]", "")) %>%
  mutate(text_clean = str_squish(text_clean))

# Train word2vec model
# word2vec expects a character vector of sentences (one per document)
w2v_model <- word2vec(tokenized$text_clean, type = "skip-gram",
                      dim = 100, window = 5, min_count = 2,
                      threads = 4, iter = 5)

cat("\n--- Word2Vec vocabulary size ---\n")
cat(nrow(as.matrix(w2v_model)), "\n")

# Build document vectors by averaging word vectors per document
w2v_matrix <- as.matrix(w2v_model)

doc_vectors <- matrix(0, nrow = nrow(df), ncol = 100)

for (i in seq_len(nrow(tokenized))) {
  tokens <- strsplit(tokenized$text_clean[i], "\\s+")[[1]]
  in_vocab <- tokens[tokens %in% rownames(w2v_matrix)]

  if (length(in_vocab) > 0) {
    vecs <- w2v_matrix[in_vocab, , drop = FALSE]
    doc_vectors[i, ] <- colMeans(vecs)
  }
}

cat("\n--- Document embedding matrix ---\n")
cat("Shape:", nrow(doc_vectors), "x", ncol(doc_vectors), "\n")

# Train/test split (75/25)
n_obs <- nrow(df)
train_idx <- sample(seq_len(n_obs), size = floor(0.75 * n_obs))
test_idx <- setdiff(seq_len(n_obs), train_idx)

X_train <- doc_vectors[train_idx, ]
X_test  <- doc_vectors[test_idx, ]
y_train <- df$y_outcome[train_idx]
y_test  <- df$y_outcome[test_idx]

# Ridge regression (alpha = 0 in glmnet is ridge)
ridge_fit <- glmnet(X_train, y_train, alpha = 0, lambda = 1.0)

y_pred <- predict(ridge_fit, newx = X_test, s = 1.0)[, 1]

mae  <- mean(abs(y_test - y_pred))
rmse <- sqrt(mean((y_test - y_pred)^2))
ss_res <- sum((y_test - y_pred)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r2 <- 1 - ss_res / ss_tot

cat("\n--- Word2Vec regression (Ridge) out-of-sample performance ---\n")
cat("MAE: ", round(mae, 4), "\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("R^2: ", round(r2, 4), "\n")

metrics <- data.frame(model = "word2vec_ridge", mae = mae, rmse = rmse, r2 = r2)
write_csv(metrics, "04_text_as_data/demo/outputs/week_word2vec_regression_metrics.csv")

# Plot actual vs predicted
png("04_text_as_data/demo/figures/word2vec_regression_actual_vs_predicted.png",
    width = 800, height = 600)
plot(y_test, y_pred,
     xlab = "Actual Outcome (y_test)",
     ylab = "Predicted Outcome (y_pred)",
     main = "Ridge Regression: Actual vs. Predicted Values",
     pch = 16, col = rgb(0, 0, 0, 0.5))
abline(lm(y_pred ~ y_test), col = "red", lwd = 2)
grid()
dev.off()

###############################################################################
# End of script
###############################################################################
cat("\nDone. Outputs written to:\n")
cat("  04_text_as_data/demo/outputs/\n")
cat("  04_text_as_data/demo/figures/\n")
cat("  04_text_as_data/demo/data_processed/\n")
