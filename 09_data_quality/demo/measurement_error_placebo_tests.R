###############################################################################
# Measurement Error + Placebo Tests Tutorial: R
# Author: Jared Edgerton
#
# This script demonstrates:
#   1) Classic measurement error in covariates (X_true vs X_observed)
#   2) How measurement error can bias regression estimates (including confounding)
#   3) Simple correction idea using a "validation subsample" (regression calibration)
#   4) Placebo tests as pipeline diagnostics:
#        - Outcome placebo (negative control outcome)
#        - Treatment permutation placebo (randomization inference)
#
# Dependencies: base R only (no packages required)
#
# Teaching note:
# - Written as a sequential workflow so students can see how the pipeline unfolds.
# - No user-defined functions.
# - Minimal control flow (loops are used for simulation/permutations).
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
set.seed(123)

dir.create("data_raw", showWarnings = FALSE)
dir.create("data_processed", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)
dir.create("src", showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Part 1: A simple data generating process with confounding
# -----------------------------------------------------------------------------
# We simulate:
#   X_true: a true confounder (unobserved truth)
#   D: a treatment/exposure correlated with X_true
#   Y: an outcome affected by both D and X_true
#
# Then we observe:
#   X_obs = X_true + U   (measurement error in X)
#
# Key intuition:
# - If X_true is a confounder, we need to control for it to estimate the effect of D.
# - If we only have a noisy measure X_obs, we do not fully control for confounding.
# - As measurement error increases, bias in the estimated effect of D can increase.

n <- 5000

# True confounder
x_true <- rnorm(n, mean = 0, sd = 1)

# Treatment assignment correlated with x_true (logistic link)
# (This creates confounding: D is not independent of x_true.)
logit_p <- 1.0 * x_true
p <- 1 / (1 + exp(-logit_p))
d <- rbinom(n, size = 1, prob = p)

# True outcome model
tau <- 1.0     # true effect of D on Y
beta <- 1.0    # effect of X_true on Y
eps_y <- rnorm(n, mean = 0, sd = 1)

y <- tau * d + beta * x_true + eps_y

# Placebo outcome (negative control outcome): NOT affected by D by construction
eps_pl <- rnorm(n, mean = 0, sd = 1)
y_placebo <- 0.0 * d + beta * x_true + eps_pl

df_base <- data.frame(
  y = y,
  y_placebo = y_placebo,
  d = d,
  x_true = x_true
)

cat("\n--- Data preview ---\n")
print(head(df_base))
cat("\nTreatment rate:", round(mean(df_base$d), 4), "\n")

# -----------------------------------------------------------------------------
# Part 2: Measurement error settings
# -----------------------------------------------------------------------------
# We will vary the measurement error in X:
#   X_obs = X_true + U,  U ~ Normal(0, sigma_u)
#
# For each sigma_u, we will estimate:
#   (A) Oracle model: y ~ d + x_true           (best case; uses truth)
#   (B) Naive model:  y ~ d + x_obs            (what you do with a noisy measure)
#   (C) Calibration:  estimate x_true ~ x_obs in a validation sample, predict x_hat,
#                    then run y ~ d + x_hat
#
# We will also run an outcome placebo:
#   y_placebo ~ d + x_obs

sigma_u_grid <- c(0.0, 0.2, 0.5, 1.0, 2.0)
R <- 30  # repetitions (to see variability from measurement error draws)

# Fix a "validation sample" index set (20% of observations)
validation_share <- 0.20
validation_idx <- sample(1:n, size = floor(validation_share * n), replace = FALSE)
is_validation <- rep(FALSE, n)
is_validation[validation_idx] <- TRUE

# Storage
results_list <- list()

cat("\n--- Running measurement error simulations ---\n")
for (sigma_u in sigma_u_grid) {
  tau_oracle_vec <- numeric(R)
  tau_naive_vec <- numeric(R)
  tau_cal_vec <- numeric(R)
  tau_placebo_vec <- numeric(R)

  beta_oracle_vec <- numeric(R)
  beta_naive_vec <- numeric(R)
  beta_cal_vec <- numeric(R)

  for (r in 1:R) {
    # Draw measurement error and observed covariate
    u <- rnorm(n, mean = 0, sd = sigma_u)
    x_obs <- x_true + u

    df <- df_base
    df$x_obs <- x_obs

    # (A) Oracle regression: y ~ d + x_true
    fit_oracle <- lm(y ~ d + x_true, data = df)
    tau_oracle_vec[r] <- coef(fit_oracle)["d"]
    beta_oracle_vec[r] <- coef(fit_oracle)["x_true"]

    # (B) Naive regression with error-prone confounder: y ~ d + x_obs
    fit_naive <- lm(y ~ d + x_obs, data = df)
    tau_naive_vec[r] <- coef(fit_naive)["d"]
    beta_naive_vec[r] <- coef(fit_naive)["x_obs"]

    # (C) Regression calibration (validation subsample):
    #     estimate x_true ~ x_obs on validation sample, predict x_hat for all.
    df_val <- df[is_validation, ]
    fit_cal <- lm(x_true ~ x_obs, data = df_val)
    df$x_hat <- predict(fit_cal, newdata = df)

    fit_calibrated <- lm(y ~ d + x_hat, data = df)
    tau_cal_vec[r] <- coef(fit_calibrated)["d"]
    beta_cal_vec[r] <- coef(fit_calibrated)["x_hat"]

    # Outcome placebo: y_placebo ~ d + x_obs
    fit_placebo <- lm(y_placebo ~ d + x_obs, data = df)
    tau_placebo_vec[r] <- coef(fit_placebo)["d"]
  }

  # Summaries per sigma_u
  results_list[[length(results_list) + 1]] <- data.frame(
    sigma_u = sigma_u,
    tau_true = tau,
    tau_oracle_mean = mean(tau_oracle_vec),
    tau_naive_mean = mean(tau_naive_vec),
    tau_cal_mean = mean(tau_cal_vec),
    tau_placebo_mean = mean(tau_placebo_vec),
    tau_oracle_q025 = quantile(tau_oracle_vec, 0.025),
    tau_oracle_q975 = quantile(tau_oracle_vec, 0.975),
    tau_naive_q025 = quantile(tau_naive_vec, 0.025),
    tau_naive_q975 = quantile(tau_naive_vec, 0.975),
    tau_cal_q025 = quantile(tau_cal_vec, 0.025),
    tau_cal_q975 = quantile(tau_cal_vec, 0.975),
    beta_true = beta,
    beta_oracle_mean = mean(beta_oracle_vec),
    beta_naive_mean = mean(beta_naive_vec),
    beta_cal_mean = mean(beta_cal_vec)
  )

  cat("  done sigma_u =", sigma_u, "\n")
}

results <- do.call(rbind, results_list)
rownames(results) <- NULL

cat("\n--- Summary (means) ---\n")
print(results[, c("sigma_u", "tau_true", "tau_oracle_mean", "tau_naive_mean",
                   "tau_cal_mean", "tau_placebo_mean")])

write.csv(results, "outputs/measurement_error_results.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# Part 3: Plot how measurement error changes estimates
# -----------------------------------------------------------------------------
# Plot tau estimates vs sigma_u
png("figures/measurement_error_tau_vs_sigma.png", width = 800, height = 500, res = 150)
plot(results$sigma_u, results$tau_oracle_mean, type = "b", pch = 16,
     ylim = c(-0.2, 2.0), xlab = "Measurement error SD (sigma_u)",
     ylab = "Estimated coefficient on d",
     main = "Estimated treatment effect vs measurement error in confounder")
lines(results$sigma_u, results$tau_naive_mean, type = "b", pch = 17, col = "red")
lines(results$sigma_u, results$tau_cal_mean, type = "b", pch = 18, col = "blue")
lines(results$sigma_u, results$tau_placebo_mean, type = "b", pch = 15, col = "darkgreen")
abline(h = tau, lty = 2, col = "gray40")
legend("topleft", legend = c("Oracle: y ~ d + x_true", "Naive: y ~ d + x_obs",
                              "Calibration: y ~ d + x_hat", "Outcome placebo",
                              "True tau"),
       col = c("black", "red", "blue", "darkgreen", "gray40"),
       pch = c(16, 17, 18, 15, NA), lty = c(1, 1, 1, 1, 2), cex = 0.7)
dev.off()

# Plot beta estimates vs sigma_u (confounder coefficient attenuation)
png("figures/measurement_error_beta_vs_sigma.png", width = 800, height = 500, res = 150)
plot(results$sigma_u, results$beta_oracle_mean, type = "b", pch = 16,
     ylim = c(0, 1.2), xlab = "Measurement error SD (sigma_u)",
     ylab = "Estimated coefficient on confounder term",
     main = "Estimated confounder effect vs measurement error (attenuation)")
lines(results$sigma_u, results$beta_naive_mean, type = "b", pch = 17, col = "red")
lines(results$sigma_u, results$beta_cal_mean, type = "b", pch = 18, col = "blue")
abline(h = beta, lty = 2, col = "gray40")
legend("bottomleft", legend = c("Oracle: coef on x_true", "Naive: coef on x_obs",
                                 "Calibration: coef on x_hat", "True beta"),
       col = c("black", "red", "blue", "gray40"),
       pch = c(16, 17, 18, NA), lty = c(1, 1, 1, 2), cex = 0.7)
dev.off()

# -----------------------------------------------------------------------------
# Part 4: Treatment permutation placebo (randomization inference)
# -----------------------------------------------------------------------------
# Here we treat the observed estimate as a test statistic:
#   tau_hat_obs = coef on d in y ~ d + x_obs
# Then we build a null distribution by permuting d.

sigma_u_perm <- 1.0
u_perm <- rnorm(n, mean = 0, sd = sigma_u_perm)
x_obs_perm <- x_true + u_perm

df_perm_base <- df_base
df_perm_base$x_obs <- x_obs_perm

fit_obs <- lm(y ~ d + x_obs, data = df_perm_base)
tau_hat_obs <- coef(fit_obs)["d"]

cat("\n--- Permutation placebo setup ---\n")
cat("sigma_u used:", sigma_u_perm, "\n")
cat("Observed tau_hat (naive model):", round(tau_hat_obs, 4), "\n")

B <- 500
tau_perm <- numeric(B)

for (b in 1:B) {
  d_perm <- sample(df_perm_base$d)
  df_b <- df_perm_base
  df_b$d_perm <- d_perm

  fit_b <- lm(y ~ d_perm + x_obs, data = df_b)
  tau_perm[b] <- coef(fit_b)["d_perm"]
}

# Empirical two-sided p-value
p_emp <- (1 + sum(abs(tau_perm) >= abs(tau_hat_obs))) / (B + 1)
cat("Empirical p-value (two-sided):", round(p_emp, 4), "\n")

perm_df <- data.frame(tau_perm = tau_perm)
write.csv(perm_df, "outputs/permutation_tau_distribution.csv", row.names = FALSE)

# Plot permutation distribution + observed line
png("figures/permutation_placebo_tau_hist.png", width = 800, height = 500, res = 150)
hist(tau_perm, breaks = 30, col = "steelblue", border = "white",
     main = paste0("Treatment permutation placebo (sigma_u=", sigma_u_perm,
                   ")\nEmpirical p-value = ", round(p_emp, 3)),
     xlab = "Coefficient on permuted treatment", ylab = "Count")
abline(v = tau_hat_obs, lty = 2, lwd = 2, col = "red")
abline(v = -tau_hat_obs, lty = 2, lwd = 1, col = "red")
legend("topright", legend = paste0("Observed tau_hat = ", round(tau_hat_obs, 3)),
       lty = 2, lwd = 2, col = "red", cex = 0.8)
dev.off()

# -----------------------------------------------------------------------------
# End of script
# -----------------------------------------------------------------------------
cat("\nDone. Outputs written to:\n")
cat("  outputs/measurement_error_results.csv\n")
cat("  outputs/permutation_tau_distribution.csv\n")
cat("  figures/measurement_error_tau_vs_sigma.png\n")
cat("  figures/measurement_error_beta_vs_sigma.png\n")
cat("  figures/permutation_placebo_tau_hist.png\n")
###############################################################################
