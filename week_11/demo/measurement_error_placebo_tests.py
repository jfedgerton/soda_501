###############################################################################
# Measurement Error + Placebo Tests Tutorial: Python
# Author: Jared Edgerton
# Date: (use your local date/time)
#
# This script demonstrates:
#   1) Classic measurement error in covariates (X_true vs X_observed)
#   2) How measurement error can bias regression estimates (including confounding)
#   3) Simple correction idea using a "validation subsample" (regression calibration)
#   4) Placebo tests as pipeline diagnostics:
#        - Outcome placebo (negative control outcome)
#        - Treatment permutation placebo (randomization inference)
#
# Teaching note:
# - Written as a sequential workflow so students can see how the pipeline unfolds.
# - No user-defined functions (no def ...).
# - Minimal control flow (loops are used for simulation/permutations).
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Recommended installs (run once in terminal):
#   pip install numpy pandas matplotlib statsmodels

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Reproducibility
np.random.seed(123)

# Create common project folders (safe to run repeatedly)
os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("src", exist_ok=True)

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

n = 5000

# True confounder
x_true = np.random.normal(loc=0.0, scale=1.0, size=n)

# Treatment assignment correlated with x_true (logistic link)
# (This creates confounding: D is not independent of x_true.)
logit_p = 1.0 * x_true
p = 1.0 / (1.0 + np.exp(-logit_p))
d = np.random.binomial(n=1, p=p, size=n)

# True outcome model
tau = 1.0     # true effect of D on Y
beta = 1.0    # effect of X_true on Y
eps_y = np.random.normal(loc=0.0, scale=1.0, size=n)

y = tau * d + beta * x_true + eps_y

# Placebo outcome (negative control outcome): NOT affected by D by construction
eps_pl = np.random.normal(loc=0.0, scale=1.0, size=n)
y_placebo = 0.0 * d + beta * x_true + eps_pl

df_base = pd.DataFrame(
    {
        "y": y,
        "y_placebo": y_placebo,
        "d": d,
        "x_true": x_true,
    }
)

print("\n--- Data preview ---")
print(df_base.head())
print("\nTreatment rate:", df_base["d"].mean())

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
#
# NOTE: This is synthetic, so we *know* x_true. In real work, "x_true" might be
# available only in a validation dataset or via higher-quality measurement.

sigma_u_grid = [0.0, 0.2, 0.5, 1.0, 2.0]
R = 30  # repetitions (to see variability from measurement error draws)

# Fix a "validation sample" index set (20% of observations)
validation_share = 0.20
validation_idx = np.random.choice(np.arange(n), size=int(validation_share * n), replace=False)
is_validation = np.zeros(n, dtype=bool)
is_validation[validation_idx] = True

# Storage
rows = []

print("\n--- Running measurement error simulations ---")
for sigma_u in sigma_u_grid:
    tau_oracle_list = []
    tau_naive_list = []
    tau_cal_list = []
    tau_placebo_list = []

    beta_oracle_list = []
    beta_naive_list = []
    beta_cal_list = []

    for r in range(R):
        # Draw measurement error and observed covariate
        u = np.random.normal(loc=0.0, scale=sigma_u, size=n)
        x_obs = x_true + u

        df = df_base.copy()
        df["x_obs"] = x_obs

        # (A) Oracle regression: y ~ d + x_true
        fit_oracle = smf.ols("y ~ d + x_true", data=df).fit()

        tau_oracle_list.append(fit_oracle.params["d"])
        beta_oracle_list.append(fit_oracle.params["x_true"])

        # (B) Naive regression with error-prone confounder: y ~ d + x_obs
        fit_naive = smf.ols("y ~ d + x_obs", data=df).fit()

        tau_naive_list.append(fit_naive.params["d"])
        beta_naive_list.append(fit_naive.params["x_obs"])

        # (C) Regression calibration (validation subsample):
        #     estimate x_true ~ x_obs on validation sample, then predict x_hat for all.
        df_val = df.loc[is_validation, ["x_true", "x_obs"]].copy()
        fit_cal = smf.ols("x_true ~ x_obs", data=df_val).fit()

        df["x_hat"] = fit_cal.predict(df[["x_obs"]])

        fit_calibrated = smf.ols("y ~ d + x_hat", data=df).fit()

        tau_cal_list.append(fit_calibrated.params["d"])
        beta_cal_list.append(fit_calibrated.params["x_hat"])

        # Outcome placebo: y_placebo ~ d + x_obs
        # If the pipeline is "finding effects" here, that suggests bias/artifacts.
        fit_placebo = smf.ols("y_placebo ~ d + x_obs", data=df).fit()
        tau_placebo_list.append(fit_placebo.params["d"])

    # Summaries per sigma_u
    rows.append(
        {
            "sigma_u": sigma_u,
            "tau_true": tau,
            "tau_oracle_mean": float(np.mean(tau_oracle_list)),
            "tau_naive_mean": float(np.mean(tau_naive_list)),
            "tau_cal_mean": float(np.mean(tau_cal_list)),
            "tau_placebo_mean": float(np.mean(tau_placebo_list)),
            "tau_oracle_q025": float(np.quantile(tau_oracle_list, 0.025)),
            "tau_oracle_q975": float(np.quantile(tau_oracle_list, 0.975)),
            "tau_naive_q025": float(np.quantile(tau_naive_list, 0.025)),
            "tau_naive_q975": float(np.quantile(tau_naive_list, 0.975)),
            "tau_cal_q025": float(np.quantile(tau_cal_list, 0.025)),
            "tau_cal_q975": float(np.quantile(tau_cal_list, 0.975)),
            "beta_true": beta,
            "beta_oracle_mean": float(np.mean(beta_oracle_list)),
            "beta_naive_mean": float(np.mean(beta_naive_list)),
            "beta_cal_mean": float(np.mean(beta_cal_list)),
        }
    )

    print(f"  done sigma_u={sigma_u}")

results = pd.DataFrame(rows)
print("\n--- Summary (means) ---")
print(results[["sigma_u", "tau_true", "tau_oracle_mean", "tau_naive_mean", "tau_cal_mean", "tau_placebo_mean"]])

results.to_csv("outputs/measurement_error_results.csv", index=False)

# -----------------------------------------------------------------------------
# Part 3: Plot how measurement error changes estimates
# -----------------------------------------------------------------------------
# Plot tau estimates vs sigma_u
plt.figure(figsize=(8, 5))
plt.plot(results["sigma_u"], results["tau_oracle_mean"], marker="o", label="Oracle: y ~ d + x_true")
plt.plot(results["sigma_u"], results["tau_naive_mean"], marker="o", label="Naive: y ~ d + x_obs")
plt.plot(results["sigma_u"], results["tau_cal_mean"], marker="o", label="Calibration: y ~ d + x_hat")
plt.plot(results["sigma_u"], results["tau_placebo_mean"], marker="o", label="Outcome placebo: y_pl ~ d + x_obs")
plt.axhline(tau, linestyle="--", label="True tau")
plt.title("Estimated treatment effect vs measurement error in confounder")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on d")
plt.legend()
plt.tight_layout()
plt.savefig("figures/measurement_error_tau_vs_sigma.png", dpi=200)
plt.close()

# Plot beta estimates vs sigma_u (confounder coefficient attenuation)
plt.figure(figsize=(8, 5))
plt.plot(results["sigma_u"], results["beta_oracle_mean"], marker="o", label="Oracle: coef on x_true")
plt.plot(results["sigma_u"], results["beta_naive_mean"], marker="o", label="Naive: coef on x_obs")
plt.plot(results["sigma_u"], results["beta_cal_mean"], marker="o", label="Calibration: coef on x_hat")
plt.axhline(beta, linestyle="--", label="True beta")
plt.title("Estimated confounder effect vs measurement error (attenuation)")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on confounder term")
plt.legend()
plt.tight_layout()
plt.savefig("figures/measurement_error_beta_vs_sigma.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Part 4: Treatment permutation placebo (randomization inference)
# -----------------------------------------------------------------------------
# Here we treat the observed estimate as a test statistic:
#   tau_hat_obs = coef on d in y ~ d + x_obs
# Then we build a null distribution by permuting d.

sigma_u_perm = 1.0
u_perm = np.random.normal(loc=0.0, scale=sigma_u_perm, size=n)
x_obs_perm = x_true + u_perm

df_perm_base = df_base.copy()
df_perm_base["x_obs"] = x_obs_perm

fit_obs = smf.ols("y ~ d + x_obs", data=df_perm_base).fit()
tau_hat_obs = float(fit_obs.params["d"])

print("\n--- Permutation placebo setup ---")
print("sigma_u used:", sigma_u_perm)
print("Observed tau_hat (naive model):", round(tau_hat_obs, 4))

B = 500
tau_perm = []

for b in range(B):
    d_perm = np.random.permutation(df_perm_base["d"].values)
    df_b = df_perm_base.copy()
    df_b["d_perm"] = d_perm

    fit_b = smf.ols("y ~ d_perm + x_obs", data=df_b).fit()
    tau_perm.append(float(fit_b.params["d_perm"]))

tau_perm = np.array(tau_perm)

# Empirical two-sided p-value
p_emp = (1.0 + np.sum(np.abs(tau_perm) >= np.abs(tau_hat_obs))) / (B + 1.0)
print("Empirical p-value (two-sided):", round(p_emp, 4))

perm_df = pd.DataFrame({"tau_perm": tau_perm})
perm_df.to_csv("outputs/permutation_tau_distribution.csv", index=False)

# Plot permutation distribution + observed line
plt.figure(figsize=(8, 5))
plt.hist(tau_perm, bins=30, alpha=0.8)
plt.axvline(tau_hat_obs, linestyle="--", linewidth=2, label=f"Observed tau_hat = {tau_hat_obs:.3f}")
plt.axvline(-tau_hat_obs, linestyle="--", linewidth=1)
plt.title(f"Treatment permutation placebo (sigma_u={sigma_u_perm})\nEmpirical p-value = {p_emp:.3f}")
plt.xlabel("Coefficient on permuted treatment")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/permutation_placebo_tau_hist.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# End of script
# -----------------------------------------------------------------------------
print("\nDone. Outputs written to:")
print("  outputs/measurement_error_results.csv")
print("  outputs/permutation_tau_distribution.csv")
print("  figures/measurement_error_tau_vs_sigma.png")
print("  figures/measurement_error_beta_vs_sigma.png")
print("  figures/permutation_placebo_tau_hist.png")
###############################################################################
