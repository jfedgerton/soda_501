###############################################################################
# Quantum Computing Tutorial (R — Non-Executable Teaching Script)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates (conceptually, with matrix math in R):
#   1) What a qubit is (state vectors + measurement)
#   2) Basic single-qubit gates (X, H, Z) as matrices
#   3) Multi-qubit states (tensor products via Kronecker product)
#   4) Entanglement via CNOT (Bell state)
#   5) Simulating measurement (sampling from probabilities)
#   6) Key pitfalls (exponential state size, measurement collapse)
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - Students should READ and RUN this script as an annotated walkthrough.
# - All quantum operations are done with base R matrix algebra.
#
# No special packages required (base R only).
###############################################################################

# Reproducibility
set.seed(123)

# -----------------------------------------------------------------------------
# Part 0: Why quantum computing looks "weird"
# -----------------------------------------------------------------------------
# Classical bit:
# - A classical bit is either 0 or 1.
#
# Qubit:
# - A qubit is a 2D complex vector (a "state vector") with unit length:
#
#     |psi> = alpha|0> + beta|1>
#
# where alpha and beta are complex numbers and:
#
#     |alpha|^2 + |beta|^2 = 1
#
# Measurement:
# - When you measure |psi> in the computational basis:
#   - you see 0 with probability |alpha|^2
#   - you see 1 with probability |beta|^2
# - After measurement, the state "collapses" to the outcome state.

# -----------------------------------------------------------------------------
# Part 1: State vectors for one qubit
# -----------------------------------------------------------------------------
# Computational basis states:
# - |0> = [1, 0]^T
# - |1> = [0, 1]^T

ket0 <- matrix(c(1, 0), ncol = 1)
ket1 <- matrix(c(0, 1), ncol = 1)

cat("\n--- Basis states ---\n")
cat("|0> =\n"); print(ket0)
cat("|1> =\n"); print(ket1)

# Example superposition:
# |psi> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
alpha <- 1 / sqrt(2)
beta  <- 1 / sqrt(2)
psi <- alpha * ket0 + beta * ket1

cat("\n--- Superposition state ---\n")
cat("|psi> = (1/sqrt(2))|0> + (1/sqrt(2))|1> =\n")
print(psi)

# Verify normalization: |alpha|^2 + |beta|^2 = 1
cat("Norm check (should be 1):", sum(Mod(psi)^2), "\n")

# Measurement probabilities
prob_0 <- Mod(alpha)^2
prob_1 <- Mod(beta)^2
cat("P(measure 0):", prob_0, "\n")
cat("P(measure 1):", prob_1, "\n")

# -----------------------------------------------------------------------------
# Part 2: Single-qubit gates (unitary matrices)
# -----------------------------------------------------------------------------
# Gates are matrices that transform state vectors:
#   |psi_out> = U |psi_in>

# X (NOT gate): flips |0> <-> |1>
X_gate <- matrix(c(0, 1, 1, 0), nrow = 2, byrow = TRUE)

# Z (phase flip): leaves |0> alone, flips sign of |1>
Z_gate <- matrix(c(1, 0, 0, -1), nrow = 2, byrow = TRUE)

# H (Hadamard): creates superposition from basis states
H_gate <- (1 / sqrt(2)) * matrix(c(1, 1, 1, -1), nrow = 2, byrow = TRUE)

cat("\n--- Single-qubit gates ---\n")
cat("X gate:\n"); print(X_gate)
cat("Z gate:\n"); print(Z_gate)
cat("H gate:\n"); print(round(H_gate, 4))

# Apply X to |0>: should give |1>
result_X0 <- X_gate %*% ket0
cat("\nX|0> =\n"); print(result_X0)

# Apply H to |0>: should give (|0> + |1>)/sqrt(2)
result_H0 <- H_gate %*% ket0
cat("H|0> =\n"); print(round(result_H0, 4))

# Apply H to |1>: should give (|0> - |1>)/sqrt(2)
result_H1 <- H_gate %*% ket1
cat("H|1> =\n"); print(round(result_H1, 4))

# Verify unitarity: H^T H = I
cat("\nH^T H (should be identity):\n")
print(round(t(H_gate) %*% H_gate, 10))

# -----------------------------------------------------------------------------
# Part 3: Two qubits (tensor products)
# -----------------------------------------------------------------------------
# A two-qubit state lives in a 4D complex vector space.
#
# Basis states: |00>, |01>, |10>, |11>
# Tensor product in R: kronecker(a, b)

ket00 <- kronecker(ket0, ket0)
ket01 <- kronecker(ket0, ket1)
ket10 <- kronecker(ket1, ket0)
ket11 <- kronecker(ket1, ket1)

cat("\n--- Two-qubit basis states ---\n")
cat("|00> =\n"); print(t(ket00))
cat("|01> =\n"); print(t(ket01))
cat("|10> =\n"); print(t(ket10))
cat("|11> =\n"); print(t(ket11))

# -----------------------------------------------------------------------------
# Part 4: Entanglement with CNOT (controlled NOT)
# -----------------------------------------------------------------------------
# CNOT: if control qubit is |1>, flip target qubit
CNOT <- matrix(c(
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, 1,
  0, 0, 1, 0
), nrow = 4, byrow = TRUE)

cat("\n--- CNOT gate ---\n")
print(CNOT)

# Bell state construction:
# Step 1: Start with |00>
state <- ket00
cat("\nStep 1: Start with |00>\n"); print(t(state))

# Step 2: Apply H to first qubit: (H tensor I)|00>
I2 <- diag(2)
HI <- kronecker(H_gate, I2)
state <- HI %*% state
cat("Step 2: Apply (H x I)|00> =\n"); print(t(round(state, 4)))

# Step 3: Apply CNOT
state <- CNOT %*% state
cat("Step 3: Apply CNOT =\n"); print(t(round(state, 4)))

cat("\nThis is the Bell state |Phi+> = (|00> + |11>)/sqrt(2)\n")

# Verify it matches the expected Bell state
bell_expected <- (1 / sqrt(2)) * (ket00 + ket11)
cat("Match expected Bell state:", all.equal(as.vector(state),
                                             as.vector(bell_expected)), "\n")

# -----------------------------------------------------------------------------
# Part 5: Simulating measurement of the Bell state
# -----------------------------------------------------------------------------
# Measurement probabilities: |amplitude|^2 for each basis state
probs <- Mod(state)^2
basis_labels <- c("|00>", "|01>", "|10>", "|11>")

cat("\n--- Measurement probabilities ---\n")
for (j in seq_along(basis_labels)) {
  cat(basis_labels[j], ":", round(probs[j], 4), "\n")
}

# Simulate 1000 measurements
n_shots <- 1000
outcomes <- sample(0:3, size = n_shots, replace = TRUE, prob = probs)
outcome_labels <- basis_labels[outcomes + 1]
counts <- table(outcome_labels)

cat("\n--- Simulated measurement results (", n_shots, "shots) ---\n")
print(counts)

cat("\nNote: You get approximately 50% |00> and 50% |11>,\n")
cat("and (almost) never |01> or |10>. This is entanglement.\n")

# Bar plot of measurement outcomes
barplot(counts, main = paste("Bell state measurement (", n_shots, "shots)"),
        xlab = "Outcome", ylab = "Count", col = "steelblue")

# -----------------------------------------------------------------------------
# Part 6: What makes quantum computing hard (key pitfalls)
# -----------------------------------------------------------------------------
# 1) Exponential state growth:
# - n qubits require 2^n complex amplitudes.
# - Show this concretely:
cat("\n--- Exponential state size ---\n")
for (nq in c(1, 2, 5, 10, 20, 30)) {
  cat(nq, "qubits ->", 2^nq, "amplitudes\n")
}
cat("30 qubits already means ~1 billion amplitudes.\n")

# 2) Measurement collapse:
# - You cannot "peek" at a quantum state without changing it.

# 3) Noise and decoherence:
# - Real hardware introduces errors; ideal math does not.

# 4) Algorithm design:
# - Quantum advantage requires careful algorithms (e.g., Grover, Shor).
# - Many problems do not get speedups.

# -----------------------------------------------------------------------------
# Practice Tasks (Students)
# -----------------------------------------------------------------------------
# 1) Conceptual:
# - Explain in words why H|0> produces a 50/50 measurement result.
#
# 2) Gate reasoning:
# - What does X do to (alpha|0> + beta|1>)?
# - What does Z do to (alpha|0> + beta|1>)?
#
# 3) Bell state:
# - Write out the algebra showing how H and CNOT create (|00> + |11>)/sqrt(2).
#
# 4) "Sampling view":
# - If you run the Bell circuit for 1000 shots, why do you expect ~500 "00" and
#   ~500 "11" (not exactly 500/500)?
#
# 5) Extension (optional, outside class):
# - Modify this script to create the other Bell states (Phi-, Psi+, Psi-).

###############################################################################
# Where to buy quantum computing time (cloud access)
###############################################################################
# 1) IBM Quantum Platform (IBM)
# 2) Amazon Braket (AWS)
# 3) Azure Quantum (Microsoft)
# 4) D-Wave Leap (D-Wave)
# 5) IonQ Quantum Cloud (IonQ direct)
# 6) Quantinuum (direct subscription)
# 7) Rigetti Quantum Cloud Services (Rigetti QCS)
#
# Teaching note:
# - "Buying time" usually means either (a) per-job/per-shot usage billing,
#   or (b) reserved QPU time blocks/subscriptions. Pricing and availability
#   change, so always check the current pricing pages.
###############################################################################
