#!/usr/bin/env python3
"""
Option 3: Neural network learns to predict bits of p from bits of N.

Pairwise MI is ~0 for middle bits. But a neural net can learn JOINT
dependencies across all bits simultaneously. The carry structure creates
global correlations that pairwise MI misses.

Experiment:
  1. Generate many (N, p) pairs where N = p*q, both k-bit primes
  2. Input: 2k bits of N
  3. Output: k bits of p (the smaller factor)
  4. Train a small MLP, measure per-bit accuracy
  5. Key question: does accuracy for MIDDLE bits exceed 50%?
     If yes → the network learned something about carry structure
     If no → even joint dependencies don't help

Scale: start at k=8 (16-bit N), go up to k=16 (32-bit N).
"""

import math
import random
import sys
import time
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# Use numpy-only MLP (no torch/tensorflow dependency)


class MLP:
    """Simple feedforward neural network using numpy."""
    def __init__(self, layer_sizes, lr=0.01):
        self.weights = []
        self.biases = []
        self.lr = lr
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(x, 0)  # ReLU
            self.activations.append(x)
        # Last layer: sigmoid for binary output
        x = x @ self.weights[-1] + self.biases[-1]
        x = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        self.activations.append(x)
        return x

    def backward(self, y_true):
        m = y_true.shape[0]
        # Output layer: binary cross-entropy gradient
        delta = self.activations[-1] - y_true  # (batch, out)

        for i in range(len(self.weights) - 1, -1, -1):
            dw = self.activations[i].T @ delta / m
            db = np.mean(delta, axis=0)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self.activations[i] > 0)  # ReLU grad

    def train_step(self, x, y):
        pred = self.forward(x)
        loss = -np.mean(y * np.log(pred + 1e-10) + (1-y) * np.log(1-pred + 1e-10))
        self.backward(y)
        return loss


def generate_data(k, num_samples):
    """Generate (N_bits, p_bits) pairs."""
    from sympy import nextprime

    X = np.zeros((num_samples, 2 * k), dtype=np.float32)
    Y = np.zeros((num_samples, k), dtype=np.float32)

    count = 0
    while count < num_samples:
        lo, hi = 1 << (k - 1), (1 << k) - 1
        p = nextprime(random.randint(lo, hi))
        if p > hi:
            continue
        q = nextprime(random.randint(lo, hi))
        if q > hi or q == p:
            continue
        if p > q:
            p, q = q, p
        N = p * q

        # Bits of N (2k bits)
        for j in range(2 * k):
            X[count, j] = (N >> j) & 1
        # Bits of p (k bits)
        for j in range(k):
            Y[count, j] = (p >> j) & 1

        count += 1

    return X, Y


def experiment(k, num_train=50000, num_test=5000, hidden=256, epochs=50, lr=0.005):
    """Train MLP to predict p bits from N bits."""
    print(f"\n{'='*70}")
    print(f"  NEURAL FACTORING: k={k} ({2*k}-bit N → {k}-bit p)")
    print(f"  {num_train} train, {num_test} test, hidden={hidden}, epochs={epochs}")
    print(f"{'='*70}")

    t0 = time.time()
    print("  Generating data...", end=" ")
    X_train, Y_train = generate_data(k, num_train)
    X_test, Y_test = generate_data(k, num_test)
    print(f"done ({time.time()-t0:.1f}s)")

    # Architecture: 2k → hidden → hidden → k
    net = MLP([2 * k, hidden, hidden, k], lr=lr)

    batch_size = 256

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(num_train)
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        total_loss = 0
        n_batches = 0
        for start in range(0, num_train, batch_size):
            xb = X_shuf[start:start+batch_size]
            yb = Y_shuf[start:start+batch_size]
            loss = net.train_step(xb, yb)
            total_loss += loss
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Test accuracy
            pred = net.forward(X_test)
            pred_bits = (pred > 0.5).astype(float)

            per_bit_acc = np.mean(pred_bits == Y_test, axis=0)
            overall_acc = np.mean(pred_bits == Y_test)
            avg_loss = total_loss / n_batches

            # Full number recovery
            full_correct = 0
            for i in range(num_test):
                p_pred = sum(int(pred_bits[i, j]) << j for j in range(k))
                p_true = sum(int(Y_test[i, j]) << j for j in range(k))
                if p_pred == p_true:
                    full_correct += 1

            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                  f"bit_acc={overall_acc:.3f} full_recover={full_correct}/{num_test}")

    # Final detailed per-bit analysis
    pred = net.forward(X_test)
    pred_bits = (pred > 0.5).astype(float)
    per_bit_acc = np.mean(pred_bits == Y_test, axis=0)

    print(f"\n  --- Per-bit accuracy ---")
    print(f"  {'bit':>4} | {'accuracy':>8} | {'above_50%':>9} | {'signal'}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*9}-+--------")
    for i in range(k):
        above = per_bit_acc[i] - 0.5
        signal = "SIGNAL" if above > 0.05 else ("weak" if above > 0.02 else "noise")
        print(f"  p[{i:2d}] | {per_bit_acc[i]:>7.3f} | {above:>+8.3f} | {signal}")

    # Bit categories
    trivial = [0, k-1]  # LSB and MSB are always 1
    low = list(range(1, k//4))
    mid = list(range(k//4, 3*k//4))
    high = list(range(3*k//4, k-1))

    for name, bits in [("trivial (0,k-1)", trivial), ("low (1..k/4)", low),
                        ("middle (k/4..3k/4)", mid), ("high (3k/4..k-1)", high)]:
        if bits:
            acc = np.mean([per_bit_acc[i] for i in bits])
            print(f"  {name}: avg accuracy = {acc:.3f} ({acc - 0.5:+.3f} above random)")

    return per_bit_acc


def generalization_test(k_train, k_test, num_train=50000, num_test=5000,
                        hidden=256, epochs=50, lr=0.003):
    """THE TEST THAT MATTERS: train on k_train bits, test on k_test bits.
    If accuracy holds → learning structure. If drops to 50% → memorized."""
    print(f"\n{'='*70}")
    print(f"  GENERALIZATION TEST: train on k={k_train}, test on k={k_test}")
    print(f"{'='*70}")

    # Train
    print("  Generating train data...", end=" ", flush=True)
    X_train, Y_train = generate_data(k_train, num_train)
    print("done")

    net = MLP([2 * k_train, hidden, hidden, k_train], lr=lr)
    batch_size = 256
    for epoch in range(epochs):
        perm = np.random.permutation(num_train)
        for start in range(0, num_train, batch_size):
            xb = X_train[perm[start:start+batch_size]]
            yb = Y_train[perm[start:start+batch_size]]
            net.train_step(xb, yb)

    # Test on SAME size (sanity check)
    X_same, Y_same = generate_data(k_train, num_test)
    pred_same = (net.forward(X_same) > 0.5).astype(float)
    acc_same = np.mean(pred_same == Y_same)

    # Test on DIFFERENT size
    # Need to pad/truncate input to match network dimensions
    # Input is 2*k bits. For larger k_test: truncate to 2*k_train bits (lose MSBs of N)
    # For smaller k_test: pad with zeros
    X_diff, Y_diff = generate_data(k_test, num_test)
    if k_test > k_train:
        # Truncate: use only first 2*k_train bits of N, first k_train bits of p
        X_diff_adj = X_diff[:, :2*k_train]
        Y_diff_adj = Y_diff[:, :k_train]
    else:
        # Pad with zeros
        X_diff_adj = np.zeros((num_test, 2*k_train), dtype=np.float32)
        X_diff_adj[:, :2*k_test] = X_diff
        Y_diff_adj = np.zeros((num_test, k_train), dtype=np.float32)
        Y_diff_adj[:, :k_test] = Y_diff

    pred_diff = (net.forward(X_diff_adj) > 0.5).astype(float)

    # Compare overlapping bits only
    overlap_k = min(k_train, k_test)
    acc_diff = np.mean(pred_diff[:, :overlap_k] == Y_diff_adj[:, :overlap_k])

    # Per-bit for overlapping range
    per_bit_same = np.mean(pred_same == Y_same, axis=0)
    per_bit_diff = np.mean(pred_diff[:, :overlap_k] == Y_diff_adj[:, :overlap_k], axis=0)

    print(f"\n  Same-size accuracy (k={k_train}→k={k_train}): {acc_same:.3f}")
    print(f"  Cross-size accuracy (k={k_train}→k={k_test}): {acc_diff:.3f}")
    print(f"  Random baseline: 0.500")
    print(f"  Generalization gap: {acc_same - acc_diff:.3f}")

    if acc_diff - 0.5 < 0.02:
        print(f"  >>> MEMORIZED: cross-size accuracy is random <<<")
    elif acc_diff - 0.5 > 0.05:
        print(f"  >>> STRUCTURAL: accuracy transfers across sizes! <<<")
    else:
        print(f"  >>> WEAK: marginal transfer <<<")

    # Per-bit detail
    print(f"\n  Per-bit (middle bits only, excluding trivial 0 and k-1):")
    for i in range(1, overlap_k - 1):
        s = per_bit_same[i] if i < len(per_bit_same) else 0
        d = per_bit_diff[i]
        print(f"    p[{i:2d}]: same={s:.3f} cross={d:.3f} gap={s-d:+.3f}")

    return acc_same, acc_diff


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Quick same-size test first
    for k in [8, 10, 12]:
        experiment(k, num_train=30000, num_test=3000, hidden=256, epochs=30, lr=0.005)

    # THE REAL TEST: generalization across sizes
    print("\n\n" + "=" * 70)
    print("  THE GENERALIZATION TESTS")
    print("=" * 70)
    generalization_test(8, 10, num_train=50000, epochs=40)
    generalization_test(8, 12, num_train=50000, epochs=40)
    generalization_test(10, 12, num_train=50000, epochs=40)
    generalization_test(10, 14, num_train=50000, epochs=40)
