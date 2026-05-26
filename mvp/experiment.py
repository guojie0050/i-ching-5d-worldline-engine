#!/usr/bin/env python3
"""
V3 实验: 易经贝叶斯世界模型 — 数据效率验证
============================================

5 模型 × 5 种子 × 6 训练量 = 150 组独立实验。
非平稳 Markov 环境 (4 机制切换), 95% 置信区间。

用法:
    python experiment.py

输出:
    - iching_efficiency.png           ← 数据效率曲线
    - iching_efficiency_ablation.png  ← 先验消融柱状图
    - results/v3_results.csv          ← 原始数据
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import csv
import os
warnings.filterwarnings("ignore")

# Import model classes
from model import (
    IChingBayesianModel, MarkovModel, NeuralNetModel,
    HEXAGRAM_AFFINITIES, N_HEXAGRAMS, N_WEATHER, WEATHER_TYPES,
)

# ============================================================================
# Configuration
# ============================================================================

LABELS = [
    "IChing-Bayes", "IChing-Uniform", "IChing-Random",
    "Markov", "Neural-Net",
]
SIZES = [100, 200, 500, 1000, 2000, 3000]
EVAL_WINDOW = 200
SEEDS = [42, 123, 456, 789, 1011]
PRIOR_STR = 5.0
TEMPERATURE = 0.3
NN_LR = 0.005

# ============================================================================
# Weather Generator: Multi-Regime Non-Stationary Markov
# ============================================================================


def generate_weather(n_days, seed):
    """
    4 机制非平稳 Markov 天气生成器。
    每 80-250 天切换转移矩阵, 模拟多变的真实世界动态。
    """
    rng = np.random.default_rng(seed)

    T_base = np.array([
        [0.42, 0.12, 0.08, 0.03, 0.03, 0.18, 0.06, 0.08],
        [0.10, 0.38, 0.08, 0.06, 0.12, 0.04, 0.14, 0.08],
        [0.10, 0.12, 0.34, 0.10, 0.08, 0.08, 0.08, 0.10],
        [0.06, 0.14, 0.10, 0.28, 0.22, 0.04, 0.10, 0.06],
        [0.06, 0.14, 0.08, 0.14, 0.30, 0.04, 0.16, 0.08],
        [0.22, 0.06, 0.06, 0.03, 0.04, 0.42, 0.10, 0.07],
        [0.06, 0.14, 0.06, 0.08, 0.14, 0.08, 0.32, 0.12],
        [0.10, 0.12, 0.08, 0.05, 0.10, 0.06, 0.12, 0.37],
    ])

    modifiers = [
        (5, 0.25, 0, 0.15),   # R0: hot+clear
        (4, 0.25, 3, 0.20),   # R1: rain+thunder
        (2, 0.30, 1, 0.15),   # R2: wind+overcast
        (7, 0.25, 6, 0.15),   # R3: fog+humid
    ]

    def make_regime_T(regime_idx):
        T = T_base.copy()
        w1, b1, w2, b2 = modifiers[regime_idx]
        T[:, w1] += b1
        T[:, w2] += b2
        boost = b1 + b2
        others = [w for w in range(N_WEATHER) if w not in (w1, w2)]
        for w in others:
            T[:, w] -= boost / len(others)
        T = np.maximum(T, 0.005)
        T /= T.sum(axis=1, keepdims=True)
        return T

    regime_Ts = [make_regime_T(i) for i in range(4)]

    # Generate regime schedule
    durations = []
    remaining = n_days
    cur = 0
    while remaining > 0:
        dur = rng.integers(80, 250)
        dur = min(dur, remaining)
        durations.append((cur, dur))
        remaining -= dur
        cur = (cur + 1) % 4

    w = np.zeros(n_days, dtype=int)
    w[0] = 0
    pos = 0
    for regime, dur in durations:
        T = regime_Ts[regime]
        for i in range(dur):
            t = pos + i
            if t == 0:
                continue
            Tt = T.copy()
            ph = 2.0 * np.pi * (t % 365) / 365.0
            su = max(0.0, np.sin(ph - np.pi / 2))
            wi = max(0.0, -np.sin(ph - np.pi / 2))
            Tt[:, 5] += 0.08 * su
            Tt[:, 7] -= 0.06 * su
            Tt[:, 7] += 0.08 * wi
            Tt[:, 5] -= 0.06 * wi
            Tt = np.maximum(Tt, 0.005)
            Tt /= Tt.sum(axis=1, keepdims=True)
            w[t] = rng.choice(N_WEATHER, p=Tt[w[t - 1]])
        pos += dur
    return w


# ============================================================================
# Model Factory
# ============================================================================

def make_models(seed):
    """Create 5 model instances."""
    rng = np.random.default_rng(seed)
    random_aff = rng.dirichlet(np.ones(N_WEATHER), N_HEXAGRAMS)
    uniform_aff = np.ones((N_HEXAGRAMS, N_WEATHER)) / N_WEATHER

    return [
        IChingBayesianModel(HEXAGRAM_AFFINITIES, PRIOR_STR, TEMPERATURE),
        IChingBayesianModel(uniform_aff, PRIOR_STR, TEMPERATURE),
        IChingBayesianModel(random_aff, PRIOR_STR, TEMPERATURE),
        MarkovModel(),
        NeuralNetModel(ctx_win=3, hidden=32, lr=NN_LR, seed=seed),
    ], LABELS


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_model(model, train_data):
    """Online training on train_data."""
    for t in range(1, len(train_data)):
        hist = train_data[:t].tolist()
        model.update(hist, train_data[t])


def eval_model(model, train_data, eval_data):
    """Evaluate on eval_data WITHOUT further model updates."""
    hist = train_data.tolist()
    correct = 0
    for act in eval_data:
        pred = model.predict(hist)
        if np.argmax(pred) == act:
            correct += 1
        hist.append(act)
    return correct / len(eval_data)


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    max_days = max(SIZES) + EVAL_WINDOW
    results = {N: {lb: [] for lb in LABELS} for N in SIZES}

    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si + 1}/{len(SEEDS)} (s={seed})...", end=" ", flush=True)
        weather = generate_weather(max_days, seed)

        for N in SIZES:
            models, labels = make_models(seed)
            train_w = weather[:N]
            eval_w = weather[N:N + EVAL_WINDOW]

            for m in models:
                train_model(m, train_w)

            for lb, m in zip(labels, models):
                acc = eval_model(m, train_w, eval_w)
                results[N][lb].append(acc)
        print("done")

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_results(results):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    colors = {
        "IChing-Bayes": "#1a5276", "IChing-Uniform": "#2980b9",
        "IChing-Random": "#e67e22", "Markov": "#27ae60",
        "Neural-Net": "#e74c3c",
    }
    markers = {
        "IChing-Bayes": "o", "IChing-Uniform": "s",
        "IChing-Random": "^", "Markov": "D", "Neural-Net": "x",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    for lb in LABELS:
        means = [np.mean(results[N][lb]) for N in SIZES]
        stds = [np.std(results[N][lb]) for N in SIZES]
        cis = [1.96 * s / np.sqrt(len(SEEDS)) for s in stds]
        ax1.errorbar(SIZES, means, yerr=cis, color=colors[lb],
                     marker=markers[lb], linewidth=2, markersize=7,
                     capsize=4, label=lb)
    ax1.axhline(1.0 / N_WEATHER, color="gray", ls=":", alpha=0.5,
                label=f"Random ({1.0 / N_WEATHER:.1%})")
    ax1.set_xlabel("Training Days")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Data Efficiency: Accuracy vs Training Data")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks(SIZES)
    ax1.set_xticklabels([str(s) for s in SIZES])

    small = [s for s in SIZES if s <= 500]
    for lb in LABELS:
        means = [np.mean(results[N][lb]) for N in small]
        stds = [np.std(results[N][lb]) for N in small]
        cis = [1.96 * s / np.sqrt(len(SEEDS)) for s in stds]
        ax2.errorbar(small, means, yerr=cis, color=colors[lb],
                     marker=markers[lb], linewidth=2, markersize=8,
                     capsize=4, label=lb)
    ax2.axhline(1.0 / N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax2.set_xlabel("Training Days")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Zoom: Small-Data Regime (100-500 days)")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"I Ching Bayesian World Model — Data Efficiency\n"
        f"({len(SEEDS)} seeds, 95% CI, prior_strength={PRIOR_STR}, "
        f"temperature={TEMPERATURE})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("iching_efficiency.png", dpi=150, bbox_inches="tight")
    print("[图1] iching_efficiency.png")
    plt.close(fig)

    # Ablation bar chart
    fig2, ax = plt.subplots(figsize=(10, 5))
    prior_ms = ["IChing-Bayes", "IChing-Uniform", "IChing-Random"]
    x = np.arange(len(SIZES))
    width = 0.25
    for i, lb in enumerate(prior_ms):
        means = [np.mean(results[N][lb]) for N in SIZES]
        stds = [np.std(results[N][lb]) for N in SIZES]
        cis = [1.96 * s / np.sqrt(len(SEEDS)) for s in stds]
        off = (i - 1) * width
        ax.bar(x + off, means, width, yerr=cis, color=colors[lb],
               label=lb, capsize=3, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Training Days")
    ax.set_ylabel("Accuracy")
    ax.set_title("Prior Ablation: Traditional vs Uniform vs Random")
    ax.legend(fontsize=9)
    ax.axhline(1.0 / N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig("iching_efficiency_ablation.png", dpi=150, bbox_inches="tight")
    print("[图2] iching_efficiency_ablation.png")
    plt.close(fig2)


# ============================================================================
# CSV Export
# ============================================================================

def save_csv(results, path="results/v3_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Days", "Model", "Mean_Accuracy", "Std", "CI95"])
        for N in SIZES:
            for lb in LABELS:
                arr = np.array(results[N][lb])
                m = np.mean(arr)
                s = np.std(arr)
                ci = 1.96 * s / np.sqrt(len(arr))
                writer.writerow([N, lb, f"{m:.4f}", f"{s:.4f}", f"{ci:.4f}"])
    print(f"[CSV] {path} ({len(SIZES) * len(LABELS)} rows)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 64)
    print("  I Ching Bayesian World Model — V3 Experiment")
    print("=" * 64)
    print(f"  Training sizes: {SIZES}")
    print(f"  Seeds: {len(SEEDS)} (95% CI)")
    print(f"  prior_strength={PRIOR_STR}, temperature={TEMPERATURE}")
    print(f"  Weather: 4-regime non-stationary Markov")
    print()

    print("[1/3] Running experiment...")
    results = run_experiment()

    print("\n[2/3] Results:")
    header = f"  {'Days':<8}"
    for lb in LABELS:
        header += f" {lb:<16}"
    print(header)
    print(f"  {'-' * 70}")
    for N in SIZES:
        row = f"  {N:<8}"
        for lb in LABELS:
            row += f" {np.mean(results[N][lb]):<15.1%}"
        print(row)

    print("\n[3/3] Generating outputs...")
    plot_results(results)
    save_csv(results)

    print("\n" + "=" * 64)
    print("  Experiment complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
