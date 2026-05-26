#!/usr/bin/env python3
"""
V5 实验: 完整易经世界模型 — 三项优化消融分析
===============================================

对比模型:
  1. Full-Model       — 非对称权重 + delta + 六爻上下文 (全部开启)
  2. No-Yao            — 关闭六爻上下文 (yao_boost=1.0)
  3. No-Delta          — 关闭个体偏差 (l2_delta=0)
  4. Symmetric         — 对称权重 (所有卦 w_upper=0.5)
  5. Trigram-Bayes     — V4 基线 (512参数, 无三项优化)
  6. Neural-Net        — 纯数据驱动

数据: Beijing 2015-2024, 3653 天
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, urllib.request, os, csv, warnings
warnings.filterwarnings("ignore")

from model import (
    HexagramWorldModel, TrigramBayesianModel, NeuralNetModel,
    TRIGRAM_WEATHER, N_WEATHER, WEATHER_TYPES,
)

# ============================================================================
# 真实天气数据 (同 V4)
# ============================================================================

WMO_MAP = {
    0:0,1:0, 2:2, 3:1, 45:7,48:7, 51:4,53:4,55:4,
    61:4,63:4,65:4, 71:7,73:7,75:7, 80:4,81:4,82:4, 95:3,96:3,99:3,
}
HOT = 32.0

def fetch_weather(cache="/tmp/beijing_weather.json"):
    if os.path.exists(cache):
        return json.load(open(cache))
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           "latitude=39.9&longitude=116.4&start_date=2015-01-01&end_date=2024-12-31"
           "&daily=weather_code,temperature_2m_max&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(url).read())
    json.dump(d, open(cache,"w"))
    return d

def to_sequence(data):
    codes = data["daily"]["weather_code"]
    temps = data["daily"]["temperature_2m_max"]
    seq = np.zeros(len(codes), dtype=int)
    for i,(c,t) in enumerate(zip(codes,temps)):
        seq[i] = 5 if (t and t>HOT) else WMO_MAP.get(c,1)
    return seq

# ============================================================================
# 实验配置
# ============================================================================

LABELS = [
    "Full-Model", "No-Yao", "No-Delta",
    "Symmetric", "Trigram-V4", "Neural-Net",
]
SIZES = [100, 200, 500, 1000, 2000]
EVAL_W = 365
SEEDS = [42, 123, 456, 789, 1011]
PS, T = 1.0, 0.5

def make_models(seed):
    return [
        HexagramWorldModel(TRIGRAM_WEATHER, PS, T, l2_delta=0.05, yao_boost=1.5),
        HexagramWorldModel(TRIGRAM_WEATHER, PS, T, l2_delta=0.05, yao_boost=1.0),
        HexagramWorldModel(TRIGRAM_WEATHER, PS, T, l2_delta=0.0,  yao_boost=1.5),
        _make_symmetric(TRIGRAM_WEATHER, PS, T),
        TrigramBayesianModel(TRIGRAM_WEATHER, PS, T),
        NeuralNetModel(ctx_win=3, hidden=32, lr=0.005, seed=seed),
    ], LABELS

def _make_symmetric(aff, ps, T):
    m = HexagramWorldModel(aff, ps, T, l2_delta=0.05, yao_boost=1.5)
    m.w_upper[:] = 0.5; m.w_lower[:] = 0.5
    return m

def train(m, data):
    for t in range(1, len(data)):
        m.update(data[:t].tolist(), data[t])

def evaluate(m, train_d, eval_d):
    h = train_d.tolist(); c = 0
    for a in eval_d:
        if np.argmax(m.predict(h)) == a: c += 1
        h.append(a)
    return c / len(eval_d)

def run(weather):
    res = {N: {lb: [] for lb in LABELS} for N in SIZES}
    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)}...", end=" ", flush=True)
        for N in SIZES:
            models, labels = make_models(seed)
            tw = weather[:N]; ew = weather[N:N+EVAL_W]
            for m in models: train(m, tw)
            for lb, m in zip(labels, models):
                res[N][lb].append(evaluate(m, tw, ew))
        print("done")
    return res

# ============================================================================
# 可视化
# ============================================================================

def plot(res, weather):
    colors = {
        "Full-Model":"#1a5276","No-Yao":"#2980b9","No-Delta":"#27ae60",
        "Symmetric":"#e67e22","Trigram-V4":"#8e44ad","Neural-Net":"#e74c3c",
    }
    marks = {
        "Full-Model":"o","No-Yao":"s","No-Delta":"D",
        "Symmetric":"^","Trigram-V4":"v","Neural-Net":"x",
    }

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6.5))

    for lb in LABELS:
        means = [np.mean(res[N][lb]) for N in SIZES]
        stds = [np.std(res[N][lb]) for N in SIZES]
        cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
        ax1.errorbar(SIZES, means, yerr=cis, color=colors[lb],
                     marker=marks[lb], lw=2, ms=7, capsize=4, label=lb)
    ax1.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax1.set_xlabel("Training Days"); ax1.set_ylabel("Accuracy")
    ax1.set_title("V5 Ablation: Beijing Real Weather")
    ax1.legend(fontsize=7, loc="lower right"); ax1.grid(alpha=0.3)
    ax1.set_xscale("log"); ax1.set_xticks(SIZES)
    ax1.set_xticklabels([str(s) for s in SIZES])

    # 小数据放大
    small = [s for s in SIZES if s<=500]
    x = np.arange(len(small)); w = 0.13
    for i, lb in enumerate(LABELS):
        means = [np.mean(res[N][lb]) for N in small]
        stds = [np.std(res[N][lb]) for N in small]
        cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
        off = (i-2.5)*w
        ax2.bar(x+off, means, w, yerr=cis, color=colors[lb],
                label=lb, capsize=2, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels([str(s) for s in small])
    ax2.set_xlabel("Training Days"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Small-Data Zoom (100-500 days)")
    ax2.legend(fontsize=6); ax2.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle("Yijing World Model V5 — Feature Ablation\n"
                 f"({len(weather)} real days, {len(SEEDS)} seeds, 95% CI)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("iching_v5_ablation.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v5_ablation.png")
    plt.close(fig)

def save_csv(res, path="results/v5_ablation.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Days","Model","Mean","Std","CI95"])
        for N in SIZES:
            for lb in LABELS:
                a = np.array(res[N][lb])
                w.writerow([N,lb,f"{np.mean(a):.4f}",f"{np.std(a):.4f}",
                            f"{1.96*np.std(a)/np.sqrt(len(a)):.4f}"])
    print(f"[CSV] {path}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*64)
    print("  Yijing World Model V5 — Feature Ablation")
    print("="*64)

    print("\n[1/3] Beijing weather...")
    data = fetch_weather()
    weather = to_sequence(data)
    uniq, cnts = np.unique(weather, return_counts=True)
    for u,c in zip(uniq,cnts):
        print(f"      {WEATHER_TYPES[u]}: {c} ({c/len(weather):.1%})")

    print(f"\n[2/3] Experiment ({len(SEEDS)} seeds)...")
    results = run(weather)

    print(f"\n[3/3] Results:")
    hdr = f"  {'Days':<8}"
    for lb in LABELS: hdr += f" {lb:<14}"
    print(hdr); print(f"  {'-'*80}")
    for N in SIZES:
        row = f"  {N:<8}"
        for lb in LABELS:
            row += f" {np.mean(results[N][lb]):<13.1%}"
        print(row)

    # Ablation impact
    N_k = 100
    full = np.mean(results[N_k]["Full-Model"])
    no_yao = np.mean(results[N_k]["No-Yao"])
    no_delta = np.mean(results[N_k]["No-Delta"])
    sym = np.mean(results[N_k]["Symmetric"])
    tri = np.mean(results[N_k]["Trigram-V4"])
    nn = np.mean(results[N_k]["Neural-Net"])

    print(f"\n  消融分析 (@{N_k}天):")
    print(f"    Full-Model:     {full:.1%}  (基准)")
    print(f"    - 六爻上下文:    {no_yao:.1%}  (Δ={full-no_yao:+.1%})")
    print(f"    - 个体delta:    {no_delta:.1%}  (Δ={full-no_delta:+.1%})")
    print(f"    - 非对称权重:    {sym:.1%}  (Δ={full-sym:+.1%})")
    print(f"    V4 Trigram基线: {tri:.1%}  (Δ={full-tri:+.1%})")
    print(f"    Neural-Net:     {nn:.1%}  (Δ={full-nn:+.1%})")

    plot(results, weather)
    save_csv(results)
    print(f"\n{'='*64}")

if __name__ == "__main__":
    main()
