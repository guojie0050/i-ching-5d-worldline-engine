#!/usr/bin/env python3
"""
V8: 多步预测 — 1/2/3 天预测跨度对比
=====================================

自回归链式预测: pred[t+1]→pred[t+2]→pred[t+3]
衡量准确率随跨度的衰减速度。

对比:
  - IChing: 1-day, 2-day, 3-day
  - NN:     1-day, 2-day, 3-day
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, urllib.request, os, csv, warnings
warnings.filterwarnings("ignore")

from model import (
    TrigramBayesianModel, IChingBayesianModel, NeuralNetModel,
    TRIGRAM_WEATHER, HEXAGRAM_AFFINITIES, N_WEATHER, WEATHER_TYPES,
)

# ============================================================================
# 多步预测评估
# ============================================================================

def predict_multistep(model, history, steps=3):
    """
    自回归多步预测。
    返回: list of (8,) arrays for each step.
    """
    preds = []
    h = history.copy()
    for _ in range(steps):
        p = model.predict(h)
        preds.append(p)
        h.append(int(np.argmax(p)))  # 用预测值扩展历史
    return preds

def evaluate_multistep(model, train_w, eval_w, steps=3):
    """评估多步预测准确率。"""
    h = train_w.tolist()
    correct = np.zeros(steps, dtype=int)
    total = len(eval_w) - steps + 1

    for i in range(total):
        preds = predict_multistep(model, h, steps)
        for s in range(steps):
            if i + s < len(eval_w):
                if np.argmax(preds[s]) == eval_w[i + s]:
                    correct[s] += 1
        h.append(eval_w[i])
        model.update(h[:-1], eval_w[i])  # update with true observation

    return correct / max(total, 1)


# ============================================================================
# 数据
# ============================================================================

WMO_MAP = {0:0,1:0,2:2,3:1,45:7,48:7,51:4,53:4,55:4,61:4,63:4,65:4,
           71:7,73:7,75:7,80:4,81:4,82:4,95:3,96:3,99:3}
HOT = 32.0

def fetch_data(cache="/tmp/beijing_weather.json"):
    if os.path.exists(cache): return json.load(open(cache))
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           "latitude=39.9&longitude=116.4&start_date=2015-01-01&end_date=2024-12-31"
           "&daily=weather_code,temperature_2m_max&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(url).read())
    json.dump(d, open(cache,"w")); return d

def to_seq(data):
    c = data["daily"]["weather_code"]; t = data["daily"]["temperature_2m_max"]
    return np.array([5 if (v and v>HOT) else WMO_MAP.get(k,1) for k,v in zip(c,t)])

# ============================================================================
# 实验
# ============================================================================

LABELS = ["Tri-1d","Tri-2d","Tri-3d","Hex-1d","Hex-2d","Hex-3d","NN-1d","NN-2d","NN-3d"]
SIZES, EVAL_W = [500, 1000, 2000, 3000], 365
SEEDS = [42, 123, 456, 789, 1011]

def run(weather):
    res = {N: {lb: [] for lb in LABELS} for N in SIZES}
    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)}...", end=" ", flush=True)
        for N in SIZES:
            tw = weather[:N]; ew = weather[N:N+EVAL_W]

            # Trigram
            m_t = TrigramBayesianModel(TRIGRAM_WEATHER, 1.0, 0.5)
            for t in range(1, len(tw)):
                m_t.update(tw[:t].tolist(), tw[t])
            acc = evaluate_multistep(m_t, tw, ew, 3)
            res[N]["Tri-1d"].append(acc[0])
            res[N]["Tri-2d"].append(acc[1])
            res[N]["Tri-3d"].append(acc[2])

            # Hexagram
            m_h = IChingBayesianModel(HEXAGRAM_AFFINITIES, 1.0, 0.5)
            for t in range(1, len(tw)):
                m_h.update(tw[:t].tolist(), tw[t])
            acc = evaluate_multistep(m_h, tw, ew, 3)
            res[N]["Hex-1d"].append(acc[0])
            res[N]["Hex-2d"].append(acc[1])
            res[N]["Hex-3d"].append(acc[2])

            # NN
            m_n = NeuralNetModel(ctx_win=3, hidden=32, lr=0.005, seed=seed)
            for t in range(1, len(tw)):
                m_n.update(tw[:t].tolist(), tw[t])
            acc = evaluate_multistep(m_n, tw, ew, 3)
            res[N]["NN-1d"].append(acc[0])
            res[N]["NN-2d"].append(acc[1])
            res[N]["NN-3d"].append(acc[2])
        print("done")
    return res

def plot(res):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5.5))

    colors_t = {1:"#1a5276",2:"#2980b9",3:"#5499c7"}
    colors_h = {1:"#27ae60",2:"#2ecc71",3:"#82e0aa"}
    colors_n = {1:"#e74c3c",2:"#ec7063",3:"#f1948a"}

    for d, lb in [(1,"Tri-1d"),(2,"Tri-2d"),(3,"Tri-3d")]:
        means = [np.mean(res[N][lb]) for N in SIZES]
        ax1.plot(SIZES, means, color=colors_t[d], marker="o", lw=2, ms=7,
                 label=f"Trigram {d}-day")
    for d, lb in [(1,"Hex-1d"),(2,"Hex-2d"),(3,"Hex-3d")]:
        means = [np.mean(res[N][lb]) for N in SIZES]
        ax1.plot(SIZES, means, color=colors_h[d], marker="s", lw=2, ms=7,
                 label=f"Hexagram {d}-day")
    for d, lb in [(1,"NN-1d"),(2,"NN-2d"),(3,"NN-3d")]:
        means = [np.mean(res[N][lb]) for N in SIZES]
        ax1.plot(SIZES, means, color=colors_n[d], marker="x", lw=2, ms=7,
                 label=f"NN {d}-day")
    ax1.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax1.set_xlabel("Training Days"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Multi-Step Prediction Accuracy")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(alpha=0.3)

    # Decay rate at 3000 days
    N_k = 3000
    t_acc = [np.mean(res[N_k][f"Tri-{d}d"]) for d in [1,2,3]]
    h_acc = [np.mean(res[N_k][f"Hex-{d}d"]) for d in [1,2,3]]
    n_acc = [np.mean(res[N_k][f"NN-{d}d"]) for d in [1,2,3]]
    x = np.arange(3); w = 0.25
    ax2.bar(x-w, t_acc, w, color=[colors_t[d] for d in [1,2,3]],
            label="Trigram (512)", edgecolor="white")
    ax2.bar(x, h_acc, w, color=[colors_h[d] for d in [1,2,3]],
            label="Hexagram (4096)", edgecolor="white")
    ax2.bar(x+w, n_acc, w, color=[colors_n[d] for d in [1,2,3]],
            label="NN", edgecolor="white")
    for i, (ta, ha, na) in enumerate(zip(t_acc, h_acc, n_acc)):
        ax2.text(i-w, ta+0.005, f"{ta:.1%}", ha="center", fontsize=7)
        ax2.text(i, ha+0.005, f"{ha:.1%}", ha="center", fontsize=7)
        ax2.text(i+w, na+0.005, f"{na:.1%}", ha="center", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(["1-day","2-day","3-day"])
    ax2.set_xlabel("Prediction Horizon"); ax2.set_ylabel(f"Accuracy @{N_k} days")
    ax2.set_title(f"Accuracy Decay with Horizon (@{N_k} days)")
    ax2.legend(); ax2.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle(f"Yijing World Model V8 — Multi-Step Prediction\n"
                 f"(Beijing 2015-2024, {len(SEEDS)} seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("iching_v8_multistep.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v8_multistep.png")
    plt.close(fig)

def save_csv(res, path="results/v8_multistep.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="") as f:
        w = csv.writer(f); w.writerow(["Days","Model","Mean","Std"])
        for N in SIZES:
            for lb in LABELS:
                a = np.array(res[N][lb])
                w.writerow([N,lb,f"{np.mean(a):.4f}",f"{np.std(a):.4f}"])
    print(f"[CSV] {path}")

def main():
    print("="*64)
    print("  V8: Multi-Step Prediction (1/2/3 days)")
    print("="*64)

    print("\n[1/3] Beijing weather...")
    d = fetch_data(); w = to_seq(d)
    print(f"      {len(w)} days")

    print(f"\n[2/3] Experiment ({len(SEEDS)} seeds)...")
    results = run(w)

    print(f"\n[3/3] Results (@3000 days):")
    for md in range(1,4):
        ta = np.mean(results[3000][f"Tri-{md}d"])
        ha = np.mean(results[3000][f"Hex-{md}d"])
        na = np.mean(results[3000][f"NN-{md}d"])
        print(f"  {md}-day: Tri={ta:.1%}  Hex={ha:.1%}  NN={na:.1%}")

    t1 = np.mean(results[3000]["Tri-1d"])
    t3 = np.mean(results[3000]["Tri-3d"])
    h1 = np.mean(results[3000]["Hex-1d"])
    h3 = np.mean(results[3000]["Hex-3d"])
    print(f"\n  3天衰减:")
    print(f"    Trigram:  {t1:.1%}→{t3:.1%} (Δ={t1-t3:.1%})")
    print(f"    Hexagram: {h1:.1%}→{h3:.1%} (Δ={h1-h3:.1%})")

    plot(results)
    save_csv(results)

if __name__ == "__main__":
    main()
