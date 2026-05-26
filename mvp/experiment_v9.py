#!/usr/bin/env python3
"""
V9: 易经隐马尔可夫模型 — 卦为隐状态, 天气为投影
=================================================

IChingHMM: Forward滤波推断隐卦象 → 预测天气发射

对比:
  - IChingHMM     (4608参数: 64×64转移 + 64×8发射)
  - Trigram-Bayes (512参数, V4基线)
  - Neural-Net    (纯数据驱动)
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, urllib.request, os, csv, warnings
warnings.filterwarnings("ignore")

from model import (
    IChingHMM, FactoredIChingHMM, TrigramBayesianModel, NeuralNetModel,
    TRIGRAM_WEATHER, HEXAGRAM_AFFINITIES, N_WEATHER, WEATHER_TYPES,
)

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

LABELS = ["HMM-Factored","HMM-Full","Trigram","Neural-Net"]
SIZES, EVAL_W = [100,200,500,1000,2000,3000], 365
SEEDS = [42,123,456,789,1011]
PS, TS, ES = 1.0, 10.0, 3.0  # prior_strength, trans_strength, emit_strength

def make_models(seed):
    return [
        FactoredIChingHMM(HEXAGRAM_AFFINITIES, trans_strength=TS, emit_strength=ES),
        IChingHMM(HEXAGRAM_AFFINITIES, trans_strength=TS, emit_strength=ES),
        TrigramBayesianModel(TRIGRAM_WEATHER, PS, 0.5),
        NeuralNetModel(ctx_win=3, hidden=32, lr=0.005, seed=seed),
    ], LABELS

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

def plot(res):
    colors = {"HMM-Factored":"#1a5276","HMM-Full":"#8e44ad","Trigram":"#2980b9","Neural-Net":"#e74c3c"}
    marks = {"HMM-Factored":"o","HMM-Full":"D","Trigram":"s","Neural-Net":"x"}

    fig, ax = plt.subplots(figsize=(12,6))
    for lb in LABELS:
        means = [np.mean(res[N][lb]) for N in SIZES]
        stds = [np.std(res[N][lb]) for N in SIZES]
        cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
        ax.errorbar(SIZES, means, yerr=cis, color=colors[lb],
                    marker=marks[lb], lw=2, ms=8, capsize=4, label=lb)
    ax.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Training Days"); ax.set_ylabel("Accuracy")
    ax.set_title("V9: I Ching HMM — Hexagrams as Hidden States")
    ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xscale("log"); ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    plt.tight_layout()
    plt.savefig("iching_v9_hmm.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v9_hmm.png")
    plt.close(fig)

def save_csv(res, path="results/v9_hmm.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="") as f:
        w = csv.writer(f); w.writerow(["Days","Model","Mean","Std","CI95"])
        for N in SIZES:
            for lb in LABELS:
                a = np.array(res[N][lb])
                w.writerow([N,lb,f"{np.mean(a):.4f}",f"{np.std(a):.4f}",
                            f"{1.96*np.std(a)/np.sqrt(len(a)):.4f}"])
    print(f"[CSV] {path}")

def main():
    print("="*64)
    print("  V9: I Ching Hidden Markov Model")
    print("="*64)
    print(f"  Architecture: 64 hidden hexagrams → 8 weather emissions")
    print(f"  Prior: 序卦传 transition + 八卦 emission")

    print("\n[1/3] Beijing weather...")
    d = fetch_data(); w = to_seq(d)
    print(f"      {len(w)} days")

    print(f"\n[2/3] Experiment ({len(SEEDS)} seeds)...")
    results = run(w)

    print(f"\n[3/3] Results:")
    hdr = f"  {'Days':<8}"
    for lb in LABELS: hdr += f" {lb:<14}"
    print(hdr); print(f"  {'-'*48}")
    for N in SIZES:
        row = f"  {N:<8}"
        for lb in LABELS:
            row += f" {np.mean(results[N][lb]):<13.1%}"
        print(row)

    N_k = 100
    hf = np.mean(results[N_k]["HMM-Factored"])
    hfull = np.mean(results[N_k]["HMM-Full"])
    tri = np.mean(results[N_k]["Trigram"])
    nn = np.mean(results[N_k]["Neural-Net"])
    print(f"\n  @{N_k}天: Factored={hf:.1%}  Full={hfull:.1%}  Trigram={tri:.1%}  NN={nn:.1%}")
    print(f"  Factored vs Full: Δ={hf-hfull:+.1%}")
    print(f"  Factored vs Trigram: Δ={hf-tri:+.1%}")

    plot(results)
    save_csv(results)
    print(f"\n{'='*64}")

if __name__ == "__main__":
    main()
