#!/usr/bin/env python3
"""
V7 实验: 更长上下文窗口 — 卦象近期表现加权 + NN 上下文扩展
============================================================

I Ching 模型: 在 hex_ll 基础上, 叠加最近 K 天预测表现的 transient bonus
  context_bonus[h] = exp(Σ_{k=1}^{K} log p(w_{t-k+1} | w_{t-k}, h) / T)

NN 模型: 直接扩展 context_window

对比: K ∈ {1(基线), 3, 5, 7}
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, urllib.request, os, csv, warnings
warnings.filterwarnings("ignore")

from model import (
    TrigramBayesianModel, NeuralNetModel,
    TRIGRAM_WEATHER, N_WEATHER, WEATHER_TYPES,
)

# ============================================================================
# 上下文增强 I Ching 模型
# ============================================================================

class ContextTrigramModel(TrigramBayesianModel):
    """
    上下文增强三爻模型。
    在标准 hex_ll 权重之上, 叠加最近 K 天的预测表现 bonus。
    """

    def __init__(self, trigram_affinities, prior_strength=1.0, temperature=0.5,
                 context_K=3, context_decay=0.9):
        super().__init__(trigram_affinities, prior_strength, temperature)
        self.ctx_K = context_K
        self.decay = context_decay

    def _context_bonus(self, history):
        """最近 K 天预测表现的指数加权 bonus。"""
        if len(history) < self.ctx_K + 1:
            return np.ones(self.nh)

        bonus = np.zeros(self.nh)
        weight = 1.0
        for k in range(1, self.ctx_K + 1):
            wp = history[-k-1]
            wn = history[-k]
            for h in range(self.nh):
                alpha_h = self._hex_alpha(h)
                p = alpha_h[wp, :] / alpha_h[wp, :].sum()
                bonus[h] += weight * np.log(max(p[wn], 1e-12))
            weight *= self.decay

        bonus -= bonus.max()
        return np.exp(bonus / max(self.T, 0.01))

    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        wc = history[-1]

        lw = self.hex_ll / max(self.T, 0.01)
        lw -= lw.max()
        wts = np.exp(lw)
        wts *= self._context_bonus(history)

        s = wts.sum()
        wts = wts / s if s > 1e-12 else np.ones(self.nh) / self.nh

        pred = np.zeros(self.nw)
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wc, :] / alpha_h[wc, :].sum()
            pred += wts[h] * p
        return pred / pred.sum()


# ============================================================================
# 数据 (同 V5)
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

def to_sequence(data):
    c = data["daily"]["weather_code"]; t = data["daily"]["temperature_2m_max"]
    return np.array([5 if (v and v>HOT) else WMO_MAP.get(k,1) for k,v in zip(c,t)])

# ============================================================================
# 实验
# ============================================================================

LABELS = ["Trigram-K1","Trigram-K3","Trigram-K5","Trigram-K7",
          "NN-K1","NN-K3","NN-K5","NN-K7"]
SIZES = [100,200,500,1000,2000,3000]
EVAL_W, SEEDS = 365, [42,123,456,789,1011]
PS, T = 1.0, 0.5

def make_models(seed):
    ks = [1,3,5,7]
    models = []
    for k in ks:
        if k == 1:
            models.append(TrigramBayesianModel(TRIGRAM_WEATHER, PS, T))
        else:
            models.append(ContextTrigramModel(TRIGRAM_WEATHER, PS, T, context_K=k))
    for k in ks:
        models.append(NeuralNetModel(ctx_win=k, hidden=32, lr=0.005, seed=seed))
    return models, LABELS

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

def plot(res):
    colors_i = {1:"#1a5276",3:"#2471a3",5:"#2980b9",7:"#5499c7"}
    colors_n = {1:"#e74c3c",3:"#ec7063",5:"#f1948a",7:"#f5b7b1"}

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6.5))

    for k in [1,3,5,7]:
        means = [np.mean(res[N][f"Trigram-K{k}"]) for N in SIZES]
        stds = [np.std(res[N][f"Trigram-K{k}"]) for N in SIZES]
        cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
        ax1.errorbar(SIZES, means, yerr=cis, color=colors_i[k],
                     marker="o", lw=2, ms=6, capsize=3, label=f"IChing K={k}")
    for k in [1,3,5,7]:
        means = [np.mean(res[N][f"NN-K{k}"]) for N in SIZES]
        stds = [np.std(res[N][f"NN-K{k}"]) for N in SIZES]
        cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
        ax1.errorbar(SIZES, means, yerr=cis, color=colors_n[k],
                     marker="x", lw=2, ms=6, capsize=3, label=f"NN K={k}")
    ax1.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax1.set_xlabel("Training Days"); ax1.set_ylabel("Accuracy")
    ax1.set_title("V7: Context Window Size Comparison")
    ax1.legend(fontsize=7, ncol=2, loc="lower right"); ax1.grid(alpha=0.3)
    ax1.set_xscale("log"); ax1.set_xticks(SIZES)
    ax1.set_xticklabels([str(s) for s in SIZES])

    # Best K per model class at 100 days
    N_k = 100
    x = np.arange(4); w = 0.35
    i_means = [np.mean(res[N_k][f"Trigram-K{k}"]) for k in [1,3,5,7]]
    n_means = [np.mean(res[N_k][f"NN-K{k}"]) for k in [1,3,5,7]]
    ax2.bar(x-w/2, i_means, w, color=[colors_i[k] for k in [1,3,5,7]],
            label="IChing", edgecolor="white")
    ax2.bar(x+w/2, n_means, w, color=[colors_n[k] for k in [1,3,5,7]],
            label="Neural-Net", edgecolor="white")
    for i, (im, nm) in enumerate(zip(i_means, n_means)):
        ax2.text(i-w/2, im+0.005, f"{im:.1%}", ha="center", fontsize=8)
        ax2.text(i+w/2, nm+0.005, f"{nm:.1%}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(["K=1","K=3","K=5","K=7"])
    ax2.set_xlabel("Context Window K"); ax2.set_ylabel(f"Accuracy @{N_k} days")
    ax2.set_title(f"Best Context Window (@{N_k} days)")
    ax2.legend(); ax2.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle(f"Yijing World Model V7 — Context Window Ablation\n"
                 f"(Beijing 2015-2024, {len(SEEDS)} seeds, 95% CI)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("iching_v7_context.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v7_context.png")
    plt.close(fig)

def save_csv(res, path="results/v7_context.csv"):
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
    print("  V7: Context Window Ablation")
    print("="*64)
    
    print("\n[1/3] Beijing weather...")
    data = fetch_data()
    weather = to_sequence(data)
    print(f"      {len(weather)} days")
    
    print(f"\n[2/3] Experiment ({len(SEEDS)} seeds, 8 models)...")
    results = run(weather)
    
    print(f"\n[3/3] Results (@100 days):")
    for k in [1,3,5,7]:
        ti = np.mean(results[100][f"Trigram-K{k}"])
        tn = np.mean(results[100][f"NN-K{k}"])
        print(f"  K={k}: IChing={ti:.1%}  NN={tn:.1%}  Δ={ti-tn:+.1%}")
    
    # Best improvement
    ti1 = np.mean(results[100]["Trigram-K1"])
    ti_best = max(np.mean(results[100][f"Trigram-K{k}"]) for k in [3,5,7])
    tn1 = np.mean(results[100]["NN-K1"])
    tn_best = max(np.mean(results[100][f"NN-K{k}"]) for k in [3,5,7])
    print(f"\n  最佳提升:")
    print(f"    IChing: K=1→best = {ti1:.1%} → {ti_best:.1%} (Δ={ti_best-ti1:+.1%})")
    print(f"    NN:     K=1→best = {tn1:.1%} → {tn_best:.1%} (Δ={tn_best-tn1:+.1%})")
    
    plot(results)
    save_csv(results)
    print(f"\n{'='*64}")

if __name__ == "__main__":
    main()
