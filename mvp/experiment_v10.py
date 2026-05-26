#!/usr/bin/env python3
"""
V10b: 五维投影观测模型（基于参考实现 + 三项关键修复）

修复:
  1. 温度缩放: probs = softmax(log(probs) + log(likes) / T), T=0.3
  2. 结构化先验: 每条世界线的 counts 从不同卦象倾向初始化
  3. 加权更新: counts[wl] += probs[wl] (高概率世界线获得更多证据)

数据: 3条世界线, 晴/雨/风分别主导, 每100-300天切换
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

N_HEX, N_W = 64, 4
WEATHER = ["晴","雨","风","雾"]
TEMP = 0.3         # 温度缩放
PRIOR_STR = 5.0    # 先验强度
SWITCH_THRESHOLD = 0.08
SWITCH_PATIENCE = 3

# ============================================================================
# 数据生成: 三条区分明确的世界线
# ============================================================================
def generate_worldline_data(n_days, seed):
    """生成卦象序列 + 天气序列 + 真实世界线, 世界线间差异显著。"""
    rng = np.random.default_rng(seed)
    
    # 每条世界线的 64×4 天气概率表
    # WL0: 晴主导 (0.55), WL1: 雨主导 (0.55), WL2: 风主导 (0.55)
    dists = []
    for wl_bias in [0, 1, 2]:  # 晴, 雨, 风
        dist = np.full((N_HEX, N_W), 0.15)
        dist[:, wl_bias] = 0.55
        # 卦象间仍有微小差异 (八卦倾向)
        for h in range(N_HEX):
            tri_u = h % 8  # simplified trigram index
            tri_l = (h // 8) % 8
            # 乾卦略更晴, 坎卦略更雨, 巽卦略更风
            if tri_u in (0, 5) or tri_l in (0, 5):  # 乾/离
                dist[h, 0] += 0.05; dist[h, wl_bias if wl_bias != 0 else 1] -= 0.05
            if tri_u in (2, 4) or tri_l in (2, 4):  # 坎/震
                dist[h, 1] += 0.05; dist[h, wl_bias if wl_bias != 1 else 0] -= 0.05
            dist[h] /= dist[h].sum()
        dists.append(dist)
    
    hexs = np.zeros(n_days, dtype=int)
    wths = np.zeros(n_days, dtype=int)
    true_wl = np.zeros(n_days, dtype=int)
    
    wl = 0; switch_at = rng.integers(100, 300)
    for t in range(n_days):
        if t >= switch_at:
            wl = rng.integers(3)
            switch_at = t + rng.integers(100, 300)
        h = rng.integers(N_HEX)
        hexs[t] = h; wths[t] = rng.choice(N_W, p=dists[wl][h]); true_wl[t] = wl
    
    return hexs, wths, true_wl, dists


# ============================================================================
# 五维投影模型 (三项修复版)
# ============================================================================
class FiveDModel:
    """五维投影观测模型。"""
    
    def __init__(self, n_wl=3):
        self.n_wl = n_wl
        # 修复2: 结构化先验 — 每条世界线有不同的初始倾向
        self.counts = np.ones((n_wl, N_HEX, N_W)) * PRIOR_STR
        # WL0偏晴, WL1偏雨, WL2偏风
        self.counts[0, :, 0] += PRIOR_STR * 0.3
        self.counts[1, :, 1] += PRIOR_STR * 0.3
        self.counts[2, :, 2] += PRIOR_STR * 0.3
        
        self.probs = np.ones(n_wl) / n_wl
        self.recent_ll = []
        self.prob_history = []
        self.anomalies = 0
    
    def predict(self, hexagram):
        h = hexagram
        pred = np.zeros(N_W)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            pred += self.probs[wl] * p
        return pred / pred.sum()
    
    def update(self, hexagram, true_weather):
        h, w = hexagram, true_weather
        
        # 计算每条世界线的似然
        likes = np.zeros(self.n_wl)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            likes[wl] = p[w]
        
        # 修复1: 温度缩放的贝叶斯更新
        log_probs = np.log(np.maximum(self.probs, 1e-12))
        log_likes = np.log(np.maximum(likes, 1e-12))
        log_probs = log_probs + log_likes / TEMP
        log_probs -= log_probs.max()
        self.probs = np.exp(log_probs)
        self.probs /= self.probs.sum()
        
        # 修复3: 加权更新 — 高概率世界线获得更多证据
        for wl in range(self.n_wl):
            self.counts[wl, h, w] += self.probs[wl]
        
        # 异常检测
        self.recent_ll.append(likes.max())
        if len(self.recent_ll) > SWITCH_PATIENCE:
            self.recent_ll.pop(0)
        self.prob_history.append(self.probs.copy())
        
        if len(self.recent_ll) == SWITCH_PATIENCE:
            if all(ll < SWITCH_THRESHOLD for ll in self.recent_ll):
                self.probs = np.ones(self.n_wl) / self.n_wl
                self.recent_ll = []
                self.anomalies += 1


class FlatBayes:
    """四维贝叶斯: 不追踪世界线。"""
    def __init__(self):
        self.counts = np.ones((N_HEX, N_W)) * PRIOR_STR
    
    def predict(self, h):
        return self.counts[h] / self.counts[h].sum()
    
    def update(self, h, w):
        self.counts[h, w] += 1.0


class NoResetModel(FiveDModel):
    """五维无重置: 跳过异常检测。"""
    def update(self, h, w):
        likes = np.zeros(self.n_wl)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            likes[wl] = p[w]
        log_probs = np.log(np.maximum(self.probs, 1e-12))
        log_probs = log_probs + np.log(np.maximum(likes, 1e-12)) / TEMP
        log_probs -= log_probs.max()
        self.probs = np.exp(log_probs); self.probs /= self.probs.sum()
        for wl in range(self.n_wl):
            self.counts[wl, h, w] += self.probs[wl]
        self.prob_history.append(self.probs.copy())


class GreedyModel(FiveDModel):
    """五维贪婪: 只用最大概率世界线预测。"""
    def predict(self, h):
        wl = np.argmax(self.probs)
        return self.counts[wl, h] / self.counts[wl, h].sum()


class SimpleNN:
    """MLP 基线。"""
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (N_HEX, 32))
        self.b1 = np.zeros(32)
        self.W2 = rng.normal(0, 0.1, (32, N_W))
        self.b2 = np.zeros(N_W); self.lr = 0.005
    
    def predict(self, h):
        x = np.zeros(N_HEX); x[h] = 1.0
        hh = np.maximum(0, x @ self.W1 + self.b1)
        l = hh @ self.W2 + self.b2; l -= l.max()
        p = np.exp(l); return p / p.sum()
    
    def update(self, h, w):
        x = np.zeros(N_HEX); x[h] = 1.0
        hp = x @ self.W1 + self.b1; hh = np.maximum(0, hp)
        l = hh @ self.W2 + self.b2; l -= l.max()
        p = np.exp(l); p /= p.sum()
        dl = p.copy(); dl[w] -= 1.0
        dW2 = np.outer(hh, dl); db2 = dl
        dh = self.W2 @ dl; dh[hp <= 0] = 0.0
        dW1 = np.outer(x, dh); db1 = dh
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1


# ============================================================================
# 实验
# ============================================================================
LABELS = ["5D-Complete","4D-FlatBayes","5D-NoReset","5D-Greedy","Neural-Net"]
SEEDS = [42, 123, 456, 789, 1011]
N_DAYS = 2000

def run():
    results = {lb: {"acc": [], "sw_acc": []} for lb in LABELS}
    results["5D-Complete"]["anomalies"] = 0
    prob_hist = None; switch_pts = None; true_wl_hist = None
    
    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)}...", end=" ", flush=True)
        hexs, wths, twl, dists = generate_worldline_data(N_DAYS, seed)
        
        models = [FiveDModel(), FlatBayes(), NoResetModel(),
                  GreedyModel(), SimpleNN(seed=seed)]
        
        switches = np.where(np.diff(twl) != 0)[0] + 1
        correct = {lb: 0 for lb in LABELS}
        sw_correct = {lb: 0 for lb in LABELS}
        sw_total = {lb: 0 for lb in LABELS}
        
        for t in range(N_DAYS):
            h, tw = hexs[t], wths[t]
            in_sw = any(sp <= t < sp + 50 for sp in switches)
            
            for lb, m in zip(LABELS, models):
                pred = m.predict(h)
                if np.argmax(pred) == tw:
                    correct[lb] += 1
                    if in_sw: sw_correct[lb] += 1
                if in_sw: sw_total[lb] += 1
                m.update(h, tw)
        
        for lb in LABELS:
            results[lb]["acc"].append(correct[lb] / N_DAYS)
            results[lb]["sw_acc"].append(sw_correct[lb] / max(sw_total[lb], 1))
        
        results["5D-Complete"]["anomalies"] += models[0].anomalies
        
        if si == 0:
            prob_hist = np.array(models[0].prob_history)
            switch_pts = switches
            true_wl_hist = twl
        
        print("done")
    
    return results, prob_hist, switch_pts, true_wl_hist


def plot_results(res, ph, sp, twl):
    fig = plt.figure(figsize=(16, 10))
    colors = {"5D-Complete":"#1a5276","4D-FlatBayes":"#2980b9",
              "5D-NoReset":"#27ae60","5D-Greedy":"#e67e22","Neural-Net":"#e74c3c"}
    
    ax1 = fig.add_subplot(2,2,1)
    x = np.arange(len(LABELS))
    means = [np.mean(res[lb]["acc"]) for lb in LABELS]
    stds = [np.std(res[lb]["acc"]) for lb in LABELS]
    cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
    ax1.bar(x, means, yerr=cis, color=[colors[lb] for lb in LABELS],
            capsize=5, edgecolor="white")
    for i, m in enumerate(means):
        ax1.text(i, m+0.01, f"{m:.1%}", ha="center", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=7)
    ax1.set_ylabel("Accuracy"); ax1.set_title("Overall (2000 days)")
    ax1.axhline(0.25, color="gray", ls=":"); ax1.grid(alpha=0.2, axis="y")
    
    ax2 = fig.add_subplot(2,2,2)
    sw_m = [np.mean(res[lb]["sw_acc"]) for lb in LABELS]
    sw_s = [np.std(res[lb]["sw_acc"]) for lb in LABELS]
    sw_c = [1.96*s/np.sqrt(len(SEEDS)) for s in sw_s]
    ax2.bar(x, sw_m, yerr=sw_c, color=[colors[lb] for lb in LABELS],
            capsize=5, edgecolor="white")
    for i, m in enumerate(sw_m):
        ax2.text(i, m+0.01, f"{m:.1%}", ha="center", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=7)
    ax2.set_ylabel("Accuracy"); ax2.set_title("Post-Switch Recovery (50d)")
    ax2.axhline(0.25, color="gray", ls=":"); ax2.grid(alpha=0.2, axis="y")
    
    ax3 = fig.add_subplot(2,1,2)
    wl_colors = ["#e74c3c","#27ae60","#2980b9"]
    for wl in range(3):
        ax3.plot(ph[:, wl], color=wl_colors[wl], lw=1.5, alpha=0.8,
                 label=f"WL-{wl}")
    for sp_ in sp:
        ax3.axvline(sp_, color="gray", ls="--", alpha=0.3, lw=0.8)
    ax3.set_xlabel("Day"); ax3.set_ylabel("Probability")
    ax3.set_title("Worldline Probability Evolution")
    ax3.legend(fontsize=8); ax3.set_ylim(0, 1.05); ax3.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("iching_v10_five_dim.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v10_five_dim.png")
    plt.close(fig)


def main():
    print("="*64)
    print("  V10b: 5D Projection Model (3 fixes applied)")
    print("="*64)
    print(f"  T={TEMP}, prior_strength={PRIOR_STR}, threshold={SWITCH_THRESHOLD}")
    
    results, ph, sp, twl = run()
    
    print(f"\n  {'='*60}")
    print(f"  {'Model':<20} {'Overall':>10} {'Post-Switch':>12} {'Anomalies':>10}")
    print(f"  {'-'*60}")
    for lb in LABELS:
        acc = np.mean(results[lb]["acc"])
        sw = np.mean(results[lb]["sw_acc"])
        a = results[lb].get("anomalies", 0)
        print(f"  {lb:<20} {acc:>9.1%} {sw:>11.1%} {a:>10}")
    print(f"  {'='*60}")
    
    n_sw = len(sp)
    n_an = results["5D-Complete"]["anomalies"]
    print(f"\n  Switches: {n_sw} real, {n_an} anomalies detected")
    
    plot_results(results, ph, sp, twl)

if __name__ == "__main__":
    main()
