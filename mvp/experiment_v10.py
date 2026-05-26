#!/usr/bin/env python3
"""
V10: 五维投影观测实验
======================
核心理论: 第五维是所有可能世界线的确定性集合。
四维时空只是其中一条世界线的投影。
64卦是标记不同世界线状态的符号系统。

实验: 3条世界线(乾元/坤元/交变), 每100-300天切换。
模型观测卦象, 预测天气, 贝叶斯更新世界线可信度。
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
import csv, os

N_HEX, N_W = 64, 4
WEATHER_NAMES = ["晴","雨","风","雾"]

# ============================================================================
# 八卦→4维天气向量
# ============================================================================
TRIGRAM_4D = np.array([
    [0.45, 0.10, 0.20, 0.25],  # 乾: 晴主导
    [0.08, 0.30, 0.22, 0.40],  # 坤: 雾/雨
    [0.15, 0.30, 0.42, 0.13],  # 震: 风/雨/雷
    [0.18, 0.20, 0.50, 0.12],  # 巽: 风主导
    [0.10, 0.48, 0.18, 0.24],  # 坎: 雨主导
    [0.50, 0.10, 0.12, 0.28],  # 离: 晴/热
    [0.15, 0.22, 0.13, 0.50],  # 艮: 雾主导
    [0.12, 0.38, 0.22, 0.28],  # 兑: 雨/雾
])

KING_WEN_TRIGRAMS = [
    (0,0),(1,1),(4,2),(6,4),(4,0),(0,4),(1,4),(4,1),
    (3,0),(0,7),(1,0),(0,1),(0,5),(5,0),(1,6),(2,1),
    (7,2),(6,3),(1,7),(3,1),(5,2),(6,5),(6,1),(1,2),
    (0,2),(6,0),(6,2),(7,3),(4,4),(5,5),(7,6),(2,3),
    (0,6),(2,0),(5,1),(1,5),(3,5),(5,7),(4,6),(2,4),
    (6,7),(3,2),(7,0),(0,3),(7,1),(1,3),(7,4),(4,3),
    (7,5),(5,3),(2,2),(6,6),(3,6),(2,7),(2,5),(5,6),
    (3,3),(7,7),(3,4),(4,7),(3,7),(2,6),(4,5),(5,4),
]

def _hex_dist(upper, lower, w_upper=0.55):
    return w_upper*TRIGRAM_4D[upper] + (1-w_upper)*TRIGRAM_4D[lower]

# ============================================================================
# 3条世界线的卦象→天气分布
# ============================================================================
def build_worldline_dists(seed_base):
    """生成3条世界线的64×4天气概率表。"""
    rng = np.random.default_rng(seed_base)
    dists = []
    
    # 世界线A (乾元): 阳卦(乾离震巽)活跃, 晴热多
    boost_a = np.array([3.0, 0.2, 2.0, 2.0, 0.2, 3.0, 0.3, 0.3])
    
    # 世界线B (坤元): 阴卦(坤坎艮兑)活跃, 阴雨多
    boost_b = np.array([0.2, 3.0, 0.3, 0.3, 3.0, 0.2, 2.0, 2.0])
    
    # 世界线C (交变): 变爻卦(震巽艮兑)活跃, 多变
    boost_c = np.array([0.3, 0.3, 3.0, 3.0, 0.3, 0.3, 3.0, 3.0])
    
    for bi, boost in enumerate([boost_a, boost_b, boost_c]):
        dist = np.zeros((N_HEX, N_W))
        for h, (u, l) in enumerate(KING_WEN_TRIGRAMS):
            base = _hex_dist(u, l)
            # 世界线调制
            modulated = base * boost[u] * (boost[l]**0.5)
            modulated /= modulated.sum()
            # 加噪声 + 世界线全局偏置
            noisy = 0.85 * modulated + 0.15 / N_W
            # WL-A: 偏晴, WL-B: 偏雨, WL-C: 偏风
            global_bias = [0, 0, 0, 0]
            if bi == 0: global_bias[0] = 0.20  # 晴
            elif bi == 1: global_bias[1] = 0.20  # 雨
            else: global_bias[2] = 0.20  # 风
            noisy = noisy + np.array(global_bias)
            noisy /= noisy.sum()
            dist[h] = noisy
        dists.append(dist)
    
    return dists


# ============================================================================
# 数据生成: 含世界线切换
# ============================================================================
def generate_data(n_days, seed):
    """生成 (hexagram, weather, true_worldline) 序列。"""
    rng = np.random.default_rng(seed)
    dists = build_worldline_dists(seed)
    
    hexagrams = np.zeros(n_days, dtype=int)
    weathers = np.zeros(n_days, dtype=int)
    true_wl = np.zeros(n_days, dtype=int)
    
    wl = 0  # 起始世界线
    switch_at = rng.integers(100, 300)
    
    for t in range(n_days):
        if t >= switch_at:
            wl = (wl + 1) % 3 if rng.random() < 0.8 else rng.integers(3)
            switch_at = t + rng.integers(100, 300)
        
        h = rng.integers(N_HEX)  # 随机卦象
        w = rng.choice(N_W, p=dists[wl][h])
        
        hexagrams[t] = h; weathers[t] = w; true_wl[t] = wl
    
    return hexagrams, weathers, true_wl, dists


# ============================================================================
# 五维投影观测模型
# ============================================================================
class FiveDModel:
    """五维投影观测模型: 维护多条世界线并追踪可信度。"""
    
    def __init__(self, prior_strength=10.0):
        self.n_wl = 3
        # 结构化先验: 每条世界线的 counts 从八卦倾向初始化
        self.counts = np.zeros((self.n_wl, N_HEX, N_W))
        # WL-A: 乾离倾向 (阳卦主导)
        boost_a = np.array([3.0,0.5,1.5,1.5,0.5,3.0,0.7,0.7])
        # WL-B: 坤坎倾向 (阴卦主导)  
        boost_b = np.array([0.5,3.0,0.7,0.7,3.0,0.5,1.5,1.5])
        # WL-C: 震巽艮兑倾向 (变爻主导)
        boost_c = np.array([0.7,0.7,3.0,3.0,0.7,0.7,2.0,2.0])
        for wl, boost in enumerate([boost_a, boost_b, boost_c]):
            for h, (u, l) in enumerate(KING_WEN_TRIGRAMS):
                base = _hex_dist(u, l)
                prior = base * boost[u] * np.sqrt(boost[l])
                prior /= prior.sum()
                self.counts[wl, h] = prior_strength * prior + 1.0
        # 世界线概率
        self.probs = np.ones(self.n_wl) / self.n_wl
        # 异常检测
        self.recent_ll = []
        self.prob_history = []
        self.anomaly_count = 0
    
    def predict(self, hexagram):
        """加权预测: Σ wl_prob × P(weather | hexagram, worldline)."""
        h = hexagram
        pred = np.zeros(N_W)
        for wl in range(self.n_wl):
            p_wl = self.counts[wl, h] / self.counts[wl, h].sum()
            pred += self.probs[wl] * p_wl
        return pred / pred.sum()
    
    def update(self, hexagram, true_weather):
        h, w = hexagram, true_weather
        
        # 计算每条世界线的似然
        likes = np.zeros(self.n_wl)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            likes[wl] = p[w]
        
        # 更新世界线概率 (温度缩放的贝叶斯更新, 确保集中)
        log_probs = np.log(np.maximum(self.probs, 1e-12)) + np.log(np.maximum(likes, 1e-12)) / 0.3
        log_probs -= log_probs.max()
        self.probs = np.exp(log_probs)
        self.probs /= self.probs.sum()
        
        # 只更新最大概率世界线 (硬分配, 防止其他世界线被污染)
        for wl in range(self.n_wl):
            self.counts[wl, h, w] += self.probs[wl]
        
        # 记录
        self.recent_ll.append(likes.max())
        if len(self.recent_ll) > 3:
            self.recent_ll.pop(0)
        self.prob_history.append(self.probs.copy())
        
        # 投影异常检测
        if len(self.recent_ll) == 3 and all(ll < 0.12 for ll in self.recent_ll):
            self.probs = np.ones(self.n_wl) / self.n_wl
            self.recent_ll = []
            self.anomaly_count += 1


class NoResetModel(FiveDModel):
    """五维无重置: 跳过异常检测。"""
    def update(self, hexagram, true_weather):
        h, w = hexagram, true_weather
        likes = np.zeros(self.n_wl)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            likes[wl] = p[w]
        log_probs = np.log(np.maximum(self.probs, 1e-12)) + np.log(np.maximum(likes, 1e-12)) / 0.3
        log_probs -= log_probs.max()
        self.probs = np.exp(log_probs)
        self.probs /= self.probs.sum()
        for wl in range(self.n_wl):
            self.counts[wl, h, w] += self.probs[wl]
        self.prob_history.append(self.probs.copy())


class GreedyModel(FiveDModel):
    """五维贪婪: 只用最大概率世界线预测。"""
    def predict(self, hexagram):
        wl = np.argmax(self.probs)
        p = self.counts[wl, hexagram] / self.counts[wl, hexagram].sum()
        return p


# ============================================================================
# 四维贝叶斯 (TrigramV4 适配版)
# ============================================================================
class FlatBayes:
    """单世界线贝叶斯模型: 64×4 Dirichlet, 不追踪世界线。"""
    def __init__(self, ps=10.0):
        self.counts = np.zeros((N_HEX, N_W))
        for h, (u, l) in enumerate(KING_WEN_TRIGRAMS):
            base = _hex_dist(u, l)
            self.counts[h] = ps * base + 1.0
    
    def predict(self, hexagram):
        return self.counts[hexagram] / self.counts[hexagram].sum()
    
    def update(self, hexagram, true_weather):
        self.counts[hexagram, true_weather] += 1.0


class SimpleNN:
    """简单 MLP: 卦象 one-hot → 隐藏 → softmax。"""
    def __init__(self, hidden=32, lr=0.005, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (N_HEX, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, (hidden, N_W))
        self.b2 = np.zeros(N_W)
        self.lr = lr
    
    def predict(self, hexagram):
        x = np.zeros(N_HEX); x[hexagram] = 1.0
        h = np.maximum(0, x @ self.W1 + self.b1)
        l = h @ self.W2 + self.b2; l -= l.max()
        p = np.exp(l); return p / p.sum()
    
    def update(self, hexagram, true_weather):
        x = np.zeros(N_HEX); x[hexagram] = 1.0
        hp = x @ self.W1 + self.b1; h = np.maximum(0, hp)
        l = h @ self.W2 + self.b2; l -= l.max()
        p = np.exp(l); p /= p.sum()
        dl = p.copy(); dl[true_weather] -= 1.0
        dW2 = np.outer(h, dl); db2 = dl
        dh = self.W2 @ dl; dh[hp <= 0] = 0.0
        dW1 = np.outer(x, dh); db1 = dh
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1


# ============================================================================
# 实验
# ============================================================================
LABELS = ["5D-Complete","4D-FlatBayes","5D-NoReset","5D-Greedy","Neural-Net"]
SEEDS = [42, 123, 456, 789, 1011]
TOTAL_DAYS = 2000

def run_experiment():
    """5种子 × 2000天 × 5模型。"""
    results = {lb: {"acc": [], "switch_acc": [], "anomalies": 0} for lb in LABELS}
    all_probs = []  # 第一个种子的世界线概率历史
    all_switches = []
    
    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)} (s={seed})...", end=" ", flush=True)
        hexs, wths, true_wl, _ = generate_data(TOTAL_DAYS, seed)
        
        models = [
            FiveDModel(), FlatBayes(), NoResetModel(),
            GreedyModel(), SimpleNN(seed=seed),
        ]
        
        switch_points = np.where(np.diff(true_wl) != 0)[0] + 1
        all_switches.append(switch_points)
        
        correct = {lb: 0 for lb in LABELS}
        switch_correct = {lb: 0 for lb in LABELS}
        switch_total = {lb: 0 for lb in LABELS}
        
        for t in range(TOTAL_DAYS):
            h, tw = hexs[t], wths[t]
            
            # 检查是否在切换后窗口
            in_window = False
            for sp in switch_points:
                if sp <= t < sp + 50:
                    in_window = True; break
            
            for lb, m in zip(LABELS, models):
                pred = m.predict(h)
                if np.argmax(pred) == tw:
                    correct[lb] += 1
                    if in_window:
                        switch_correct[lb] += 1
                if in_window:
                    switch_total[lb] += 1
                
                m.update(h, tw)
        
        for lb in LABELS:
            results[lb]["acc"].append(correct[lb] / TOTAL_DAYS)
            sw_total = switch_total[lb]
            results[lb]["switch_acc"].append(switch_correct[lb] / max(sw_total, 1) if sw_total > 0 else 0)
        
        results["5D-Complete"]["anomalies"] += models[0].anomaly_count
        
        if si == 0:
            all_probs = np.array(models[0].prob_history)
        
        print("done")
    
    return results, all_probs, all_switches[0], true_wl[:TOTAL_DAYS] if SEEDS else None


# ============================================================================
# 可视化
# ============================================================================
def plot_results(results, probs_hist, switch_points, true_wl):
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 整体准确率
    ax1 = fig.add_subplot(2, 2, 1)
    colors = {"5D-Complete":"#1a5276","4D-FlatBayes":"#2980b9",
              "5D-NoReset":"#27ae60","5D-Greedy":"#e67e22","Neural-Net":"#e74c3c"}
    x = np.arange(len(LABELS))
    means = [np.mean(results[lb]["acc"]) for lb in LABELS]
    stds = [np.std(results[lb]["acc"]) for lb in LABELS]
    cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
    bars = ax1.bar(x, means, yerr=cis, color=[colors[lb] for lb in LABELS],
                   capsize=5, edgecolor="white")
    for b, m in zip(bars, means):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{m:.1%}",
                 ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("Overall Accuracy"); ax1.set_title("V10: Overall Accuracy (2000 days)")
    ax1.axhline(0.25, color="gray", ls=":", alpha=0.5, label="Random 25%")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.2, axis="y")
    
    # 2. 切换后恢复准确率
    ax2 = fig.add_subplot(2, 2, 2)
    sw_means = [np.mean(results[lb]["switch_acc"]) for lb in LABELS]
    sw_stds = [np.std(results[lb]["switch_acc"]) for lb in LABELS]
    sw_cis = [1.96*s/np.sqrt(len(SEEDS)) for s in sw_stds]
    ax2.bar(x, sw_means, yerr=sw_cis, color=[colors[lb] for lb in LABELS],
            capsize=5, edgecolor="white")
    for i, m in enumerate(sw_means):
        ax2.text(i, m+0.005, f"{m:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("Accuracy (50d post-switch)"); ax2.set_title("Recovery After Worldline Switch")
    ax2.axhline(0.25, color="gray", ls=":", alpha=0.5)
    ax2.grid(alpha=0.2, axis="y")
    
    # 3. 世界线概率演化
    ax3 = fig.add_subplot(2, 1, 2)
    colors_wl = ["#e74c3c","#27ae60","#2980b9"]
    labels_wl = ["WL-A (乾元)","WL-B (坤元)","WL-C (交变)"]
    days = np.arange(len(probs_hist))
    for wl in range(3):
        ax3.plot(days, probs_hist[:, wl], color=colors_wl[wl], lw=1.5, alpha=0.8,
                 label=labels_wl[wl])
    # 竖虚线标记切换点
    for sp in switch_points:
        ax3.axvline(sp, color="gray", ls="--", alpha=0.3, lw=0.8)
    ax3.set_xlabel("Day"); ax3.set_ylabel("Worldline Probability")
    ax3.set_title("Worldline Probability Evolution (Seed 42)")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_ylim(0, 1.05); ax3.grid(alpha=0.2)
    
    # 标注真实世界线
    ax3_twin = ax3.twiny()
    ax3_twin.set_xlim(ax3.get_xlim())
    unique_wl = []
    prev = -1
    for t, wl in enumerate(true_wl):
        if wl != prev:
            unique_wl.append((t, wl)); prev = wl
    tick_pos = [t for t, _ in unique_wl]
    tick_labels = [f"WL{wl}" for _, wl in unique_wl]
    ax3_twin.set_xticks(tick_pos[:min(10, len(tick_pos))])
    ax3_twin.set_xticklabels(tick_labels[:10], fontsize=7, color="gray")
    
    plt.tight_layout()
    plt.savefig("iching_v10_five_dim.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v10_five_dim.png")
    plt.close(fig)


def print_results(results):
    print(f"\n  {'='*72}")
    print(f"  {'Model':<20} {'Overall':>10} {'Post-Switch':>12} {'Anomalies':>10}")
    print(f"  {'-'*72}")
    for lb in LABELS:
        acc = np.mean(results[lb]["acc"])
        sw = np.mean(results[lb]["switch_acc"])
        anom = results[lb].get("anomalies", 0)
        print(f"  {lb:<20} {acc:>9.1%} {sw:>11.1%} {anom:>10}")
    print(f"  {'='*72}")


def main():
    print("="*64)
    print("  V10: Five-Dimensional Projection Experiment")
    print("="*64)
    print(f"  3 worldlines, {TOTAL_DAYS} days, {len(SEEDS)} seeds")
    print(f"  Switches every 100-300 days")
    
    print(f"\n[1/2] Running experiment...")
    results, probs, switches, true_wl = run_experiment()
    
    print(f"\n[2/2] Results:")
    print_results(results)
    
    # 切换检测统计
    n_switches = len(switches)
    n_anomalies = results["5D-Complete"]["anomalies"]
    print(f"\n  切换检测: 真实切换={n_switches}次, 异常触发={n_anomalies}次")
    
    plot_results(results, probs, switches, true_wl)
    print(f"\n{'='*64}")

if __name__ == "__main__":
    main()
