#!/usr/bin/env python3
"""
V11: 翻转测试 — 世界线切换检测速度
=====================================
理论预测: 5D模型通过追踪世界线概率, 能在切换后更快识别出新世界线。
   如果观测到"不匹配当前世界线"的数据, 概率应快速转移到正确世界线。

验证指标: 切换发生后, 模型需要多少天使正确世界线概率 > 0.5
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

N_HEX, N_W = 64, 4
TEMP = 0.3
PRIOR_STR = 5.0
N_DAYS = 2000
SEEDS = [42, 123, 456, 789, 1011]

# ============================================================================
# 数据生成 (同V10b)
# ============================================================================
def generate_data(n_days, seed):
    rng = np.random.default_rng(seed)
    dists = []
    for wl_bias in [0, 1, 2]:
        dist = np.full((N_HEX, N_W), 0.15)
        dist[:, wl_bias] = 0.55
        for h in range(N_HEX):
            tri_u, tri_l = h % 8, (h // 8) % 8
            if tri_u in (0, 5) or tri_l in (0, 5):
                dist[h, 0] += 0.05; dist[h, wl_bias if wl_bias != 0 else 1] -= 0.05
            if tri_u in (2, 4) or tri_l in (2, 4):
                dist[h, 1] += 0.05; dist[h, wl_bias if wl_bias != 1 else 0] -= 0.05
            dist[h] /= dist[h].sum()
        dists.append(dist)
    
    hexs = np.zeros(n_days, dtype=int)
    wths = np.zeros(n_days, dtype=int)
    true_wl = np.zeros(n_days, dtype=int)
    switch_points = []
    
    wl = 0; switch_at = rng.integers(100, 300)
    for t in range(n_days):
        if t >= switch_at:
            new_wl = rng.integers(3)
            if new_wl != wl:
                switch_points.append(t)
            wl = new_wl
            switch_at = t + rng.integers(100, 300)
        h = rng.integers(N_HEX)
        hexs[t] = h; wths[t] = rng.choice(N_W, p=dists[wl][h]); true_wl[t] = wl
    
    return hexs, wths, true_wl, np.array(switch_points)


# ============================================================================
# 模型 (同V10b, 加世界线概率追踪)
# ============================================================================
class FiveDModel:
    def __init__(self):
        self.n_wl = 3
        self.counts = np.ones((3, N_HEX, N_W)) * PRIOR_STR
        self.counts[0, :, 0] += PRIOR_STR * 3.0
        self.counts[1, :, 1] += PRIOR_STR * 3.0
        self.counts[2, :, 2] += PRIOR_STR * 3.0
        self.probs = np.ones(3) / 3
        self.prob_history = []
    
    def predict(self, h):
        pred = np.zeros(N_W)
        for wl in range(3):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            pred += self.probs[wl] * p
        return pred / pred.sum()
    
    def update(self, h, w):
        likes = np.zeros(3)
        for wl in range(3):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            likes[wl] = p[w]
        lp = np.log(np.maximum(self.probs, 1e-12))
        lp = lp + np.log(np.maximum(likes, 1e-12)) / TEMP
        lp -= lp.max()
        self.probs = np.exp(lp); self.probs /= self.probs.sum()
        for wl in range(3):
            self.counts[wl, h, w] += self.probs[wl]
        self.prob_history.append(self.probs.copy())


class FlatBayes:
    def __init__(self):
        self.counts = np.ones((N_HEX, N_W)) * PRIOR_STR
    def predict(self, h):
        return self.counts[h] / self.counts[h].sum()
    def update(self, h, w):
        self.counts[h, w] += 1.0


# ============================================================================
# 翻转测试
# ============================================================================
SEARCH_WINDOW = 30  # 切换后30天内搜索检测点

def run_flip_test():
    results = {"5D-Complete": [], "4D-NoReset": []}
    all_recovery_curves = []  # (n_switches, window) array of correct WL probs
    
    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)}...", end=" ", flush=True)
        hexs, wths, twl, switches = generate_data(N_DAYS, seed)
        
        m5 = FiveDModel(); m5_nr = FiveDModel()
        
        # 收集每个切换后30天的恢复曲线
        recovery_data = {sp: [] for sp in switches}
        
        for t in range(N_DAYS):
            old_probs_5d = m5.probs.copy()
            m5.update(hexs[t], wths[t])
            m5_nr.update(hexs[t], wths[t])
            
            for sp in switches:
                offset = t - sp
                if 0 <= offset < SEARCH_WINDOW:
                    recovery_data[sp].append((offset, m5.probs[twl[t]]))
        
        # 对齐所有切换的恢复曲线
        for sp, data in recovery_data.items():
            curve = np.full(SEARCH_WINDOW, np.nan)
            for offset, prob in data:
                curve[offset] = prob
            all_recovery_curves.append(curve)
        
        print("done")
    
    recovery = np.array(all_recovery_curves)  # (n_switches, 30)
    return recovery


def main():
    print("="*64)
    print("  V11: Flip Test — Recovery Curve Analysis")
    print("="*64)
    
    recovery = run_flip_test()
    n_switches = recovery.shape[0]
    mean_curve = np.nanmean(recovery, axis=0)
    
    print(f"\n  Switches analyzed: {n_switches}")
    print(f"  Recovery curve (mean P(correct WL) at each day):")
    for d in [0, 1, 2, 3, 5, 7, 10, 15, 20, 29]:
        if d < len(mean_curve):
            print(f"    Day {d:2d}: {mean_curve[d]:.3f}")
    
    print(f"\n  Days to reach P>0.5: ", end="")
    above = np.where(mean_curve > 0.5)[0]
    if len(above) > 0:
        print(f"{above[0]} days")
    else:
        print("never")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_curve, color="#1a5276", lw=2, label="Mean P(correct WL)")
    ax.fill_between(range(len(mean_curve)),
                    np.nanpercentile(recovery, 25, axis=0),
                    np.nanpercentile(recovery, 75, axis=0),
                    alpha=0.2, color="#1a5276")
    ax.axhline(0.5, color="gray", ls="--", label="P=0.5")
    ax.axhline(1/3, color="gray", ls=":", alpha=0.5, label="Uniform (1/3)")
    ax.set_xlabel("Days after switch"); ax.set_ylabel("P(correct worldline)")
    ax.set_title(f"V11: Worldline Recovery Curve (mean over {n_switches} switches, {len(SEEDS)} seeds)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("iching_v11_flip_test.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v11_flip_test.png")
    plt.close(fig)
    
    print(f"\n{'='*64}")

if __name__ == "__main__":
    main()
