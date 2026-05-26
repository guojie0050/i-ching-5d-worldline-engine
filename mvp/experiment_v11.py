#!/usr/bin/env python3
"""
V11: 真实季节五维投影实验
==========================
数据: 北京 2015-2024, 干支纪日→卦象, 4天气(晴雨阴雪)
世界线: 4条 = 春夏秋冬
模型: 4世界线五维投影 + 4对照组
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, urllib.request, os, warnings
from datetime import date
warnings.filterwarnings("ignore")

N_HEX, N_W = 64, 4
WEATHER = ["晴","雨","阴","雪"]
TEMP = 0.3; PRIOR_STR = 1.0; BIAS = 3.0
N_DAYS_PER_SEED = 1460  # 4年

# ============================================================================
# 干支→卦象映射
# ============================================================================
REF_DATE = date(1900, 1, 1)

def date_to_ganzhi_hex(d):
    """日期→干支索引(0-59)→卦象(0-63)"""
    days = (d - REF_DATE).days
    ganzhi = days % 60
    return ganzhi % 64  # 确定性映射

# ============================================================================
# 真实天气获取
# ============================================================================
WMO_MAP = {0:0,1:0, 2:2,3:2, 45:2,48:2, 51:1,53:1,55:1,61:1,63:1,65:1,
           71:3,73:3,75:3,77:3,85:3,86:3, 80:1,81:1,82:1, 95:1,96:1,99:1}

def fetch_weather(cache="/tmp/beijing_snow.json"):
    if os.path.exists(cache): return json.load(open(cache))
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           "latitude=39.9&longitude=116.4&start_date=2015-01-01&end_date=2024-12-31"
           "&daily=weather_code,temperature_2m_min,snowfall_sum&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(url).read())
    json.dump(d, open(cache,"w")); return d

def to_sequence(data):
    """转换为 (dates, hexagrams, weather_types)"""
    daily = data["daily"]
    times = daily["time"]
    codes = daily["weather_code"]
    temps = daily["temperature_2m_min"]
    snows = daily["snowfall_sum"]
    
    n = len(times)
    hexs = np.zeros(n, dtype=int)
    wths = np.zeros(n, dtype=int)
    dates = []
    
    for i in range(n):
        y, m, d = times[i].split("-")
        dt = date(int(y), int(m), int(d))
        dates.append(dt)
        hexs[i] = date_to_ganzhi_hex(dt)
        
        code = codes[i]; temp = temps[i] if temps[i] else 10
        snow = snows[i] if snows[i] else 0
        
        if snow and snow > 0 and temp < 2:
            wths[i] = 3  # 雪
        else:
            wths[i] = WMO_MAP.get(code, 2)  # 晴0/阴2/雨3
    
    return np.array(dates), hexs, wths

def get_season(dt):
    """返回季节 0=春3-5, 1=夏6-8, 2=秋9-11, 3=冬12-2"""
    m = dt.month
    if 3 <= m <= 5: return 0
    if 6 <= m <= 8: return 1
    if 9 <= m <= 11: return 2
    return 3

# ============================================================================
# 4世界线五维模型
# ============================================================================
class FiveDModel4:
    def __init__(self, n_wl=4):
        self.n_wl = n_wl
        self.counts = np.ones((n_wl, N_HEX, N_W)) * PRIOR_STR
        # 每条世界线不同倾向 (与季节无关的先验, 模型自己学)
        for wl in range(n_wl):
            self.counts[wl, :, wl % N_W] += PRIOR_STR * BIAS
        self.probs = np.ones(n_wl) / n_wl
        self.prob_history = []
        self.recent_ll = []
        self.anomalies = 0
    
    def predict(self, h):
        pred = np.zeros(N_W)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            pred += self.probs[wl] * p
        return pred / pred.sum()
    
    def update(self, h, w):
        likes = np.zeros(self.n_wl)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            likes[wl] = p[w]
        lp = np.log(np.maximum(self.probs, 1e-12))
        lp = lp + np.log(np.maximum(likes, 1e-12)) / TEMP
        lp -= lp.max()
        self.probs = np.exp(lp); self.probs /= self.probs.sum()
        for wl in range(self.n_wl):
            self.counts[wl, h, w] += self.probs[wl]
        self.prob_history.append(self.probs.copy())
        self.recent_ll.append(likes.max())
        if len(self.recent_ll) > 3: self.recent_ll.pop(0)
        if len(self.recent_ll) == 3 and all(ll < 0.05 for ll in self.recent_ll):
            self.probs = np.ones(self.n_wl) / self.n_wl
            self.recent_ll = []; self.anomalies += 1

class NoReset4(FiveDModel4):
    def update(self, h, w):
        likes = np.zeros(self.n_wl)
        for wl in range(self.n_wl):
            p = self.counts[wl, h] / self.counts[wl, h].sum()
            likes[wl] = p[w]
        lp = np.log(np.maximum(self.probs, 1e-12))
        lp = lp + np.log(np.maximum(likes, 1e-12)) / TEMP
        lp -= lp.max()
        self.probs = np.exp(lp); self.probs /= self.probs.sum()
        for wl in range(self.n_wl):
            self.counts[wl, h, w] += self.probs[wl]
        self.prob_history.append(self.probs.copy())

class Greedy4(FiveDModel4):
    def predict(self, h):
        wl = np.argmax(self.probs)
        return self.counts[wl, h] / self.counts[wl, h].sum()

class FlatBayes4:
    def __init__(self):
        self.counts = np.ones((N_HEX, N_W)) * PRIOR_STR
    def predict(self, h):
        return self.counts[h] / self.counts[h].sum()
    def update(self, h, w):
        self.counts[h, w] += 1.0

class SimpleNN4:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (N_HEX, 32)); self.b1 = np.zeros(32)
        self.W2 = rng.normal(0, 0.1, (32, N_W)); self.b2 = np.zeros(N_W)
        self.lr = 0.005
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
        dW2 = np.outer(hh, dl); db2 = dl; dh = self.W2 @ dl; dh[hp <= 0] = 0
        dW1 = np.outer(x, dh); db1 = dh
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

# ============================================================================
# 实验
# ============================================================================
LABELS = ["5D-Complete","4D-Flat","5D-NoReset","5D-Greedy","Neural-Net"]
SEEDS = [42, 123, 456, 789, 1011]
CHECKPOINTS = [100, 500, 1000, 1460]
SWITCH_DATES = [date(y, 3, 1) for y in range(2015, 2025)] + \
               [date(y, 6, 1) for y in range(2015, 2025)] + \
               [date(y, 9, 1) for y in range(2015, 2025)] + \
               [date(y, 12, 1) for y in range(2015, 2025)]

def run():
    print("[1/3] Loading weather...")
    data = fetch_weather()
    dates, hexs, wths = to_sequence(data)
    print(f"  {len(dates)} days loaded")
    
    # 天气分布
    for i, name in enumerate(WEATHER):
        cnt = (wths == i).sum()
        print(f"    {name}: {cnt} ({cnt/len(wths):.1%})")
    
    results = {lb: {"acc": {cp: [] for cp in CHECKPOINTS},
                    "sw_acc": [], "season_acc": {s: [] for s in range(4)}}
               for lb in LABELS}
    results["5D-Complete"]["anomalies"] = 0
    prob_hist = None; season_true = None
    
    print(f"\n[2/3] Experiment ({len(SEEDS)} seeds)...")
    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)}...", end=" ", flush=True)
        rng = np.random.default_rng(seed)
        start = rng.integers(0, len(dates) - N_DAYS_PER_SEED)
        d_sub = dates[start:start+N_DAYS_PER_SEED]
        h_sub = hexs[start:start+N_DAYS_PER_SEED]
        w_sub = wths[start:start+N_DAYS_PER_SEED]
        seasons = np.array([get_season(d) for d in d_sub])
        
        models = [FiveDModel4(), FlatBayes4(), NoReset4(),
                  Greedy4(), SimpleNN4(seed=seed)]
        
        # 切换点: 季节变化
        sw_idx = np.where(np.diff(seasons) != 0)[0] + 1
        
        correct = {lb: 0 for lb in LABELS}
        sw_correct = {lb: 0 for lb in LABELS}
        sw_total = {lb: 0 for lb in LABELS}
        season_correct = {lb: {s: 0 for s in range(4)} for lb in LABELS}
        season_total = {lb: {s: 0 for s in range(4)} for lb in LABELS}
        
        for t in range(N_DAYS_PER_SEED):
            h, w = h_sub[t], w_sub[t]; s = seasons[t]
            in_sw = any(si <= t < si + 30 for si in sw_idx)
            
            for lb, m in zip(LABELS, models):
                pred = m.predict(h)
                if np.argmax(pred) == w:
                    correct[lb] += 1
                    if in_sw: sw_correct[lb] += 1
                    season_correct[lb][s] += 1
                if in_sw: sw_total[lb] += 1
                season_total[lb][s] += 1
                m.update(h, w)
            
            for cp in CHECKPOINTS:
                if t == cp - 1:
                    for lb in LABELS:
                        results[lb]["acc"][cp].append(correct[lb] / (t + 1))
        
        for lb in LABELS:
            results[lb]["sw_acc"].append(
                sw_correct[lb] / max(sw_total[lb], 1))
            for s in range(4):
                results[lb]["season_acc"][s].append(
                    season_correct[lb][s] / max(season_total[lb][s], 1))
        
        results["5D-Complete"]["anomalies"] += models[0].anomalies
        
        if si == 0:
            prob_hist = np.array(models[0].prob_history)
            season_true = seasons
        
        print("done")
    
    print(f"\n[3/3] Results:")
    return results, prob_hist, season_true, sw_idx


def print_results(results):
    print(f"\n  {'='*80}")
    print(f"  {'Model':<18} {'100d':>8} {'500d':>8} {'1000d':>8} {'1460d':>8} {'Post-Sw':>8}")
    print(f"  {'-'*80}")
    for lb in LABELS:
        row = f"  {lb:<18}"
        for cp in CHECKPOINTS:
            row += f" {np.mean(results[lb]['acc'][cp]):>7.1%}"
        row += f" {np.mean(results[lb]['sw_acc']):>7.1%}"
        print(row)
    
    print(f"\n  {'='*60}")
    print(f"  {'Model':<18} {'春':>10} {'夏':>10} {'秋':>10} {'冬':>10}")
    print(f"  {'-'*60}")
    for lb in LABELS:
        row = f"  {lb:<18}"
        for s in range(4):
            row += f" {np.mean(results[lb]['season_acc'][s]):>9.1%}"
        print(row)
    
    anom = results["5D-Complete"]["anomalies"]
    print(f"\n  Anomalies detected: {anom}")


def plot_results(results, prob_hist, seasons, sw_idx):
    fig = plt.figure(figsize=(16, 12))
    
    # Overall accuracy
    ax1 = fig.add_subplot(2, 2, 1)
    colors = {"5D-Complete":"#1a5276","4D-Flat":"#2980b9",
              "5D-NoReset":"#27ae60","5D-Greedy":"#e67e22","Neural-Net":"#e74c3c"}
    x = np.arange(len(LABELS))
    means = [np.mean(results[lb]["acc"][1460]) for lb in LABELS]
    stds = [np.std(results[lb]["acc"][1460]) for lb in LABELS]
    cis = [1.96*s/np.sqrt(5) for s in stds]
    ax1.bar(x, means, yerr=cis, color=[colors[lb] for lb in LABELS],
            capsize=5, edgecolor="white")
    for i, m in enumerate(means):
        ax1.text(i, m+0.01, f"{m:.1%}", ha="center", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=7)
    ax1.set_ylabel("Accuracy"); ax1.set_title("Overall (1460 days, Beijing 2015-2024)")
    ax1.axhline(0.25, color="gray", ls=":"); ax1.grid(alpha=0.2, axis="y")
    
    # Seasonal breakdown
    ax2 = fig.add_subplot(2, 2, 2)
    sx = np.arange(4); w = 0.15
    for i, lb in enumerate(LABELS):
        sm = [np.mean(results[lb]["season_acc"][s]) for s in range(4)]
        ax2.bar(sx + (i-2)*w, sm, w, color=colors[lb], label=lb, edgecolor="white")
    ax2.set_xticks(sx); ax2.set_xticklabels(["春","夏","秋","冬"])
    ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy by Season")
    ax2.legend(fontsize=7, ncol=2); ax2.grid(alpha=0.2, axis="y")
    
    # Worldline probability evolution
    ax3 = fig.add_subplot(2, 1, 2)
    wl_colors = ["#e74c3c","#27ae60","#e67e22","#2980b9"]
    wl_names = ["Spring","Summer","Autumn","Winter"]
    days = np.arange(len(prob_hist))
    for wl in range(4):
        ax3.plot(days, prob_hist[:, wl], color=wl_colors[wl], lw=1.2, alpha=0.8,
                 label=wl_names[wl])
    for si in sw_idx:
        ax3.axvline(si, color="gray", ls="--", alpha=0.3, lw=0.8)
    ax3.set_xlabel("Day"); ax3.set_ylabel("Probability")
    ax3.set_title("Worldline Probability Evolution (Seed 42)")
    ax3.legend(fontsize=8); ax3.set_ylim(0, 1.05); ax3.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("iching_v11_seasonal.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v11_seasonal.png")
    plt.close(fig)


def main():
    print("="*64)
    print("  V11: Seasonal 5D Projection — Real Beijing Weather")
    print("="*64)
    
    results, ph, seasons, sw = run()
    print_results(results)
    plot_results(results, ph, seasons, sw)

if __name__ == "__main__":
    main()
