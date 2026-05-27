#!/usr/bin/env python3
"""
V10 消融实验 — 四城市数据
==========================
验证五维投影理论各组件的因果贡献:
  ① 温度缩放 (T)
  ② 结构化先验 (prior)
  ③ 加权更新 (weight)
  ④ 异常检测 (anomaly)

城市: Beijing, Shanghai, Guangzhou, Chengdu
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, warnings
from datetime import date
warnings.filterwarnings("ignore")

N_HEX, N_W = 64, 4
TEMP = 0.3; PRIOR_STR = 5.0
TOTAL_DAYS = 2000
SEEDS = [42, 123, 456, 789, 1011]

# 八卦天气向量
TRIGRAM_4D = np.array([
    [0.55, 0.10, 0.15, 0.20],[0.05, 0.25, 0.30, 0.40],
    [0.15, 0.35, 0.35, 0.15],[0.20, 0.15, 0.50, 0.15],
    [0.10, 0.50, 0.20, 0.20],[0.50, 0.05, 0.15, 0.30],
    [0.10, 0.20, 0.15, 0.55],[0.15, 0.40, 0.20, 0.25],
])

KING_WEN_TRIGRAMS = [
    (0,0),(7,7),(2,4),(7,2),(2,0),(0,2),(7,7),(2,7),
    (5,0),(0,6),(7,0),(0,7),(0,3),(3,0),(7,1),(4,7),
    (6,4),(1,5),(7,6),(5,7),(3,4),(1,3),(1,7),(5,1),
    (0,4),(1,0),(1,4),(6,5),(2,2),(3,3),(6,3),(4,3),
    (0,1),(4,0),(3,7),(7,3),(5,5),(6,4),(2,1),(1,2),
    (1,6),(5,4),(6,0),(0,5),(6,7),(7,1),(6,6),(5,1),
    (6,5),(3,5),(4,4),(1,1),(5,1),(4,6),(4,5),(3,6),
    (5,5),(6,6),(5,2),(2,6),(5,6),(4,1),(2,3),(3,2),
]

def natural_dists():
    nd = np.zeros((N_HEX, N_W))
    for h, (u, l) in enumerate(KING_WEN_TRIGRAMS):
        nd[h] = 0.55 * TRIGRAM_4D[u] + 0.45 * TRIGRAM_4D[l]
        nd[h] /= nd[h].sum()
    return nd

# ============================================================================
# 真实天气 → 4类
# ============================================================================
WMO = {0:0,1:0,2:2,3:1,45:7,48:7,51:4,53:4,55:4,61:4,63:4,65:4,
       71:7,73:7,75:7,80:4,81:4,82:4,95:3,96:3,99:3}
# 映射到4类: 0=clear,1=rain,2=overcast,3=fog/snow

def load_city(name):
    path = f"/tmp/{name}_weather.json"
    if not os.path.exists(path): return None
    d = json.load(open(path))
    codes = d["daily"]["weather_code"]
    temps = d["daily"]["temperature_2m_max"]
    precip = d["daily"]["precipitation_sum"]
    
    wtypes = np.zeros(len(codes), dtype=int)
    for i, (c, t, p) in enumerate(zip(codes, temps, precip)):
        if t and t > 32: wtypes[i] = 0  # hot → clear
        else:
            w = WMO.get(c, 1)
            wtypes[i] = w % 4  # map to 0-3
    return wtypes


# ============================================================================
# 消融模型
# ============================================================================
class AblationModel:
    def __init__(self, n_wl=3, temp=TEMP, use_prior=True, use_weight=True, 
                 use_anomaly=True, natural=None):
        self.n_wl = n_wl; self.T = temp; self.use_anomaly = use_anomaly
        self.counts = np.ones((n_wl, N_HEX, N_W)) * PRIOR_STR
        if use_prior and natural is not None:
            for wl in range(n_wl):
                self.counts[wl] += PRIOR_STR * natural
        self.probs = np.ones(n_wl) / n_wl
        self.prob_history = []; self.anomalies = 0
        self.use_weight = use_weight
    
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
        
        if self.T != 1.0:
            lp = np.log(np.maximum(self.probs, 1e-12))
            lp = lp + np.log(np.maximum(likes, 1e-12)) / self.T
            lp -= lp.max()
            self.probs = np.exp(lp); self.probs /= self.probs.sum()
        else:
            self.probs = self.probs * likes
            self.probs /= self.probs.sum()
        
        for wl in range(self.n_wl):
            self.counts[wl, h, w] += self.probs[wl] if self.use_weight else 1.0
        
        self.prob_history.append(self.probs.copy())
        if self.use_anomaly:
            self._check_anomaly(likes.max())
    
    def _check_anomaly(self, max_ll):
        if not hasattr(self, '_recent'): self._recent = []
        self._recent.append(max_ll)
        if len(self._recent) > 3: self._recent.pop(0)
        if len(self._recent) == 3 and all(ll < 0.10 for ll in self._recent):
            self.probs = np.ones(self.n_wl) / self.n_wl
            self._recent = []; self.anomalies += 1


class FlatBayes:
    def __init__(self, natural=None):
        self.counts = np.ones((N_HEX, N_W)) * PRIOR_STR
        if natural is not None: self.counts += PRIOR_STR * natural
    def predict(self, h):
        return self.counts[h] / self.counts[h].sum()
    def update(self, h, w):
        self.counts[h, w] += 1.0


class SimpleNN:
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
        dW2 = np.outer(hh, dl); db2 = dl
        dh = self.W2 @ dl; dh[hp <= 0] = 0.0
        dW1 = np.outer(x, dh); db1 = dh
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1


# ============================================================================
# 实验
# ============================================================================
ABLATION_LABELS = [
    "5D-Complete","5D-NoTemp","5D-NoPrior",
    "5D-NoWeight","5D-NoAnomaly","4D-Flat","Neural-Net"
]
CITIES = ["beijing","shanghai","guangzhou","chengdu"]

def make_ablation_models(natural):
    nd = natural if natural is not None else natural_dists()
    return [
        AblationModel(temp=TEMP, use_prior=True, use_weight=True, use_anomaly=True, natural=nd),
        AblationModel(temp=1.0, use_prior=True, use_weight=True, use_anomaly=True, natural=nd),
        AblationModel(temp=TEMP, use_prior=False, use_weight=True, use_anomaly=True, natural=nd),
        AblationModel(temp=TEMP, use_prior=True, use_weight=False, use_anomaly=True, natural=nd),
        AblationModel(temp=TEMP, use_prior=True, use_weight=True, use_anomaly=False, natural=nd),
        FlatBayes(natural=nd),
        SimpleNN(seed=42),
    ]

def run_city(weather_seq):
    n = len(weather_seq)
    results = {lb: [] for lb in ABLATION_LABELS}
    
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        start = rng.integers(0, max(1, n - TOTAL_DAYS))
        w_sub = weather_seq[start:start+TOTAL_DAYS]
        hexs = rng.integers(0, N_HEX, TOTAL_DAYS)
        
        models = make_ablation_models(natural_dists())
        correct = {lb: 0 for lb in ABLATION_LABELS}
        
        for t in range(TOTAL_DAYS):
            h, tw = hexs[t], w_sub[t]
            for lb, m in zip(ABLATION_LABELS, models):
                if np.argmax(m.predict(h)) == tw:
                    correct[lb] += 1
                m.update(h, tw)
        
        for lb in ABLATION_LABELS:
            results[lb].append(correct[lb] / TOTAL_DAYS)
    
    return results


def main():
    print("="*64)
    print("  V10 Ablation — 4 Cities")
    print("="*64)
    
    all_results = {}
    for city in CITIES:
        wseq = load_city(city)
        if wseq is None:
            print(f"  {city}: NO DATA")
            continue
        print(f"\n  {city}: {len(wseq)} days")
        res = run_city(wseq)
        all_results[city] = res
        for lb in ABLATION_LABELS:
            print(f"    {lb:<16}: {np.mean(res[lb]):.1%}")
    
    # 汇总表
    print(f"\n  {'='*80}")
    print(f"  {'City':<12}", end="")
    for lb in ABLATION_LABELS:
        print(f" {lb:<12}", end="")
    print(f"\n  {'-'*80}")
    for city in CITIES:
        if city in all_results:
            row = f"  {city:<12}"
            for lb in ABLATION_LABELS:
                row += f" {np.mean(all_results[city][lb]):<11.1%}"
            print(row)
    
    # 消融贡献
    print(f"\n  消融因果链 (Beijing):")
    bj = all_results.get("beijing", {})
    if bj:
        complete = np.mean(bj["5D-Complete"])
        for name, lb in [("温度缩放", "5D-NoTemp"), ("结构化先验", "5D-NoPrior"),
                          ("加权更新", "5D-NoWeight"), ("异常检测", "5D-NoAnomaly")]:
            val = np.mean(bj[lb])
            print(f"    -{name}: {val:.1%} (Δ={complete-val:+.1%})")


if __name__ == "__main__":
    main()
