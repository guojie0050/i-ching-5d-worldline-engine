#!/usr/bin/env python3
"""
V6 实验: 多特征输入 — 温度/降水/风速/湿度增强预测
=====================================================

对比:
  - Trigram-Bayes + features
  - Trigram-Bayes (weather-code only, V5 baseline)
  - Neural-Net + features (60-dim input)
  - Neural-Net (24-dim input, V5 baseline)

数据: Beijing 2015-2024, 5 特征 × 3653 天
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, urllib.request, os, csv, warnings
warnings.filterwarnings("ignore")

from model import (
    TrigramBayesianModel, NeuralNetModel,
    TRIGRAM_WEATHER, N_WEATHER, WEATHER_TYPES, HEX_TO_TRIGRAMS,
)

# ============================================================================
# 多特征数据处理
# ============================================================================

WMO_MAP = {0:0,1:0,2:2,3:1,45:7,48:7,51:4,53:4,55:4,61:4,63:4,65:4,
           71:7,73:7,75:7,80:4,81:4,82:4,95:3,96:3,99:3}
HOT_T = 32.0
N_FEAT = 5  # weather_code + temp + precip + wind + humidity

def fetch_data(cache="/tmp/beijing_multi.json"):
    if os.path.exists(cache):
        return json.load(open(cache))
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           "latitude=39.9&longitude=116.4&start_date=2015-01-01&end_date=2024-12-31"
           "&daily=weather_code,temperature_2m_max,precipitation_sum,"
           "wind_speed_10m_max,relative_humidity_2m_mean&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(url).read())
    json.dump(d, open(cache,"w"))
    return d

def to_arrays(data):
    """返回 (weather_seq, feature_matrix)。feature_matrix: (n_days, 5) 归一化。"""
    daily = data["daily"]
    n = len(daily["weather_code"])
    
    # Weather types (0-7)
    codes = daily["weather_code"]
    temps = daily["temperature_2m_max"]
    wtypes = np.zeros(n, dtype=int)
    for i in range(n):
        t = temps[i]
        wtypes[i] = 5 if (t and t > HOT_T) else WMO_MAP.get(codes[i], 1)
    
    # Feature matrix: [temp, precip, wind, humidity] (归一化到 [0,1])
    feats = np.zeros((n, N_FEAT))
    # 0: weather_code (normalized to 0-1 range)
    feats[:,0] = wtypes / 7.0
    
    # 1: temperature_2m_max
    t_vals = np.array([v if v else 15.0 for v in temps])
    feats[:,1] = (t_vals + 15) / 60.0  # map [-15,45] to [0,1]
    
    # 2: precipitation_sum (log-scale)
    p_vals = np.array([v if v else 0.0 for v in daily["precipitation_sum"]])
    feats[:,2] = np.log1p(p_vals) / np.log1p(200)  # log(1+x)/log(201)
    
    # 3: wind_speed_10m_max
    w_vals = np.array([v if v else 10.0 for v in daily["wind_speed_10m_max"]])
    feats[:,3] = w_vals / 60.0
    
    # 4: relative_humidity_2m_mean
    h_vals = np.array([v if v else 50.0 for v in daily["relative_humidity_2m_mean"]])
    feats[:,4] = h_vals / 100.0
    
    feats = np.clip(feats, 0, 1)
    return wtypes, feats


# ============================================================================
# 增强版 I Ching 模型: 特征相似度调制卦象权重
# ============================================================================

class FeaturedTrigramModel(TrigramBayesianModel):
    """
    多特征三爻共享模型。
    
    在原有 hex_ll 权重基础上，加入特征匹配奖励:
      relevance[h] = softmax(hex_ll[h]/T + feat_match[h] * feat_weight)
    
    每个卦象维护一个"特征原型" (5维均值向量),
    特征匹配度 = 当前特征向量与原型的内积相似度。
    """
    
    def __init__(self, trigram_affinities, prior_strength=1.0, temperature=0.5,
                 feat_weight=0.5):
        super().__init__(trigram_affinities, prior_strength, temperature)
        self.feat_w = feat_weight
        # 每个卦象的特征原型: (64, 5), 初始化为卦象亲和度的简单编码
        self.prototypes = np.zeros((self.nh, N_FEAT))
        for h in range(self.nh):
            # 原型初始 = 卦象主导天气倾向 (feat 0)
            tu, tl = HEX_TO_TRIGRAMS[h]
            avg_aff = (trigram_affinities[tu] + trigram_affinities[tl]) / 2
            self.prototypes[h, 0] = avg_aff.argmax() / 7.0
        self.prototypes[:, 1:] = 0.5  # 中性初始化
        self.proto_count = np.ones(self.nh)  # 更新计数
    
    def _feat_match(self, features):
        """计算当前特征与各卦象原型的余弦相似度。"""
        if features is None or len(features) == 0:
            return np.ones(self.nh)
        f = np.array(features[-1])  # 最新一天的特征
        # 余弦相似度
        proto_norm = np.linalg.norm(self.prototypes, axis=1) + 1e-8
        feat_norm = np.linalg.norm(f) + 1e-8
        sim = (self.prototypes @ f) / (proto_norm * feat_norm)
        return np.clip(sim + 1.0, 0.5, 2.0)  # map [-1,1] to [0.5, 2.0]
    
    def predict(self, history, features=None):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        wc = history[-1]
        
        lw = self.hex_ll / max(self.T, 0.01)
        lw -= lw.max()
        wts = np.exp(lw)
        
        # 特征匹配奖励
        if features is not None and len(features) >= 1:
            wts *= self._feat_match(features) ** self.feat_w
        
        s = wts.sum()
        wts = wts / s if s > 1e-12 else np.ones(self.nh) / self.nh
        
        pred = np.zeros(self.nw)
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wc, :] / alpha_h[wc, :].sum()
            pred += wts[h] * p
        return pred / pred.sum()
    
    def update(self, history, observed, features=None, lr=1.0):
        super().update(history, observed, lr)
        # 更新特征原型
        if features is not None and len(features) >= 1:
            f = np.array(features[-1])
            for h in range(self.nh):
                # EMA 更新原型
                alpha_ema = 0.01
                self.prototypes[h] = (1 - alpha_ema) * self.prototypes[h] + alpha_ema * f


# ============================================================================
# 增强版 NN: 更多输入维度
# ============================================================================

class FeaturedNN(NeuralNetModel):
    """NN with feature vector input instead of one-hot."""
    
    def __init__(self, ctx_win=3, hidden=32, lr=0.005, seed=42):
        d = ctx_win * N_FEAT  # 3 * 5 = 15
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2./d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2./hidden), (hidden, N_WEATHER))
        self.b2 = np.zeros(N_WEATHER)
        self.lr = lr; self.cw = ctx_win
    
    def _enc(self, features):
        """Encode feature history into vector."""
        r = features[-self.cw:] if len(features) >= self.cw else features
        v = np.zeros(self.cw * N_FEAT)
        off = (self.cw - len(r)) * N_FEAT
        for i, f in enumerate(r):
            v[off + i*N_FEAT : off + (i+1)*N_FEAT] = np.array(f)
        return v
    
    def predict(self, history, features=None):
        if features is None or len(features) == 0:
            return np.ones(N_WEATHER) / N_WEATHER
        x = self._enc(features)
        h = np.maximum(0, x @ self.W1 + self.b1)
        l = h @ self.W2 + self.b2; l -= l.max()
        p = np.exp(l); return p / p.sum()
    
    def update(self, history, observed, features=None, lr=None):
        if features is None or len(features) == 0:
            return
        if lr is None: lr = self.lr
        x = self._enc(features)
        hp = x @ self.W1 + self.b1; h = np.maximum(0, hp)
        l = h @ self.W2 + self.b2; l -= l.max()
        p = np.exp(l); p /= p.sum()
        dl = p.copy(); dl[observed] -= 1.
        dW2 = np.outer(h, dl); db2 = dl
        dh = self.W2 @ dl; dh[hp <= 0] = 0.
        dW1 = np.outer(x, dh); db1 = dh
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1


# ============================================================================
# 实验
# ============================================================================

LABELS = ["Trigram+Feat", "Trigram-Base", "NN+Feat", "NN-Base"]
SIZES = [100, 200, 500, 1000, 2000, 3000]
EVAL_W = 365; SEEDS = [42, 123, 456, 789, 1011]
PS, T, FW = 1.0, 0.5, 0.5

def make_models(seed):
    return [
        FeaturedTrigramModel(TRIGRAM_WEATHER, PS, T, feat_weight=FW),
        TrigramBayesianModel(TRIGRAM_WEATHER, PS, T),
        FeaturedNN(ctx_win=3, hidden=32, lr=0.005, seed=seed),
        NeuralNetModel(ctx_win=3, hidden=32, lr=0.005, seed=seed),
    ], LABELS

class WrapperNN:
    """Wrapper: NN-Base only uses weather_code, not features."""
    def __init__(self, nn): self.nn = nn
    def predict(self, h, f=None): return self.nn.predict(h)
    def update(self, h, o, f=None, lr=None): self.nn.update(h, o, lr)

def train(m, wtypes, feats):
    for t in range(1, len(wtypes)):
        hist = wtypes[:t].tolist()
        if isinstance(m, (FeaturedTrigramModel, FeaturedNN)):
            ft = feats[:t].tolist()
            m.update(hist, wtypes[t], ft)
        else:
            m.update(hist, wtypes[t])

def evaluate(m, train_w, eval_w, train_f, eval_f):
    h = train_w.tolist(); c = 0
    has_f = isinstance(m, (FeaturedTrigramModel, FeaturedNN))
    fh = train_f.tolist() if has_f else None
    for i in range(len(eval_w)):
        pred = m.predict(h, fh) if has_f else m.predict(h)
        if np.argmax(pred) == eval_w[i]: c += 1
        h.append(eval_w[i])
        if fh is not None: fh.append(eval_f[i].tolist())
    return c / len(eval_w)

def run(wtypes, feats):
    res = {N: {lb: [] for lb in LABELS} for N in SIZES}
    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)}...", end=" ", flush=True)
        for N in SIZES:
            models, labels = make_models(seed)
            tw = wtypes[:N]; ew = wtypes[N:N+EVAL_W]
            tf = feats[:N]; ef = feats[N:N+EVAL_W]
            for m in models:
                train(m, tw, tf)
            for lb, m in zip(labels, models):
                res[N][lb].append(evaluate(m, tw, ew, tf, ef))
        print("done")
    return res

def plot(res):
    colors = {"Trigram+Feat":"#1a5276","Trigram-Base":"#2980b9",
              "NN+Feat":"#e74c3c","NN-Base":"#e67e22"}
    marks = {"Trigram+Feat":"o","Trigram-Base":"s","NN+Feat":"D","NN-Base":"^"}
    
    fig, ax = plt.subplots(figsize=(12,6))
    for lb in LABELS:
        means = [np.mean(res[N][lb]) for N in SIZES]
        stds = [np.std(res[N][lb]) for N in SIZES]
        cis = [1.96*s/np.sqrt(len(SEEDS)) for s in stds]
        ax.errorbar(SIZES, means, yerr=cis, color=colors[lb],
                    marker=marks[lb], lw=2, ms=8, capsize=4, label=lb)
    ax.axhline(1./N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Training Days"); ax.set_ylabel("Accuracy")
    ax.set_title("V6: Multi-Feature Input (temp+precip+wind+humidity)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xscale("log"); ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    plt.tight_layout()
    plt.savefig("iching_v6_features.png", dpi=150, bbox_inches="tight")
    print("[图] iching_v6_features.png")
    plt.close(fig)

def save_csv(res, path="results/v6_features.csv"):
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
    print("  V6: Multi-Feature Weather Prediction")
    print("="*64)
    
    print("\n[1/3] Loading data...")
    data = fetch_data()
    wtypes, feats = to_arrays(data)
    print(f"      {len(wtypes)} days, {N_FEAT} features")
    
    print(f"\n[2/3] Experiment...")
    results = run(wtypes, feats)
    
    print(f"\n[3/3] Results:")
    hdr = f"  {'Days':<8}"
    for lb in LABELS: hdr += f" {lb:<14}"
    print(hdr); print(f"  {'-'*60}")
    for N in SIZES:
        row = f"  {N:<8}"
        for lb in LABELS:
            row += f" {np.mean(results[N][lb]):<13.1%}"
        print(row)
    
    # Feature impact
    N_k = 100
    t_f = np.mean(results[N_k]["Trigram+Feat"])
    t_b = np.mean(results[N_k]["Trigram-Base"])
    n_f = np.mean(results[N_k]["NN+Feat"])
    n_b = np.mean(results[N_k]["NN-Base"])
    
    print(f"\n  特征增益 (@{N_k}天):")
    print(f"    Trigram: +features → {t_f:.1%} (base={t_b:.1%}, Δ={t_f-t_b:+.1%})")
    print(f"    NN:      +features → {n_f:.1%} (base={n_b:.1%}, Δ={n_f-n_b:+.1%})")
    
    plot(results)
    save_csv(results)
    print(f"\n{'='*64}")

if __name__ == "__main__":
    main()
