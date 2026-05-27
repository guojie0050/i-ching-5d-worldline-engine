#!/usr/bin/env python3
"""
V12: 过去→未来 — 时序分离验证
==============================
核心问题: 5D模型的优势是"拟合过去"还是"预测未来"？

设计:
  - Years 1-5 (2015-2019): 训练集 (各模型内部更新)
  - Years 6-10 (2020-2024): 测试集 (不更新, 纯预测)
  - 若5D的优势来自拟合, 则测试集上优势消失
  - 若5D的优势来自世界线追踪, 则测试集上保持优势

对比: 5D-Complete vs 4D-FlatBayes vs Neural-Net
"""

import numpy as np
import json, os, warnings, urllib.request
from datetime import date
warnings.filterwarnings("ignore")

N_HEX, N_W = 64, 4
TEMP = 0.3; PRIOR_STR = 5.0

# 八卦 + 64卦定义(同V11)
TRIGRAM_4D = np.array([
    [0.55,0.10,0.15,0.20],[0.05,0.25,0.30,0.40],
    [0.15,0.35,0.35,0.15],[0.20,0.15,0.50,0.15],
    [0.10,0.50,0.20,0.20],[0.50,0.05,0.15,0.30],
    [0.10,0.20,0.15,0.55],[0.15,0.40,0.20,0.25],
])
KWT = [(0,0),(7,7),(2,4),(7,2),(2,0),(0,2),(7,7),(2,7),
       (5,0),(0,6),(7,0),(0,7),(0,3),(3,0),(7,1),(4,7),
       (6,4),(1,5),(7,6),(5,7),(3,4),(1,3),(1,7),(5,1),
       (0,4),(1,0),(1,4),(6,5),(2,2),(3,3),(6,3),(4,3),
       (0,1),(4,0),(3,7),(7,3),(5,5),(6,4),(2,1),(1,2),
       (1,6),(5,4),(6,0),(0,5),(6,7),(7,1),(6,6),(5,1),
       (6,5),(3,5),(4,4),(1,1),(5,1),(4,6),(4,5),(3,6),
       (5,5),(6,6),(5,2),(2,6),(5,6),(4,1),(2,3),(3,2)]

def nat_dists():
    nd = np.zeros((N_HEX,N_W))
    for h,(u,l) in enumerate(KWT):
        nd[h] = 0.55*TRIGRAM_4D[u] + 0.45*TRIGRAM_4D[l]
        nd[h] /= nd[h].sum()
    return nd

# 干支→卦象
GZ_REF = date(1900,1,1)
def date_to_hex(d):
    return ((d - GZ_REF).days % 60) % 64

# 天气
WMO = {0:0,1:0,2:2,3:1,45:2,48:2,51:4,53:4,55:4,61:4,63:4,65:4,
       71:3,73:3,75:3,77:3,85:3,86:3,80:4,81:4,82:4,95:3,96:3,99:3}

def load_beijing():
    cache = "/tmp/beijing_weather.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           "latitude=39.9&longitude=116.4&start_date=2015-01-01&end_date=2024-12-31"
           "&daily=weather_code,temperature_2m_max,snowfall_sum&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(url).read())
    json.dump(d, open(cache,"w")); return d

def to_arrays(data):
    daily = data["daily"]; n = len(daily["time"])
    ts = daily["time"]; cs = daily["weather_code"]
    tm = daily["temperature_2m_max"]; sn = daily["snowfall_sum"]
    
    hexs = np.zeros(n, dtype=int); wths = np.zeros(n, dtype=int)
    for i in range(n):
        y,m,d = ts[i].split("-"); dt = date(int(y),int(m),int(d))
        hexs[i] = date_to_hex(dt)
        c, t, s = cs[i], tm[i] if tm[i] else 10, sn[i] if sn[i] else 0
        if s and s>0 and t<2: wths[i] = 3
        else: wths[i] = WMO.get(c,2) % 4
    return hexs, wths


# ============================================================================
# 模型(简化版)
# ============================================================================
class FiveDModel:
    def __init__(self, nd):
        self.n_wl = 4  # 四季
        self.counts = np.ones((4,N_HEX,N_W))*PRIOR_STR
        for wl in range(4): self.counts[wl] += PRIOR_STR*nd
        self.probs = np.ones(4)/4
    
    def predict(self, h):
        pred = np.zeros(N_W)
        for wl in range(4):
            p = self.counts[wl,h]/self.counts[wl,h].sum()
            pred += self.probs[wl]*p
        return pred/pred.sum()
    
    def update(self, h, w):
        likes = np.zeros(4)
        for wl in range(4):
            p=self.counts[wl,h]/self.counts[wl,h].sum(); likes[wl]=p[w]
        lp=np.log(np.maximum(self.probs,1e-12))+np.log(np.maximum(likes,1e-12))/TEMP
        lp-=lp.max(); self.probs=np.exp(lp); self.probs/=self.probs.sum()
        for wl in range(4): self.counts[wl,h,w]+=self.probs[wl]

class FlatBayes:
    def __init__(self, nd):
        self.counts = np.ones((N_HEX,N_W))*PRIOR_STR + PRIOR_STR*nd
    def predict(self, h):
        return self.counts[h]/self.counts[h].sum()
    def update(self, h, w):
        self.counts[h,w] += 1.0

class SimpleNN:
    def __init__(self, seed=42):
        rng=np.random.default_rng(seed)
        self.W1=rng.normal(0,0.1,(N_HEX,32)); self.b1=np.zeros(32)
        self.W2=rng.normal(0,0.1,(32,N_W)); self.b2=np.zeros(N_W); self.lr=0.005
    def predict(self, h):
        x=np.zeros(N_HEX);x[h]=1.0
        hh=np.maximum(0,x@self.W1+self.b1)
        l=hh@self.W2+self.b2;l-=l.max();p=np.exp(l);return p/p.sum()
    def update(self, h, w):
        x=np.zeros(N_HEX);x[h]=1.0
        hp=x@self.W1+self.b1;hh=np.maximum(0,hp)
        l=hh@self.W2+self.b2;l-=l.max();p=np.exp(l);p/=p.sum()
        dl=p.copy();dl[w]-=1.0
        self.W2-=self.lr*np.outer(hh,dl);self.b2-=self.lr*dl
        dh=self.W2@dl;dh[hp<=0]=0.0
        self.W1-=self.lr*np.outer(x,dh);self.b1-=self.lr*dh


# ============================================================================
# 时序分离实验
# ============================================================================
SEEDS = [42,123,456,789,1011]

def run():
    hexs_all, wths_all = to_arrays(load_beijing())
    n_total = len(hexs_all)
    nd = nat_dists()
    
    # 5年训练 + 5年测试 窗口, 随机偏移
    results = {"5D-Complete": {"train":[],"test":[],"season":[]},
               "4D-FlatBayes":{"train":[],"test":[],"season":[]},
               "Neural-Net":  {"train":[],"test":[],"season":[]}}
    
    season_labels = {0:"冬",1:"春",2:"夏",3:"秋"}
    
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        start = rng.integers(0, n_total - 365*10)
        mid = start + 365*5
        end = start + 365*10
        
        train_h = hexs_all[start:mid]; train_w = wths_all[start:mid]
        test_h  = hexs_all[mid:end];   test_w  = wths_all[mid:end]
        
        models = [FiveDModel(nd), FlatBayes(nd), SimpleNN(seed=seed)]
        names = ["5D-Complete","4D-FlatBayes","Neural-Net"]
        
        # 训练
        for t in range(len(train_h)):
            for m in models:
                m.update(train_h[t], train_w[t])
        
        # 训练集评估
        train_corr = {n:0 for n in names}
        for t in range(len(train_h)):
            for n, m in zip(names, models):
                if np.argmax(m.predict(train_h[t])) == train_w[t]:
                    train_corr[n] += 1
                # 不更新! 纯评估
        
        # 测试集评估
        test_corr = {n:0 for n in names}
        season_corr = {n:{s:0 for s in range(4)} for n in names}
        season_cnt = {s:0 for s in range(4)}
        
        for t in range(len(test_h)):
            dt = date(2020 + t//365, (t%365)//30 + 1, (t%365)%28 + 1)
            season = (dt.month % 12) // 3  # 0=冬,1=春,2=夏,3=秋
            season_cnt[season] += 1
            
            for n, m in zip(names, models):
                if np.argmax(m.predict(test_h[t])) == test_w[t]:
                    test_corr[n] += 1
                    season_corr[n][season] += 1
        
        for n in names:
            results[n]["train"].append(train_corr[n]/len(train_h))
            results[n]["test"].append(test_corr[n]/len(test_h))
            for s in range(4):
                if season_cnt[s] > 0:
                    results[n]["season"].append(season_corr[n][s]/season_cnt[s])
        
        print(f"  Seed {si+1}: 5D train={train_corr['5D-Complete']:.1%} "
              f"test={test_corr['5D-Complete']:.1%} | "
              f"Flat test={test_corr['4D-FlatBayes']:.1%} | "
              f"NN test={test_corr['Neural-Net']:.1%}")
    
    print(f"\n  {'='*55}")
    print(f"  {'Model':<16} {'Train':>10} {'Test':>10} {'Gap':>8}")
    print(f"  {'-'*45}")
    for n in ["5D-Complete","4D-FlatBayes","Neural-Net"]:
        tr = np.mean(results[n]["train"])
        te = np.mean(results[n]["test"])
        print(f"  {n:<16} {tr:>9.1%} {te:>9.1%} {tr-te:>+7.1%}")
    
    s5d = results["5D-Complete"]["test"]
    s4d = results["4D-FlatBayes"]["test"]
    nn  = results["Neural-Net"]["test"]
    print(f"\n  5D vs Flat on test: {np.mean(s5d):.1%} vs {np.mean(s4d):.1%} "
          f"(Δ={np.mean(s5d)-np.mean(s4d):+.1%})")
    print(f"  5D vs NN on test:   {np.mean(s5d):.1%} vs {np.mean(nn):.1%} "
          f"(Δ={np.mean(s5d)-np.mean(nn):+.1%})")


if __name__ == "__main__":
    run()
