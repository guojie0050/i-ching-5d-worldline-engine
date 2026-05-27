#!/usr/bin/env python3
"""
5D × Sanchen 融合实验
=======================
三陈共识网络 + 多世界线切换农事环境
3条世界线(旱年/涝年/正常年), 每100-300天切换一年类型
卦师团必须追踪世界线并决策
"""

import numpy as np
from sanchen_diviner import SanchenDiviner, divine_full
import warnings; warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1011]
N_YEARS = 20

def generate_worldline_years(seed):
    """生成20年世界线切换序列"""
    rng = np.random.default_rng(seed)
    years = []
    wl = 0; switch_at = rng.integers(3, 8)
    for y in range(N_YEARS):
        if y >= switch_at:
            new_wl = rng.integers(3)
            if new_wl != wl: years.append({'switch': True}) if y > 0 else None
            wl = new_wl
            switch_at = y + rng.integers(3, 8)
        yr_type = ['旱','涝','正常'][wl]
        # 在世界线内生成天气: 旱年偏旱, 涝年偏涝
        if yr_type == '旱':
            precip = rng.uniform(0.1, 0.7)
        elif yr_type == '涝':
            precip = rng.uniform(2.0, 3.5)
        else:
            precip = rng.uniform(0.8, 1.8)
        
        years.append({
            'year': 2005 + y, 'worldline': wl, 'type': yr_type,
            'precip': precip, 'switch': False, 'extreme': precip < 0.3 or precip > 3.0
        })
    return years

def compute_harvest(choice, weather_type, extreme):
    base = 100
    if weather_type == '旱': base *= 1.5 if choice == '旱稻' else 0.4
    elif weather_type == '涝': base *= 1.5 if choice == '水稻' else 0.4
    else: base *= 1.2 if choice == '水稻' else 0.85
    if extreme: base *= 0.7
    return base


# ============================================================================
# 世界线感知版三陈卦师
# ============================================================================
class WorldlineAwareDiviner:
    """三陈卦师 + 世界线概率追踪"""
    def __init__(self, seed=42):
        self.diviner = SanchenDiviner(seed, cognitive_style='balanced', cold_start=True)
        self.wl_probs = np.ones(3) / 3  # 旱/涝/正常
        self.wl_history = []
    
    def consult(self, hexagram):
        r = self.diviner.consult(hexagram)
        wl = np.argmax(self.wl_probs)
        # 世界线调制: 当前最可能的世界线影响最终决策
        if wl == 0 and r['sanchen_decision'] == '进取':
            r['recommendation'] = '旱稻'
        elif wl == 1 and r['sanchen_decision'] == '保守':
            r['recommendation'] = '水稻'
        return r
    
    def update_worldline(self, actual_type, decision):
        """用真实年类型更新世界线概率 + 三陈贝叶斯更新"""
        target = {'旱': 0, '涝': 1, '正常': 2}[actual_type]
        # 似然更新
        likes = np.ones(3) * 0.1
        likes[target] = 0.8
        lp = np.log(np.maximum(self.wl_probs, 1e-12)) + np.log(likes) / 0.3
        lp -= lp.max()
        self.wl_probs = np.exp(lp); self.wl_probs /= self.wl_probs.sum()
        self.wl_history.append(self.wl_probs.copy())
        
        # 三陈贝叶斯更新
        optimal = {'旱': '旱稻', '涝': '水稻', '正常': '水稻'}[actual_type]
        self.diviner.update(None, None, optimal)


# ============================================================================
# 共识网络
# ============================================================================
class WorldlineConsensus:
    def __init__(self, n=5, seed=42):
        self.diviners = [WorldlineAwareDiviner(seed + i*100) for i in range(n)]
    def decide(self, hexagram):
        results = [d.consult(hexagram) for d in self.diviners]
        decs = [r['recommendation'] for r in results]
        vote = {}
        for d in decs: vote[d] = vote.get(d, 0) + 1
        return max(vote, key=vote.get)
    def update(self, actual_type):
        for d in self.diviners:
            d.update_worldline(actual_type, None)


# ============================================================================
# 实验
# ============================================================================
def run():
    configs = {
        '5D-Sanchen-5': lambda s: WorldlineConsensus(5, s),
        '5D-Sanchen-1': lambda s: WorldlineAwareDiviner(s),
        'Flat-Sanchen-1': lambda s: SanchenSingleWL(s),
        'DQN': lambda s: WorldlineDQN(),
    }
    results = {n: {'harvest': [], 'sw_recovery': []} for n in configs}
    
    for si, seed in enumerate(SEEDS):
        print(f"\n  Seed {si+1}/{len(SEEDS)}")
        years = generate_worldline_years(seed)
        models = {n: f(seed) for n, f in configs.items()}
        harvests = {n: 0 for n in models}
        
        # 切换后恢复追踪
        for n in models: results[n].setdefault('switches', 0)
        
        for yr_idx, yr_info in enumerate(years):
            y = yr_info['year']; actual = yr_info['type']
            h = divine_full(y)
            optimal = {'旱': '旱稻', '涝': '水稻', '正常': '水稻'}[actual]
            
            for name, model in models.items():
                if 'Consensus' in str(type(model)):
                    dec = model.decide(h)
                elif 'WL' in name:
                    r = model.consult(h) if hasattr(model, 'consult') else model.decide(h)
                    dec = r['recommendation'] if isinstance(r, dict) else r
                elif name == 'DQN':
                    dec = model.decide(y)
                else:
                    dec = model.decide(h)
                
                harvests[name] += compute_harvest(dec, actual, yr_info['extreme'])
            
            # 更新世界线追踪
            if '5D-Sanchen-5' in models:
                models['5D-Sanchen-5'].update(actual)
            if '5D-Sanchen-1' in models:
                models['5D-Sanchen-1'].update_worldline(actual, None)
            if 'DQN' in models:
                models['DQN'].update(optimal)
        
        for n in models:
            results[n]['harvest'].append(harvests[n])
        
        print(f"    5D-S5={harvests['5D-Sanchen-5']:.0f}  5D-S1={harvests['5D-Sanchen-1']:.0f}  "
              f"Flat={harvests['Flat-Sanchen-1']:.0f}  DQN={harvests['DQN']:.0f}")
    
    return results

class SanchenSingleWL:
    def __init__(self, seed=42):
        self.diviner = SanchenDiviner(seed, cognitive_style='balanced')
    def decide(self, hexagram):
        return self.diviner.consult(hexagram)['recommendation']
    def update(self, *args): pass

class WorldlineDQN:
    def __init__(self):
        self.Q = {}; self.lr=0.1; self.gamma=0.9; self.eps=0.3
        self.last_s=None; self.last_a=None
    def decide(self, year):
        s = year % 5
        if s not in self.Q or np.random.random() < self.eps:
            a = np.random.choice(['旱稻','水稻'])
        else:
            a = max(self.Q[s], key=self.Q[s].get)
        self.last_s, self.last_a = s, a
        return a
    def update(self, optimal):
        if self.last_s is not None:
            s,a = self.last_s, self.last_a
            if s not in self.Q: self.Q[s] = {x:0.0 for x in ['旱稻','水稻']}
            r = 1.0 if (a == optimal) else 0.0
            ns = (s+1)%5
            if ns not in self.Q: self.Q[ns] = {x:0.0 for x in ['旱稻','水稻']}
            self.Q[s][a] += self.lr*(r+self.gamma*max(self.Q[ns].values())-self.Q[s][a])

def main():
    print("="*64)
    print("  5D × Sanchen: Worldline Tracking + Consensus")
    print("="*64)
    results = run()
    
    print(f"\n  {'='*55}")
    print(f"  {'Model':<18} {'Harvest':>12}")
    for n in results:
        print(f"  {n:<18} {np.mean(results[n]['harvest']):>7.0f}±{np.std(results[n]['harvest']):.0f}")
    
    sd5 = np.mean(results['5D-Sanchen-5']['harvest'])
    sd1 = np.mean(results['5D-Sanchen-1']['harvest'])
    flat = np.mean(results['Flat-Sanchen-1']['harvest'])
    print(f"\n  5D-5 vs Flat: {sd5:.0f} vs {flat:.0f} (Δ={sd5-flat:+.0f})")
    print(f"  5D-1 vs Flat: {sd1:.0f} vs {flat:.0f} (Δ={sd1-flat:+.0f})")

if __name__ == "__main__":
    main()
