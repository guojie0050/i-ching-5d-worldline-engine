#!/usr/bin/env python3
"""
多观测者共识农业决策系统 — 最终版
====================================
三理论融合: 五维投影 + 观测者交集 + 双缝干涉
架构: 5独立卦师 → 共识投票 → 坍缩修正 → 共识演化
十年农事决策, 北京真实天气统计
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, warnings, urllib.request
from yijing_engine import YijingEngine
warnings.filterwarnings("ignore")

# ============================================================================
# 任务1: 多观测者共识网络
# ============================================================================
class ConsensusNetwork:
    def __init__(self, n_observers=5):
        self.n = n_observers
        self.observers = [YijingEngine() for _ in range(n_observers)]
        self.accuracy = [0.5] * n_observers
        self.consensus_history = []
        self.all_reasonings = []
    
    def get_consensus_decision(self, year):
        decisions = []; reasonings = []
        for i, obs in enumerate(self.observers):
            r = obs.consult_year(year)
            decisions.append(r['recommendation'])
            reasonings.append({
                'observer_id': i, 'recommendation': r['recommendation'],
                'tiyong': r['tiyong'], 'jixiong': r['jixiong'],
                'stage': r['stage'], 'confidence': r['confidence'],
                'reason': r['reason'],
            })
        
        vote = {}
        for d in decisions: vote[d] = vote.get(d, 0) + 1
        max_v = max(vote.values())
        ratio = max_v / self.n
        
        if ratio >= 0.8:
            final = max(vote, key=vote.get); level = '高共识'
        elif ratio >= 0.6:
            final = max(vote, key=vote.get); level = '中等共识'
        else:
            w = {'旱稻': 0, '水稻': 0}
            for i, d in enumerate(decisions): w[d] += self.accuracy[i]
            final = max(w, key=w.get); level = '低共识（加权决断）'
        
        self.consensus_history.append(ratio)
        self.all_reasonings.append(reasonings)
        return final, level, ratio, decisions, reasonings
    
    def collapse_and_update(self, year, actual_weather, harvest):
        for i, obs in enumerate(self.observers):
            r = obs.consult_year(year)
            was_correct = (
                (actual_weather == '旱' and r['recommendation'] == '旱稻') or
                (actual_weather == '涝' and r['recommendation'] == '水稻') or
                (actual_weather == '正常')
            )
            obs.update_experience_from_result(was_correct)
            self.accuracy[i] = 0.9 * self.accuracy[i] + 0.1 * (1.0 if was_correct else 0.0)


# ============================================================================
# 任务3: 对比模型
# ============================================================================
class SingleDiviner:
    def __init__(self):
        self.engine = YijingEngine()
    def decide(self, year):
        return self.engine.consult_year(year)['recommendation']
    def update(self, year, actual):
        r = self.engine.consult_year(year)
        ok = (actual=='旱' and r['recommendation']=='旱稻') or \
             (actual=='涝' and r['recommendation']=='水稻') or (actual=='正常')
        self.engine.update_experience_from_result(ok)

class MajorityNoCorrection:
    def __init__(self):
        self.engines = [YijingEngine() for _ in range(5)]
    def decide(self, year):
        votes = [e.consult_year(year)['recommendation'] for e in self.engines]
        return max(set(votes), key=votes.count)
    def update(self, year, actual): pass  # 不修正

class DQNBaseline:
    def __init__(self):
        self.Q = {}
        self.lr, self.gamma, self.eps = 0.1, 0.9, 0.3
        self.actions = ['旱稻','水稻']
        self.last_state, self.last_action = None, None
    def decide(self, year):
        state = year % 3
        if state not in self.Q or np.random.random() < self.eps:
            a = np.random.choice(self.actions)
        else:
            a = max(self.Q[state], key=self.Q[state].get)
        self.last_state, self.last_action = state, a
        return a
    def update(self, year, actual):
        if self.last_state is not None:
            s, a = self.last_state, self.last_action
            if s not in self.Q: self.Q[s] = {act: 0.0 for act in self.actions}
            r = 1.0 if ((actual=='旱' and a=='旱稻') or (actual=='涝' and a=='水稻') or actual=='正常') else 0.0
            ns = (year+1) % 3
            if ns not in self.Q: self.Q[ns] = {act: 0.0 for act in self.actions}
            fut = max(self.Q[ns].values())
            self.Q[s][a] += self.lr * (r + self.gamma * fut - self.Q[s][a])

class FixedStrategy:
    def decide(self, year): return '水稻'
    def update(self, year, actual): pass

class RandomStrategy:
    def decide(self, year): return np.random.choice(['旱稻','水稻'])
    def update(self, year, actual): pass


# ============================================================================
# 天气与收成
# ============================================================================
def load_beijing():
    c = "/tmp/beijing_weather.json"
    if os.path.exists(c): return json.load(open(c))
    u = ("https://archive-api.open-meteo.com/v1/archive?"
         "latitude=39.9&longitude=116.4&start_date=2015-01-01&end_date=2024-12-31"
         "&daily=weather_code,precipitation_sum,temperature_2m_max&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(u).read())
    json.dump(d, open(c,"w")); return d

def generate_weather_sequence(data, n_years=10, seed=42):
    """基于真实北京天气的10年序列"""
    rng = np.random.default_rng(seed)
    ts = data["daily"]["time"]
    precip = data["daily"]["precipitation_sum"]
    temps = data["daily"]["temperature_2m_max"]
    
    years_data = {}
    for i, t in enumerate(ts):
        y = int(t.split("-")[0])
        if y not in years_data:
            years_data[y] = {"precip": 0, "days": 0, "hot": 0}
        years_data[y]["precip"] += (precip[i] if precip[i] else 0)
        years_data[y]["days"] += 1
        years_data[y]["hot"] += 1 if (temps[i] and temps[i] > 32) else 0
    
    years = sorted(years_data.keys())[:n_years]
    weather = []
    for y in years:
        s = years_data[y]
        avg_p = s["precip"] / s["days"]
        hot_r = s["hot"] / s["days"]
        if avg_p < 0.8: typ = '旱'
        elif avg_p > 1.8: typ = '涝'
        else: typ = '正常'
        extreme = avg_p < 0.5 or avg_p > 2.5 or hot_r > 0.5
        weather.append({'year': y, 'type': typ, 'extreme': extreme, 'avg_precip': avg_p})
    
    # 确保至少1-2个极端年
    has_extreme = sum(1 for w in weather if w['extreme'])
    if has_extreme < 1:
        weather[rng.integers(0, n_years)]['extreme'] = True
    if has_extreme < 2 and len(weather) > 5:
        idx = rng.integers(0, n_years)
        if not weather[idx]['extreme']:
            weather[idx]['extreme'] = True
    
    return weather

def compute_harvest(choice, weather_type, extreme):
    base = 100
    if weather_type == '旱':
        base *= 1.5 if choice == '旱稻' else 0.4
    elif weather_type == '涝':
        base *= 1.5 if choice == '水稻' else 0.4
    else:
        base *= 1.2 if choice == '水稻' else 0.85
    base += 20  # 夏季施肥
    if extreme:
        base *= 0.6
    return base


# ============================================================================
# 任务3: 完整实验
# ============================================================================
MODELS = ["Consensus-5","Single","Majority","DQN","Fixed","Random"]
SEEDS = [42, 123, 456, 789, 1011]

def run_full_experiment():
    data = load_beijing()
    results = {m: {'harvest': [], 'extreme': 0} for m in MODELS}
    results['Consensus-5']['consensus_curves'] = []
    log_entries = []
    
    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"  Seed {si+1}/{len(SEEDS)} (s={seed})")
        print(f"{'='*60}")
        
        weather = generate_weather_sequence(data, 10, seed)
        types = [w['type'] for w in weather]
        extremes = [w['extreme'] for w in weather]
        
        print(f"  天气: {' '.join(t[0] for t in types)}")
        print(f"  极端: {' '.join('⚠' if e else ' ' for e in extremes)}")
        
        # 初始化模型
        consensus = ConsensusNetwork(5)
        single = SingleDiviner()
        majority = MajorityNoCorrection()
        dqn = DQNBaseline()
        fixed = FixedStrategy()
        rand = RandomStrategy()
        
        harvests = {m: 0 for m in MODELS}
        ext_surv = {m: 0 for m in MODELS}
        
        for yr, w in enumerate(weather):
            y = w['year']
            
            # === 共识网络 ===
            final, level, ratio, decs, reas = consensus.get_consensus_decision(y)
            
            # === 基线 ===
            s_dec = single.decide(y)
            m_dec = majority.decide(y)
            d_dec = dqn.decide(y)
            f_dec = fixed.decide(y)
            r_dec = rand.decide(y)
            
            dec_map = {'Consensus-5': final, 'Single': s_dec, 'Majority': m_dec,
                       'DQN': d_dec, 'Fixed': f_dec, 'Random': r_dec}
            
            # === 收成 ===
            actual_h = {}
            for m_name, choice in dec_map.items():
                h = compute_harvest(choice, w['type'], w['extreme'])
                harvests[m_name] += h
                actual_h[m_name] = h
                if w['extreme'] and h > 80:
                    ext_surv[m_name] += 1
            
            # === 坍缩修正 ===
            consensus.collapse_and_update(y, w['type'], actual_h['Consensus-5'])
            single.update(y, w['type'])
            dqn.update(y, w['type'])
            
            # === 日志 ===
            if si == 0 or yr < 3:
                if si == 0 and yr < 3:
                    log = f"Year {y} ({w['type']}{'⚠' if w['extreme'] else ''})\n"
                    log += f"  决策: {final}({level},{ratio:.0%}) "
                    for i, r in enumerate(reas):
                        log += f"#{i}={r['recommendation']} "
                    log += f"→ 收成={actual_h['Consensus-5']:.0f}\n"
                    log_entries.append(log)
        
        for m in MODELS:
            results[m]['harvest'].append(harvests[m])
            results[m]['extreme'] += ext_surv[m]
        results['Consensus-5']['consensus_curves'].append(
            np.mean(consensus.consensus_history))
        
        best = max(harvests, key=harvests.get)
        print(f"  Best: {best}={harvests[best]:.0f}")
        for m in MODELS:
            print(f"    {m:<14}: {harvests[m]:.0f}")
    
    return results, log_entries, consensus


# ============================================================================
# 任务4: 输出
# ============================================================================
def plot_and_print(results, log_entries, consensus_net):
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS (10yrs × {len(SEEDS)} seeds)")
    print(f"{'='*60}")
    print(f"  {'Model':<16} {'Harvest':>12} {'Extreme':>10}")
    print(f"  {'-'*40}")
    for m in MODELS:
        hv = np.mean(results[m]['harvest'])
        hs = np.std(results[m]['harvest'])
        ex = results[m]['extreme']
        print(f"  {m:<16} {hv:>8.0f}±{hs:.0f} {ex:>10}")
    
    cs = results['Consensus-5']
    sh = results['Single']
    print(f"\n  共识 vs 单一: Δ={np.mean(cs['harvest'])-np.mean(sh['harvest']):+.0f}")
    
    # 共识度曲线
    curves = cs['consensus_curves']
    if curves:
        fig, ax = plt.subplots(figsize=(10, 4))
        mean_c = np.mean(curves, axis=0)
        ax.plot(range(1, len(mean_c)+1), mean_c, 'o-', color='#1a5276', lw=2, ms=8)
        ax.fill_between(range(1, len(mean_c)+1),
                        np.percentile(curves, 25, axis=0),
                        np.percentile(curves, 75, axis=0),
                        alpha=0.2, color='#1a5276')
        ax.set_xlabel('Year'); ax.set_ylabel('Consensus Ratio')
        ax.set_title('Consensus Evolution Over 10 Years')
        ax.set_ylim(0, 1.1); ax.grid(alpha=0.3)
        ax.axhline(0.6, color='gray', ls='--', alpha=0.5, label='Majority')
        ax.axhline(0.8, color='gray', ls=':', alpha=0.5, label='Strong')
        ax.legend()
        plt.tight_layout()
        plt.savefig('consensus_evolution.png', dpi=150, bbox_inches='tight')
        print("[图] consensus_evolution.png")
        plt.close(fig)
    
    # 日志
    if log_entries:
        print(f"\n{'='*60}")
        print("  关键日志 (Seed 1, 前3年):")
        for log in log_entries[:3]:
            print(f"  {log}")


if __name__ == "__main__":
    print("="*64)
    print("  Multi-Observer Consensus Farming — Final")
    print("="*64)
    results, logs, net = run_full_experiment()
    plot_and_print(results, logs, net)
