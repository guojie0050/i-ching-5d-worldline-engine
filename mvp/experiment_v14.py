#!/usr/bin/env python3
"""
V17: 全融合 — 5卦师×5卦象×时序上下文
========================================
方向1: 五卦师世界线分化 (5 diviners)
方向2: 五卦象视角 (本+互+变+综+错)
方向3: 时序上下文 (去年天气→今年先验)

完整架构: 25个独立信息源 → 加权共识 → 年决策
"""

import numpy as np
import json, os, urllib.request, warnings
from yijing_engine import YijingEngine, HEXAGRAMS, BAGUA, BAGUA_WUXING
warnings.filterwarnings("ignore")

GZ_YEAR_REF = 1984; N_HEX = 64

def load_beijing():
    c = "/tmp/beijing_20yr.json"
    if os.path.exists(c): return json.load(open(c))
    u = ("https://archive-api.open-meteo.com/v1/archive?"
         "latitude=39.9&longitude=116.4&start_date=2015-01-01&end_date=2024-12-31"
         "&daily=weather_code,precipitation_sum,temperature_2m_max&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(u).read())
    json.dump(d, open(c,"w")); return d

def classify_years(data, n=20):
    ts = data["daily"]["time"]; pr = data["daily"]["precipitation_sum"]
    ys = {}
    for i, t in enumerate(ts):
        y = int(t.split("-")[0])
        if y not in ys: ys[y] = {"p": 0, "d": 0}
        ys[y]["p"] += (pr[i] if pr[i] else 0); ys[y]["d"] += 1
    return [{'year': y, 'type': '旱' if s["p"]/s["d"]<0.8 else ('涝' if s["p"]/s["d"]>1.8 else '正常')}
            for y, s in sorted(ys.items())[:n]]


# ============================================================================
# 五卦象结构: 本+互+变+综+错
# ============================================================================
# 错卦映射: trigram→complement
CUO_MAP = [7,6,5,4,3,2,1,0]  # 乾↔坤, 兑↔艮, 离↔坎, 震↔巽

def divine_5views(year):
    """年干支 → 本卦/互卦/变卦/综卦/错卦"""
    gz_idx = (year - GZ_YEAR_REF) % 60
    yg = gz_idx % 10; yz = gz_idx % 12
    shang = (yg + yz) % 8; xia = (yg * yz + yz) % 8
    dongyao = (yg + yz * 2) % 6
    hex_bin = (shang << 3) | xia
    
    # 本卦
    ben = hex_bin
    
    # 互卦
    mid = (hex_bin >> 1) & 0b1111
    hu = ((mid >> 1) & 0b111) * 8 + (mid & 0b111)
    
    # 变卦
    bian_bin = hex_bin ^ (1 << (5 - dongyao))
    
    # 综卦: swap upper/lower
    zong = (xia << 3) | shang
    
    # 错卦: complement each trigram
    upper = hex_bin >> 3; lower = hex_bin & 0b111
    cuo = (CUO_MAP[upper] << 3) | CUO_MAP[lower]
    
    return {
        'ben': ben % N_HEX, 'hu': hu % N_HEX, 'bian': bian_bin % N_HEX,
        'zong': zong % N_HEX, 'cuo': cuo % N_HEX, 'dongyao': dongyao,
    }


# ============================================================================
# 全融合卦师: 5卦象 × 世界线专长 × 时序
# ============================================================================
class FullDiviner:
    def __init__(self, wl_specialty, view_weight_profile='balanced'):
        self.specialty = wl_specialty  # 世界线专长: 0=旱,1=涝,2=正常
        self.engine = YijingEngine()
        self.wl_probs = np.ones(3) / 3
        self.wl_probs[wl_specialty] = 0.5; self.wl_probs /= self.wl_probs.sum()
        self.confidence = 0.5
        self.correct_history = []
        
        # 五卦象各自的权重 (不同卦师可以有不同的视角偏好)
        if view_weight_profile == 'balanced':
            self.view_weights = {'ben':1.0, 'hu':0.8, 'bian':0.9, 'zong':0.7, 'cuo':0.6}
        elif view_weight_profile == 'ben_heavy':
            self.view_weights = {'ben':1.5, 'hu':0.5, 'bian':0.5, 'zong':0.3, 'cuo':0.2}
        elif view_weight_profile == 'structure':
            self.view_weights = {'ben':0.8, 'hu':1.2, 'bian':1.0, 'zong':1.0, 'cuo':0.8}
    
    def consult(self, year, views, last_type=None, dongyao=None):
        self._last_dongyao = dongyao
        """五卦象综合推演 + 时序上下文"""
        base_r = self.engine.consult_year(year)
        
        # 五卦象加权投票 (卦师内部)
        view_votes = {'旱稻': 0, '水稻': 0}
        view_total = 0
        
        for vname, hex_idx in views.items():
            w = self.view_weights.get(vname, 0.5)
            r = self._view_consult(hex_idx, base_r, self._last_dongyao)
            view_votes[r['recommendation']] += w * r.get('confidence', 0.5)
            view_total += w
        
        internal_rec = max(view_votes, key=view_votes.get)
        internal_conf = view_votes[internal_rec] / max(view_total, 0.01)
        
        # 世界线调制
        wl = np.argmax(self.wl_probs)
        if wl == 0 and internal_conf < 0.7:
            internal_rec = '旱稻'
        elif wl == 1 and internal_conf < 0.7:
            internal_rec = '水稻'
        
        # 时序上下文: 去年天气→调整世界线概率
        if last_type:
            temporal_prior = {'旱': [0.4, 0.2, 0.3], '涝': [0.2, 0.4, 0.3], '正常': [0.25, 0.25, 0.35]}
            tp = temporal_prior.get(last_type, [1/3]*3)
            self.wl_probs = 0.7 * self.wl_probs + 0.3 * np.array(tp)
            self.wl_probs /= self.wl_probs.sum()
        
        return {
            'recommendation': internal_rec,
            'confidence': 0.6 * internal_conf + 0.4 * self.confidence,
            'worldline': wl,
            'wl_probs': self.wl_probs.copy(),
        }
    
    def _view_consult(self, hex_idx, base_r, dongyao=None):
        """用特定卦象+爻辞做推演"""
        h = HEXAGRAMS[hex_idx % N_HEX]
        upper, lower = h[2], h[3]
        ti_wx = BAGUA_WUXING[upper]; yong_wx = BAGUA_WUXING[lower]
        
        SHENG = {'木':'火','火':'土','土':'金','金':'水','水':'木'}
        KE = {'木':'土','土':'水','水':'火','火':'金','金':'木'}
        
        rec = base_r['recommendation']
        conf = 0.5
        
        if SHENG.get(yong_wx) == ti_wx:  # 用生体
            rec = '水稻'; conf = 0.7
        elif KE.get(ti_wx) == yong_wx:  # 体克用
            rec = base_r['recommendation']; conf = 0.65
        elif KE.get(yong_wx) == ti_wx:  # 用克体
            rec = '旱稻'; conf = 0.7
        elif ti_wx == yong_wx:  # 比和
            rec = '水稻'; conf = 0.55
        else:
            rec = '旱稻'; conf = 0.5
        
        # 爻辞层: 动爻位置调制
        if dongyao is not None:
            yao_stages = [
                (0.75, 0.65, '潜伏'),   # 初爻: conservative
                (0.70, 0.68, '初现'),   # 二爻: slightly conservative
                (0.65, 0.70, '发展'),   # 三爻: neutral
                (0.60, 0.72, '上升'),   # 四爻: slightly aggressive
                (0.55, 0.78, '鼎盛'),   # 五爻: aggressive
                (0.65, 0.72, '转折'),   # 上爻: cautionary
            ]
            dry_w, wet_w, stage = yao_stages[dongyao]
            # 根据爻辞调整: 潜伏/转折→旱稻, 鼎盛→水稻
            if dongyao == 0:  # 初爻潜伏
                rec = '旱稻'; conf += 0.1
            elif dongyao == 4:  # 五爻鼎盛
                rec = '水稻'; conf += 0.1
            elif dongyao == 5:  # 上爻转折
                if rec == '水稻': rec = '旱稻'; conf += 0.05
            # 其他爻保持原判, 微调置信度
            conf += 0.05 if dongyao in (2,3) else 0
        
        return {'recommendation': rec, 'confidence': min(conf, 0.95)}
    
    def update(self, year, actual_type, last_type=None):
        # 时序先验
        if last_type:
            tp = {'旱': [0.4, 0.2, 0.3], '涝': [0.2, 0.4, 0.3], '正常': [0.25, 0.25, 0.35]}
            self.wl_probs = 0.7 * self.wl_probs + 0.3 * np.array(tp[last_type])
            self.wl_probs /= self.wl_probs.sum()
        
        target = {'旱': 0, '涝': 1, '正常': 2}[actual_type]
        self.wl_probs[target] *= 1.5; self.wl_probs /= self.wl_probs.sum()
        
        correct = (self.specialty == target)
        self.confidence = 0.9 * self.confidence + 0.1 * (1.0 if correct else 0.0)
        self.correct_history.append(correct)


# ============================================================================
# 共识层
# ============================================================================
def full_consensus(diviners, year, views, last_type=None):
    decs = []; wts = []
    for d in diviners:
        r = d.consult(year, views, last_type, views.get('dongyao', 0))
        decs.append(r['recommendation'])
        wts.append(r['confidence'] * np.max(r['wl_probs']))
    
    wv = {'旱稻': 0, '水稻': 0}
    for dec, w in zip(decs, wts): wv[dec] += w
    winner = max(wv, key=wv.get)
    ratio = wv[winner] / max(sum(wts), 0.01)
    return winner, ratio, decs


# ============================================================================
# 实验
# ============================================================================
SEEDS = [42, 123, 456, 789, 1011]

def run():
    data = load_beijing()
    configs = {
        'Full-5x5+Temporal': {
            'diviners': 5, 'views': 5, 'temporal': True,
        },
        '5x5-NoTemporal': {
            'diviners': 5, 'views': 5, 'temporal': False,
        },
        '3x3-NoTemporal (V16)': {
            'diviners': 3, 'views': 3, 'temporal': False,
        },
        'Single-Ben': {
            'diviners': 1, 'views': 1, 'temporal': False,
        },
    }
    
    results = {n: {'acc': [], 'cons': []} for n in configs}
    
    for si, seed in enumerate(SEEDS):
        print(f"\n  Seed {si+1}/{len(SEEDS)}")
        years = classify_years(data, 10)
        
        models = {}
        for name, cfg in configs.items():
            n_div = cfg['diviners']
            profiles = ['balanced','ben_heavy','structure','balanced','ben_heavy'][:n_div]
            divs = [FullDiviner(wl_specialty=i%3, view_weight_profile=profiles[i]) 
                    for i in range(n_div)]
            models[name] = {'diviners': divs, 'cfg': cfg}
        
        last_type = None
        for yr_info in years:
            y = yr_info['year']; actual = yr_info['type']
            views_5 = divine_5views(y)
            views_3 = {k: views_5[k] for k in ['ben','hu','bian']}
            
            for name, model in models.items():
                cfg = model['cfg']
                divs = model['diviners']
                v = views_5 if cfg['views'] >= 5 else views_3
                lt = last_type if cfg['temporal'] else None
                for d in divs:
                    d.update(y, actual, lt)
            
            last_type = actual
        
        for name, model in models.items():
            correct = 0; ratios = []
            last_type = None
            for yr_info in years:
                y = yr_info['year']; actual = yr_info['type']
                cfg = model['cfg']; divs = model['diviners']
                v = divine_5views(y) if cfg['views'] >= 5 else {k: divine_5views(y)[k] for k in ['ben','hu','bian']}
                lt = last_type if cfg['temporal'] else None
                winner, ratio, _ = full_consensus(divs, y, v, lt)
                optimal = {'旱': '旱稻', '涝': '水稻', '正常': '水稻'}
                if winner == optimal.get(actual, '水稻'): correct += 1
                ratios.append(ratio)
                last_type = actual
            results[name]['acc'].append(correct / len(years))
            results[name]['cons'].append(np.mean(ratios))
        
        for name in configs:
            print(f"    {name:<25}: {results[name]['acc'][-1]:.0%}")
    
    return results


def main():
    print("="*64)
    print("  V17: Full Fusion — 5×5 Views + Temporal Context")
    print("="*64)
    results = run()
    
    print(f"\n  {'='*70}")
    print(f"  {'Architecture':<30} {'Accuracy':>10} {'Consensus':>10}")
    for name, res in results.items():
        print(f"  {name:<30} {np.mean(res['acc']):>9.1%} {np.mean(res['cons']):>9.1%}")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
