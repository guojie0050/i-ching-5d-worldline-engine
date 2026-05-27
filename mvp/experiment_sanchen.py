#!/usr/bin/env python3
"""
V20: 先验分化共识网络 — 五种认知风格
======================================
5个三陈卦师, 各自不同认知风格:
  保守型: 分歧时选最保守 ← 极端年预警
  进取型: 分歧时选最进取 ← 正常年拉高收益
  时位型: 分歧时听时位的 ← 季节转换敏感
  体用型: 分歧时听体用的 ← 旱涝明确年果断
  反直觉: 分歧时选少数派 ← 打破集体盲区

对比: 分化5卦师 vs 同质5卦师 vs 三陈单人 vs 原始
"""

import numpy as np
import json, os, urllib.request, warnings
from sanchen_diviner import SanchenDiviner, divine_full
from yijing_engine import YijingEngine

def divine_multi_method(year):
    """同一年, 五种不同大衍筮法 → 五个不同卦象"""
    gz = (year - 1984) % 60
    yg = gz % 10; yz = gz % 12
    methods = [
        # 方法1: 标准
        lambda: ((yg + yz) % 8, (yg * yz + yz) % 8, (yg + yz * 2) % 6),
        # 方法2: 月日主导
        lambda: ((yg * 3 + yz) % 8, (yz * 5 + yg) % 8, (yg + yz * 3) % 6),
        # 方法3: 奇偶分离
        lambda: ((yg ^ yz) % 8, ((yg << 1) + yz) % 8, (yg * yz) % 6),
        # 方法4: 天地数
        lambda: ((yg + 1) % 8, (yz + 3) % 8, (yg + yz + 5) % 6),
        # 方法5: 先后天
        lambda: ((yg * 7 + yz * 11) % 8, (yz * 13 + yg * 5) % 8, (yg * 3 + yz * 7) % 6),
    ]
    results = []
    for i, method in enumerate(methods):
        shang, xia, dongyao = method()
        hex_bin = (shang << 3) | xia
        from sanchen_diviner import HEXAGRAMS, BAGUA, BAGUA_WUXING, N_HEX
        mid = (hex_bin >> 1) & 0b1111
        hu = ((mid >> 1) & 0b111) * 8 + (mid & 0b111)
        bian = (hex_bin ^ (1 << (5 - dongyao))) % N_HEX
        zong = (xia << 3) | shang
        cuo_map = [7,6,5,4,3,2,1,0]
        cuo = (cuo_map[hex_bin >> 3] << 3) | cuo_map[hex_bin & 0b111]
        results.append({
            'hexagram_idx': hex_bin % N_HEX, 'shang': shang, 'xia': xia,
            'dongyao': dongyao, 'hugua_idx': hu % N_HEX,
            'biangua_idx': bian, 'zong_idx': zong % N_HEX, 'cuo_idx': cuo % N_HEX,
            'hexagram_name': HEXAGRAMS[hex_bin % N_HEX][1],
            'shang_name': BAGUA[shang], 'xia_name': BAGUA[xia],
            'shang_wx': BAGUA_WUXING[shang], 'xia_wx': BAGUA_WUXING[xia],
            'method': f'M{i+1}',
        })
    return results

def divine_seasonal(year):
def divine_seasonal(year):
    """一年三次卜卦: 春筮/夏筮/秋筮 → 三个不同卦象"""
    spring = divine_full(year * 100 + 3)   # 3月
    summer = divine_full(year * 100 + 6)   # 6月  
    autumn = divine_full(year * 100 + 9)   # 9月
    return {'spring': spring, 'summer': summer, 'autumn': autumn}
warnings.filterwarnings("ignore")

def load_beijing():
    c = "/tmp/beijing_20yr.json"
    if os.path.exists(c): return json.load(open(c))
    u = ("https://archive-api.open-meteo.com/v1/archive?"
         "latitude=39.9&longitude=116.4&start_date=2005-01-01&end_date=2024-12-31"
         "&daily=weather_code,precipitation_sum&timezone=Asia/Shanghai")
    d = json.loads(urllib.request.urlopen(u).read())
    json.dump(d, open(c,"w")); return d

def classify_years(data, n=20):
    ts = data["daily"]["time"]; pr = data["daily"]["precipitation_sum"]
    ys = {}
    for i, t in enumerate(ts):
        y = int(t.split("-")[0])
        if y not in ys: ys[y] = {"p": 0, "d": 0}
        ys[y]["p"] += (pr[i] if pr[i] else 0); ys[y]["d"] += 1
    return [{'year': y, 'type': '旱' if s["p"]/s["d"]<0.8 else ('涝' if s["p"]/s["d"]>1.8 else '正常'),
             'extreme': s["p"]/s["d"]<0.5 or s["p"]/s["d"]>2.5}
            for y, s in sorted(ys.items())[:n]]

def compute_harvest(choice, weather_type, extreme=False):
    base = 100
    if weather_type == '旱': base *= 1.5 if choice == '旱稻' else 0.4
    elif weather_type == '涝': base *= 1.5 if choice == '水稻' else 0.4
    else: base *= 1.2 if choice == '水稻' else 0.85
    if extreme: base *= 0.7
    return base


# ============================================================================
# 分化共识网络
# ============================================================================
STYLES = ['conservative', 'aggressive', 'shiwei_first', 'tiyong_first', 'contrarian']
STYLE_NAMES = {'conservative':'保守','aggressive':'进取','shiwei_first':'时位',
               'tiyong_first':'体用','contrarian':'反直觉'}

class DifferentiatedConsensus:
    def __init__(self, seed=42):
        self.diviners = [SanchenDiviner(seed + i*100, cognitive_style=s) for i, s in enumerate(STYLES)]
        self.style_accuracy = {s: 0.5 for s in STYLES}
        self.consensus_history = []
        self.style_correct = {s: 0 for s in STYLES}
        self.style_total = {s: 0 for s in STYLES}
    
    def decide(self, hexagram, use_confidence_weight=False, use_fourth=False):
        results = [d.consult(hexagram, use_fourth=use_fourth) for d in self.diviners]
        decs = [r['recommendation'] for r in results]
        # 获取三陈置信度
        confs = [r.get('confidence', '中') for r in results]
        conf_map = {'高': 3.0, '中': 1.5, '低': 0.5}
        conf_weights = [conf_map.get(c, 1.0) for c in confs]
        
        if use_confidence_weight:
            # 四陈模式: 用三陈置信度做加权投票
            weighted = {}
            for d in set(decs): weighted[d] = 0
            for i, d in enumerate(decs):
                weighted[d] += conf_weights[i]
            final = max(weighted, key=weighted.get)
            total_w = sum(weighted.values())
            ratio = weighted[final] / max(total_w, 0.01)
            self.consensus_history.append(ratio)
            
            max_v = max(weighted.values()) / max(conf_weights)
            if max_v >= 3.5: level = '高'
            elif max_v >= 2.0: level = '中'
            else: level = '低(加权)'
        else:
            # 原始模式
            vote = {}
            for d in decs: vote[d] = vote.get(d, 0) + 1
            max_v = max(vote.values())
            ratio = max_v / len(self.diviners)
            self.consensus_history.append(ratio)
            
            if max_v >= 4:
                final = max(vote, key=vote.get); level = '高'
            elif max_v == 3:
                final = max(vote, key=vote.get); level = '中'
            else:
                weighted = {}
                for d in set(decs): weighted[d] = 0
                for i, d in enumerate(decs):
                    weighted[d] += self.style_accuracy[STYLES[i]]
                final = max(weighted, key=weighted.get); level = '低(加权)'
        
        return final, level, ratio, results
    
    def update(self, was_correct_per_style):
        """更新每种风格的准确率"""
        styles = [d.cognitive_style for d in self.diviners]
        for style, correct in was_correct_per_style.items():
            if style in self.style_accuracy:
                self.style_accuracy[style] = 0.9 * self.style_accuracy[style] + 0.1 * (1.0 if correct else 0.0)
                self.style_total[style] += 1
                if correct: self.style_correct[style] += 1


# ============================================================================
# 对照组
# ============================================================================
class HomogeneousSanchen:
    """5个同质三陈卦师 (V19基线)"""
    def __init__(self, seed=42):
        self.diviners = [SanchenDiviner(seed + i*100, cognitive_style='balanced') for i in range(5)]
    def decide(self, hexagram):
        results = [d.consult(hexagram, use_fourth=use_fourth) for d in self.diviners]
        decs = [r['recommendation'] for r in results]
        vote = {}
        for d in decs: vote[d] = vote.get(d, 0) + 1
        return max(vote, key=vote.get), None, None, results
    def update(self, *args): pass

class MultiHexDiviner:
    """多卦象卦师: 收3个季节卦象, 每个独立做三陈, 内部加权投票"""
    def __init__(self, seed=42, hex_subset=None):
        self.engine = SanchenDiviner(seed, cognitive_style='balanced')
        self.subset = hex_subset or ['spring','summer','autumn']
    
    def consult(self, seasonal_hexes):
        results = []
        for season in self.subset:
            h = seasonal_hexes[season]
            r = self.engine.consult(h)
            results.append(r)
        # 内部加权投票: 所有季节卦象的三陈结果取多数
        decs = [r['recommendation'] for r in results]
        vote = {}
        for d in decs: vote[d] = vote.get(d, 0) + 1
        final = max(vote, key=vote.get)
        return final, results

class MultiHexConsensus:
    """多卦象分化共识: 5个卦师, 每人分到不同的季节卦象子集"""
    def __init__(self, seed=42):
        subsets = [
            ['spring'], ['summer'], ['autumn'],
            ['spring','summer'], ['summer','autumn'],
        ]
        self.diviners = [MultiHexDiviner(seed + i*100, hex_subset=s) for i, s in enumerate(subsets)]
        self.consensus_history = []
    
    def decide(self, seasonal_hexes):
        results_list = []
        decs = []
        for d in self.diviners:
            final, details = d.consult(seasonal_hexes)
            decs.append(final)
            results_list.append(details)
        vote = {}
        for d in decs: vote[d] = vote.get(d, 0) + 1
        max_v = max(vote.values())
        ratio = max_v / len(self.diviners)
        self.consensus_history.append(ratio)
        final = max(vote, key=vote.get)
        return final, f'{max_v}/5', ratio, results_list
    
    def update(self, *args): pass

class FullMultiMethodConsensus:
    """五个完整三陈卦师, 各自从不同大衍筮法获得不同卦象输入"""
    def __init__(self, seed=42, n=5):
        self.diviners = [SanchenDiviner(seed + i*100, cognitive_style='balanced') for i in range(n)]
        self.consensus_history = []
    
    def decide(self, multi_hexes):
        results = [d.consult(h) for d, h in zip(self.diviners, multi_hexes)]
        decs = [r['recommendation'] for r in results]
        vote = {}
        for d in decs: vote[d] = vote.get(d, 0) + 1
        max_v = max(vote.values())
        ratio = max_v / len(self.diviners)
        self.consensus_history.append(ratio)
        final = max(vote, key=vote.get)
        return final, f'{max_v}/{len(self.diviners)}', ratio, results
    
    def update(self, *args): pass

class GuaQiConsensus(FullMultiMethodConsensus):
    """卦气增强版: 投票时用卦气加权"""
    def decide(self, multi_hexes, month=3):
        results = []
        weights = []
        for d, h in zip(self.diviners, multi_hexes):
            r = d.consult(h)
            results.append(r)
            # 卦气权重: hexagram的季节适配度
            gq_w = guaqi_weight(h, month)
            weights.append(gq_w)
        
        # 加权投票
        wv = {}
        for r, w in zip(results, weights):
            dec = r['recommendation']
            wv[dec] = wv.get(dec, 0) + w
        
        total_w = sum(weights)
        final = max(wv, key=wv.get)
        ratio = wv[final] / max(total_w, 0.01)
        self.consensus_history.append(ratio)
        return final, f'GQ{ratio:.0%}', ratio, results

class GuaQiSingle:
    """卦气增强单人: 多卦象+卦气加权"""
    def __init__(self, seed=42):
        self.diviner = SanchenDiviner(seed, cognitive_style='balanced')
    
    def decide(self, multi_hexes, month=3):
        results = [self.diviner.consult(h) for h in multi_hexes]
        weights = [guaqi_weight(h, month) for h in multi_hexes]
        wv = {}
        for r, w in zip(results, weights):
            wv[r['recommendation']] = wv.get(r['recommendation'], 0) + w
        final = max(wv, key=wv.get)
        return final, None, None, results
    
    def update(self, *args): pass

class GuaQiTeam5:
    """五个卦气增强版单卦师组队"""
    def __init__(self, seed=42):
        self.members = [GuaQiSingle(seed + i*100) for i in range(5)]
        self.consensus_history = []
    
    def decide(self, multi_hexes, month=3):
        # 每人独立推演 (内部卦气加权)
        member_decs = []
        member_weights = []
        for m in self.members:
            # GuaQiSingle内部用卦气加权多卦象
            final, _, _, _ = m.decide(multi_hexes, month)
            member_decs.append(final)
            # 团队层权重: 用卦气均值
            gq_avg = np.mean([guaqi_weight(h, month) for h in multi_hexes])
            member_weights.append(gq_avg)
        
        wv = {}
        for dec, w in zip(member_decs, member_weights):
            wv[dec] = wv.get(dec, 0) + w
        final = max(wv, key=wv.get)
        ratio = wv[final] / max(sum(member_weights), 0.01)
        self.consensus_history.append(ratio)
        return final, f'GQ5T{ratio:.0%}', ratio, None
    
    def update(self, *args): pass

class SanchenSingle:
    def __init__(self, seed=42):
        self.diviner = SanchenDiviner(seed, cognitive_style='balanced')
    def decide(self, hexagram):
        r = self.diviner.consult(hexagram)
        return r['recommendation'], None, None, [r]
    def update(self, optimal_dec=None):
        if optimal_dec:
            self.diviner.update(None, None, optimal_dec)

class SanchenSingleBayes(SanchenSingle):
    """贝叶斯版: 冷启动, 从零学习推演器准确率"""
    def __init__(self, seed=42):
        self.diviner = SanchenDiviner(seed, cognitive_style='balanced', cold_start=True)

class DifferentiatedConsensus3(DifferentiatedConsensus):
    """三人团: 体用型 + 时位型 + 卦变型 (三才)"""
    def __init__(self, seed=42):
        styles = ['tiyong_first', 'shiwei_first', 'balanced']
        self.diviners = [SanchenDiviner(seed + i*100, cognitive_style=s) for i, s in enumerate(styles)]
        self.style_accuracy = {s: 0.5 for s in styles}
        self.consensus_history = []
        self.style_correct = {s: 0 for s in styles}
        self.style_total = {s: 0 for s in styles}

class DifferentiatedConsensus9(DifferentiatedConsensus):
    """九人团: 五种风格 × 2副本 + 1平衡 (九宫)"""
    def __init__(self, seed=42):
        styles = (STYLES * 2)[:9]  # 5种风格各2份, 取前9
        self.diviners = [SanchenDiviner(seed + i*100, cognitive_style=s) for i, s in enumerate(styles)]
        self.style_accuracy = {s: 0.5 for s in set(styles)}
        self.consensus_history = []
        self.style_correct = {s: 0 for s in set(styles)}
        self.style_total = {s: 0 for s in set(styles)}

class SanchenSingleFourth:
    def __init__(self, seed=42):
        self.diviner = SanchenDiviner(seed, cognitive_style='balanced')
    def decide(self, hexagram):
        r = self.diviner.consult(hexagram, use_fourth=True)
        return r['recommendation'], None, None, [r]
    def update(self, *args): pass

class OriginalSingle:
    def __init__(self):
        self.engine = YijingEngine()
    def decide(self, hexagram, year):
        return self.engine.consult_year(year)['recommendation'], None, None, None
    def update(self, *args): pass

class DQN:
    def __init__(self):
        self.Q = {}; self.lr=0.1; self.gamma=0.9; self.eps=0.3
        self.last_s=None; self.last_a=None
    def decide(self, year):
        s = year % 3
        if s not in self.Q or np.random.random() < self.eps:
            a = np.random.choice(['旱稻','水稻'])
        else:
            a = max(self.Q[s], key=self.Q[s].get)
        self.last_s, self.last_a = s, a
        return a, None, None, None
    def update(self, optimal):
        if self.last_s is not None:
            s,a = self.last_s, self.last_a
            if s not in self.Q: self.Q[s] = {x:0.0 for x in ['旱稻','水稻']}
            r = 1.0 if (a == optimal) else 0.0
            ns = (s+1)%3
            if ns not in self.Q: self.Q[ns] = {x:0.0 for x in ['旱稻','水稻']}
            self.Q[s][a] += self.lr * (r + self.gamma * max(self.Q[ns].values()) - self.Q[s][a])

class Fixed:
    def decide(self, *args): return '水稻', None, None, None
    def update(self, *args): pass


# ============================================================================
# 实验
# ============================================================================
SEEDS = [42, 123, 456, 789, 1011]

def run():
    data = load_beijing()
    configs = {
        'SC1-Fourth (四陈单人)': lambda s: SanchenSingleFourth(s),
        'Diff-3 (三人)': lambda s: DifferentiatedConsensus3(s),
        'Diff-9 (九人)': lambda s: DifferentiatedConsensus9(s),
        'Full-5x5Meth (全卦师)': lambda s: FullMultiMethodConsensus(s, 5),
        'GuaQi-5 (卦气团)': lambda s: GuaQiConsensus(s, 5),
        'SC1-Bayes (贝叶斯单)': lambda s: SanchenSingleBayes(s),
        'GQ-Team5 (卦气组队)': lambda s: GuaQiTeam5(s),
        'GuaQi-1 (卦气单)': lambda s: GuaQiSingle(s),
        'Full-3x3Meth (全卦师3)': lambda s: FullMultiMethodConsensus(s, 3),
        'MultiHex-5 (多卦)': lambda s: MultiHexConsensus(s),
        'Diff-5+ConfW (四陈)': lambda s: DifferentiatedConsensus(s),
        'Diff-5 (分化)': lambda s: DifferentiatedConsensus(s),
        'Homo-5 (同质)': lambda s: HomogeneousSanchen(s),
        'Sanchen-1 (单人)': lambda s: SanchenSingle(s),
        'Original-1': lambda s: OriginalSingle(),
        'DQN': lambda s: DQN(),
        'Fixed': lambda s: Fixed(),
    }
    
    results = {n: {'harvest': [], 'extreme_surv': 0, 'consensus': [], 'style_stats': []} for n in configs}
    # 贝叶斯学习曲线: 每年累计准确率
    bayes_learning_curve = []  # (seed, year_idx, cumulative_acc)
    sample_logs = []
    
    for si, seed in enumerate(SEEDS):
        print(f"\n  Seed {si+1}/{len(SEEDS)}")
        train_years = [y for y in classify_years(data, 20) if y['year'] <= 2014]
    test_years = [y for y in classify_years(data, 20) if y['year'] > 2014]
    # 卦气从训练年学到权重
    # (already done in _build_guaqi_from_data which uses <= 2014)
    years = test_years  # 只评估测试年
    # DQN: 先在训练年在线学习, 再测测试年
    print(f'    Train/Test: {len(train_years)}+{len(test_years)}yr')
    print(f'    Test years:  {[y["year"] for y in test_years]} ({len(test_years)})')
        models = {n: f(seed) for n, f in configs.items()}
        harvests = {n: 0 for n in models}
        ext_surv = {n: 0 for n in models}
        
        for yr_info in years:
            y = yr_info['year']; actual = yr_info['type']; extreme = yr_info['extreme']
            hexagram = divine_full(y); seasonal = divine_seasonal(y); multi_hexes = divine_multi_method(y)
            optimal = {'旱': '旱稻', '涝': '水稻', '正常': '水稻'}
            
            for name, model in models.items():
                if 'Diff' in name or 'Homo' in name or 'Sanchen' in name:
                    use_cw = ('ConfW' in name)
                    use_4th = ('4th' in name or 'Fourth' in name)
                    dec, level, ratio, details = model.decide(hexagram, use_confidence_weight=use_cw, use_fourth=use_4th)
                elif 'GQ-Team' in name:
                    dec, level, ratio, details = model.decide(multi_hexes, month=3)
                elif 'GuaQi-5' in name or 'GuaQi-1' in name:
                    dec, level, ratio, details = model.decide(multi_hexes, month=3)  # 春分
                elif 'Full-5' in name or 'Full-3' in name:
                    dec, level, ratio, details = model.decide(multi_hexes)
                elif 'MultiHex' in name:
                    dec, level, ratio, details = model.decide(seasonal)
                    elif 'Original' in name:
                    dec, _, _, _ = model.decide(hexagram, y)
                elif name == 'DQN':
                    dec, _, _, _ = model.decide(y)
                else:
                    dec, _, _, _ = model.decide()
                
                harvests[name] += compute_harvest(dec, actual, extreme)
                if extreme and compute_harvest(dec, actual, extreme) > 60:
                    ext_surv[name] += 1
            
            # DQN更新
            if 'DQN' in models:
                models['DQN'].update(optimal.get(actual))
            
            # 分化共识: 更新风格准确率
            for mname in ['Diff-5 (分化)', 'Diff-3 (三人)', 'Diff-9 (九人)']:
                if mname in models:
                    m = models[mname]
                    per_style = {}
                    for i, d in enumerate(m.diviners):
                        s = d.cognitive_style
                        r = d.log[-1] if d.log else {'recommendation': '旱稻'}
                        per_style[s] = (r['recommendation'] == optimal.get(actual, '水稻'))
                    m.update(per_style)
            
            # 日志
            if si == 0 and yr_info < 2:
                if 'Diff-5' in models:
                    r = models['Diff-5 (分化)'].diviners[0].log[-1]
                    log = f"Y{y}({actual}) {hexagram['hexagram_name']} "
                    log += f"体:{r['chen1']['relation']}→{r['chen1']['decision']} "
                    log += f"时:{r['chen2']['yao_name']}→{r['chen2']['decision']} "
                    log += f"变→{r['chen3']['decision']} "
                    log += f"→{r['sanchen_decision']}({r['confidence']})"
                    sample_logs.append(log)
        
        for n in models:
            results[n]['harvest'].append(harvests[n])
        
        # 贝叶斯学习曲线
        if 'SC1-Bayes' in models:
            curve = []
            cum_correct = 0
            for t_idx, yr_info in enumerate(years):
                y = yr_info['year']; actual = yr_info['type']
                h = divine_full(y)
                dec, _, _, _ = models['SC1-Bayes (贝叶斯单)'].decide(h)
                optimal = {'旱': '旱稻', '涝': '水稻', '正常': '水稻'}
                if dec == optimal.get(actual, '水稻'): cum_correct += 1
                curve.append(cum_correct / (t_idx + 1))
            bayes_learning_curve.append(curve)
            results[n]['extreme_surv'] += ext_surv[n]
        
        if 'Diff-5' in models:
            cr = np.mean(models['Diff-5 (分化)'].consensus_history)
            results['Diff-5 (分化)']['consensus'].append(cr)
        
        best = max(harvests, key=harvests.get)
        print(f"    Best={best}:{harvests[best]:.0f}  "
              f"GQ5={harvests.get('GuaQi-5 (卦气团)',0):.0f}  GQ1={harvests.get('GuaQi-1 (卦气单)',0):.0f}  GQT5={harvests.get('GQ-Team5 (卦气组队)',0):.0f}  F5M={harvests.get('Full-5x5Meth (全卦师)',0):.0f}  F3M={harvests.get('Full-3x3Meth (全卦师3)',0):.0f}  Diff3={harvests.get('Diff-3 (三人)',0):.0f}  Diff5={harvests.get('Diff-5 (分化)',0):.0f}  Diff9={harvests.get('Diff-9 (九人)',0):.0f}  "
              f"Homo5={harvests.get('Homo-5 (同质)',0):.0f}  "
              f"SC1B={harvests.get('SC1-Bayes (贝叶斯单)',0):.0f}  SC1={harvests.get('Sanchen-1 (单人)',0):.0f}")
    
    return results, sample_logs


def main():
    print("="*64)
    print("  V20: Differentiated Cognitive Styles")
    print("="*64)
    results, logs = run()
    
    print(f"\n  {'='*65}")
    print(f"  {'Model':<20} {'Harvest':>12} {'Extreme':>10} {'Consensus':>10}")
    print(f"  {'-'*55}")
    fixed_h = np.mean(results['Fixed']['harvest'])
    for n in results:
        hv = np.mean(results[n]['harvest'])
        hs = np.std(results[n]['harvest'])
        ex = results[n]['extreme_surv']
        cs = np.mean(results[n].get('consensus', [0]))
        print(f"  {n:<20} {hv:>7.0f}±{hs:.0f} {ex:>10} {cs:>9.0%}")
    
    diff = np.mean(results['Diff-5 (分化)']['harvest'])
    homo = np.mean(results['Homo-5 (同质)']['harvest'])
    sc1  = np.mean(results['Sanchen-1 (单人)']['harvest'])
    print(f"\n  分化 vs 同质: {diff:.0f} vs {homo:.0f} (Δ={diff-homo:+.0f})")
    print(f"  分化 vs 单人: {diff:.0f} vs {sc1:.0f} (Δ={diff-sc1:+.0f})")
    
    if logs:
        print(f"\n  {'='*60}")
        print("  关键日志:")
        for l in logs[:2]: print(f"  {l}")


if bayes_learning_curve and len(bayes_learning_curve) > 0:
        print(f"\n  === 贝叶斯学习曲线 (5种子平均) ===")
        curves = np.array(bayes_learning_curve)
        mean_curve = np.mean(curves, axis=0)
        for yr in [0, 2, 4, 6, 9]:
            print(f"    Year {yr+1}: cumulative acc = {mean_curve[yr]:.0%}")

if __name__ == "__main__":
    main()
