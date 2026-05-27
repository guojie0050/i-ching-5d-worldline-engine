#!/usr/bin/env python3
"""
农事决策实验 — Full Architecture
==================================
使用完整易经决策引擎（筮法→体用→时位→决策→反馈）
10年模拟, 每日天气(晴/阴/雨), 3种年类型(旱/涝/正常)
对比: 易经引擎 vs 随机 vs 固定 vs Q学习
"""

import numpy as np
from yijing_engine import YijingEngine, BAGUA_WUXING, BAGUA_YINYANG

# ============================================================================
# 天气生成器
# ============================================================================
class WeatherGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.year_types = []
        self.daily_weather = []
        self._generate()
    
    def _generate(self):
        for yr in range(10):
            r = self.rng.random()
            if r < 0.3: yt = '旱年'
            elif r < 0.6: yt = '涝年'
            else: yt = '正常年'
            self.year_types.append(yt)
            
            year_wx = []
            for doy in range(365):
                # 天气受干支影响: 阳日偏晴, 阴日偏雨, 与年类型叠加
                day_idx = (yr * 365 + doy) % 60  # 干支60日循环
                is_yang = day_idx % 2 == 0  # 阳干日 vs 阴干日
                
                if yt == '旱年':
                    base = [0.45, 0.25, 0.30]
                    base[0] += 0.15 if is_yang else -0.05  # 阳日更晴
                    base[2] -= 0.05 if is_yang else -0.05
                elif yt == '涝年':
                    base = [0.15, 0.20, 0.65]
                    base[2] += 0.05 if not is_yang else -0.05  # 阴日更雨
                    base[0] += 0.05 if is_yang else -0.05
                else:
                    base = [0.35, 0.30, 0.35]
                    base[0] += 0.08 if is_yang else -0.08
                    base[2] -= 0.08 if is_yang else -0.08
                
                base = np.clip(base, 0.05, 0.90)
                base = np.array(base) / sum(base)
                year_wx.append(self.rng.choice(['晴','阴','雨'], p=base))
            self.daily_weather.append(year_wx)


# ============================================================================
# 易经农事决策模型
# ============================================================================
class YijingFarmingModel:
    def __init__(self):
        self.engine = YijingEngine()
        self.log = []
        self.daily_advice = []
    
    def decide(self, year, month, day):
        """每日占筮 → 累积建议 → 春/夏/秋决策点整合"""
        result = self.engine.consult(year, month, day)
        dec = result['decision']
        self.daily_advice.append({
            'action': dec['action'],
            'confidence': dec['confidence'],
            'jixiong': dec['jixiong'],
        })
        return result
    
    def seasonal_decision(self, season):
        """
        每季末整合该季所有日占结果 → 决策。
        season: 'spring'(3-5月), 'summer'(6-8月), 'autumn'(9-11月)
        """
        if len(self.daily_advice) == 0:
            return '旱稻','施肥','晚收'
        
        # 统计该季推荐的action类型
        actions = [d['action'] for d in self.daily_advice]
        jixiong_list = [d['jixiong'] for d in self.daily_advice]
        
        ji_ratio = sum(1 for j in jixiong_list if j == '吉') / max(len(jixiong_list), 1)
        xiong_ratio = sum(1 for j in jixiong_list if j == '凶') / max(len(jixiong_list), 1)
        
        # Spring: 选种决策
        if season == 'spring':
            # 吉多→水稻(需水多), 凶多→旱稻
            if ji_ratio > 0.4 and xiong_ratio < 0.3:
                spring = '水稻'
            else:
                spring = '旱稻'
            return spring, None, None
        
        # Summer: 施肥决策
        if season == 'summer':
            if ji_ratio > 0.35:
                summer = '施肥'
            else:
                summer = '不施肥'
            return None, summer, None
        
        # Autumn: 收成决策
        if season == 'autumn':
            # act_bold多→晚收, defend/hold多→早收
            bold_ratio = sum(1 for a in actions if a == 'act_bold') / max(len(actions), 1)
            hold_ratio = sum(1 for a in actions if a in ('defend','hold')) / max(len(actions), 1)
            if bold_ratio > 0.25:
                autumn = '晚收'
            elif hold_ratio > 0.4:
                autumn = '早收'
            else:
                autumn = '晚收' if ji_ratio > 0.35 else '早收'
            return None, None, autumn
        
        return None, None, None
    
    def feedback(self, was_good, decision_info):
        """反馈学习: 更新体用判断可靠性"""
        if decision_info and 'tiyong' in decision_info:
            self.engine.learn_from_outcome(decision_info, was_good)


# ============================================================================
# 基线模型
# ============================================================================
class RandomModel:
    def __init__(self):
        self.rng = np.random.default_rng()
    def seasonal_decision(self, season):
        if season == 'spring':
            return self.rng.choice(['旱稻','水稻']), None, None
        elif season == 'summer':
            return None, self.rng.choice(['施肥','不施肥']), None
        else:
            return None, None, self.rng.choice(['早收','晚收'])

class FixedModel:
    def seasonal_decision(self, season):
        if season == 'spring': return '水稻', None, None
        elif season == 'summer': return None, '施肥', None
        else: return None, None, '晚收'

class QLearningModel:
    def __init__(self):
        self.Q = {}
        self.alpha, self.gamma, self.epsilon = 0.1, 0.9, 0.3
        self.spring_choices = ['旱稻','水稻']
        self.summer_choices = ['施肥','不施肥']
        self.autumn_choices = ['早收','晚收']
    
    def seasonal_decision(self, season):
        state = season
        if season not in self.Q or np.random.random() < self.epsilon:
            if season == 'spring': return np.random.choice(self.spring_choices), None, None
            elif season == 'summer': return None, np.random.choice(self.summer_choices), None
            else: return None, None, np.random.choice(self.autumn_choices)
        
        choices = {'spring': self.spring_choices, 'summer': self.summer_choices,
                    'autumn': self.autumn_choices}[season]
        best = max(self.Q[season], key=self.Q[season].get)
        if season == 'spring': return best, None, None
        elif season == 'summer': return None, best, None
        else: return None, None, best


# ============================================================================
# 年模拟
# ============================================================================
def simulate_year(env, year, models):
    """模拟一年, 返回各模型的收获。"""
    yt = env.year_types[year]
    daily_wx = env.daily_weather[year]
    
    spring_choices = {}; summer_choices = {}; autumn_choices = {}
    
    # 易经模型: 逐日占筮
    yj = models.get('Yijing')
    if yj:
        yj.daily_advice = []
        for day in range(90):  # 春季 (简化: 前90天)
            yj.decide(2020 + year, 3 + day // 30, (day % 30) + 1)
        spring_choices['Yijing'] = yj.seasonal_decision('spring')
        for day in range(91):  # 夏季
            yj.decide(2020 + year, 6 + day // 30, (day % 30) + 1)
        summer_choices['Yijing'] = yj.seasonal_decision('summer')
        for day in range(91):  # 秋季
            yj.decide(2020 + year, 9 + day // 30, (day % 30) + 1)
        autumn_choices['Yijing'] = yj.seasonal_decision('autumn')
    
    for name in ['Random','Fixed','QLearn']:
        m = models[name]
        spring_choices[name] = m.seasonal_decision('spring')
        summer_choices[name] = m.seasonal_decision('summer')
        autumn_choices[name] = m.seasonal_decision('autumn')
    
    # 计算收获
    harvests = {}
    qing_days = sum(1 for w in daily_wx if w == '晴')
    yin_days = sum(1 for w in daily_wx if w == '阴')
    yu_days = sum(1 for w in daily_wx if w == '雨')
    
    for name, (spring, _, _) in spring_choices.items():
        base = 100
        if spring == '旱稻':
            base *= 1.0 + 0.3 * (qing_days / 365)
            base *= 1.0 - 0.2 * (yu_days / 365)
        else:  # 水稻
            base *= 1.0 + 0.3 * (yu_days / 365)
            base *= 1.0 - 0.2 * (qing_days / 365)
        
        _, summer, _ = summer_choices[name]
        base += 30 if summer == '施肥' else 10
        
        _, _, autumn = autumn_choices[name]
        if autumn == '早收':
            base *= 0.9
        else:
            base *= 1.05 if yt == '正常年' else 0.85
        
        harvests[name] = base
    
    return harvests, yt, {
        'spring': spring_choices, 'summer': summer_choices, 'autumn': autumn_choices
    }


# ============================================================================
# 实验
# ============================================================================
def run():
    seeds = [42, 123, 456, 789, 1011]
    model_names = ['Yijing','Random','Fixed','QLearn']
    results = {m: {'harvest':[], 'regret':[], 'decisions':[]} for m in model_names}
    
    for si, seed in enumerate(seeds):
        print(f"\n  === Seed {si+1}/{len(seeds)} (s={seed}) ===")
        env = WeatherGenerator(seed=seed)
        
        models = {
            'Yijing': YijingFarmingModel(),
            'Random': RandomModel(),
            'Fixed': FixedModel(),
            'QLearn': QLearningModel(),
        }
        
        total_h = {m: 0 for m in model_names}
        total_r = {m: 0 for m in model_names}
        
        for yr in range(10):
            harvests, yt, choices = simulate_year(env, yr, models)
            
            for m in model_names:
                total_h[m] += harvests[m]
                best = 180  # heuristics
                total_r[m] += max(0, best - harvests[m])
            
            if yr < 2:
                print(f"  Year {yr+1} ({yt}): YJ→{choices['spring']['Yijing'][0]}/{choices['summer']['Yijing'][1]}/{choices['autumn']['Yijing'][2]} "
                      f"h={harvests['Yijing']:.0f} | "
                      f"Random={harvests['Random']:.0f} Fixed={harvests['Fixed']:.0f} Q={harvests['QLearn']:.0f}")
        
        for m in model_names:
            results[m]['harvest'].append(total_h[m])
            results[m]['regret'].append(total_r[m])
        
        print(f"  {'Yijing':<10} h={total_h['Yijing']:.0f} r={total_r['Yijing']:.0f}")
        print(f"  {'Random':<10} h={total_h['Random']:.0f} r={total_r['Random']:.0f}")
        print(f"  {'Fixed':<10} h={total_h['Fixed']:.0f} r={total_r['Fixed']:.0f}")
        print(f"  {'QLearn':<10} h={total_h['QLearn']:.0f} r={total_r['QLearn']:.0f}")
    
    print(f"\n  {'='*50}")
    print(f"  Final (10yrs×5 seeds)")
    print(f"  {'Model':<10} {'Harvest':>12} {'Regret':>10}")
    print(f"  {'-'*35}")
    for m in model_names:
        hv = np.mean(results[m]['harvest'])
        rg = np.mean(results[m]['regret'])
        print(f"  {m:<10} {hv:>10.0f}±{np.std(results[m]['harvest']):.0f} {rg:>9.0f}")
    print(f"  {'='*50}")


if __name__ == "__main__":
    run()
