#!/usr/bin/env python3
"""GitHub Actions 版预测脚本 —— 输出 JSON + HTML"""
import numpy as np
import json, os, urllib.request, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(__file__))
from sanchen_diviner import SanchenDiviner, N_HEX

WMO = {0:0,1:0,2:2,3:1,45:2,48:2,51:4,53:4,55:4,61:4,63:4,65:4,
       71:3,73:3,75:3,77:3,85:3,86:3,80:4,81:4,82:4,95:3,96:3,99:3}
WEATHER_NAMES = ['晴','阴','风','雨']

def fetch_recent(days=7):
    end = date.today(); start = end - timedelta(days=days)
    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude=39.9&longitude=116.4"
           f"&start_date={start}&end_date={end}"
           f"&daily=weather_code,temperature_2m_max&timezone=Asia/Shanghai")
    data = json.loads(urllib.request.urlopen(url).read())['daily']
    results = []
    for i in range(len(data['time'])):
        code = data['weather_code'][i]; temp = data['temperature_2m_max'][i] or 15
        w = 0 if temp > 32 else WMO.get(code,1) % 4
        results.append({'date': data['time'][i], 'weather': w})
    return results

def weather_to_yao(w): return 1 if w in (0,2) else 0

def encode_hexagram(history):
    if len(history) < 6: return None
    recent = [weather_to_yao(h['weather']) for h in history[-6:]]
    return sum(recent[i] << i for i in range(6)) % N_HEX

def make_hexagram(hex_idx, history):
    h_upper = hex_idx // 8; h_lower = hex_idx % 8
    dongyao = sum(weather_to_yao(h['weather']) for h in history[-3:]) % 6
    return {
        'hexagram_idx': hex_idx, 'shang': h_upper, 'xia': h_lower,
        'dongyao': dongyao,
        'hugua_idx': ((h_upper ^ h_lower)*7 + dongyao) % N_HEX,
        'biangua_idx': ((h_upper ^ 0b111)*8 + h_lower) % N_HEX,
        'zong_idx': (h_lower << 3) | h_upper,
        'cuo_idx': ((7-h_upper) << 3) | (7-h_lower),
    }

class Auto5DEngine:
    def __init__(self):
        self.diviners = [SanchenDiviner(42 + i*100, cold_start=True) for i in range(5)]
        self.wl_probs = np.ones(3) / 3
        self.state_file = 'data/5d_state.json'
        self.history_file = 'data/prediction_history.json'
        os.makedirs('data', exist_ok=True)
        self.load_state()
    
    def predict(self, weather_history):
        hex_idx = encode_hexagram(weather_history)
        if hex_idx is None: return None
        hexagram = make_hexagram(hex_idx, weather_history)
        
        decisions = [d.consult(hexagram) for d in self.diviners]
        decs = [r['recommendation'] for r in decisions]
        vote = {}; [vote.update({d: vote.get(d,0)+1}) for d in decs]
        consensus = max(vote, key=vote.get)
        ratio = vote[consensus] / 5
        
        wl = np.argmax(self.wl_probs)
        if consensus in ('进取','常规'):
            pred = '晴朗' if wl == 0 else ('降雨' if wl == 1 else '晴朗')
        elif consensus == '保守':
            pred = '降雨' if wl == 1 else '阴天'
        else:
            pred = '阴天' if wl == 2 else ('降雨' if wl == 1 else '晴朗')
        
        return {'consensus': consensus, 'ratio': round(ratio,2), 
                'prediction': pred, 'wl': int(wl), 
                'wl_probs': self.wl_probs.round(3).tolist()}
    
    def collapse(self, actual_weather, prediction):
        w = actual_weather['weather']
        likes = np.ones(3) * 0.15
        if w == 0: likes[0] = 0.7; likes[1] = 0.1
        elif w in (1,3): likes[1] = 0.7; likes[0] = 0.1
        else: likes[2] = 0.5
        lp = np.log(np.maximum(self.wl_probs,1e-12)) + np.log(likes)/0.3
        lp -= lp.max(); self.wl_probs = np.exp(lp); self.wl_probs /= self.wl_probs.sum()
        if prediction:
            for d in self.diviners:
                d.update(None, None, '旱稻' if w == 0 else '水稻')
        self.save_state()
    
    def save_state(self):
        json.dump({'wl_probs': self.wl_probs.tolist(),
                   'display_acc': [d.display_acc for d in self.diviners]},
                  open(self.state_file, 'w'))
    
    def load_state(self):
        if os.path.exists(self.state_file):
            s = json.load(open(self.state_file))
            self.wl_probs = np.array(s['wl_probs'])
            for i, acc in enumerate(s.get('display_acc',[])):
                if i < len(self.diviners): self.diviners[i].display_acc = acc

# ============================================================================
# 主逻辑
# ============================================================================
engine = Auto5DEngine()
weather_buffer = fetch_recent(7)

# 坍缩昨天
history = json.load(open(engine.history_file)) if os.path.exists(engine.history_file) else []
if history:
    latest = weather_buffer[-1]
    last_entry = history[-1]
    engine.collapse(latest, last_entry.get('prediction'))

# 预测明天
pred = engine.predict(weather_buffer)

entry = {
    'date': date.today().strftime('%Y-%m-%d'),
    'prediction': pred['prediction'] if pred else '?',
    'consensus': pred['consensus'] if pred else '?',
    'ratio': pred['ratio'] if pred else 0,
    'wl': pred['wl'] if pred else 0,
    'wl_probs': pred['wl_probs'] if pred else [0.33,0.33,0.33],
}
history.append(entry)

# 只保留最近60天
if len(history) > 60: history = history[-60:]
json.dump(history, open(engine.history_file, 'w'), ensure_ascii=False)

# 输出
today_str = date.today().strftime('%Y年%m月%d日')
wl_names = ['晴主导', '雨主导', '正常']
print(f"{'='*50}")
print(f"  🔮 五维易经天气预测 — {today_str}")
print(f"  {'='*50}")
print(f"  明日预测: {entry['prediction']}")
print(f"  共识度:   {entry['ratio']:.0%} ({entry['consensus']})")
print(f"  世界线:   {wl_names[entry['wl']]} ({entry['wl_probs']})")
print(f"  {'='*50}")
