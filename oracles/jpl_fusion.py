#!/usr/bin/env python3
"""
完整融合版 — 亲和向量 × 三陈调制 × 卦气权重 × 世界线 × 贝叶斯
五层架构叠在一起做天气预测
"""
import sys, os, json, hashlib, time
import numpy as np
from datetime import datetime
from collections import Counter

# ═══════════════ Layer 0: 64卦亲和向量 ═══════════
TRIGRAM_W = np.array([
    [0.35,0.05,0.05,0.05,0.05,0.30,0.05,0.10],[0.05,0.30,0.10,0.05,0.10,0.05,0.25,0.10],
    [0.10,0.15,0.10,0.35,0.10,0.05,0.10,0.05],[0.15,0.10,0.35,0.10,0.10,0.05,0.10,0.05],
    [0.05,0.15,0.05,0.10,0.35,0.05,0.15,0.10],[0.15,0.05,0.05,0.05,0.05,0.40,0.15,0.10],
    [0.10,0.20,0.10,0.05,0.10,0.05,0.15,0.25],[0.10,0.15,0.10,0.05,0.15,0.05,0.15,0.25],
])

KW = [(1,"乾",0,0),(2,"坤",1,1),(3,"屯",4,3),(4,"蒙",6,4),(5,"需",4,0),(6,"讼",0,4),(7,"师",1,4),(8,"比",4,1),
    (9,"小畜",3,0),(10,"履",0,7),(11,"泰",1,0),(12,"否",0,1),(13,"同人",0,5),(14,"大有",5,0),(15,"谦",1,6),
    (16,"豫",2,1),(17,"随",7,2),(18,"蛊",6,3),(19,"临",1,7),(20,"观",3,1),(21,"噬嗑",5,2),(22,"贲",6,5),
    (23,"剥",6,1),(24,"复",1,2),(25,"无妄",0,2),(26,"大畜",6,0),(27,"颐",6,2),(28,"大过",7,3),(29,"坎",4,4),
    (30,"离",5,5),(31,"咸",7,6),(32,"恒",2,3),(33,"遁",0,6),(34,"大壮",2,0),(35,"晋",5,1),(36,"明夷",1,5),
    (37,"家人",3,5),(38,"睽",5,7),(39,"蹇",4,6),(40,"解",2,4),(41,"损",6,7),(42,"益",3,2),(43,"夬",7,0),
    (44,"姤",0,3),(45,"萃",7,1),(46,"升",1,3),(47,"困",7,4),(48,"井",4,3),(49,"革",7,5),(50,"鼎",5,3),
    (51,"震",2,2),(52,"艮",6,6),(53,"渐",3,6),(54,"归妹",2,7),(55,"丰",2,5),(56,"旅",5,6),(57,"巽",3,3),
    (58,"兑",7,7),(59,"涣",3,4),(60,"节",4,7),(61,"中孚",3,7),(62,"小过",2,6),(63,"既济",4,5),(64,"未济",5,4)]
HEX_U = [kw[2] for kw in KW]; HEX_L = [kw[3] for kw in KW]

WUXING = ["金","土","木","木","水","火","土","金"]  # 乾坤震巽坎离艮兑
SHENG = {("木","火"),("火","土"),("土","金"),("金","水"),("水","木")}
KE   = {("木","土"),("土","水"),("水","火"),("火","金"),("金","木")}

def hex_affinity(h):
    u,l = HEX_U[h%64], HEX_L[h%64]
    return 0.55*TRIGRAM_W[u] + 0.45*TRIGRAM_W[l]

def hex_name(h): return KW[h%64][1]

# ═══════════════ Layer 1: 三陈调制 ═══════════════
def sanchen_modulate(probs, h_idx, dongyao, month):
    """三陈推演 → 调制亲和向量的概率分布"""
    u,l = HEX_U[h_idx%64], HEX_L[h_idx%64]
    ti_wx = WUXING[u if dongyao < 3 else l]
    yong_wx = WUXING[l if dongyao < 3 else u]
    
    # 第一陈: 体用生克 → 幅值调制
    if (yong_wx, ti_wx) in SHENG:     boost = 1.3   # 用生体: 吉, 强化主导倾向
    elif (ti_wx, yong_wx) in KE:      boost = 1.15  # 体克用: 吉, 小幅强化
    elif (ti_wx, yong_wx) in SHENG:   boost = 0.85  # 体生用: 耗, 弱化
    elif (yong_wx, ti_wx) in KE:      boost = 0.7   # 用克体: 凶, 大幅弱化→拉平
    else:                             boost = 1.0   # 比和: 不变
    
    # 第二陈: 时位 → 持续性偏置
    persistence_bias = [0.1, 0.15, 0.05, 0.0, -0.1, 0.1][dongyao % 6]
    
    # 第三陈: 卦变 → 方向偏置
    hu_h = ((u ^ l) * 7 + dongyao) % 64
    hu_u, hu_l = HEX_U[hu_h], HEX_L[hu_h]
    hu_wx_u, hu_wx_l = WUXING[hu_u], WUXING[hu_l]
    direction_bias = 0.05 if (hu_wx_u, hu_wx_l) in SHENG else (-0.05 if (hu_wx_u, hu_wx_l) in KE else 0)
    
    # 应用调制: boost主导维度, 拉平其他维度
    top = np.argmax(probs)
    result = probs.copy()
    result[top] *= boost
    result /= result.sum()
    
    # 持续性: 晴天→晴天概率高
    result[0] += persistence_bias
    result = np.clip(result, 0.02, 0.95)
    result /= result.sum()
    
    return result

# ═══════════════ Layer 2: 卦气权重 ═══════════════
TRIGRAM_SEASON = np.array([
    [0.7, 1.0, 0.8, 0.5],  # 乾: 春夏强
    [0.4, 0.5, 0.9, 0.7],  # 兑: 秋冬强
    [0.9, 1.5, 0.6, 0.3],  # 离: 夏至强
    [1.5, 0.7, 0.4, 0.3],  # 震: 春至强
    [0.8, 0.5, 0.8, 0.7],  # 巽: 春秋
    [0.5, 0.5, 0.6, 1.5],  # 坎: 冬至强
    [0.4, 0.5, 0.7, 1.2],  # 艮: 秋冬
    [0.6, 0.6, 0.8, 0.8],  # 坤: 土四季
])

def guaqi_weight(h_idx, month):
    season = (month - 1) // 3  # 0=春,1=夏,2=秋,3=冬
    u,l = HEX_U[h_idx%64], HEX_L[h_idx%64]
    w_u = TRIGRAM_SEASON[u, season]
    w_l = TRIGRAM_SEASON[l, season]
    return 0.55*w_u + 0.45*w_l

# ═══════════════ 辅助 ═══════════════
def weather_to_idx(wmo, tmp):
    if tmp and tmp > 32: return 5
    return {0:0,1:0,2:2,3:1,45:7,48:7,51:4,53:4,55:4,61:4,63:4,65:4,
            71:7,73:7,75:7,80:4,81:4,82:4,95:3,96:3,99:3}.get(wmo,1)

def is_clear(w): return w in (0,5)
def cast_hex(seed):
    rng = np.random.default_rng(seed)
    shang = rng.integers(0,8); xia = rng.integers(0,8)
    return shang*8+xia, rng.integers(0,6)

def jpl_seed(lon_j, lon_e, obs_id=0):
    s = f"{lon_j:.10f}{lon_e:.10f}{int(time.time()/300)}{obs_id}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:16],16) % (2**31-1)

def get_jpl(name, default):
    try:
        import urllib.request
        cmd = "599" if name=="jupiter" else "399"
        url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='{cmd}'&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='ECLIPTIC'&CENTER='500@0'&START_TIME='2026-05-29'&STOP_TIME='2026-05-30'&STEP_SIZE='1h'&QUANTITIES='31'"
        data = json.loads(urllib.request.urlopen(url, timeout=10).read())
        return float(data['result'].split('$$SOE')[1].split()[2])
    except: return default

def get_weather():
    try:
        import urllib.request
        url = "https://api.open-meteo.com/v1/forecast?latitude=39.9&longitude=116.4&current=weather_code,temperature_2m_max&timezone=Asia/Shanghai"
        d = json.loads(urllib.request.urlopen(url, timeout=5).read())
        return weather_to_idx(d['current']['weather_code'], d['current'].get('temperature_2m_max'))
    except: return 0

# ═══════════════ 主循环 ═══════════════
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data_jpl')
os.makedirs(DATA_DIR, exist_ok=True)
N_OBS, N_HEX, N_W = 5, 64, 8

# Dirichlet counts
counts = np.zeros((N_OBS, N_HEX, N_W))
for oi in range(N_OBS):
    for h in range(N_HEX):
        counts[oi,h] = 5.0*hex_affinity(h) + 1.0

# 世界线
wl = np.array([[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6],[0.5,0.3,0.2],[0.3,0.3,0.4]])

# 加载持久化状态
if os.path.exists(os.path.join(DATA_DIR, 'model_state.json')):
    saved = json.load(open(os.path.join(DATA_DIR, 'model_state.json')))
    wl = np.array(saved.get('wl', wl.tolist()))

# 三陈 display 准确率追踪 (EMA)
display_acc = np.full((N_OBS, 3), 0.5)  # [体用, 时位, 卦变]
display_cnt = np.zeros((N_OBS, 3))

# 爻辞编码: 动爻位置→晴/阴偏向
YAOCI_BIAS = {
    0: -0.05,  # 初爻潜伏: 微偏阴 (潜龙勿用→保守)
    1:  0.00,  # 二爻初现: 中性
    2:  0.03,  # 三爻发展: 微偏晴
    3:  0.05,  # 四爻上升: 偏晴
    4:  0.08,  # 五爻鼎盛: 强偏晴 (飞龙在天→进取)
    5: -0.03,  # 上爻转折: 微偏阴 (亢龙有悔→收敛)
}

def yaoci_modulate(probs, dongyao, display_belief):
    """爻辞调制 + 历史准确率加权"""
    bias = YAOCI_BIAS.get(dongyao % 6, 0.0)
    # display_belief: 体用/时位/卦变各自的历史准确率→权重
    trust = np.clip(display_belief, 0.3, 1.0)
    bias *= np.mean(trust)  # 如果三陈都不准, 爻辞偏置减半
    probs[0] += bias
    probs = np.clip(probs, 0.02, 0.95)
    return probs / probs.sum()

now = datetime.now(); ts = now.strftime('%Y-%m-%d %H:%M')
month = now.month

# ── 预测 ──
lon_j, lon_e = get_jpl("jupiter", 234.5), get_jpl("earth", 78.9)
preds = []
for oi in range(N_OBS):
    h_idx, dongyao = cast_hex(jpl_seed(lon_j, lon_e, 0)  # 同卦: 五人共享同一卦象)
    h_idx %= 64
    
    # Layer 0: 亲和向量
    probs = counts[oi, h_idx] / counts[oi, h_idx].sum()
    
    # Layer 1: 三陈调制
    probs = sanchen_modulate(probs, h_idx, dongyao, month)
    
    # Layer 1.5: 爻辞调制
    probs = yaoci_modulate(probs, dongyao, display_acc[oi])
    
    # Layer 2: 卦气权重
    gq = guaqi_weight(h_idx, month)
    probs = probs * (0.5 + 0.5*gq)  # gq归一化到[0.5,1.5]
    probs /= probs.sum()
    
    # Layer 3: 世界线调制
    wi = np.argmax(wl[oi])
    if wi == 0:   probs[0] *= 1.8; probs /= probs.sum()
    elif wi == 2: probs[0] *= 0.6; probs /= probs.sum()
    
    clear_prob = probs[0] + probs[5]
    preds.append(0 if clear_prob > 0.35 else 1)

vote = Counter(preds); consensus = vote.most_common(1)[0][0]
cname = "晴" if consensus == 0 else "阴雨"
print(f"[{ts}] 预测: {cname} (共识{vote[consensus]}/5)")
print(f"  卦: {hex_name(cast_hex(jpl_seed(lon_j,lon_e,0))[0])} | " + " ".join(["晴" if p==0 else "雨" for p in preds]))

# 保存预测
pred_log = {'time': ts, 'prediction': cname, 'preds': preds, 'consensus': vote[consensus]}
json.dump(pred_log, open(os.path.join(DATA_DIR, 'last_pred.json'), 'w'))

# ── 验证上轮 ──
vpath = os.path.join(DATA_DIR, 'verify_log.json')
ppath = os.path.join(DATA_DIR, 'last_pred.json')
if os.path.exists(ppath):
    prev = json.load(open(ppath))
    actual = get_weather(); aname = "晴" if is_clear(actual) else "阴雨"
    pcorr = (1 if prev['prediction']=='晴' else 0) == (1 if is_clear(actual) else 0)
    
    for oi in range(N_OBS):
        h_idx, _ = cast_hex(jpl_seed(lon_j, lon_e, 0)  # 同卦: 五人共享同一卦象)
        h_idx %= 64
        ocorr = (prev['preds'][oi] == (1 if is_clear(actual) else 0))
        counts[oi, h_idx, actual] += 1.0
        twl = 0 if is_clear(actual) else 1
        wl[oi][twl] *= 1.5 if ocorr else 0.8
        wl[oi] /= wl[oi].sum()
        # 更新 display_acc: 三个陈各自的准确率
        for di in range(3):
            display_cnt[oi][di] += 1
            display_acc[oi][di] = 0.95*display_acc[oi][di] + 0.05*(1.0 if ocorr else 0.0)
    
    vlog = json.load(open(vpath)) if os.path.exists(vpath) else []
    vlog.append({'time': ts, 'prediction': prev['prediction'], 'actual': aname, 'correct': pcorr})
    json.dump(vlog[-1000:], open(vpath, 'w'), ensure_ascii=False)
    
    c = sum(1 for v in vlog if v['correct']); t = len(vlog)
    print(f"[{ts}] 验证: {prev['prediction']} vs {aname} {'✓' if pcorr else '✗'} | {c}/{t}={c/max(t,1):.0%}")
    if t % 10 == 0:
        print(f"  世界线: " + " | ".join([f"O{i}:WL{int(np.argmax(wl[i]))}" for i in range(5)]))
        # 保存 display_acc 状态
        state = {'display_acc': display_acc.tolist(), 'display_cnt': display_cnt.tolist(), 'wl': wl.tolist()}
        with open(os.path.join(DATA_DIR, 'model_state.json'), 'w') as sf:
            json.dump(state, sf)