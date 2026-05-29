#!/usr/bin/env python3
"""
内部熵池融合版 — os.urandom() 起卦 × 六层单卦师架构 × 世界线分化
与 JPL 版架构完全一致，唯一变量：随机源
"""
import sys, os, json, time, numpy as np
from datetime import datetime
from collections import Counter

# ═══════════════ Layer 0: 64卦亲和向量 ═══════════════
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
WUXING = ["金","土","木","木","水","火","土","金"]
SHENG = {("木","火"),("火","土"),("土","金"),("金","水"),("水","木")}
KE   = {("木","土"),("土","水"),("水","火"),("火","金"),("金","木")}
TRIGRAM_SEASON = np.array([[0.7,1.0,0.8,0.5],[0.4,0.5,0.9,0.7],[0.9,1.5,0.6,0.3],[1.5,0.7,0.4,0.3],
    [0.8,0.5,0.8,0.7],[0.5,0.5,0.6,1.5],[0.4,0.5,0.7,1.2],[0.6,0.6,0.8,0.8]])
YAOCI_BIAS = {0:-0.05,1:0.0,2:0.03,3:0.05,4:0.08,5:-0.03}

def hex_aff(h):return 0.55*TRIGRAM_W[HEX_U[h%64]]+0.45*TRIGRAM_W[HEX_L[h%64]]
def guaqi_w(h,m):s=(m-1)//3;return 0.55*TRIGRAM_SEASON[HEX_U[h%64],s]+0.45*TRIGRAM_SEASON[HEX_L[h%64],s]

def sanchen_mod(probs,h,d,_):
    u,l=HEX_U[h%64],HEX_L[h%64];ti=WUXING[u if d<3 else l];yo=WUXING[l if d<3 else u]
    if (yo,ti) in SHENG: b=1.3
    elif (ti,yo) in KE: b=1.15
    elif (ti,yo) in SHENG: b=0.85
    elif (yo,ti) in KE: b=0.7
    else: b=1.0
    r=probs.copy();r[np.argmax(r)]*=b;r/=r.sum()
    r[0]+=[0.1,0.15,0.05,0.0,-0.1,0.1][d%6];return np.clip(r,0.02,0.95)/np.clip(r,0.02,0.95).sum()

def yaoci_mod(probs,d,da):
    probs[0]+=YAOCI_BIAS.get(d%6,0)*np.clip(np.mean(da),0.3,1.0)
    return np.clip(probs,0.02,0.95)/np.clip(probs,0.02,0.95).sum()

def weather_to_idx(wmo,tmp):
    if tmp and tmp>32:return 5
    return {0:0,1:0,2:2,3:1,45:7,48:7,51:4,53:4,55:4,61:4,63:4,65:4,71:7,73:7,75:7,80:4,81:4,82:4,95:3,96:3,99:3}.get(wmo,1)

def is_clear(w):return w in(0,5)

# ═══════════════ 随机源 ═══════════════
def entropy_seed(obs_id=0):
    return int.from_bytes(os.urandom(8), 'big') % (2**31-1) + obs_id * 7919

def cast_hex(seed):
    rng=np.random.default_rng(seed)
    return rng.integers(0,64), rng.integers(0,6)

# ═══════════════ 天气 ═══════════════
def get_weather():
    try:
        import urllib.request
        url="https://api.open-meteo.com/v1/forecast?latitude=39.9&longitude=116.4&current=weather_code,temperature_2m_max&timezone=Asia/Shanghai"
        d=json.loads(urllib.request.urlopen(url,timeout=5).read())
        return weather_to_idx(d['current']['weather_code'],d['current'].get('temperature_2m_max'))
    except:return 0

# ═══════════════ 初始化 ═══════════════
DATA_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data_entropy')
os.makedirs(DATA_DIR,exist_ok=True)
N_OBS,N_HEX,N_W=5,64,8

counts=np.zeros((N_OBS,N_HEX,N_W))
for oi in range(N_OBS):
    for h in range(N_HEX):counts[oi,h]=5.0*hex_aff(h)+1.0

wl=np.array([[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6],[0.5,0.3,0.2],[0.3,0.3,0.4]])
display_acc=np.full((N_OBS,3),0.5)

# 加载状态
sp=os.path.join(DATA_DIR,'model_state.json')
if os.path.exists(sp):
    s=json.load(open(sp)); wl=np.array(s.get('wl',wl.tolist())); display_acc=np.array(s.get('display_acc',display_acc.tolist()))

now=datetime.now();ts=now.strftime('%Y-%m-%d %H:%M');month=now.month

# ── 预测 (5人同卦, 不同世界线) ──
preds=[]
for oi in range(N_OBS):
    h_idx,dongyao=cast_hex(entropy_seed(0)  # 同卦)
    h_idx%=64
    probs=counts[oi,h_idx]/counts[oi,h_idx].sum()
    probs=sanchen_mod(probs,h_idx,dongyao,month)
    probs=yaoci_mod(probs,dongyao,display_acc[oi])
    gq=guaqi_w(h_idx,month);probs*=(0.5+0.5*gq);probs/=probs.sum()
    wi=np.argmax(wl[oi])
    if wi==0:probs[0]*=1.8;probs/=probs.sum()
    elif wi==2:probs[0]*=0.6;probs/=probs.sum()
    preds.append(0 if probs[0]+probs[5]>0.35 else 1)

vote=Counter(preds);consensus=vote.most_common(1)[0][0]
cname="晴" if consensus==0 else "阴雨"
print(f"[{ts}] 预测: {cname} (共识{vote[consensus]}/5) | " + " ".join(["晴" if p==0 else "雨" for p in preds]))

pred_log={'time':ts,'prediction':cname,'preds':preds,'consensus':vote[consensus]}
json.dump(pred_log,open(os.path.join(DATA_DIR,'last_pred.json'),'w'))

# ── 验证上轮 ──
vpath=os.path.join(DATA_DIR,'verify_log.json')
if os.path.exists(os.path.join(DATA_DIR,'last_pred.json')):
    prev=json.load(open(os.path.join(DATA_DIR,'last_pred.json')))
    actual=get_weather();aname="晴" if is_clear(actual) else "阴雨"
    pcorr=(1 if prev['prediction']=='晴' else 0)==(1 if is_clear(actual) else 0)
    
    for oi in range(N_OBS):
        h_idx,_=cast_hex(entropy_seed(0)  # 同卦);h_idx%=64
        ocorr=(prev['preds'][oi]==(1 if is_clear(actual) else 0))
        counts[oi,h_idx,actual]+=1.0
        twl=0 if is_clear(actual) else 1
        wl[oi][twl]*=1.5 if ocorr else 0.8;wl[oi]/=wl[oi].sum()
        for di in range(3):display_acc[oi][di]=0.95*display_acc[oi][di]+0.05*(1.0 if ocorr else 0.0)
    
    vlog=json.load(open(vpath)) if os.path.exists(vpath) else []
    vlog.append({'time':ts,'prediction':prev['prediction'],'actual':aname,'correct':pcorr})
    json.dump(vlog[-1000:],open(vpath,'w'),ensure_ascii=False)
    
    c=sum(1 for v in vlog if v['correct']);t=len(vlog)
    print(f"[{ts}] 验证: {prev['prediction']} vs {aname} {'✓' if pcorr else '✗'} | {c}/{t}={c/max(t,1):.0%}")
    
    state={'wl':wl.tolist(),'display_acc':display_acc.tolist()}
    json.dump(state,open(sp,'w'))
