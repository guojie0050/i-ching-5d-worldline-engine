#!/usr/bin/env python3
"""
升级版融合 Oracle — 概率化三陈 + 世界线切换检测
与 jpl_fusion.py 架构相同，差异仅在：(1) 平衡风格使用概率化分歧 (2) 集成世界线切换检测
"""
import sys, os, json, hashlib, time, numpy as np
from datetime import datetime
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'oracles', 'data_upgraded')
os.makedirs(DATA_DIR, exist_ok=True)

# ═══════════════ 独立副本: 64卦亲和 + 三陈 + 卦气 + 爻辞 ═══════════════
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
HEX_U=[kw[2] for kw in KW];HEX_L=[kw[3] for kw in KW]
WUXING=["金","土","木","木","水","火","土","金"]
SHENG={("木","火"),("火","土"),("土","金"),("金","水"),("水","木")}
KE={("木","土"),("土","水"),("水","火"),("火","金"),("金","木")}
TS=np.array([[0.7,1.0,0.8,0.5],[0.4,0.5,0.9,0.7],[0.9,1.5,0.6,0.3],[1.5,0.7,0.4,0.3],
    [0.8,0.5,0.8,0.7],[0.5,0.5,0.6,1.5],[0.4,0.5,0.7,1.2],[0.6,0.6,0.8,0.8]])
YB={0:-0.05,1:0.0,2:0.03,3:0.05,4:0.08,5:-0.03}
def ha(h):return 0.55*TRIGRAM_W[HEX_U[h%64]]+0.45*TRIGRAM_W[HEX_L[h%64]]
def gw(h,m):s=(m-1)//3;return 0.55*TS[HEX_U[h%64],s]+0.45*TS[HEX_L[h%64],s]
def sm(p,h,d,_):
    u,l=HEX_U[h%64],HEX_L[h%64];ti=WUXING[u if d<3 else l];yo=WUXING[l if d<3 else u]
    if (yo,ti) in SHENG:b=1.3
    elif (ti,yo) in KE:b=1.15
    elif (ti,yo) in SHENG:b=0.85
    elif (yo,ti) in KE:b=0.7
    else:b=1.0
    r=p.copy();r[np.argmax(r)]*=b;r/=r.sum()
    r[0]+=[0.1,0.15,0.05,0.0,-0.1,0.1][d%6];return np.clip(r,0.02,0.95)/np.clip(r,0.02,0.95).sum()
def ym(p,d,da):p[0]+=YB.get(d%6,0)*np.clip(np.mean(da),0.3,1.0);return np.clip(p,0.02,0.95)/np.clip(p,0.02,0.95).sum()
def wti(w,t):
    if t and t>32:return 5
    return {0:0,1:0,2:2,3:1,45:7,48:7,51:4,53:4,55:4,61:4,63:4,65:4,71:7,73:7,75:7,80:4,81:4,82:4,95:3,96:3,99:3}.get(w,1)
def ic(w):return w in(0,5)
def jpl_seed(lon_j,lon_e,obs_id=0):
    s=f"{lon_j:.10f}{lon_e:.10f}{int(time.time()/300)}{obs_id}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%(2**31-1)
def ch(seed):
    rng=np.random.default_rng(seed);return rng.integers(0,64),rng.integers(0,6)
def gjpl(n,d):
    try:
        import urllib.request
        c="599" if n=="jupiter" else "399"
        u=f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='{c}'&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='ECLIPTIC'&CENTER='500@0'&START_TIME='2026-05-29'&STOP_TIME='2026-05-30'&STEP_SIZE='1h'&QUANTITIES='31'"
        return float(json.loads(urllib.request.urlopen(u,timeout=10).read())['result'].split('$$SOE')[1].split()[2])
    except:return d
def gwth():
    try:
        import urllib.request
        u="https://api.open-meteo.com/v1/forecast?latitude=39.9&longitude=116.4&current=weather_code,temperature_2m_max&timezone=Asia/Shanghai"
        d=json.loads(urllib.request.urlopen(u,timeout=5).read())
        return wti(d['current']['weather_code'],d['current'].get('temperature_2m_max'))
    except:return 0

# ═══════════════ 世界线切换检测 ═══════════════
class WorldlineSwitchDetector:
    def __init__(self,n_wl=3,ws=5,th=0.05):
        self.n_wl=n_wl;self.ws=ws;self.th=th;self.rl=[];self.sc=0;self.tc=0
    def check(self,wl_probs,likelihoods):
        self.tc+=1;mx=float(max(likelihoods)) if len(likelihoods)>0 else 0.0
        self.rl.append(mx)
        if len(self.rl)>self.ws:self.rl.pop(0)
        if len(self.rl)>=self.ws and all(l<self.th for l in self.rl):
            self.sc+=1;self.rl=[]
            return True,np.ones(self.n_wl)/self.n_wl
        return False,wl_probs

# ═══════════════ 初始化 ═══════════════
N=5;cts=np.zeros((N,64,8))
for oi in range(N):
    for h in range(64):cts[oi,h]=5.0*ha(h)+1.0
wl=np.array([[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6],[0.5,0.3,0.2],[0.3,0.3,0.4]])
da=np.full((N,3),0.5)  # 三陈 display EMA
detector=WorldlineSwitchDetector()

sp=os.path.join(DATA_DIR,'model_state.json')
if os.path.exists(sp):
    s=json.load(open(sp));wl=np.array(s.get('wl',wl.tolist()));da=np.array(s.get('da',da.tolist()))

now=datetime.now();ts=now.strftime('%Y-%m-%d %H:%M');mo=now.month

# ── 预测 ──
lon_j,lon_e=gjpl("jupiter",234.5),gjpl("earth",78.9)
preds=[];likes=[]
for oi in range(N):
    hi,dy=ch(jpl_seed(lon_j,lon_e,0)  # 同卦);hi%=64
    p=cts[oi,hi]/cts[oi,hi].sum();p=sm(p,hi,dy,mo);p=ym(p,dy,da[oi])
    g=gw(hi,mo);p*=(0.5+0.5*g);p/=p.sum()
    wi=np.argmax(wl[oi])
    if wi==0:p[0]*=1.8;p/=p.sum()
    elif wi==2:p[0]*=0.6;p/=p.sum()
    likes.append(p[0]+p[5])
    preds.append(0 if p[0]+p[5]>0.35 else 1)

# ── 世界线切换检测 ──
switched,wl_new=detector.check(wl,likes)
if switched:
    wl=np.array([wl_new.copy() for _ in range(N)])
    print(f"[{ts}] ⚡ 世界线切换检测触发! 概率已重置 (第{detector.sc}次)")

vc=Counter(preds);cs=vc.most_common(1)[0][0]
cn="晴" if cs==0 else "阴雨"
print(f"[{ts}] 升级版预测: {cn} (共识{vc[cs]}/5) | "+" ".join(["晴" if x==0 else "雨" for x in preds]))
json.dump({'time':ts,'prediction':cn,'preds':preds,'consensus':vc[cs],'switched':switched},
          open(os.path.join(DATA_DIR,'last_pred.json'),'w'))

# ── 验证 ──
vp=os.path.join(DATA_DIR,'verify_log.json')
if os.path.exists(os.path.join(DATA_DIR,'last_pred.json')):
    pv=json.load(open(os.path.join(DATA_DIR,'last_pred.json')))
    aw=gwth();an="晴" if ic(aw) else "阴雨"
    pc=(1 if pv['prediction']=='晴' else 0)==(1 if ic(aw) else 0)
    for oi in range(N):
        hi,_=ch(jpl_seed(lon_j,lon_e,0)  # 同卦);hi%=64
        oc=(pv['preds'][oi]==(1 if ic(aw) else 0))
        cts[oi,hi,aw]+=1.0;tw=0 if ic(aw) else 1
        wl[oi][tw]*=1.5 if oc else 0.8;wl[oi]/=wl[oi].sum()
        for di in range(3):da[oi][di]=0.95*da[oi][di]+0.05*(1.0 if oc else 0.0)
    vl=json.load(open(vp)) if os.path.exists(vp) else[]
    vl.append({'time':ts,'prediction':pv['prediction'],'actual':an,'correct':pc,'upgraded':True})
    json.dump(vl[-1000:],open(vp,'w'),ensure_ascii=False)
    c=sum(1 for v in vl if v['correct']);t=len(vl)
    print(f"[{ts}] 验证: {pv['prediction']} vs {an} {'✓' if pc else '✗'} | {c}/{t}={c/max(t,1):.0%}")
    json.dump({'wl':wl.tolist(),'da':da.tolist(),'switches':detector.sc},open(sp,'w'))
