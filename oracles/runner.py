#!/usr/bin/env python3
"""
统一 Oracle Runner — 参数化控制：信源、观察者数、是否升级
用法:
  --source jpl|entropy|date
  --observers N        (1=单人, 5=五人团)
  --upgraded           (启用概率化三陈+世界线切换)
  --data-dir PATH      (可选, 默认自动生成)
"""
import sys, os, json, hashlib, time, numpy as np
from datetime import datetime
from collections import Counter

# ═══════════════ 参数解析 ═══════════════
args = {k:v for k,v in zip(sys.argv[1::2], sys.argv[2::2])}
source = args.get('--source', 'jpl')
n_obs = int(args.get('--observers', '5'))
upgraded = '--upgraded' in sys.argv
data_dir = args.get('--data-dir', f'data_{source}{"_up" if upgraded else ""}_{n_obs}obs')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
os.makedirs(DATA_DIR, exist_ok=True)

N_OBS, N_HEX, N_W = n_obs, 64, 8

# ═══════════════ Layer 0: 亲和向量 ═══════════════
TRIGRAM_W = np.array([
    [0.35,0.05,0.05,0.05,0.05,0.30,0.05,0.10],[0.05,0.30,0.10,0.05,0.10,0.05,0.25,0.10],
    [0.10,0.15,0.10,0.35,0.10,0.05,0.10,0.05],[0.15,0.10,0.35,0.10,0.10,0.05,0.10,0.05],
    [0.05,0.15,0.05,0.10,0.35,0.05,0.15,0.10],[0.15,0.05,0.05,0.05,0.05,0.40,0.15,0.10],
    [0.10,0.20,0.10,0.05,0.10,0.05,0.15,0.25],[0.10,0.15,0.10,0.05,0.15,0.05,0.15,0.25]])
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

def sanchen_mod(probs,h,d,da_oi,upgraded):
    """三陈调制: upgraded版用EMA加权"""
    u,l=HEX_U[h%64],HEX_L[h%64];ti=WUXING[u if d<3 else l];yo=WUXING[l if d<3 else u]
    if (yo,ti) in SHENG:b=1.3
    elif (ti,yo) in KE:b=1.15
    elif (ti,yo) in SHENG:b=0.85
    elif (yo,ti) in KE:b=0.7
    else:b=1.0
    if upgraded:
        # EMA加权: 如果体用display历史不准, 弱化它的调制幅度
        tiyong_trust = np.clip(da_oi[0], 0.3, 1.0) if da_oi is not None else 0.5
        b = 1.0 + (b-1.0)*tiyong_trust
    r=probs.copy();r[np.argmax(r)]*=b;r/=r.sum()
    r[0]+=[0.1,0.15,0.05,0.0,-0.1,0.1][d%6];return np.clip(r,0.02,0.95)/np.clip(r,0.02,0.95).sum()

def yaoci_mod(probs,d,da_oi):
    bias=YB.get(d%6,0)
    if da_oi is not None: bias*=np.clip(np.mean(da_oi),0.3,1.0)
    probs[0]+=bias;return np.clip(probs,0.02,0.95)/np.clip(probs,0.02,0.95).sum()

def wti(w,t):
    if t and t>32:return 5
    return {0:0,1:0,2:2,3:1,45:7,48:7,51:4,53:4,55:4,61:4,63:4,65:4,71:7,73:7,75:7,80:4,81:4,82:4,95:3,96:3,99:3}.get(w,1)
def ic(w):return w in(0,5)

# ═══════════════ 随机源 ═══════════════
def jpl_seed(lon_j,lon_e):
    s=f"{lon_j:.10f}{lon_e:.10f}{int(time.time()/300)}";return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%(2**31-1)
def entropy_seed():return int.from_bytes(os.urandom(8),'big')%(2**31-1)
def cast_hex(seed):rng=np.random.default_rng(seed);return rng.integers(0,64),rng.integers(0,6)

def weather_hex():
    """6维天气→6爻→卦象: 温度/湿度/风速/气压/云量/降水 → 阴阳爻"""
    import urllib.request
    url = "https://api.open-meteo.com/v1/forecast?latitude=39.9&longitude=116.4&current=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,cloud_cover,precipitation&timezone=Asia/Shanghai"
    c = json.loads(urllib.request.urlopen(url, timeout=5).read())['current']
    yao = 0
    yao |= (1 if c['temperature_2m'] > 15 else 0) << 0    # 1爻(初): 温度>15→阳
    yao |= (1 if c['relative_humidity_2m'] > 60 else 0) << 1  # 2爻: 湿度>60→阳
    yao |= (1 if c['wind_speed_10m'] > 3 else 0) << 2     # 3爻: 风速>3→阳
    yao |= (1 if c['pressure_msl'] > 1013 else 0) << 3    # 4爻: 气压>1013→阳
    yao |= (1 if c['cloud_cover'] > 50 else 0) << 4       # 5爻: 云量>50→阳
    yao |= (1 if c['precipitation'] > 0 else 0) << 5      # 6爻(上): 降水>0→阳
    # 动爻: 偏离中位最大的变量
    devs = [abs(c['temperature_2m']-15)/20, abs(c['relative_humidity_2m']-60)/40,
            abs(c['wind_speed_10m']-3)/5, abs(c['pressure_msl']-1013)/20,
            abs(c['cloud_cover']-50)/50, abs(c['precipitation']-0)/1]
    return yao % 64, devs.index(max(devs))

def get_seed(source):
    if source=='jpl':
        try:
            import urllib.request
            for cmd,name in[('599','jupiter'),('399','earth')]:
                u=f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='{cmd}'&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='ECLIPTIC'&CENTER='500@0'&START_TIME='2026-05-29'&STOP_TIME='2026-05-30'&STEP_SIZE='1h'&QUANTITIES='31'"
                d=json.loads(urllib.request.urlopen(u,timeout=10).read())
                if name=='jupiter':lj=float(d['result'].split('$$SOE')[1].split()[2])
                else:le=float(d['result'].split('$$SOE')[1].split()[2])
            return jpl_seed(lj,le),lj,le
        except:return jpl_seed(234.5,78.9),234.5,78.9
    elif source=='entropy':return entropy_seed(),0,0
    elif source=='weather':hx,dy=weather_hex();return hx,dy,0
    else:now=datetime.now();return date_hex(now.year,now.month,now.day)[0],now.year,now.month

def get_weather():
    """返回3分类: 0=晴, 1=阴, 2=雨"""
    try:
        import urllib.request
        u="https://api.open-meteo.com/v1/forecast?latitude=39.9&longitude=116.4&current=weather_code,temperature_2m_max&timezone=Asia/Shanghai"
        d=json.loads(urllib.request.urlopen(u,timeout=5).read())
        w = wti(d['current']['weather_code'],d['current'].get('temperature_2m_max'))
        if w in (0,5): return 0  # 晴
        if w in (3,4): return 2  # 雨
        return 1  # 阴/风/雾
    except:return 0

# ═══════════════ 世界线切换检测 (upgraded only) ═══════════════
class WorldlineSwitchDetector:
    def __init__(self,ws=5,th=0.05):self.ws=ws;self.th=th;self.rl=[];self.sc=0
    def check(self,likelihoods):
        mx=float(max(likelihoods)) if len(likelihoods)>0 else 0.0;self.rl.append(mx)
        if len(self.rl)>self.ws:self.rl.pop(0)
        if len(self.rl)>=self.ws and all(l<self.th for l in self.rl):self.sc+=1;self.rl=[];return True
        return False

# ═══════════════ 初始化 ═══════════════
cts=np.zeros((N_OBS,N_HEX,N_W))
for oi in range(N_OBS):
    for h in range(N_HEX):cts[oi,h]=5.0*ha(h)+1.0

wl=np.array([[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6],[0.5,0.3,0.2],[0.3,0.3,0.4]][:N_OBS])
da=np.full((N_OBS,3),0.5)
detector=WorldlineSwitchDetector() if upgraded else None

sp=os.path.join(DATA_DIR,'model_state.json')
if os.path.exists(sp):s=json.load(open(sp));wl=np.array(s.get('wl',wl.tolist()));da=np.array(s.get('da',da.tolist()))

now=datetime.now();ts=now.strftime('%Y-%m-%d %H:%M');mo=now.month
label=f"{source}{'_up' if upgraded else ''}-{n_obs}obs"

# ── 验证上轮预测 (先读取旧last_pred, 再覆盖新预测) ──
vp=os.path.join(DATA_DIR,'verify_log.json')
old_pred_path=os.path.join(DATA_DIR,'last_pred.json')
if os.path.exists(old_pred_path):
    pv=json.load(open(old_pred_path))
    # Only verify if the prediction is from a PREVIOUS hour (not just-created)
    pred_time = datetime.strptime(pv['time'], '%Y-%m-%d %H:%M')
    if pred_time < now:
        aw=get_weather();an=W3[aw]
        pc=(pv['prediction']==an)
        for oi in range(N_OBS):
            # Reconstruct hexagram from saved prediction context
            if source in ('date','weather'): hi2=pv.get('hex_idx',seed);dy2=pv.get('dongyao',0)%6
            else: hi2,_=cast_hex(jpl_seed(pv.get('jpl_lon',234.5),pv.get('jpl_lon',78.9)));hi2%=64;dy2=0
            oc=(pv['preds'][oi]==W3[aw] if 'preds' in pv else False)
            if not isinstance(oc, bool): oc=(pv['preds'][oi]==aw)
            cts[oi,hi2%64,aw]+=1.0;tw=0 if aw==0 else (1 if aw==1 else 2)
            if tw<3:wl[oi][tw]*=1.5 if oc else 0.8;wl[oi]/=wl[oi].sum()
            if upgraded:
                for di in range(3):da[oi][di]=0.95*da[oi][di]+0.05*(1.0 if oc else 0.0)
        vl=json.load(open(vp)) if os.path.exists(vp) else[]
        vl.append({'time':ts,'prediction':pv['prediction'],'actual':an,'correct':pc,'label':label})
        json.dump(vl[-1000:],open(vp,'w'),ensure_ascii=False)
        c=sum(1 for v in vl if v['correct']);t=len(vl)
        print(f"[{ts}] {label} 验证: {pv['prediction']} vs {an} {'✓' if pc else '✗'} | {c}/{t}={c/max(t,1):.0%}")
    else:
        # 首次运行, 无旧预测可验证
        pass

# ── 预测 (同卦: 所有卦师同一卦象, 仅世界线不同) ──
seed,aux1,aux2=get_seed(source)
if source in ('date','weather'):hi,dy=seed,aux2%6
else:hi,dy=cast_hex(seed);hi%=64

preds=[];likes=[]
for oi in range(N_OBS):
    p=cts[oi,hi]/cts[oi,hi].sum()
    p=sanchen_mod(p,hi,dy,da[oi] if upgraded else None, upgraded)
    p=yaoci_mod(p,dy,da[oi] if upgraded else None)
    g=gw(hi,mo);p*=(0.5+0.5*g);p/=p.sum()
    wi=np.argmax(wl[oi])
    if wi==0:p[0]*=1.8;p/=p.sum()
    elif wi==2:p[0]*=0.6;p/=p.sum()
    # 3分类预测: 晴/阴/雨
    if p[0]+p[5] > 0.35: pred = 0  # 晴概率高
    elif p[3]+p[4] > 0.25: pred = 2  # 雨概率高
    else: pred = 1  # 阴
    likes.append(max(p))
    preds.append(pred)

if upgraded and detector and detector.check(likes):
    wl=np.ones_like(wl)/wl.shape[1]
    print(f"[{ts}] ⚡ {label}: 世界线切换检测触发 (第{detector.sc}次)")

vc=Counter(preds);cs=vc.most_common(1)[0][0]
W3=["晴","阴","雨"]
cn=W3[cs]
print(f"[{ts}] {label}: {cn} (共识{vc[cs]}/{N_OBS}) | "+" ".join([W3[x] for x in preds]))

pred_log={'time':ts,'prediction':cn,'preds':preds,'consensus':vc[cs],'label':label}
json.dump(pred_log,open(os.path.join(DATA_DIR,'last_pred.json'),'w'))

    json.dump({'wl':wl.tolist(),'da':da.tolist(),'switches':detector.sc if detector else 0},open(sp,'w'))
