from sanchen_diviner import SanchenDiviner, divine_full
import json, os

c='/tmp/beijing_20yr.json'
d=json.load(open(c))['daily']
ts=d['time'];pr=d['precipitation_sum']
ys={}
for i,t in enumerate(ts):
    y=int(t.split('-')[0])
    if y not in ys: ys[y]={'p':0,'d':0}
    ys[y]['p']+=(pr[i] if pr[i] else 0); ys[y]['d']+=1
years=[{'year':y,'type':'旱' if s['p']/s['d']<0.8 else ('涝' if s['p']/s['d']>1.8 else '正常')}
       for y,s in sorted(ys.items())[:20]]

sd=SanchenDiviner()
total={0:0,1:0,2:0}; correct={0:0,1:0,2:0}
chen_names=['体用','时位','卦变']

# Per-weather-type stats
per_type={t:{'total':{0:0,1:0,2:0},'correct':{0:0,1:0,2:0}} for t in ['旱','涝','正常']}

agree=0; total_y=0
un_c=0; un_t=0; sp_c=0; sp_t=0

for yr in years:
    h=divine_full(yr['year'])
    r=sd.consult(h)
    actual=yr['type']
    optimal={'旱':'旱稻','涝':'水稻','正常':'水稻'}[actual]
    
    chen_decs=[
        ('旱稻' if r['chen1']['decision'] in ('防守','保守') else '水稻'),
        ('旱稻' if r['chen2']['decision'] in ('防守','保守') else '水稻'),
        ('旱稻' if r['chen3']['decision'] in ('防守','保守') else '水稻'),
    ]
    
    for i,dec in enumerate(chen_decs):
        total[i]+=1
        if dec==optimal: correct[i]+=1
        per_type[actual]['total'][i]+=1
        if dec==optimal: per_type[actual]['correct'][i]+=1
    
    raw=[r['chen1']['decision'],r['chen2']['decision'],r['chen3']['decision']]
    total_y+=1
    if len(set(raw))==1:
        agree+=1
        un_t+=1
        if r['recommendation']==optimal: un_c+=1
    else:
        sp_t+=1
        if r['recommendation']==optimal: sp_c+=1

print('=== 各陈独立准确率 ===')
for i in range(3):
    print(f'{chen_names[i]}: {correct[i]}/{total[i]} = {correct[i]/total[i]:.0%}')

print(f'\n三陈一致率: {agree}/{total_y} = {agree/total_y:.0%}')
print(f'分歧率: {1-agree/total_y:.0%}')
print(f'一致时正确率: {un_c}/{un_t} = {un_c/un_t:.0%}' if un_t else 'N/A')
print(f'分歧时正确率: {sp_c}/{sp_t} = {sp_c/sp_t:.0%}' if sp_t else 'N/A')

for yt in ['旱','涝','正常']:
    print(f'\n{yt}年:')
    for i in range(3):
        t=per_type[yt]['total'][i]; c=per_type[yt]['correct'][i]
        print(f'  {chen_names[i]}: {c}/{t} = {c/t:.0%}' if t else f'  {chen_names[i]}: N/A')

# 分歧模式统计
pattern={}
for yr in years:
    h=divine_full(yr['year'])
    r=sd.consult(h)
    raw=tuple([r['chen1']['decision'],r['chen2']['decision'],r['chen3']['decision']])
    pattern[raw]=pattern.get(raw,0)+1
print(f'\n=== 三陈分歧模式分布 ===')
for k,v in sorted(pattern.items(), key=lambda x:-x[1])[:8]:
    print(f'  {k}: {v}次')
