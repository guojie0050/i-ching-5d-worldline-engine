#!/usr/bin/env python3
"""
三陈九卦推演器 — SanchenDiviner
================================
孔子《易传·系辞》: "三陈九卦" — 卦德、卦用、卦时三次推演, 独立判断, 内部共识。

架构:
  第一陈(体用): 只看五行生克 → 吉凶判断
  第二陈(时位): 只看动爻位置+爻辞 → 时机判断  
  第三陈(卦变): 只看互卦+变卦 → 趋势判断
  内部共识: 三陈一致(高)/两陈一致(中)/全分歧(低)→防守
"""

import numpy as np

# ============================================================================
# 基础数据 (与 yijing_engine.py 共享但独立维护)
# ============================================================================
BAGUA = ['乾','兑','离','震','巽','坎','艮','坤']
BAGUA_WUXING = ['金','金','火','木','木','水','土','土']
BAGUA_NATURE = ['天','泽','火','雷','风','水','山','地']

SHENG = {'木':'火','火':'土','土':'金','金':'水','水':'木'}
KE   = {'木':'土','土':'水','水':'火','火':'金','金':'木'}

YAO_STRATEGIES = {
    0: ('防守', '初爻潜伏，时机未到，潜龙勿用'),
    1: ('保守', '二爻初现，崭露头角，可小步试探'),
    2: ('常规', '三爻发展，勤勉谨慎，防微杜渐'),
    3: ('进取', '四爻上升，接近核心，察言观色'),
    4: ('进取', '五爻鼎盛，时机成熟，飞龙在天'),
    5: ('保守', '上爻转折，盛极将衰，功成身退'),
}

GZ_YEAR_REF = 1984
TIAN_GAN = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
DI_ZHI = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']

# 六十四卦 (id, name, upper_trigram_idx, lower_trigram_idx)
HEXAGRAMS = [
    (1,'乾为天',0,0),(2,'坤为地',7,7),(3,'水雷屯',6,4),(4,'山水蒙',1,6),
    (5,'水天需',6,0),(6,'天水讼',0,6),(7,'地水师',7,6),(8,'水地比',6,7),
    (9,'风天小畜',5,0),(10,'天泽履',0,2),(11,'地天泰',7,0),(12,'天地否',0,7),
    (13,'天火同人',0,3),(14,'火天大有',3,0),(15,'地山谦',7,1),(16,'雷地豫',4,7),
    (17,'泽雷随',2,4),(18,'山风蛊',1,5),(19,'地泽临',7,2),(20,'风地观',5,7),
    (21,'火雷噬嗑',3,4),(22,'山火贲',1,3),(23,'山地剥',1,7),(24,'地雷复',7,4),
    (25,'天雷无妄',0,4),(26,'山天大畜',1,0),(27,'山雷颐',1,4),(28,'泽风大过',2,5),
    (29,'坎为水',6,6),(30,'离为火',3,3),(31,'泽山咸',2,1),(32,'雷风恒',4,5),
    (33,'天山遁',0,1),(34,'雷天大壮',4,0),(35,'火地晋',3,7),(36,'地火明夷',7,3),
    (37,'风火家人',5,3),(38,'火泽睽',3,2),(39,'水山蹇',6,1),(40,'雷水解',4,6),
    (41,'山泽损',1,2),(42,'风雷益',5,4),(43,'泽天夬',2,0),(44,'天风姤',0,5),
    (45,'泽地萃',2,7),(46,'地风升',7,5),(47,'泽水困',2,6),(48,'水风井',6,5),
    (49,'泽火革',2,3),(50,'火风鼎',3,5),(51,'震为雷',4,4),(52,'艮为山',1,1),
    (53,'风山渐',5,1),(54,'雷泽归妹',4,2),(55,'雷火丰',4,3),(56,'火山旅',3,1),
    (57,'巽为风',5,5),(58,'兑为泽',2,2),(59,'风水涣',5,6),(60,'水泽节',6,2),
    (61,'风泽中孚',5,2),(62,'雷山小过',4,1),(63,'水火既济',6,3),(64,'火水未济',3,6),
]

N_HEX = 64


# ============================================================================
# 筮法: 年干支→卦象结构
# ============================================================================
def divine_full(year):
    """年干支→本卦/互卦/变卦/动爻/综卦/错卦"""
    gz_idx = (year - GZ_YEAR_REF) % 60
    yg = gz_idx % 10; yz = gz_idx % 12
    shang = (yg + yz) % 8; xia = (yg * yz + yz) % 8
    dongyao = (yg + yz * 2) % 6
    hex_bin = (shang << 3) | xia
    
    # 互卦
    mid = (hex_bin >> 1) & 0b1111
    hu_shang = (mid >> 1) & 0b111; hu_xia = mid & 0b111
    hu = (hu_shang % 8) * 8 + (hu_xia % 8)
    hu_name = HEXAGRAMS[hu % N_HEX][1]
    
    # 变卦
    bian_bin = hex_bin ^ (1 << (5 - dongyao))
    bian = bian_bin % N_HEX
    bian_name = HEXAGRAMS[bian][1]
    
    # 综卦: swap
    zong = (xia << 3) | shang
    
    # 错卦: complement
    cuo_map = [7,6,5,4,3,2,1,0]
    upper = hex_bin >> 3; lower = hex_bin & 0b111
    cuo = (cuo_map[upper] << 3) | cuo_map[lower]
    
    return {
        'hexagram_idx': hex_bin % N_HEX,
        'hexagram_name': HEXAGRAMS[hex_bin % N_HEX][1],
        'shang': shang, 'xia': xia, 'dongyao': dongyao,
        'hugua_idx': hu % N_HEX, 'hugua_name': hu_name,
        'hugua_shang': hu_shang % 8, 'hugua_xia': hu_xia % 8,
        'biangua_idx': bian, 'biangua_name': bian_name,
        'zong_idx': zong % N_HEX, 'cuo_idx': cuo % N_HEX,
        'shang_name': BAGUA[shang], 'xia_name': BAGUA[xia],
        'shang_wx': BAGUA_WUXING[shang], 'xia_wx': BAGUA_WUXING[xia],
    }


# ============================================================================
# 推演引擎
# ============================================================================
class WuxingEngine:
    """五行生克引擎 — 供体用推演使用"""
    def get_element(self, gua_idx):
        return BAGUA_WUXING[gua_idx % 8]
    
    def get_relation(self, yong_wx, ti_wx):
        """用对体的关系"""
        if SHENG.get(yong_wx) == ti_wx: return '用生体'
        elif KE.get(ti_wx) == yong_wx: return '体克用'
        elif SHENG.get(ti_wx) == yong_wx: return '体生用'
        elif KE.get(yong_wx) == ti_wx: return '用克体'
        elif ti_wx == yong_wx: return '比和'
        return '未知'

class ShiweiEngine:
    """时位判断引擎"""
    def get_strategy(self, dongyao):
        return YAO_STRATEGIES.get(dongyao, ('常规', ''))

class GuabianEngine:
    """卦变推演引擎"""
    def analyze(self, hexagram):
        hu = hexagram.get('hugua_idx')
        bian = hexagram.get('biangua_idx')
        ben = hexagram['hexagram_idx']
        
        wx = WuxingEngine()
        
        # 互卦分析
        hu_shang = hexagram.get('hugua_shang', 0)
        hu_xia = hexagram.get('hugua_xia', 0)
        if hu is not None:
            hu_ti_wx = wx.get_element(hu_shang if hexagram['dongyao'] >= 3 else hu_xia)
            hu_yong_wx = wx.get_element(hu_xia if hexagram['dongyao'] >= 3 else hu_shang)
            hu_rel = wx.get_relation(hu_yong_wx, hu_ti_wx)
            hu_good = hu_rel in ('用生体', '体克用')
        else:
            hu_good = True
        
        # 变卦分析
        if bian is not None:
            bian_shang = HEXAGRAMS[bian][2]; bian_xia = HEXAGRAMS[bian][3]
            bian_ti_wx = wx.get_element(bian_shang if hexagram['dongyao'] >= 3 else bian_xia)
            bian_yong_wx = wx.get_element(bian_xia if hexagram['dongyao'] >= 3 else bian_shang)
            bian_rel = wx.get_relation(bian_yong_wx, bian_ti_wx)
            bian_good = bian_rel in ('用生体', '体克用')
        else:
            bian_good = True
        
        return hu_good, bian_good


# ============================================================================
# 三陈推演器
# ============================================================================
class SanchenDiviner:
    """
    三陈九卦推演器
    同一卦师, 三次独立推演同一组卦象, 内部取共识
    """
    def __init__(self, seed=42, cognitive_style='balanced', cold_start=True):
        self.wuxing = WuxingEngine()
        self.cognitive_style = cognitive_style
        self.shiwei = ShiweiEngine()
        self.guabian = GuabianEngine()
        self.rng = np.random.default_rng(seed)
        self.log = []
        self.total_consults = 0
        self.disagreement_count = 0
        # 贝叶斯更新: 三个推演器的历史准确率 (EMA)
        self.display_acc = [0.5, 0.5, 0.5]
        if not cold_start:
            _warm = _get_display_accuracy_from_data()
            if _warm is not None:
                self.display_acc = list(_warm)
        self.display_total = [0, 0, 0]
    
    def first_chen_tiyong(self, hexagram):
        """第一陈: 体用推演 — 只看五行生克"""
        shang = hexagram['shang']; xia = hexagram['xia']
        dongyao = hexagram['dongyao']
        
        if dongyao < 3:
            ti_gua, yong_gua = shang, xia
        else:
            ti_gua, yong_gua = xia, shang
        
        ti_wx = self.wuxing.get_element(ti_gua)
        yong_wx = self.wuxing.get_element(yong_gua)
        relation = self.wuxing.get_relation(yong_wx, ti_wx)
        
        if relation in ('用生体',):
            dec, reason = '进取', f'用生体({yong_wx}生{ti_wx})，外部滋养主体，顺势可得'
        elif relation in ('体克用',):
            dec, reason = '进取', f'体克用({ti_wx}克{yong_wx})，主体掌控外部，需努力可成'
        elif relation in ('体生用',):
            dec, reason = '保守', f'体生用({ti_wx}生{yong_wx})，主体消耗自身，应保守'
        elif relation in ('用克体',):
            dec, reason = '防守', f'用克体({yong_wx}克{ti_wx})，外部不利，宜守不宜攻'
        else:
            dec, reason = '常规', f'比和({ti_wx}同{yong_wx})，内外均衡，按常规'
        
        return {
            'decision': dec, 'reason': reason,
            'ti_gua': BAGUA[ti_gua], 'yong_gua': BAGUA[yong_gua],
            'ti_wx': ti_wx, 'yong_wx': yong_wx, 'relation': relation,
        }
    
    def second_chen_shiwei(self, hexagram):
        """第二陈: 时位推演 — 只看动爻位置"""
        dongyao = hexagram['dongyao']
        decision, desc = YAO_STRATEGIES[dongyao]
        
        yao_names = ['初爻','二爻','三爻','四爻','五爻','上爻']
        return {
            'decision': decision,
            'reason': desc,
            'dongyao': dongyao,
            'yao_name': yao_names[dongyao],
        }
    
    def third_chen_guabian(self, hexagram):
        """第三陈: 卦变推演 — 只看互卦和变卦"""
        hu_good, bian_good = self.guabian.analyze(hexagram)
        
        if hu_good and bian_good:
            return {'decision': '进取', 'reason': '互卦支持+变卦向好，趋势有利'}
        elif hu_good and not bian_good:
            return {'decision': '保守', 'reason': '互卦尚可但变卦示警，趋势转差'}
        elif not hu_good and bian_good:
            return {'decision': '保守', 'reason': '互卦困难但变卦向好，谨慎乐观'}
        else:
            return {'decision': '防守', 'reason': '互卦不利+变卦恶化，全面防守'}
    
    def _apply_style_bias(self, decisions, c1, c2, c3):
        """认知风格介入：仅在三陈分歧时生效"""
        vote = {}
        for d in decisions: vote[d] = vote.get(d, 0) + 1
        max_v = max(vote.values())
        
        style = self.cognitive_style
        
        if style == 'conservative':
            # 保守型：分歧时选最保守的选项
            priority = ['防守', '保守', '常规', '进取']
            for p in priority:
                if p in decisions:
                    return p, '中', f'保守型介入，分歧中选{p}'
        
        elif style == 'aggressive':
            # 进取型：分歧时选最进取的选项
            priority = ['进取', '常规', '保守', '防守']
            for p in priority:
                if p in decisions:
                    return p, '中', f'进取型介入，分歧中选{p}'
        
        elif style == 'shiwei_first':
            # 时位优先：分歧时采纳时位推演
            return c2['decision'], '中', f'时位优先介入，采纳{c2["decision"]}'
        
        elif style == 'tiyong_first':
            # 体用优先：分歧时采纳体用推演
            return c1['decision'], '中', f'体用优先介入，采纳{c1["decision"]}'
        
        elif style == 'contrarian':
            # 反直觉：分歧时选少数派
            min_opt = min(vote, key=vote.get)
            return min_opt, '低', f'逆向介入，分歧中选少数派{min_opt}'
        
        else:
            # balanced: 三陈全分歧→防守; 二陈一致→多数但降级
            if max_v == 2:
                final = max(vote, key=vote.get)
                if final == '进取': final = '保守'
                elif final == '常规': final = '保守'
                names = ['体用','时位','卦变']
                diss = None
                for i, d in enumerate(decisions):
                    if d != final: diss = names[i]; break
                return final, '中', f'平衡型，二陈一致({final})但{diss}分歧，降级保守'
            else:
                return '防守', '低', f'平衡型，三陈全分歧，全面防守'
    
    def fourth_chen_zongcuo(self, hexagram):
            """第四陈: 综错推演 — 只看向综卦和错卦, 不关心体用时位卦变"""
        zong = hexagram.get('zong_idx'); cuo = hexagram.get('cuo_idx')
        wx = self.wuxing
        
        # 综卦分析: 上下颠倒后的五行关系
        if zong is not None:
            h_zong = HEXAGRAMS[zong]
            zong_shang, zong_xia = h_zong[2], h_zong[3]
            zong_ti_wx = wx.get_element(zong_shang if hexagram['dongyao'] >= 3 else zong_xia)
            zong_yong_wx = wx.get_element(zong_xia if hexagram['dongyao'] >= 3 else zong_shang)
            zong_rel = wx.get_relation(zong_yong_wx, zong_ti_wx)
            zong_good = zong_rel in ('用生体', '体克用')
        else:
            zong_good = True
        
        # 错卦分析: 阴阳全反后的五行关系
        if cuo is not None:
            h_cuo = HEXAGRAMS[cuo]
            cuo_shang, cuo_xia = h_cuo[2], h_cuo[3]
            cuo_ti_wx = wx.get_element(cuo_shang if hexagram['dongyao'] >= 3 else cuo_xia)
            cuo_yong_wx = wx.get_element(cuo_xia if hexagram['dongyao'] >= 3 else cuo_shang)
            cuo_rel = wx.get_relation(cuo_yong_wx, cuo_ti_wx)
            cuo_good = cuo_rel in ('用生体', '体克用')
        else:
            cuo_good = True
        
        if zong_good and cuo_good:
            return {'decision': '进取', 'reason': '综卦错卦皆有利，反向视角也支持'}
        elif zong_good and not cuo_good:
            return {'decision': '保守', 'reason': '综卦有利但错卦示警，正面可行反面有隐患'}
        elif not zong_good and cuo_good:
            return {'decision': '保守', 'reason': '综卦不利但错卦反向有利'}
        else:
            return {'decision': '防守', 'reason': '综卦错卦皆不利，逆向验证不通过'}
    
    def consult(self, hexagram, use_fourth=False):
    def fourth_chen_zongcuo(self, hexagram):
            """第四陈: 综错推演 — 只看向综卦和错卦, 不关心体用时位卦变"""
        zong = hexagram.get('zong_idx'); cuo = hexagram.get('cuo_idx')
        wx = self.wuxing
        
        # 综卦分析: 上下颠倒后的五行关系
        if zong is not None:
            h_zong = HEXAGRAMS[zong]
            zong_shang, zong_xia = h_zong[2], h_zong[3]
            zong_ti_wx = wx.get_element(zong_shang if hexagram['dongyao'] >= 3 else zong_xia)
            zong_yong_wx = wx.get_element(zong_xia if hexagram['dongyao'] >= 3 else zong_shang)
            zong_rel = wx.get_relation(zong_yong_wx, zong_ti_wx)
            zong_good = zong_rel in ('用生体', '体克用')
        else:
            zong_good = True
        
        # 错卦分析: 阴阳全反后的五行关系
        if cuo is not None:
            h_cuo = HEXAGRAMS[cuo]
            cuo_shang, cuo_xia = h_cuo[2], h_cuo[3]
            cuo_ti_wx = wx.get_element(cuo_shang if hexagram['dongyao'] >= 3 else cuo_xia)
            cuo_yong_wx = wx.get_element(cuo_xia if hexagram['dongyao'] >= 3 else cuo_shang)
            cuo_rel = wx.get_relation(cuo_yong_wx, cuo_ti_wx)
            cuo_good = cuo_rel in ('用生体', '体克用')
        else:
            cuo_good = True
        
        if zong_good and cuo_good:
            return {'decision': '进取', 'reason': '综卦错卦皆有利，反向视角也支持'}
        elif zong_good and not cuo_good:
            return {'decision': '保守', 'reason': '综卦有利但错卦示警，正面可行反面有隐患'}
        elif not zong_good and cuo_good:
            return {'decision': '保守', 'reason': '综卦不利但错卦反向有利'}
        else:
            return {'decision': '防守', 'reason': '综卦错卦皆不利，逆向验证不通过'}
    
    def consult(self, hexagram, use_fourth=False):
        """主接口: 三陈推演 + 内部共识"""
        self.total_consults += 1
        
        c1 = self.first_chen_tiyong(hexagram)
        c2 = self.second_chen_shiwei(hexagram)
        c3 = self.third_chen_guabian(hexagram)
        
        if use_fourth:
            c4 = self.fourth_chen_zongcuo(hexagram)
            decisions = [c1['decision'], c2['decision'], c3['decision'], c4['decision']]
            chen_display = [c1, c2, c3, c4]
        else:
            decisions = [c1['decision'], c2['decision'], c3['decision']]
            chen_display = [c1, c2, c3]
        vote = {}
        for d in decisions: vote[d] = vote.get(d, 0) + 1
        max_v = max(vote.values())
        n_chen = len(decisions)
        
        if max_v == n_chen:
            # 全票通过
            # 三陈一致 — 认知风格不介入
            final = decisions[0]; conf = '高'
            if n_chen == 4:
                note = f'四陈一致({final})，体用时位卦变综错全部指向同一结论'
            else:
                note = f'三陈一致({final})，体用时位卦变全部指向同一结论'
        else:
            # 分歧 — 认知风格介入
            final, conf, note = self._apply_style_bias(decisions, c1, c2, c3)
            self.disagreement_count += 1
        
        # 决策映射到农事
        farm_map = {'进取': '水稻', '常规': '水稻', '保守': '旱稻', '防守': '旱稻'}
        farm_dec = farm_map[final]
        
        result = {
            'recommendation': farm_dec,
            'sanchen_decision': final,
            'confidence': conf,
            'consensus_note': note,
            'chen1': c1, 'chen2': c2, 'chen3': c3,
            'raw_votes': vote,
        }
        if use_fourth:
            result['chen4'] = c4
        
        self.log.append(result)
        return result
    
    def update(self, hexagram, was_correct, optimal_decision=None):
        """贝叶斯更新: 追踪每个推演器的历史准确率"""
        if optimal_decision is None: return
        if not self.log: return
        r = self.log[-1]
        
        # 各推演器的独立决策 → 是否正确
        chen_decs = [
            ('旱稻' if r['chen1']['decision'] in ('防守','保守') else '水稻'),
            ('旱稻' if r['chen2']['decision'] in ('防守','保守') else '水稻'),
            ('旱稻' if r['chen3']['decision'] in ('防守','保守') else '水稻'),
        ]
        for i in range(3):
            self.display_total[i] += 1
            correct = (chen_decs[i] == optimal_decision)
            self.display_acc[i] = 0.9 * self.display_acc[i] + 0.1 * (1.0 if correct else 0.0)
    
    def get_disagreement_rate(self):
        if self.total_consults == 0: return 0
        return self.disagreement_count / self.total_consults


# 预计算: 从10年训练数据获取各推演器的初始准确率
_DISPLAY_ACC_CACHE = None

def _get_display_accuracy_from_data():
    global _DISPLAY_ACC_CACHE
    if _DISPLAY_ACC_CACHE is not None:
        return _DISPLAY_ACC_CACHE
    
    import json, os
    c = '/tmp/beijing_20yr.json'
    if not os.path.exists(c): return None
    d = json.load(open(c))['daily']
    ts = d['time']; pr = d['precipitation_sum']
    
    sd = SanchenDiviner()
    correct = [0, 0, 0]; total = [0, 0, 0]
    
    for i, t in enumerate(ts):
        y = int(t.split('-')[0])
        if y > 2014: continue
        h = divine_full(y)
        r = sd.consult(h)
        actual_p = pr[i] if pr[i] else 0
        yt = '旱' if actual_p < 0.8 else ('涝' if actual_p > 1.8 else '正常')
        optimal = {'旱': '旱稻', '涝': '水稻', '正常': '水稻'}[yt]
        
        chen_decs = [
            ('旱稻' if r['chen1']['decision'] in ('防守','保守') else '水稻'),
            ('旱稻' if r['chen2']['decision'] in ('防守','保守') else '水稻'),
            ('旱稻' if r['chen3']['decision'] in ('防守','保守') else '水稻'),
        ]
        for j in range(3):
            total[j] += 1
            if chen_decs[j] == optimal: correct[j] += 1
    
    _DISPLAY_ACC_CACHE = tuple(correct[j]/max(total[j],1) for j in range(3))
    return _DISPLAY_ACC_CACHE


# ============================================================================
# 自测
# ============================================================================
if __name__ == '__main__':
    diviner = SanchenDiviner()
    
    for year in [2015, 2020, 2024]:
        h = divine_full(year)
        r = diviner.consult(h)
        print(f"\n{'='*60}")
        print(f"  年份: {year}")
        print(f"  卦象: {h['hexagram_name']} 上{h['shang_name']}下{h['xia_name']} 动爻{r['chen2']['dongyao']}")
        print(f"  {'='*60}")
        print(f"  第一陈(体用): 体{r['chen1']['ti_gua']}({r['chen1']['ti_wx']}) "
              f"用{r['chen1']['yong_gua']}({r['chen1']['yong_wx']}) "
              f"{r['chen1']['relation']} → {r['chen1']['decision']}")
        print(f"  第二陈(时位): {r['chen2']['yao_name']} {r['chen2']['reason']} → {r['chen2']['decision']}")
        print(f"  第三陈(卦变): {r['chen3']['reason']} → {r['chen3']['decision']}")
        print(f"  {'='*60}")
        print(f"  投票: {r['raw_votes']} → {r['sanchen_decision']}({r['confidence']}置信)")
        print(f"  农事: {r['recommendation']}")
