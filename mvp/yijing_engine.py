#!/usr/bin/env python3
"""
易经决策引擎 — YijingEngine (Full Architecture)
==================================================
六模块架构:
  1. 日历引擎   — 年月日时 → 天干地支
  2. 筮法引擎   — 干支 → 本卦/互卦/变卦/动爻
  3. 体用分析器 — 体用生克判断
  4. 时位判断器 — 动爻位置 → 阶段诊断
  5. 决策生成器 — 综合体用×时位×经验 → 行动决策
  6. 反馈学习器 — 结果反馈 → 更新体用可靠性
"""

import numpy as np
from datetime import date

# ============================================================================
# 模块1: 天干地支系统
# ============================================================================

TIAN_GAN = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
DI_ZHI   = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
GAN_WUXING = ['木','木','火','火','土','土','金','金','水','水']
ZHI_WUXING = ['水','土','木','木','土','火','火','土','金','金','土','水']

# 年干支 (以甲子年=4AD为参考, 1984是甲子年)
GANZI_YEAR_REF = 1984  # 甲子年

def ganzhi_year(year):
    idx = (year - GANZI_YEAR_REF) % 60
    return TIAN_GAN[idx % 10] + DI_ZHI[idx % 12]

# 月干支 (年上起月法: 甲己之年丙作首)
YUE_GAN_START = {
    '甲': 2, '己': 2,  # 丙寅
    '乙': 4, '庚': 4,  # 戊寅
    '丙': 6, '辛': 6,  # 庚寅
    '丁': 8, '壬': 8,  # 壬寅
    '戊': 0, '癸': 0,  # 甲寅
}

def ganzhi_month(year, month):
    yg = TIAN_GAN[(year - GANZI_YEAR_REF) % 10]
    start = YUE_GAN_START[yg]
    gan_idx = (start + month - 1) % 10  # month=1 (寅月, Feb)
    zhi_idx = (month + 1) % 12  # 子(11)丑(12)寅(1)→index
    return TIAN_GAN[gan_idx] + DI_ZHI[zhi_idx]

# 日干支 (简化: date→偏移)
GANZI_DAY_REF = date(1900, 1, 1)

def ganzhi_day(d):
    days = (d - GANZI_DAY_REF).days
    return TIAN_GAN[days % 10] + DI_ZHI[days % 12]

# 时干支 (日上起时法: 甲己还加甲)
def ganzhi_hour(d, hour):
    dg = TIAN_GAN.index(ganzhi_day(d)[0])
    start = (dg % 5) * 2  # 甲己→0, 乙庚→2, 丙辛→4, 丁壬→6, 戊癸→8
    zhi_idx = hour // 2  # 0=子时, 1=丑时, ...
    gan_idx = (start + zhi_idx) % 10
    return TIAN_GAN[gan_idx] + DI_ZHI[zhi_idx]


# ============================================================================
# 模块2: 八卦与六十四卦系统
# ============================================================================

BAGUA = ['乾','兑','离','震','巽','坎','艮','坤']
BAGUA_SYMBOL = ['☰','☱','☲','☳','☴','☵','☶','☷']
BAGUA_WUXING = ['金','金','火','木','木','水','土','土']
BAGUA_NATURE = ['天','泽','火','雷','风','水','山','地']
BAGUA_YINYANG = [1,0,0,1,0,1,1,0]  # 1=阳

# 六十四卦: (卦名, 上卦idx, 下卦idx, 卦辞概要)
HEXAGRAMS = [
    (1,'乾为天',0,0,'元亨利贞'),
    (2,'坤为地',1,1,'元亨，利牝马之贞'),
    (3,'水雷屯',4,2,'元亨利贞，勿用有攸往'),
    (4,'山水蒙',6,4,'亨。匪我求童蒙，童蒙求我'),
    (5,'水天需',4,0,'有孚，光亨，贞吉'),
    (6,'天水讼',0,4,'有孚窒惕，中吉终凶'),
    (7,'地水师',1,4,'贞，丈人吉，无咎'),
    (8,'水地比',4,1,'吉。原筮元永贞'),
    (9,'风天小畜',3,0,'亨。密云不雨'),
    (10,'天泽履',0,7,'履虎尾，不咥人，亨'),
    (11,'地天泰',1,0,'小往大来，吉亨'),
    (12,'天地否',0,1,'否之匪人，不利君子贞'),
    (13,'天火同人',0,5,'同人于野，亨'),
    (14,'火天大有',5,0,'元亨'),
    (15,'地山谦',1,6,'亨，君子有终'),
    (16,'雷地豫',2,1,'利建侯行师'),
    (17,'泽雷随',7,2,'元亨利贞，无咎'),
    (18,'山风蛊',6,3,'元亨，利涉大川'),
    (19,'地泽临',1,7,'元亨利贞'),
    (20,'风地观',3,1,'盥而不荐，有孚颙若'),
    (21,'火雷噬嗑',5,2,'亨，利用狱'),
    (22,'山火贲',6,5,'亨，小利有攸往'),
    (23,'山地剥',6,1,'不利有攸往'),
    (24,'地雷复',1,2,'亨，出入无疾'),
    (25,'天雷无妄',0,2,'元亨利贞'),
    (26,'山天大畜',6,0,'利贞，不家食吉'),
    (27,'山雷颐',6,2,'贞吉，观颐自求口实'),
    (28,'泽风大过',7,3,'栋桡，利有攸往'),
    (29,'坎为水',4,4,'习坎，有孚维心亨'),
    (30,'离为火',5,5,'利贞亨，畜牝牛吉'),
    (31,'泽山咸',7,6,'亨利贞，取女吉'),
    (32,'雷风恒',2,3,'亨，无咎，利贞'),
    (33,'天山遁',0,6,'亨，小利贞'),
    (34,'雷天大壮',2,0,'利贞'),
    (35,'火地晋',5,1,'康侯用锡马蕃庶'),
    (36,'地火明夷',1,5,'利艰贞'),
    (37,'风火家人',3,5,'利女贞'),
    (38,'火泽睽',5,7,'小事吉'),
    (39,'水山蹇',4,6,'利西南，不利东北'),
    (40,'雷水解',2,4,'利西南，无所往'),
    (41,'山泽损',6,7,'有孚元吉，无咎可贞'),
    (42,'风雷益',3,2,'利有攸往，利涉大川'),
    (43,'泽天夬',7,0,'扬于王庭，孚号有厉'),
    (44,'天风姤',0,3,'女壮，勿用取女'),
    (45,'泽地萃',7,1,'亨，王假有庙'),
    (46,'地风升',1,3,'元亨，用见大人'),
    (47,'泽水困',7,4,'亨贞，大人吉无咎'),
    (48,'水风井',4,3,'改邑不改井'),
    (49,'泽火革',7,5,'巳日乃孚，元亨利贞'),
    (50,'火风鼎',5,3,'元吉亨'),
    (51,'震为雷',2,2,'亨，震来虩虩'),
    (52,'艮为山',6,6,'艮其背不获其身'),
    (53,'风山渐',3,6,'女归吉，利贞'),
    (54,'雷泽归妹',2,7,'征凶，无攸利'),
    (55,'雷火丰',2,5,'亨，王假之'),
    (56,'火山旅',5,6,'小亨，旅贞吉'),
    (57,'巽为风',3,3,'小亨，利有攸往'),
    (58,'兑为泽',7,7,'亨利贞'),
    (59,'风水涣',3,4,'亨，王假有庙'),
    (60,'水泽节',4,7,'亨，苦节不可贞'),
    (61,'风泽中孚',3,7,'豚鱼吉，利涉大川'),
    (62,'雷山小过',2,6,'亨利贞，可小事不可大事'),
    (63,'水火既济',4,5,'亨小利贞，初吉终乱'),
    (64,'火水未济',5,4,'亨，小狐汔济'),
]

# 爻辞模板 (实际应有384条，此处为规则化简版)
YAO_STAGES = [
    (0, '潜伏期', '时机未到，宜隐忍待时，积蓄力量'),
    (1, '初现期', '可小步试探，观察反应，不可冒进'),
    (2, '发展期', '勤勉谨慎，防微杜渐，扎实前行'),
    (3, '上升期', '积极作为，察言观色，顺势而上'),
    (4, '鼎盛期', '时机成熟，放手一搏，但要防满招损'),
    (5, '转折期', '盛极将衰，功成身退，准备下一轮'),
]

# 五行生克
WUXING_SHENG = {'木':'火','火':'土','土':'金','金':'水','水':'木'}
WUXING_KE   = {'木':'土','土':'水','水':'火','火':'金','金':'木'}


# ============================================================================
# 主引擎类
# ============================================================================

class YijingEngine:
    """易经决策引擎 —— 六模块集成。"""
    
    def __init__(self):
        self.experience = {}
        self.decision_log = []
    
    # ── 日历 ──
    def get_ganzhi(self, year, month, day, hour=0):
        return {
            'year': ganzhi_year(year),
            'month': ganzhi_month(year, month),
            'day': ganzhi_day(date(year, month, day)),
            'hour': ganzhi_hour(date(year, month, day), hour),
        }
    
    # ── 筮法: 大衍筮法简化 ──
    def divine(self, year, month, day, hour=0):
        """
        简化大衍筮法:
          上卦 = (年支数 + 月支数) % 8
          下卦 = (日支数 + 时支数) % 8
          动爻 = (年支数 + 月支数 + 日支数 + 时支数) % 6
        """
        gz = self.get_ganzhi(year, month, day, hour)
        y_zhi = DI_ZHI.index(gz['year'][1])
        m_zhi = DI_ZHI.index(gz['month'][1])
        d_zhi = DI_ZHI.index(gz['day'][1])
        h_zhi = DI_ZHI.index(gz['hour'][1])
        
        shang = (y_zhi + m_zhi) % 8
        xia   = (d_zhi + h_zhi) % 8
        dongyao = (y_zhi + m_zhi + d_zhi + h_zhi) % 6
        
        ben_gua_idx = shang * 8 + xia
        # 变卦: 动爻翻转
        bian_shang = shang if dongyao >= 3 else (shang ^ (1 << (2 - dongyao))) if dongyao < 3 else shang
        bian_xia = xia if dongyao < 3 else (xia ^ (1 << (5 - dongyao))) if dongyao >= 3 else xia
        # 简化: dongyao位置的bit翻转
        hex_bin = (shang << 3) | xia
        hex_bin ^= (1 << (5 - dongyao))
        bian_shang = (hex_bin >> 3) & 7
        bian_xia = hex_bin & 7
        bian_gua_idx = bian_shang * 8 + bian_xia
        
        # 互卦: 本卦的2-4爻为下互, 3-5爻为上互
        # 简化: 取本卦二进制的中4位
        mid = (hex_bin >> 1) & 0b1111
        hu_shang = (mid >> 2) & 3
        hu_xia = mid & 3
        # 需要映射到3位: 取本卦的上下卦各取部分
        hu_shang = (shang & 0b110) | ((xia >> 2) & 0b001)
        hu_xia = ((shang & 0b001) << 2) | ((xia >> 1) & 0b011)
        hu_gua_idx = (hu_shang % 8) * 8 + (hu_xia % 8)
        
        ben = HEXAGRAMS[ben_gua_idx]
        bian = HEXAGRAMS[bian_gua_idx % 64]
        hu = HEXAGRAMS[hu_gua_idx % 64]
        
        return {
            'ben_gua': {'name': ben[1], 'idx': ben_gua_idx, 'upper': ben[2], 'lower': ben[3]},
            'hu_gua':  {'name': hu[1], 'idx': hu_gua_idx % 64, 'upper': hu[2], 'lower': hu[3]},
            'bian_gua':{'name': bian[1], 'idx': bian_gua_idx % 64, 'upper': bian[2], 'lower': bian[3]},
            'dongyao': dongyao,
            'ben_desc': ben[4],
        }
    
    # ── 体用分析 ──
    def analyze_tiyong(self, divine_result):
        """
        体用判断:
          动爻在0-2爻(下卦)→下卦为用,上卦为体
          动爻在3-5爻(上卦)→上卦为用,下卦为体
        生克关系:
          用生体=吉, 体克用=吉, 体生用=耗, 用克体=凶, 比和=平
        """
        dy = divine_result['dongyao']
        ben = divine_result['ben_gua']
        
        if dy < 3:
            ti_gua = ben['upper']
            yong_gua = ben['lower']
        else:
            ti_gua = ben['lower']
            yong_gua = ben['upper']
        
        ti_wx = BAGUA_WUXING[ti_gua]
        yong_wx = BAGUA_WUXING[yong_gua]
        ti_yin = BAGUA_YINYANG[ti_gua]
        yong_yin = BAGUA_YINYANG[yong_gua]
        
        # 生克判断
        if WUXING_SHENG.get(yong_wx) == ti_wx:
            tiyong, jixiong, reason = '用生体','吉','外部滋养主体，顺势可得成功'
        elif WUXING_KE.get(ti_wx) == yong_wx:
            tiyong, jixiong, reason = '体克用','吉','主体有能力掌控，需付出努力'
        elif WUXING_SHENG.get(ti_wx) == yong_wx:
            tiyong, jixiong, reason = '体生用','耗','主体消耗自身资源维持外部'
        elif WUXING_KE.get(yong_wx) == ti_wx:
            tiyong, jixiong, reason = '用克体','凶','外部环境不利，宜守不宜攻'
        elif ti_wx == yong_wx:
            tiyong, jixiong, reason = '比和','平','内外均衡，按常规行事'
        else:
            tiyong, jixiong, reason = '未知','平','关系不明确'
        
        # 阴阳分析
        yinyang_info = f"体{'阳' if ti_yin else '阴'}用{'阳' if yong_yin else '阴'}"
        
        return {
            'ti_gua': ti_gua, 'yong_gua': yong_gua,
            'ti_name': BAGUA[ti_gua], 'yong_name': BAGUA[yong_gua],
            'ti_symbol': BAGUA_SYMBOL[ti_gua], 'yong_symbol': BAGUA_SYMBOL[yong_gua],
            'ti_wuxing': ti_wx, 'yong_wuxing': yong_wx,
            'tiyong': tiyong, 'jixiong': jixiong, 'reason': reason,
            'yinyang': yinyang_info,
        }
    
    # ── 时位判断 ──
    def analyze_shiwei(self, divine_result):
        """动爻位置→时位阶段。"""
        dy = divine_result['dongyao']
        stage_idx, stage_name, stage_advice = YAO_STAGES[dy]
        
        # 互卦补充
        hu = divine_result['hu_gua']
        hu_upper_wx = BAGUA_WUXING[hu['upper']]
        hu_lower_wx = BAGUA_WUXING[hu['lower']]
        
        return {
            'dongyao': dy,
            'stage_idx': stage_idx,
            'stage_name': stage_name,
            'stage_advice': stage_advice,
            'hu_analysis': f"互卦: {BAGUA_NATURE[hu['upper']]}在上, {BAGUA_NATURE[hu['lower']]}在下",
        }
    
    # ── 决策生成 ──
    def generate_decision(self, tiyong_result, shiwei_result, divine_result, context=None):
        """综合体用×时位→决策。"""
        jx = tiyong_result['jixiong']
        stage = shiwei_result['stage_name']
        ti_name = tiyong_result['ti_name']
        yong_name = tiyong_result['yong_name']
        
        # 决策规则
        if jx == '吉':
            if '潜伏' in stage:
                action = 'prepare'
                advice = f'{ti_name}体{yong_name}用，{tiyong_result["tiyong"]}且时未至，先备后动'
            elif '鼎盛' in stage:
                action = 'act_bold'
                advice = f'{ti_name}体{yong_name}用，{tiyong_result["tiyong"]}且天时已至，全力一搏'
            else:
                action = 'act_normal'
                advice = f'{ti_name}体{yong_name}用，{tiyong_result["tiyong"]}，按计划推进'
        elif jx == '凶':
            if '潜伏' in stage:
                action = 'defend'
                advice = f'{ti_name}体{yong_name}用，{tiyong_result["tiyong"]}且时不利，收缩防守'
            elif '鼎盛' in stage:
                action = 'cautious_act'
                advice = f'{ti_name}体{yong_name}用，{tiyong_result["tiyong"]}但鼎盛时位可抵，谨慎突破'
            else:
                action = 'hold'
                advice = f'{ti_name}体{yong_name}用，{tiyong_result["tiyong"]}，宜守待机'
        else:
            action = 'normal'
            advice = shiwei_result['stage_advice']
        
        # 置信度
        conf = 0.9 if jx == '吉' else (0.5 if jx == '凶' else 0.7)
        
        return {
            'action': action,
            'advice': advice,
            'confidence': conf,
            'hex_name': divine_result['ben_gua']['name'],
            'tiyong': tiyong_result['tiyong'],
            'jixiong': jx,
            'stage': stage,
        }
    
    # ── 反馈学习 ──
    def learn_from_outcome(self, decision, was_good):
        """结果反馈→更新体用判断可靠性。"""
        key = decision['tiyong']
        if key not in self.experience:
            self.experience[key] = {'good': 0, 'total': 0}
        self.experience[key]['total'] += 1
        if was_good:
            self.experience[key]['good'] += 1
    
    def get_experience_stats(self):
        stats = {}
        for key, counts in self.experience.items():
            if counts['total'] > 0:
                stats[key] = counts['good'] / counts['total']
        return stats
    
    # ── 主接口: 完整推演 ──
    def consult(self, year, month, day, hour=0, context=None):
        """一次完整咨询 → 返回决策。"""
        gz = self.get_ganzhi(year, month, day, hour)
        div = self.divine(year, month, day, hour)
        tiyong = self.analyze_tiyong(div)
        shiwei = self.analyze_shiwei(div)
        decision = self.generate_decision(tiyong, shiwei, div, context)
        
        return {
            'ganzhi': gz,
            'divine': div,
            'tiyong': tiyong,
            'shiwei': shiwei,
            'decision': decision,
        }

    # ── 简化接口: 仅年份 ──
    def consult_year(self, year):
        """仅接受年份, 用干支推演 → 返回含推荐种植类型的决策"""
        result = self.consult(year, 3, 15)  # 春分前后
        self._mark_last_tiyong(result['tiyong']['tiyong'])
        dec = result['decision']
        tiyong = result['tiyong']
        shiwei = result['shiwei']
        
        # 推荐映射: 体用×时位 → 旱稻/水稻
        jx = dec['jixiong']; stage = dec['stage']
        rel = tiyong['tiyong']
        
        if jx == '吉':
            if '潜伏' in stage:
                rec = '旱稻'
                reason = f'{rel}({jx})但时位潜伏→保守等待'
            elif '鼎盛' in stage:
                rec = '水稻'
                reason = f'{rel}({jx})且时位鼎盛→全力一搏'
            else:
                rec = '水稻'
                reason = f'{rel}({jx})→按计划推进'
        elif jx == '凶':
            rec = '旱稻'
            reason = f'{rel}({jx})→防守策略'
        elif '耗' in rel:
            if '鼎盛' in stage:
                rec = '水稻'
                reason = f'{rel}但鼎盛→可尝试'
            else:
                rec = '旱稻'
                reason = f'{rel}→保守'
        else:
            rec = '旱稻'
            reason = f'{rel}→保守默认'
        
        # 置信度
        exp_stats = self.get_experience_stats()
        if rel in exp_stats:
            conf = 0.5 + 0.4 * exp_stats[rel]
        else:
            conf = dec.get('confidence', 0.5)
        
        return {
            'year': year,
            'recommendation': rec,
            'reason': reason,
            'confidence': conf,
            'hex_name': dec['hex_name'],
            'tiyong': rel,
            'jixiong': jx,
            'stage': stage,
            'ti_name': tiyong['ti_name'],
            'yong_name': tiyong['yong_name'],
            'ti_wuxing': tiyong['ti_wuxing'],
            'yong_wuxing': tiyong['yong_wuxing'],
            'dongyao': shiwei['dongyao'],
            'full': result,
        }
    
    # ── 结果反馈 ──
    def update_experience_from_result(self, was_correct):
        """坍缩后更新: 记录最后一次推演是否正确"""
        # 使用最近一次 consult_year 的结果
        if hasattr(self, '_last_tiyong'):
            self.learn_from_outcome(
                {'tiyong': self._last_tiyong},
                was_correct
            )
    
    def _mark_last_tiyong(self, tiyong_rel):
        self._last_tiyong = tiyong_rel




# ============================================================================
# 自测
# ============================================================================
if __name__ == '__main__':
    engine = YijingEngine()
    # 测试: 2025年6月15日午时
    result = engine.consult(2025, 6, 15, 12)
    print(f"干支: {result['ganzhi']}")
    print(f"卦象: {result['divine']['ben_gua']['name']} "
          f"({BAGUA_SYMBOL[result['divine']['ben_gua']['upper']]}"
          f"{BAGUA_SYMBOL[result['divine']['ben_gua']['lower']]})")
    print(f"互卦: {result['divine']['hu_gua']['name']}")
    print(f"变卦: {result['divine']['bian_gua']['name']}")
    print(f"动爻: 第{result['divine']['dongyao']+1}爻")
    print(f"体用: {result['tiyong']['tiyong']}({result['tiyong']['jixiong']}) "
          f"体{result['tiyong']['ti_name']}({result['tiyong']['ti_wuxing']}) "
          f"用{result['tiyong']['yong_name']}({result['tiyong']['yong_wuxing']})")
    print(f"时位: {result['shiwei']['stage_name']}")
    print(f"决策: {result['decision']['action']} — {result['decision']['advice']}")
