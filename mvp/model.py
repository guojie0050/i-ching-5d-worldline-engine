"""
易经贝叶斯世界模型 —— 核心模型实现
======================================

每个卦象维护一个 8×8 条件转移 Dirichlet 后验:
  p(w_{t+1} | w_t, hexagram h) ~ Dirichlet(alpha_h[w_t, :])

64 卦 × 8 × 8 = 4096 个 Dirichlet 参数。
预测: 按历史对数似然加权的混合专家聚合。
更新: Dirichlet-Multinomial 共轭软更新。

Reference:
  - 《周易》(I Ching), circa 1000 BCE
  - Jaynes, E.T. (2003). Probability Theory: The Logic of Science.
"""

import numpy as np

# ============================================================================
# 八卦 (Ba Gua) — 结构化先验的构建块
# ============================================================================

WEATHER_TYPES = ["晴", "阴", "风", "雷", "雨", "暑", "湿", "雾"]
N_WEATHER = len(WEATHER_TYPES)

TRIGRAM_KEYS = ["乾", "坤", "震", "巽", "坎", "离", "艮", "兑"]

# 八卦 → 天气倾向向量 (Prior Causal Framework)
TRIGRAM_WEATHER = np.array([
    [0.45, 0.05, 0.10, 0.05, 0.05, 0.25, 0.03, 0.02],  # 乾: 清朗炎热
    [0.03, 0.40, 0.05, 0.05, 0.08, 0.02, 0.25, 0.12],  # 坤: 阴沉潮湿
    [0.05, 0.10, 0.12, 0.38, 0.22, 0.05, 0.05, 0.03],  # 震: 雷暴降雨
    [0.10, 0.15, 0.42, 0.10, 0.08, 0.05, 0.05, 0.05],  # 巽: 大风阴天
    [0.03, 0.10, 0.05, 0.10, 0.40, 0.02, 0.20, 0.10],  # 坎: 降雨潮湿
    [0.20, 0.03, 0.05, 0.05, 0.02, 0.50, 0.08, 0.07],  # 离: 炎热清朗
    [0.05, 0.25, 0.08, 0.05, 0.08, 0.04, 0.15, 0.30],  # 艮: 雾气阴沉
    [0.05, 0.10, 0.08, 0.05, 0.20, 0.04, 0.25, 0.23],  # 兑: 潮湿雾雨
])



# ============================================================================
# 六十四卦 (64 Hexagrams) — King Wen Sequence
# ============================================================================

# (id, name, upper_trigram_index, lower_trigram_index)
KING_WEN = [
    (1, "乾", 0, 0), (2, "坤", 1, 1), (3, "屯", 4, 2), (4, "蒙", 6, 4),
    (5, "需", 4, 0), (6, "讼", 0, 4), (7, "师", 1, 4), (8, "比", 4, 1),
    (9, "小畜", 3, 0), (10, "履", 0, 7), (11, "泰", 1, 0), (12, "否", 0, 1),
    (13, "同人", 0, 5), (14, "大有", 5, 0), (15, "谦", 1, 6), (16, "豫", 2, 1),
    (17, "随", 7, 2), (18, "蛊", 6, 3), (19, "临", 1, 7), (20, "观", 3, 1),
    (21, "噬嗑", 5, 2), (22, "贲", 6, 5), (23, "剥", 6, 1), (24, "复", 1, 2),
    (25, "无妄", 0, 2), (26, "大畜", 6, 0), (27, "颐", 6, 2), (28, "大过", 7, 3),
    (29, "坎", 4, 4), (30, "离", 5, 5), (31, "咸", 7, 6), (32, "恒", 2, 3),
    (33, "遁", 0, 6), (34, "大壮", 2, 0), (35, "晋", 5, 1), (36, "明夷", 1, 5),
    (37, "家人", 3, 5), (38, "睽", 5, 7), (39, "蹇", 4, 6), (40, "解", 2, 4),
    (41, "损", 6, 7), (42, "益", 3, 2), (43, "夬", 7, 0), (44, "姤", 0, 3),
    (45, "萃", 7, 1), (46, "升", 1, 3), (47, "困", 7, 4), (48, "井", 4, 3),
    (49, "革", 7, 5), (50, "鼎", 5, 3), (51, "震", 2, 2), (52, "艮", 6, 6),
    (53, "渐", 3, 6), (54, "归妹", 2, 7), (55, "丰", 2, 5), (56, "旅", 5, 6),
    (57, "巽", 3, 3), (58, "兑", 7, 7), (59, "涣", 3, 4), (60, "节", 4, 7),
    (61, "中孚", 3, 7), (62, "小过", 2, 6), (63, "既济", 4, 5), (64, "未济", 5, 4),
]

N_HEXAGRAMS = len(KING_WEN)


def build_hexagram_affinities():
    """
    构建六十四卦的天气亲和度矩阵。
    每个卦象 = 上卦 55% + 下卦 45%, 归一化。
    """
    aff = np.zeros((N_HEXAGRAMS, N_WEATHER))
    for i, (_, _, ui, li) in enumerate(KING_WEN):
        a = 0.55 * TRIGRAM_WEATHER[ui] + 0.45 * TRIGRAM_WEATHER[li]
        aff[i] = a / a.sum()
    return aff


HEXAGRAM_AFFINITIES = build_hexagram_affinities()


# ============================================================================
# 易经贝叶斯模型
# ============================================================================

class IChingBayesianModel:
    """
    易经贝叶斯天气预测模型。

    每个卦象 h 维护 8×8 Dirichlet 转移矩阵:
      p(w_{t+1} | w_t, h) ~ Dirichlet(alpha_h[w_t, :])

    64 × 8 × 8 = 4096 Dirichlet 参数。

    先验:
      alpha_h[w_from, w_to] = prior_strength × affinity_h[w_to] + 1.0

    预测:
      p(w_{t+1} | history) = Σ_h p(w_{t+1} | w_t, h) × weight(h)
      权重由各卦象的历史对数似然决定 (softmax with temperature)。

    更新:
      观测 (w_t, w_{t+1}) 后:
        alpha_h[w_t, w_{t+1}] += lr × contribution(h)
      contribution(h) ∝ 卦象 h 对此正确转移的预测概率。
    """

    def __init__(self, affinities, prior_strength=1.0, temperature=0.5):
        """
        Args:
            affinities: (64, 8) 天气亲和度矩阵
            prior_strength: 先验强度 (高 = 更信任传统知识)
            temperature: 混合权重温度 (低 = 更集中在最佳卦象)
        """
        self.ps = prior_strength
        self.T = temperature
        self.nh = N_HEXAGRAMS
        self.nw = N_WEATHER

        # Dirichlet concentration parameters: (64, 8, 8)
        self.alpha = np.zeros((self.nh, self.nw, self.nw))
        for h in range(self.nh):
            for wf in range(self.nw):
                self.alpha[h, wf, :] = prior_strength * affinities[h, :] + 1.0

        self.prior_alpha = self.alpha.copy()
        # 各卦象累积对数似然 — 用于混合权重
        self.hex_ll = np.zeros(self.nh)

    def predict(self, history):
        """
        预测下一个天气分布。

        Args:
            history: list of int, 历史天气序列

        Returns:
            (8,) 概率分布
        """
        if len(history) == 0:
            return np.ones(self.nw) / self.nw

        wc = history[-1]

        # 温度缩放的 softmax 权重
        lw = self.hex_ll / max(self.T, 0.01)
        lw -= lw.max()
        wts = np.exp(lw)
        s = wts.sum()
        wts = wts / s if s > 1e-12 else np.ones(self.nh) / self.nh

        # 加权聚合
        pred = np.zeros(self.nw)
        for h in range(self.nh):
            p = self.alpha[h, wc, :] / self.alpha[h, wc, :].sum()
            pred += wts[h] * p

        return pred / pred.sum()

    def update(self, history, observed, lr=1.0):
        """
        贝叶斯更新 — 以真实转移为证据。

        Args:
            history: list of int, 历史天气
            observed: int, 观测到的 w_{t+1}
            lr: float, 学习率
        """
        if len(history) == 0:
            return

        wp, wn = history[-1], observed

        # 各卦象对正确结果的贡献度
        contrib = np.zeros(self.nh)
        for h in range(self.nh):
            p = self.alpha[h, wp, :] / self.alpha[h, wp, :].sum()
            contrib[h] = p[wn]

        cs = contrib.sum()
        contrib = contrib / cs if cs > 1e-12 else np.ones(self.nh) / self.nh

        # Dirichlet 软更新
        for h in range(self.nh):
            self.alpha[h, wp, wn] += lr * contrib[h]

        # 更新对数似然追踪
        for h in range(self.nh):
            p = self.alpha[h, wp, :] / self.alpha[h, wp, :].sum()
            self.hex_ll[h] += np.log(max(p[wn], 1e-12))

    def entropy(self, history):
        """当前预测的熵 (不确定性度量)。"""
        p = self.predict(history)
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p))

    def get_mixture_weights(self):
        """返回当前各卦象的混合权重。"""
        lw = self.hex_ll / max(self.T, 0.01)
        lw -= lw.max()
        wts = np.exp(lw)
        return wts / wts.sum()


# ============================================================================
# 三爻共享贝叶斯模型 (Trigram-Level Parameter Sharing)
# ============================================================================

# Build hex-to-trigram index mapping
HEX_TO_TRIGRAMS = np.array([(ui, li) for _, _, ui, li in KING_WEN], dtype=int)

# ============================================================================
# 六爻编码 (Yao Encoding) — 天气 ↔ 阴阳 ↔ 卦象
# ============================================================================

# 八卦二进制 (伏羲次序: 乾7→坤0, 上爻=高位)
TRIGRAM_BIN = np.array([7, 0, 4, 3, 2, 5, 1, 6], dtype=int)
#                  乾 坤 震 巽 坎 离 艮 兑

# 六十四卦二进制: hex_bin[h] = (上卦bin<<3) | 下卦bin
HEX_BIN = np.array([
    (TRIGRAM_BIN[ui] << 3) | TRIGRAM_BIN[li]
    for _, _, ui, li in KING_WEN
], dtype=int)

# 二进制→卦象索引 (反向查找表, 0-63)
BIN_TO_HEX = np.zeros(64, dtype=int)
for h, b in enumerate(HEX_BIN):
    BIN_TO_HEX[b] = h


def weather_to_yao(w):
    """天气→阴阳爻: 阳(1)=晴/暑/风, 阴(0)=其他"""
    return 1 if w in (0, 5, 2) else 0


def history_to_hexagram(history):
    """过去6天天气→6爻模式→卦象索引。不足6天返回None。"""
    if len(history) < 6:
        return None
    recent = history[-6:]
    yao_pattern = sum(weather_to_yao(w) << i for i, w in enumerate(recent))
    return BIN_TO_HEX[yao_pattern]




class TrigramBayesianModel:
    """
    三爻共享贝叶斯模型 —— 卦象参数由其上下三爻卦共享。

    每个三爻卦 t 维护独立 8×8 Dirichlet 转移矩阵:
      卦象 h = (t_u, t_l) 的参数 = (alpha[t_u] + alpha[t_l]) / 2

    8 三爻卦 × 8 × 8 = 512 个 Dirichlet 参数 (减少 87.5%)。
    共享三爻的卦象自动共享信息 → 小数据效率显著提升。

    先验: alpha[t, wf, :] = prior_strength × trigram_affinity[t, :] + 1.0
    预测: 同 hexagram 模型 (加权混合)
    更新: 按贡献分配给上下三爻卦 → alpha[t_u] 和 alpha[t_l] 各得一半更新
    """

    def __init__(self, trigram_affinities, prior_strength=1.0, temperature=0.5):
        self.ps = prior_strength
        self.T = temperature
        self.nt = 8       # 8 trigrams
        self.nh = 64       # 64 hexagrams
        self.nw = 8       # 8 weather types

        # 三爻卦级参数: (8, 8, 8)
        self.alpha = np.zeros((self.nt, self.nw, self.nw))
        for t in range(self.nt):
            for wf in range(self.nw):
                self.alpha[t, wf, :] = prior_strength * trigram_affinities[t, :] + 1.0

        self.prior_alpha = self.alpha.copy()
        self.hex_ll = np.zeros(self.nh)

    def _hex_alpha(self, h):
        """卦象 h 的有效 alpha: 上下三爻卦的均值。"""
        tu, tl = HEX_TO_TRIGRAMS[h]
        return (self.alpha[tu] + self.alpha[tl]) / 2.0

    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw

        wc = history[-1]

        lw = self.hex_ll / max(self.T, 0.01)
        lw -= lw.max()
        wts = np.exp(lw)
        s = wts.sum()
        wts = wts / s if s > 1e-12 else np.ones(self.nh) / self.nh

        pred = np.zeros(self.nw)
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wc, :] / alpha_h[wc, :].sum()
            pred += wts[h] * p

        return pred / pred.sum()

    def update(self, history, observed, lr=1.0):
        if len(history) == 0:
            return

        wp, wn = history[-1], observed

        # Per-hexagram contribution (same as hexagram model)
        contrib = np.zeros(self.nh)
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wp, :] / alpha_h[wp, :].sum()
            contrib[h] = p[wn]

        cs = contrib.sum()
        contrib = contrib / cs if cs > 1e-12 else np.ones(self.nh) / self.nh

        # Update trigram-level parameters (split contribution to upper/lower)
        for h in range(self.nh):
            tu, tl = HEX_TO_TRIGRAMS[h]
            self.alpha[tu, wp, wn] += lr * contrib[h] * 0.5
            self.alpha[tl, wp, wn] += lr * contrib[h] * 0.5

        # Update per-hexagram log-likelihood
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wp, :] / alpha_h[wp, :].sum()
            self.hex_ll[h] += np.log(max(p[wn], 1e-12))

    def entropy(self, history):
        p = self.predict(history)
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p))

    def get_mixture_weights(self):
        lw = self.hex_ll / max(self.T, 0.01)
        lw -= lw.max()
        wts = np.exp(lw)
        return wts / wts.sum()

    def param_count(self):
        """返回有效参数数量。"""
        return self.nt * self.nw * self.nw



# ============================================================================
# 完整易经世界模型 — 非对称权重 + 层次化delta + 六爻上下文
# ============================================================================

class HexagramWorldModel:
    """
    完整易经世界模型 (V5):

    ① 非对称权重: 上卦/下卦权重因卦而异 (纯卦对称, 混合卦可调)
    ② 层次化 delta: alpha[h] = shared + delta[h] (群体共享 + 个体偏差)
    ③ 六爻上下文: 过去6天阴阳模式→匹配卦象→提升权重

    参数量:
      - alpha: 8×8×8 = 512 (三爻共享)
      - delta: 64×8×8 = 4096 (卦象个体, L2 正则化约束)
      - w_upper: 64 (位置权重)
    """

    def __init__(self, trigram_affinities, prior_strength=1.0, temperature=0.5,
                 l2_delta=0.05, yao_boost=1.5):
        self.ps = prior_strength
        self.T = temperature
        self.l2 = l2_delta
        self.yao_boost = yao_boost
        self.nt = 8; self.nh = 64; self.nw = 8

        self.alpha = np.zeros((self.nt, self.nw, self.nw))
        for t in range(self.nt):
            for wf in range(self.nw):
                self.alpha[t, wf, :] = prior_strength * trigram_affinities[t, :] + 1.0

        self.delta = np.zeros((self.nh, self.nw, self.nw))

        self.w_upper = np.full(self.nh, 0.55)
        self.w_lower = np.full(self.nh, 0.45)
        pure_ids = [0, 1, 28, 29, 50, 51, 56, 57]
        for pid in pure_ids:
            self.w_upper[pid] = 0.50
            self.w_lower[pid] = 0.50

        self.hex_ll = np.zeros(self.nh)
        self.prior_alpha = self.alpha.copy()

    def _hex_alpha(self, h):
        tu, tl = HEX_TO_TRIGRAMS[h]
        shared = (self.w_upper[h] * self.alpha[tu] +
                  self.w_lower[h] * self.alpha[tl])
        return shared + self.delta[h]

    def _yao_bonus(self, history):
        matched = history_to_hexagram(history)
        bonus = np.ones(self.nh)
        if matched is not None:
            bonus[matched] = self.yao_boost
            matched_bin = HEX_BIN[matched]
            for h in range(self.nh):
                if bin(HEX_BIN[h] ^ matched_bin).count('1') == 1:
                    bonus[h] = 1.0 + (self.yao_boost - 1.0) * 0.3
        return bonus

    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        wc = history[-1]
        lw = self.hex_ll / max(self.T, 0.01)
        lw -= lw.max()
        wts = np.exp(lw)
        wts *= self._yao_bonus(history)
        s = wts.sum()
        wts = wts / s if s > 1e-12 else np.ones(self.nh) / self.nh
        pred = np.zeros(self.nw)
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wc, :] / alpha_h[wc, :].sum()
            pred += wts[h] * p
        return pred / pred.sum()

    def update(self, history, observed, lr=1.0):
        if len(history) == 0:
            return
        wp, wn = history[-1], observed
        contrib = np.zeros(self.nh)
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wp, :] / alpha_h[wp, :].sum()
            contrib[h] = p[wn]
        cs = contrib.sum()
        contrib = contrib / cs if cs > 1e-12 else np.ones(self.nh) / self.nh
        for h in range(self.nh):
            tu, tl = HEX_TO_TRIGRAMS[h]
            self.alpha[tu, wp, wn] += lr * contrib[h] * self.w_upper[h]
            self.alpha[tl, wp, wn] += lr * contrib[h] * self.w_lower[h]
            self.delta[h, wp, wn] += lr * contrib[h] * 0.1
            self.delta[h, wp, wn] *= (1.0 - self.l2)
        for h in range(self.nh):
            alpha_h = self._hex_alpha(h)
            p = alpha_h[wp, :] / alpha_h[wp, :].sum()
            self.hex_ll[h] += np.log(max(p[wn], 1e-12))

    def param_count(self):
        return self.nt * self.nw * self.nw + self.nh



# ============================================================================
# 卦象关系 (Hexagram Relationships) — 序卦传/综卦/错卦
# ============================================================================

# 综卦: swap upper/lower trigrams. zong_gua[h] = index of inverted hexagram
ZONG_GUA = np.zeros(N_HEXAGRAMS, dtype=int)
for h, (_, _, ui, li) in enumerate(KING_WEN):
    for h2, (_, _, ui2, li2) in enumerate(KING_WEN):
        if ui == li2 and li == ui2:
            ZONG_GUA[h] = h2; break

# 错卦: complement each trigram. 乾↔坤, 兑↔艮, 离↔坎, 震↔巽
CUO_MAP = {7:0, 0:7, 6:1, 1:6, 5:2, 2:5, 4:3, 3:4}  # trigram bin → complement
TRIGRAM_TO_BIN = {i: TRIGRAM_BIN[i] for i in range(8)}
BIN_TO_TRIGRAM = {v: k for k, v in TRIGRAM_TO_BIN.items()}
CUO_GUA = np.zeros(N_HEXAGRAMS, dtype=int)
for h in range(N_HEXAGRAMS):
    tb = HEX_BIN[h]
    upper_bin = tb >> 3
    lower_bin = tb & 0b111
    cuo_upper = CUO_MAP[upper_bin]
    cuo_lower = CUO_MAP[lower_bin]
    cuo_bin = (cuo_upper << 3) | cuo_lower
    CUO_GUA[h] = BIN_TO_HEX[cuo_bin]


def build_transition_prior(strength=10.0):
    """
    构建 64×64 转移先验矩阵 (Dirichlet concentration).
    基于序卦传关系: 自稳 > 综卦 > 错卦 > 邻卦 > 其他.
    """
    T = np.full((N_HEXAGRAMS, N_HEXAGRAMS), 0.1)  # baseline
    
    for h in range(N_HEXAGRAMS):
        T[h, h] = strength * 0.30               # 自稳
        T[h, ZONG_GUA[h]] = strength * 0.12     # 综卦
        T[h, CUO_GUA[h]] = strength * 0.08      # 错卦
        # 序卦传邻卦
        if h > 0: T[h, h-1] = strength * 0.05
        if h < N_HEXAGRAMS-1: T[h, h+1] = strength * 0.05
    T += 1.0  # Laplace smoothing
    return T


# ============================================================================
# 易经隐马尔可夫模型 (I Ching HMM) — 卦为隐状态, 天气为发射
# ============================================================================

class IChingHMM:
    """
    两层易经隐马尔可夫模型:
    
    隐状态层: z_t ∈ {1..64} (卦象)
      p(z_{t+1} | z_t) ~ Dirichlet(trans_alpha[z_t, :])
    
    观测层: w_t ∈ {1..8} (天气)
      p(w_t | z_t) ~ Dirichlet(emit_alpha[z_t, :])
    
    推理 (观象): Forward 滤波 → p(z_t | w_{1..t})
    预测 (玩辞): p(w_{t+1} | w_{1..t}) = Σ p(w_{t+1}|z_{t+1}) p(z_{t+1}|z_t) p(z_t|history)
    更新: 软 EM — 用滤波分布加权更新转移矩阵和发射矩阵
    """

    def __init__(self, hex_affinities, trans_strength=10.0, emit_strength=3.0):
        self.nh = N_HEXAGRAMS; self.nw = N_WEATHER
        
        # 转移矩阵 Dirichlet: (64, 64)
        self.trans_alpha = build_transition_prior(trans_strength)
        self.prior_trans = self.trans_alpha.copy()
        
        # 发射矩阵 Dirichlet: (64, 8)  
        self.emit_alpha = np.zeros((self.nh, self.nw))
        for h in range(self.nh):
            self.emit_alpha[h, :] = emit_strength * hex_affinities[h, :] + 1.0
        self.prior_emit = self.emit_alpha.copy()
        
        # 滤波信念: p(z_t | w_{1..t})
        self.belief = np.ones(self.nh) / self.nh
    
    def forward_filter(self, observed_weather):
        """
        一步 Forward 滤波: belief_{t-1} → belief_t
        α_t(z) ∝ p(w_t | z) × Σ_{z'} p(z | z') × α_{t-1}(z')
        """
        emit_p = self.emit_alpha[:, observed_weather] / self.emit_alpha.sum(axis=1)
        trans_p = self.trans_alpha / self.trans_alpha.sum(axis=1, keepdims=True)
        pred_belief = self.belief @ trans_p  # Σ_{z'} p(z|z') * α(z')
        new_belief = emit_p * pred_belief
        s = new_belief.sum()
        if s > 1e-12:
            self.belief = new_belief / s
        else:
            self.belief = np.ones(self.nh) / self.nh
    
    def predict(self, history):
        """
        预测下一个天气: p(w_{t+1} | w_{1..t})
        = Σ_{z_t, z_{t+1}} p(w_{t+1}|z_{t+1}) p(z_{t+1}|z_t) belief(z_t)
        """
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        
        trans_p = self.trans_alpha / self.trans_alpha.sum(axis=1, keepdims=True)
        emit_p = self.emit_alpha / self.emit_alpha.sum(axis=1, keepdims=True)
        
        # belief(t) × trans → belief_{t+1|t}
        next_belief = self.belief @ trans_p  # (64,) 
        
        # p(w_{t+1}|history) = Σ_z emit(z, w_{t+1}) × next_belief(z)
        pred = next_belief @ emit_p  # (8,)
        return pred / pred.sum()
    
    def update(self, history, observed, lr_trans=1.0, lr_emit=1.0):
        """
        贝叶斯更新: 用滤波分布加权更新转移和发射参数.
        """
        if len(history) == 0:
            return
        
        # 更新发射矩阵: emit_alpha[z, w_obs] += lr × belief_old(z)
        emit_update = lr_emit * self.belief
        self.emit_alpha[:, observed] += emit_update
        
        # Forward 滤波到新状态
        old_belief = self.belief.copy()
        self.forward_filter(observed)
        
        # 更新转移矩阵: trans_alpha[z_old, z_new] += lr × belief_old(z_old) × belief_new(z_new)
        trans_update = lr_trans * np.outer(old_belief, self.belief)
        self.trans_alpha += trans_update
    
    def entropy(self, history):
        p = self.predict(history)
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p))

    def param_count(self):
        return self.nh * self.nh + self.nh * self.nw


# ============================================================================
# 因子化易经 HMM — 64×64转移分解为两个8×8三爻转移
# ============================================================================

class FactoredIChingHMM:
    """
    因子化易经隐马尔可夫模型。
    
    转移矩阵: p(u_{t+1},l_{t+1} | u_t,l_t) = p(u_{t+1}|u_t) × p(l_{t+1}|l_t)
    两个 8×8 三爻转移矩阵 (共128参数), 替代 64×64 矩阵 (4096参数).
    
    发射矩阵: 64×8 (保持不变, 三爻共享先验).
    
    信念: 因式化 p(u,l) ≈ p(u) × p(l), 减少滤波计算量.
    """

    def __init__(self, hex_affinities, trans_strength=5.0, emit_strength=3.0):
        self.nt = 8; self.nh = 64; self.nw = 8
        
        # 两个 8×8 三爻转移矩阵
        self.trans_u = self._build_trigram_trans_prior(trans_strength)
        self.trans_l = self._build_trigram_trans_prior(trans_strength)
        
        # 发射矩阵: (64, 8)
        self.emit_alpha = np.zeros((self.nh, self.nw))
        for h in range(self.nh):
            self.emit_alpha[h, :] = emit_strength * hex_affinities[h, :] + 1.0
        
        # 因式化信念: p(u) × p(l)
        self.belief_u = np.ones(self.nt) / self.nt
        self.belief_l = np.ones(self.nt) / self.nt
    
    def _build_trigram_trans_prior(self, strength):
        """8×8 三爻转移先验: 自稳 + 五行生克."""
        T = np.full((8, 8), 0.1)
        for t in range(8):
            T[t, t] = strength * 0.40  # 自稳
        # 五行生克链: 木(3,2)→火(5)→土(1,6)→金(0,7)→水(4)→木
        sheng = [(3,5),(2,5),(5,1),(5,6),(1,0),(6,0),(0,4),(7,4),(4,3),(4,2)]
        for a, b in sheng:
            T[a, b] += strength * 0.06
        T += 1.0
        return T
    
    def _full_transition(self):
        """Kronecker积构建64×64转移矩阵: T = Tu ⊗ Tl, 行归一化."""
        pu = self.trans_u / self.trans_u.sum(axis=1, keepdims=True)
        pl = self.trans_l / self.trans_l.sum(axis=1, keepdims=True)
        # Kronecker: T[(u',l'), (u,l)] = pu[u,u'] × pl[l,l']
        # Shape: (8,8,8,8) → reshape to (64,64)
        T = np.einsum('ab,cd->bdac', pu, pl)  # pu[a,b]*pl[c,d] => T[b,d,a,c]
        T = T.reshape(8*8, 8*8)  # (64,64)
        T /= T.sum(axis=1, keepdims=True)
        return T
    
    def _hex_to_ul(self, h):
        return HEX_TO_TRIGRAMS[h]
    
    def forward_filter(self, observed):
        """因式化 Forward 滤波."""
        # 发射概率
        emit_p = self.emit_alpha[:, observed] / self.emit_alpha.sum(axis=1)
        # 全转移矩阵
        T = self._full_transition()
        
        # 全信念: belief[(u,l)]
        full_belief = np.outer(self.belief_u, self.belief_l).ravel()  # (64,)
        
        # Forward
        pred_belief = full_belief @ T  # (64,)
        new_full = emit_p * pred_belief
        s = new_full.sum()
        if s > 1e-12:
            new_full /= s
        else:
            new_full = np.ones(self.nh) / self.nh
        
        # 因式化: 边缘化
        new_full = new_full.reshape(8, 8)
        self.belief_u = new_full.sum(axis=1)  # 边缘化 l
        self.belief_l = new_full.sum(axis=0)  # 边缘化 u
        self.belief_u /= self.belief_u.sum()
        self.belief_l /= self.belief_l.sum()
    
    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        
        T = self._full_transition()
        emit_p = self.emit_alpha / self.emit_alpha.sum(axis=1, keepdims=True)
        
        full_belief = np.outer(self.belief_u, self.belief_l).ravel()
        next_belief = full_belief @ T  # (64,)
        pred = next_belief @ emit_p  # (8,)
        return pred / pred.sum()
    
    def update(self, history, observed, lr_trans=0.3, lr_emit=1.0):
        if len(history) == 0:
            return
        
        # 旧信念
        old_full = np.outer(self.belief_u, self.belief_l)  # (8,8)
        
        # 更新发射
        emit_update = lr_emit * old_full.ravel()
        self.emit_alpha[:, observed] += emit_update
        
        # Forward 滤波
        self.forward_filter(observed)
        
        # 因式化信念
        new_full = np.outer(self.belief_u, self.belief_l)  # (8,8)
        
        # 更新两个三爻转移矩阵: 边缘化
        # trans_u[u_old, u_new] += Σ_{l_old,l_new} old_full[u_old,l_old] × new_full[u_new,l_new]
        # = (Σ_l old_full[u_old,l]) × (Σ_l new_full[u_new,l])
        u_marginal_old = old_full.sum(axis=1)  # p(u_old)
        u_marginal_new = new_full.sum(axis=1)  # p(u_new)
        self.trans_u += lr_trans * np.outer(u_marginal_old, u_marginal_new)
        
        l_marginal_old = old_full.sum(axis=0)  # p(l_old)
        l_marginal_new = new_full.sum(axis=0)  # p(l_new)
        self.trans_l += lr_trans * np.outer(l_marginal_old, l_marginal_new)
    
    def param_count(self):
        return self.nt*self.nt*2 + self.nh*self.nw



# ============================================================================
# 规则驱动易经 HMM — 爻变转移(确定) + 卦象→天气发射(学习)
# ============================================================================

# 五行元素分配
WUXING_ELEMENT = ["金","土","木","木","水","火","土","金"]  # 乾坤震巽坎离艮兑
WUXING_SHENG = {("木","火"),("火","土"),("土","金"),("金","水"),("水","木")}
WUXING_KE   = {("木","土"),("土","水"),("水","火"),("火","金"),("金","木")}


def build_rule_transition(self_prob=0.30, d1_prob=0.08, d2_prob=0.03, sheng_boost=1.5):
    """
    根据易经爻变规则构建确定性的64×64转移矩阵。
    
    转移权重基于:
      1. Hamming距离 (爻变数): 自稳 > 1爻变 > 2爻变 > ...
      2. 五行生克: 上/下卦五行相生时提升概率
    """
    T = np.zeros((N_HEXAGRAMS, N_HEXAGRAMS))
    
    for h in range(N_HEXAGRAMS):
        h_bin = HEX_BIN[h]
        u, l = HEX_TO_TRIGRAMS[h]
        
        for h2 in range(N_HEXAGRAMS):
            h2_bin = HEX_BIN[h2]
            d = bin(h_bin ^ h2_bin).count('1')
            
            if d == 0:    w = self_prob
            elif d == 1:  w = d1_prob
            elif d == 2:  w = d2_prob
            elif d == 3:  w = d2_prob * 0.3
            elif d == 6:  w = d1_prob * 0.3  # 错卦 (全变)
            else:         w = 0.001
            
            # 五行生克调制
            u2, l2 = HEX_TO_TRIGRAMS[h2]
            eu, eu2 = WUXING_ELEMENT[u], WUXING_ELEMENT[u2]
            el, el2 = WUXING_ELEMENT[l], WUXING_ELEMENT[l2]
            
            if (eu, eu2) in WUXING_SHENG: w *= sheng_boost
            if (el, el2) in WUXING_SHENG: w *= sheng_boost
            if (eu, eu2) in WUXING_KE:    w *= (2 - sheng_boost)
            if (el, el2) in WUXING_KE:    w *= (2 - sheng_boost)
            
            T[h, h2] = max(w, 1e-6)
    
    T /= T.sum(axis=1, keepdims=True)
    return T


class RuleDrivenHMM:
    """
    规则驱动易经隐马尔可夫模型。
    
    转移矩阵: 从易经爻变+五行规则构建 (固定, 不学习)
    发射矩阵: 64×8 Dirichlet (学习: 卦象→天气的映射)
    
    理念: 易经提供了"变化法则", 贝叶斯更新学习"法则如何显化为观测".
    """
    
    def __init__(self, hex_affinities, emit_strength=3.0, 
                 self_prob=0.30, d1_prob=0.08):
        self.nh = N_HEXAGRAMS; self.nw = N_WEATHER
        
        # 固定转移矩阵 (不学习)
        self.transition = build_rule_transition(self_prob, d1_prob)
        
        # 可学习的发射矩阵
        self.emit_alpha = np.zeros((self.nh, self.nw))
        for h in range(self.nh):
            self.emit_alpha[h, :] = emit_strength * hex_affinities[h, :] + 1.0
        self.prior_emit = self.emit_alpha.copy()
        
        self.belief = np.ones(self.nh) / self.nh
        self.belief_history = []  # for debugging
    
    def forward_filter(self, observed):
        """Forward 滤波: 固定转移 + 学习到的发射."""
        emit_p = self.emit_alpha[:, observed] / self.emit_alpha.sum(axis=1)
        pred_belief = self.belief @ self.transition
        new_belief = emit_p * pred_belief
        s = new_belief.sum()
        if s > 1e-12:
            self.belief = new_belief / s
        else:
            self.belief = np.ones(self.nh) / self.nh
    
    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        
        emit_p = self.emit_alpha / self.emit_alpha.sum(axis=1, keepdims=True)
        next_belief = self.belief @ self.transition
        pred = next_belief @ emit_p
        return pred / pred.sum()
    
    def update(self, history, observed, lr=1.0):
        """只更新发射矩阵, 转移矩阵保持不变."""
        if len(history) == 0:
            return
        
        old_belief = self.belief.copy()
        self.emit_alpha[:, observed] += lr * old_belief
        self.forward_filter(observed)
    
    def belief_entropy(self):
        """当前belief的熵 (衡量推理确定性)."""
        b = np.clip(self.belief, 1e-12, 1.0)
        return -np.sum(b * np.log(b))
    
    def param_count(self):
        return self.nh * self.nw  # only emission is learned



# ============================================================================
# 八卦 HMM — 8 态隐马尔可夫, 观测=隐状态维度 (信息瓶颈消除)
# ============================================================================

class TrigramHMM:
    """
    八卦隐马尔可夫模型: 8 个三爻卦作为隐状态。
    
    转移矩阵: 8×8 (64参数, 五行规则先验 + 数据学习)
    发射矩阵: 8×8 (64参数, 卦象→天气映射, 学习)
    
    关键: 8 态隐空间 + 8 类观测 → 信息瓶颈消除.
    """
    
    def __init__(self, trigram_affinities, trans_strength=5.0, emit_strength=3.0):
        self.nt = 8; self.nw = 8
        
        # 转移矩阵: 五行规则先验
        self.trans_alpha = np.full((8,8), 1.0)
        for t in range(8):
            self.trans_alpha[t, t] = 1.0 + trans_strength * 0.40  # 自稳
        sheng = [(3,5),(2,5),(5,1),(5,6),(1,0),(6,0),(0,4),(7,4),(4,3),(4,2)]
        for a, b in sheng:
            self.trans_alpha[a, b] += trans_strength * 0.06
        ke = [(3,1),(1,4),(4,5),(5,0),(0,3),(6,4),(4,5),(5,7),(7,3),(2,6)]  # simplified
        for a, b in [(3,1),(1,4),(4,5),(5,0),(0,3)]:
            self.trans_alpha[a,b] += trans_strength * 0.02
        
        # 发射矩阵: 三爻天气先验
        self.emit_alpha = np.zeros((self.nt, self.nw))
        for t in range(self.nt):
            self.emit_alpha[t,:] = emit_strength * trigram_affinities[t,:] + 1.0
        
        self.belief = np.ones(self.nt) / self.nt
    
    def forward_filter(self, observed):
        emit_p = self.emit_alpha[:, observed] / self.emit_alpha.sum(axis=1)
        trans_p = self.trans_alpha / self.trans_alpha.sum(axis=1, keepdims=True)
        pred = self.belief @ trans_p
        new_b = emit_p * pred
        s = new_b.sum()
        self.belief = new_b / s if s > 1e-12 else np.ones(self.nt)/self.nt
    
    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        trans_p = self.trans_alpha / self.trans_alpha.sum(axis=1, keepdims=True)
        emit_p = self.emit_alpha / self.emit_alpha.sum(axis=1, keepdims=True)
        next_b = self.belief @ trans_p
        return (next_b @ emit_p)
    
    def update(self, history, observed, lr_tr=0.2, lr_em=1.0):
        if len(history) == 0: return
        old = self.belief.copy()
        self.emit_alpha[:, observed] += lr_em * old
        self.forward_filter(observed)
        tr_update = lr_tr * np.outer(old, self.belief)
        self.trans_alpha += tr_update
    
    def param_count(self):
        return self.nt * self.nt + self.nt * self.nw



# ============================================================================
# 符号-贝叶斯双引擎模型 — 符号层做"观象→求变→玩辞", 贝叶斯层校准发射
# ============================================================================

class SymbolicBayesianModel:
    """
    符号-贝叶斯双引擎模型。
    
    符号引擎 (确定性, 不学习):
      观象: 给定天气 → 找到匹配的卦象 (发射先验)
      求变: 对于匹配卦象 → 计算爻变可达的变卦
      玩辞: 变卦 → 预测天气分布 (发射后验)
    
    贝叶斯引擎 (学习):
      用真实观测更新发射矩阵, 使"观象"越来越准。
    
    与纯HMM的关键区别: 符号引擎主动修剪状态空间,
    每步只在 ~8 个卦象上推理, 而非全 64 个。
    """
    
    def __init__(self, hex_affinities, emit_strength=3.0, top_k=8):
        self.nh = N_HEXAGRAMS; self.nw = N_WEATHER; self.top_k = top_k
        
        # 可学习的发射矩阵 (仅此被更新)
        self.emit_alpha = np.zeros((self.nh, self.nw))
        for h in range(self.nh):
            self.emit_alpha[h, :] = emit_strength * hex_affinities[h, :] + 1.0
        
        # 预计算爻变邻域: 每个卦象的 1-yao-change 邻居
        self.yao_neighbors = self._build_yao_neighbors()
        
        # 信念: 当前卦象分布
        self.belief = np.ones(self.nh) / self.nh
    
    def _build_yao_neighbors(self):
        """预计算每个卦象的 1爻变邻居列表."""
        neighbors = []
        for h in range(N_HEXAGRAMS):
            h_bin = HEX_BIN[h]
            neigh = []
            for h2 in range(N_HEXAGRAMS):
                if h == h2: continue
                if bin(HEX_BIN[h2] ^ h_bin).count('1') == 1:
                    neigh.append(h2)
            neighbors.append(neigh)
        return neighbors
    
    def _symbolic_observe(self, observed):
        """
        符号引擎: 观象 — 找到与当前天气最匹配的卦象。
        使用发射先验计算匹配度。
        """
        emit_p = self.emit_alpha[:, observed] / self.emit_alpha.sum(axis=1)
        # 选 top_k 最匹配的卦象
        top = np.argsort(emit_p)[-self.top_k:]
        return top
    
    def _symbolic_transform(self, observed_hexes):
        """
        符号引擎: 求变 — 对每个匹配卦象, 计算可达的变卦。
        可达 = 自身 + 1爻变邻居 + 错卦。
        """
        reachable = set()
        for h in observed_hexes:
            reachable.add(h)  # 自稳
            reachable.update(self.yao_neighbors[h])  # 1爻变
            reachable.add(CUO_GUA[h])  # 错卦
        return np.array(sorted(reachable))
    
    def _symbolic_predict(self, observed, current_hexes, next_hexes):
        """
        符号引擎: 玩辞 — 从变卦的发射后验预测天气分布。
        
        信念流动: current_hexes → (爻变) → next_hexes → 发射 → 天气
        """
        emit_p = self.emit_alpha / self.emit_alpha.sum(axis=1, keepdims=True)
        
        # 当前卦象的信念 (归一化到 current_hexes)
        curr_belief = np.zeros(self.nh)
        curr_belief[current_hexes] = self.belief[current_hexes]
        s = curr_belief.sum()
        if s > 1e-12: curr_belief /= s
        
        # 爻变转移: curr → next (均匀分配到可达变卦)
        next_belief = np.zeros(self.nh)
        for h in current_hexes:
            if curr_belief[h] < 1e-8: continue
            targets = [h] + self.yao_neighbors[h] + [CUO_GUA[h]]
            targets = [t for t in targets if t in next_hexes]
            if not targets: continue
            per_target = curr_belief[h] / len(targets)
            for t in targets:
                next_belief[t] += per_target
        
        # 从变卦发射到天气
        pred = next_belief[next_hexes] @ emit_p[next_hexes]
        return pred / pred.sum()
    
    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        
        observed = history[-1]
        current = self._symbolic_observe(observed)
        next_h = self._symbolic_transform(current)
        return self._symbolic_predict(observed, current, next_h)
    
    def update(self, history, observed, lr=1.0):
        if len(history) == 0: return
        
        wp = history[-1] if len(history) >= 1 else observed
        current = self._symbolic_observe(wp)
        
        # 贝叶斯更新: 增强匹配卦象的发射概率
        self.emit_alpha[current, observed] += lr
        
        # 更新信念: 匹配卦象的信念提升
        emit_p = self.emit_alpha[:, observed] / self.emit_alpha.sum(axis=1)
        self.belief *= emit_p
        s = self.belief.sum()
        if s > 1e-12: self.belief /= s
    
    def param_count(self):
        return self.nh * self.nw




# ============================================================================
# 对照组模型
# ============================================================================

class MarkovModel:
    """一态贝叶斯转移学习器 (正确模型类)。"""
    def __init__(self, n_weather=N_WEATHER, prior_strength=1.0):
        self.nw = n_weather
        self.alpha = np.full((n_weather, n_weather), prior_strength)

    def predict(self, history):
        if len(history) == 0:
            return np.ones(self.nw) / self.nw
        return self.alpha[history[-1]] / self.alpha[history[-1]].sum()

    def update(self, history, observed, lr=1.0):
        if len(history) > 0:
            self.alpha[history[-1], observed] += lr


class NeuralNetModel:
    """2 层 MLP 基线 (纯数据驱动)。"""
    def __init__(self, ctx_win=3, hidden=32, lr=0.005, seed=42):
        rng = np.random.default_rng(seed)
        d = ctx_win * N_WEATHER
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, N_WEATHER))
        self.b2 = np.zeros(N_WEATHER)
        self.lr = lr
        self.cw = ctx_win

    def _enc(self, history):
        r = history[-self.cw:] if len(history) >= self.cw else history
        v = np.zeros(self.cw * N_WEATHER)
        off = (self.cw - len(r)) * N_WEATHER
        for i, w in enumerate(r):
            v[off + i * N_WEATHER + w] = 1.0
        return v

    def predict(self, history):
        x = self._enc(history)
        h = np.maximum(0, x @ self.W1 + self.b1)
        l = h @ self.W2 + self.b2
        l -= l.max()
        p = np.exp(l)
        return p / p.sum()

    def update(self, history, observed, lr=None):
        if lr is None:
            lr = self.lr
        x = self._enc(history)
        hp = x @ self.W1 + self.b1
        h = np.maximum(0, hp)
        l = h @ self.W2 + self.b2
        l -= l.max()
        p = np.exp(l)
        p /= p.sum()
        dl = p.copy()
        dl[observed] -= 1.0
        dW2 = np.outer(h, dl)
        db2 = dl
        dh = self.W2 @ dl
        dh[hp <= 0] = 0.0
        dW1 = np.outer(x, dh)
        db1 = dh
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1