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
