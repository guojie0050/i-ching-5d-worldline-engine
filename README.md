# 易经-贝叶斯世界模型

**Yijing-Bayesian World Model** — 以《易经》为骨，贝叶斯为血，构建能自我修正的世界预测 AI。

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Active Research](https://img.shields.io/badge/Status-Active_Research-blue.svg)]()

---

## 核心思想

当前大模型拟合"人类如何说"（下一个 token），我们构建一个学"世界如何是"的模型——以易经六十四卦为结构化先验因果框架，以贝叶斯更新为持续学习引擎。

**三句话概括**:
1. 六十四卦 = 对世界隐状态的离散划分（结构化先验）
2. 贝叶斯更新 = 每次真实反馈定向修正内部参数（不遗忘）
3. 确定性升级 = 模型对世界的认知越来越精确（可量化）

## 实验历程 (V3 → V9)

所有实验基于 5 种子 × 95% CI，训练/评估严格分离。

### V3-V4: 基础架构验证

| 实验 | 数据 | 关键发现 |
|------|------|---------|
| V3 | 合成 4-机制 Markov | 传统卦象先验 > 随机先验；易经 > NN |
| V4 | **北京 10年真实天气** | 三爻共享 512 参数 = 独立卦象 4096 参数精度 |

### V5-V8: 增量优化

| 实验 | 方向 | 结果 |
|------|------|------|
| V5 | 非对称权重 + delta + 六爻上下文 | 消融无差异：基线已太强 |
| V6 | 温度/降水/风速/湿度多特征 | 易经 +0%；NN +2.2pp（先验已内化特征） |
| V7 | 上下文窗口 K=1/3/5/7 | 易经不受影响；NN K=5 最佳 |
| V8 | 多步预测 1/2/3 天 | 六爻模型衰减 2.2% vs 三爻 5.5% |

### V9: HMM 架构探索（6 变体，全部负结果）

| HMM 变体 | 参数 | 100天 | 3000天 |
|----------|:--:|:--:|:--:|
| Full 64×64 | 4608 | 35.6% | 44.9% |
| Factored 8² | 640 | 35.6% | 44.9% |
| Rule-Driven (爻变) | 512 | 35.6% | 44.9% |
| 8-Trigram | 128 | 35.6% | 44.9% |
| Symbolic-Bayesian | 512 | 28.5% | 35.9% |
| **TrigramV4 (混合专家)** | **512** | **35.9%** | **54.8%** |

**HMM 结论**: 6 种隐马尔可夫架构全部不如简单混合专家模型。隐状态推断引入的信息瓶颈/正反馈锁死损害了性能。

### 🏆 最佳架构: TrigramV4

| 训练天数 | Trigram (512) | Neural-Net | 领先 |
|:--:|:--:|:--:|:--:|
| 100 | **35.9%** | 28.7% | +7.2pp |
| 500 | **48.5%** | 46.2% | +2.3pp |
| 1000 | **56.4%** | 53.8% | +2.6pp |
| 3000 | 54.8% | **56.8%** | -2.0pp |

- 小/中数据量：易经框架持续领先（先验优势）
- 大数据量（3000天）：NN 反超（通用逼近能力释放）
- 结论：结构化先验在**数据稀缺时**提供决定性优势

## 项目结构

```
yijing-bayesian-world-model/
├── README.md
├── paper/draft.md              ← 论文草稿
├── discussion/full-log.md      ← 思想演化记录
└── mvp/
    ├── model.py                ← 9 个模型类 (IChing/HMM/Symbolic/Trigram/NN)
    ├── experiment.py           ← V3 合成数据实验
    ├── experiment_real.py      ← V4 真实数据实验
    ├── experiment_v5.py        ← V5 三优化消融
    ├── experiment_v6.py        ← V6 多特征
    ├── experiment_v7.py        ← V7 上下文窗口
    ├── experiment_v8.py        ← V8 多步预测
    ├── experiment_v9.py        ← V9 HMM 架构探索
    ├── iching_*.png            ← 实验图表
    └── results/                ← CSV 数据
```

## 运行

```bash
cd mvp
pip install numpy matplotlib
python experiment_real.py  # V4: Beijing weather baseline
```

## 许可证

CC BY 4.0

---

> "易与天地准，故能弥纶天地之道。" ——《周易·系辞》
