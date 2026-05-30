# i-ching-5d-worldline-engine

> **Restarting the 3000-year-old cosmic operating system.**
> 用现代计算重新启动三千年前的宇宙认知操作系统。

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## What is this?

We propose a fundamental hypothesis: **The I Ching (易经) is not a divination tool, but a forgotten "cosmic operating system."** Its hexagram generation process (大衍筮法) is structurally isomorphic to the "crystallizing spacetime" framework proposed by Filip Strubbe (2025) — a modern physics theory that explains quantum phenomena as projections from a higher-dimensional, classically deterministic structure.

This repository is the **engineering implementation** of that hypothesis. We restarted this ancient operating system with modern computation, Bayesian updating, and a 5D worldline-tracking architecture, and validated it on real-world tasks.

**Key result**: Every time we restored a structural element of the I Ching (互卦, 变卦, 爻辞, multi-observer consensus), the system's prediction accuracy improved measurably. No component was redundant.

---

## Core Hypothesis

> *The I Ching is a forgotten "Cosmic Cognitive Operating System." Its casting ritual simulates worldline collapse dynamics, and its interpretive rules decode causal dynamics. We restarted it with code.*

This hypothesis is independently supported by **Strubbe's crystallizing spacetime framework** [arXiv:2505.10383](https://arxiv.org/abs/2505.10383), which we discovered *after* completing our experimental suite (V3–V27). Both systems share the same core logic: an external evolution parameter (τ) drives dynamic evolution, multiple possibilities coexist in parallel, a dynamic process selects a single outcome, and the observer only sees the final result — not the process itself.

The 64 hexagrams are a **structured partition of the world's possible states** — an ancient attempt to index the projections of worldlines from a 5D deterministic structure into our 4D spacetime.

---

## Architecture

Three random sources, one unified six-layer cognitive stack, twelve parallel variants running hourly:

```
Source           Stack (same for all)
───────          ────────────────────
Weather (6 vars)     ↓
  → Hexagram     Affinity Vector (64×8)
                 × Three-Display Modulation (三陈: 体用/时位/卦变)
JPL Planetary        × Yao-Ci Encoding (爻辞)
(Jupiter/Earth)      × GuaQi Seasonal Weights (卦气)
  → Hexagram     × Worldline Priors (5 observers, differentiated)
                 × Bayesian Dirichlet Update
Entropy Pool     × Consensus Voting (多数投票)
(os.urandom)         ↓
  → Hexagram     Predict: Clear(晴)/Overcast(阴)/Rain(雨)
                     ↓
                 Verify after 1 hour → Collapse → Update
```

| Variant | Source | Observers | Upgraded |
|:--:|------|:--:|:--:|
| weather-1obs | 6-dim weather encoding | 1 | — |
| weather-5obs | 6-dim weather encoding | 5 | — |
| weather-1obs↑ | 6-dim weather encoding | 1 | EMA + switch detect |
| weather-5obs↑ | 6-dim weather encoding | 5 | EMA + switch detect |
| jpl-1obs | Jupiter/Earth ecliptic | 1 | — |
| jpl-5obs | Jupiter/Earth ecliptic | 5 | — |
| jpl-1obs↑ | Jupiter/Earth ecliptic | 1 | EMA + switch detect |
| jpl-5obs↑ | Jupiter/Earth ecliptic | 5 | EMA + switch detect |
| entropy-1obs | os.urandom() | 1 | — |
| entropy-5obs | os.urandom() | 5 | — |
| entropy-1obs↑ | os.urandom() | 1 | EMA + switch detect |
| entropy-5obs↑ | os.urandom() | 5 | EMA + switch detect |

All variants use **same hexagram for all observers** (同卦). Differences come only from worldline priors and Bayesian learning history. Check results anytime:

```bash
python3 check_oracles.py
```

---

## Key Experimental Results

| Experiment | Architecture | Key Finding |
|:---|:---|:---|
| V3–V4 | Structured hexagram prior | Outperforms neural networks in data-scarce regimes (+7.2pp @100 days) |
| V10–V11 | 5D Projection Model | +11.7pp over flat baseline in non-stationary environments |
| V16 | Tri-Hexagram Consensus (本/互/变) | Heterogeneous views outperform homogeneous consensus by 4pp |
| V19 | Sanchen Diviner (三陈) | Dimensional separation: +8.0% over original engine |
| V21 | Full-Diviner Multi-Method | Different hexagram inputs: first reliable multi > single |
| V22 | GuaQi Seasonal Weights (卦气) | Strongest single component: 1954 harvest |
| V25 | Bayesian Online Learning | Cold-start EMA: +4.1% over non-learning baseline |
| V27 | 5D × Sanchen Fusion | All four theoretical pillars active, each contributes |

**Component Attribution**: 52% → 76% cumulative gain from sequentially restoring I Ching structural elements. 24pp total improvement. No component was redundant.

---

## Paper

**Title**: *《易经》：五维投影的观测接口——基于结晶化时空框架的贝叶斯计算实现*

**English**: *The I Ching as a 5D Projection Observer: Implementing Crystallizing Spacetime with Bayesian World Models*

→ [`paper/draft.md`](paper/draft.md) (Chinese) · [`paper/draft_en.md`](paper/draft_en.md) (English)

---

## Quick Start

```bash
git clone https://github.com/guojie0050/i-ching-5d-worldline-engine.git
cd i-ching-5d-worldline-engine

# Install dependencies
pip install numpy matplotlib

# Run a single oracle (weather-encoded hexagram, single observer)
python3 oracles/runner.py --source weather --observers 1

# Run all 12 variants (requires crontab)
# See cron configuration at the top of this README or in crontab -l

# Check results across all variants
python3 check_oracles.py

# Run historical experiments
cd mvp
python3 experiment_v19.py  # Sanchen Diviner
python3 experiment_v27.py  # 5D × Sanchen Fusion
```

---

## Repository Structure

```
i-ching-5d-worldline-engine/
├── paper/                     ← 论文（中英文）
│   ├── draft.md               ← 中文版
│   └── draft_en.md            ← English version
├── mvp/                       ← V3–V27 实验代码
│   ├── model.py               ← 9+ 模型类 (IChing/HMM/Trigram/Symbolic/FiveD)
│   ├── sanchen_diviner.py     ← 三陈推演器 (V19)
│   ├── experiment_v*.py       ← V3–V27 实验脚本
│   └── results/               ← CSV 实验数据
├── oracles/                   ← 实时五维投影 Oracle (12 variants)
│   ├── runner.py              ← 统一 Oracle Runner (参数化)
│   ├── worldline_detector.py  ← 世界线切换检测
│   ├── data_*/                ← 各变体独立验证数据
│   └── oracle.log             ← 运行日志
├── check_oracles.py           ← Oracle 对比工具
└── README.md
```

---

## Keywords

`I Ching` `5D Projection` `Worldline` `Bayesian Update` `Crystallizing Spacetime` `World Model` `Causal Reasoning` `Interpretable AI` `Sample Efficiency` `Cognitive Architecture` `Observer Consensus` `Hexagram`

---

## License

CC BY 4.0 — Thoughts should flow freely.

---

## 中文简介

我们在此提出一个根本性的理论假说：**《易经》是一套被遗忘的"宇宙认知操作系统"。** 其大衍筮法的推演过程，与 Filip Strubbe 提出的"结晶化时空"物理学框架在结构上惊人一致——后者认为量子现象并非自然界的本质，而是四维时空在更高层次上演化的投影。

我们用现代计算重新启动了这套古老的操作系统，并通过 V3 至 V27 一系列实验验证了其有效性。结果表明：**每当我们还原《易经》的一个完整结构元素，系统在真实世界预测任务上的性能都获得了可测量的提升。**

我们不宣称构建了一个更好的 GPT，而是指出了一个被长期忽视的方向：三千年前的东方智慧，可能已经编码了宇宙认知的底层算法。

---

## Related Links

- **Strubbe crystallizing spacetime**: [arXiv:2505.10383](https://arxiv.org/abs/2505.10383)
- **Full paper manuscript**: [`paper/draft.md`](paper/draft.md)


## Real-Time Oracle System

12 parallel oracle variants run hourly, comparing 3 random sources (weather encoding / JPL planetary / quantum entropy) × 4 modes (single / 5-observer × original / upgraded).

```bash
python3 check_oracles.py
```

Data: `oracles/data_*/verify_log.json`. Architecture: `oracles/runner.py`.
