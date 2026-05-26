#!/usr/bin/env python3
"""
V4 实验: 三爻共享架构 + 北京真实天气数据
==========================================

对比:
  - Hexagram-Bayes  (64×8×8=4096 参数, 独立卦象)
  - Trigram-Bayes    (8×8×8=512 参数, 三爻共享)
  - Uniform-Trigram  (消融: 无先验知识)
  - Random-Trigram   (消融: 错误先验)
  - Markov           (基线)
  - Neural-Net       (纯数据驱动)

数据: Beijing 2015-2024, 3653 天 Weather Code + 温度
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, urllib.request, os, csv, warnings
warnings.filterwarnings("ignore")

from model import (
    IChingBayesianModel, TrigramBayesianModel,
    MarkovModel, NeuralNetModel,
    HEXAGRAM_AFFINITIES, TRIGRAM_WEATHER,
    N_HEXAGRAMS, N_WEATHER, WEATHER_TYPES, TRIGRAM_KEYS,
)

# ============================================================================
# 真实天气数据：WMO Code → 8 种天气类型
# ============================================================================

# WMO weather code → 天气类型 index
# 结合北京气候特点: 暑 = 最高温 > 32°C
WMO_MAP = {
    0: 0, 1: 0,              # Clear / Mainly clear → 晴
    2: 2,                     # Partly cloudy → 风
    3: 1,                     # Overcast → 阴
    45: 7, 48: 7,             # Fog → 雾
    51: 4, 53: 4, 55: 4,     # Drizzle → 雨
    61: 4, 63: 4, 65: 4,     # Rain → 雨
    71: 7, 73: 7, 75: 7,     # Snow → 雾 (北京少雪, 归雾)
    80: 4, 81: 4, 82: 4,     # Rain showers → 雨
    95: 3, 96: 3, 99: 3,     # Thunderstorm → 雷
}

HOT_THRESHOLD = 32.0  # °C, 超过此温度 → 暑


def fetch_beijing_weather(cache_path="/tmp/beijing_weather.json"):
    """下载北京 2015-2024 逐日天气 (含温度)。已缓存则直接读取。"""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url = ("https://archive-api.open-meteo.com/v1/archive?"
           "latitude=39.9&longitude=116.4"
           "&start_date=2015-01-01&end_date=2024-12-31"
           "&daily=weather_code,temperature_2m_max"
           "&timezone=Asia/Shanghai")
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def weather_codes_to_sequence(data):
    """WMO codes + 温度 → 8 类天气序列。"""
    codes = data["daily"]["weather_code"]
    temps = data["daily"]["temperature_2m_max"]
    seq = np.zeros(len(codes), dtype=int)

    for i, (code, temp) in enumerate(zip(codes, temps)):
        if temp is not None and temp > HOT_THRESHOLD:
            seq[i] = 5  # 暑
        else:
            seq[i] = WMO_MAP.get(code, 1)  # unknown → 阴

    return seq


# ============================================================================
# 实验配置
# ============================================================================

LABELS = [
    "Trigram-Bayes", "Hexagram-Bayes",
    "Uniform-Trigram", "Random-Trigram",
    "Markov", "Neural-Net",
]
SIZES = [100, 200, 500, 1000, 2000]
EVAL_WINDOW = 365
SEEDS = [42, 123, 456, 789, 1011]
PRIOR_STR = 1.0
TEMPERATURE = 0.5


def make_models(seed):
    rng = np.random.default_rng(seed)
    random_tri = rng.dirichlet(np.ones(N_WEATHER), 8)
    uniform_tri = np.ones((8, N_WEATHER)) / N_WEATHER

    return [
        # 三爻共享模型 (主推)
        TrigramBayesianModel(TRIGRAM_WEATHER, PRIOR_STR, TEMPERATURE),
        # 独立卦象模型 (V3 基准)
        IChingBayesianModel(HEXAGRAM_AFFINITIES, PRIOR_STR, TEMPERATURE),
        # 消融
        TrigramBayesianModel(uniform_tri, PRIOR_STR, TEMPERATURE),
        TrigramBayesianModel(random_tri, PRIOR_STR, TEMPERATURE),
        # 基线
        MarkovModel(),
        NeuralNetModel(ctx_win=3, hidden=32, lr=0.005, seed=seed),
    ], LABELS


def train_model(model, train_data):
    for t in range(1, len(train_data)):
        model.update(train_data[:t].tolist(), train_data[t])


def eval_model(model, train_data, eval_data):
    hist = train_data.tolist()
    correct = 0
    for act in eval_data:
        pred = model.predict(hist)
        if np.argmax(pred) == act:
            correct += 1
        hist.append(act)
    return correct / len(eval_data)


def run_experiment(weather):
    max_days = min(len(weather), max(SIZES) + EVAL_WINDOW)
    results = {N: {lb: [] for lb in LABELS} for N in SIZES}

    for si, seed in enumerate(SEEDS):
        print(f"  Seed {si+1}/{len(SEEDS)} (s={seed})...", end=" ", flush=True)
        for N in SIZES:
            models, labels = make_models(seed)
            train_w = weather[:N]
            eval_w = weather[N:N + EVAL_WINDOW]

            for m in models:
                train_model(m, train_w)

            for lb, m in zip(labels, models):
                acc = eval_model(m, train_w, eval_w)
                results[N][lb].append(acc)
        print("done")

    return results


# ============================================================================
# 可视化
# ============================================================================

def plot_results(results, weather, save=True):
    colors = {
        "Trigram-Bayes": "#1a5276", "Hexagram-Bayes": "#2980b9",
        "Uniform-Trigram": "#27ae60", "Random-Trigram": "#e67e22",
        "Markov": "#8e44ad", "Neural-Net": "#e74c3c",
    }
    markers = {
        "Trigram-Bayes": "o", "Hexagram-Bayes": "s",
        "Uniform-Trigram": "D", "Random-Trigram": "^",
        "Markov": "v", "Neural-Net": "x",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    for lb in LABELS:
        means = [np.mean(results[N][lb]) for N in SIZES]
        stds = [np.std(results[N][lb]) for N in SIZES]
        cis = [1.96 * s / np.sqrt(len(SEEDS)) for s in stds]
        ax1.errorbar(SIZES, means, yerr=cis, color=colors[lb],
                     marker=markers[lb], linewidth=2, markersize=7,
                     capsize=4, label=lb)
    ax1.axhline(1.0 / N_WEATHER, color="gray", ls=":", alpha=0.5,
                label=f"Random ({1.0/N_WEATHER:.1%})")
    ax1.set_xlabel("Training Days"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Beijing Weather (2015-2024): Accuracy vs Training Data")
    ax1.legend(fontsize=7, loc="lower right"); ax1.grid(alpha=0.3)
    ax1.set_xscale("log"); ax1.set_xticks(SIZES)
    ax1.set_xticklabels([str(s) for s in SIZES])

    # Architecture comparison: Trigram vs Hexagram
    arch_models = ["Trigram-Bayes", "Hexagram-Bayes", "Markov", "Neural-Net"]
    x = np.arange(len(SIZES))
    width = 0.2
    for i, lb in enumerate(arch_models):
        means = [np.mean(results[N][lb]) for N in SIZES]
        stds = [np.std(results[N][lb]) for N in SIZES]
        cis = [1.96 * s / np.sqrt(len(SEEDS)) for s in stds]
        off = (i - 1.5) * width
        ax2.bar(x + off, means, width, yerr=cis, color=colors[lb],
                label=lb, capsize=3, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels([str(s) for s in SIZES])
    ax2.set_xlabel("Training Days"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Architecture: Trigram-Shared (512) vs Hexagram (4096 params)")
    ax2.legend(fontsize=7); ax2.axhline(1.0/N_WEATHER, color="gray", ls=":", alpha=0.5)
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"I Ching Bayesian World Model — Real Weather (Beijing)\n"
        f"({len(weather)} days, {len(SEEDS)} seeds, 95% CI)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        plt.savefig("iching_real_weather.png", dpi=150, bbox_inches="tight")
        print("[图] iching_real_weather.png")
    plt.close(fig)


def save_csv(results, path="results/v4_real_weather.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Days", "Model", "Mean_Accuracy", "Std", "CI95"])
        for N in SIZES:
            for lb in LABELS:
                arr = np.array(results[N][lb])
                w.writerow([N, lb, f"{np.mean(arr):.4f}", f"{np.std(arr):.4f}",
                            f"{1.96*np.std(arr)/np.sqrt(len(arr)):.4f}"])
    print(f"[CSV] {path} ({len(SIZES)*len(LABELS)} rows)")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 64)
    print("  I Ching Bayesian World Model — V4 Real Weather")
    print("=" * 64)

    # 获取真实天气
    print("\n[1/4] 获取北京天气数据...")
    data = fetch_beijing_weather()
    weather = weather_codes_to_sequence(data)
    n_days = len(weather)

    # 统计分布
    uniq, cnts = np.unique(weather, return_counts=True)
    print(f"      {n_days} 天, 分布:")
    for u, c in zip(uniq, cnts):
        print(f"        {WEATHER_TYPES[u]}: {c:4d} ({c/n_days:.1%})")

    # 参数统计
    print(f"\n[2/4] 模型参数:")
    print(f"      Trigram-Bayes:   512 params (8×8×8)")
    print(f"      Hexagram-Bayes: 4096 params (64×8×8)")
    print(f"      Markov:           64 params (8×8)")
    print(f"      Neural-Net:      800 params (24-32-8)")

    # 运行实验
    print(f"\n[3/4] 数据效率实验...")
    results = run_experiment(weather)

    # 结果
    print(f"\n[4/4] 结果:")
    header = f"  {'Days':<8}"
    for lb in LABELS:
        header += f" {lb:<16}"
    print(header)
    print(f"  {'-'*80}")
    for N in SIZES:
        row = f"  {N:<8}"
        best = -1; best_lb = ""
        for lb in LABELS:
            m = np.mean(results[N][lb])
            row += f" {m:<15.1%}"
            if m > best: best = m; best_lb = lb
        row += f"  best: {best_lb}"
        print(row)

    # 关键发现
    N_s, N_l = 100, 2000
    tb_s = np.mean(results[N_s]["Trigram-Bayes"])
    hb_s = np.mean(results[N_s]["Hexagram-Bayes"])
    nn_s = np.mean(results[N_s]["Neural-Net"])

    tb_l = np.mean(results[N_l]["Trigram-Bayes"])
    hb_l = np.mean(results[N_l]["Hexagram-Bayes"])
    nn_l = np.mean(results[N_l]["Neural-Net"])

    print(f"\n  关键对比:")
    print(f"    @{N_s}天: Trigram={tb_s:.1%}  Hexagram={hb_s:.1%}  "
          f"NN={nn_s:.1%}")
    print(f"    @{N_l}天: Trigram={tb_l:.1%}  Hexagram={hb_l:.1%}  "
          f"NN={nn_l:.1%}")

    if tb_s > hb_s:
        print(f"    ✓ 三爻共享在小数据区间优于独立卦象 "
              f"(+{tb_s-hb_s:.1%})")
    if tb_s > nn_s:
        print(f"    ✓ 三爻贝叶斯 > 神经网络 (+{tb_s-nn_s:.1%})")

    # 图表
    plot_results(results, weather)
    save_csv(results)

    print(f"\n{'='*64}")
    print(f"  实验完成。三爻共享 = 参数减少 87.5% + 信息共享")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
