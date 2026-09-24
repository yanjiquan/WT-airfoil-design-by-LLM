"""优化结果可视化：三张图

1. ``convergence.png``     —— 每代最优个体的收敛曲线（右上角图例标注最优值）
2. ``pop_boxplot.png``     —— 每代种群箱线图（obj / 平均Cl / 平均滞回面积）
3. ``population_dist.png`` —— 末代种群参数分布

平均滞回面积子图**仅在工况 (平均攻角, 攻角振幅) 唯一时绘制**——跨平均攻角/振幅
的滞回面积物理不可比，平均无意义。
"""
import json

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

C_BEST = '#1f77b4'
C_MEAN = '#ff7f0e'
C_FILL = '#bcd9ea'
C_CONV = '#d62728'
C_HYST = '#2ca02c'


def plot_pso_convergence(generation_history, save_path='convergence.png'):
    """美观的收敛曲线：每代最优适应度，右上角图例标注最优值。"""
    gens = [r['generation'] for r in generation_history]
    best = [r['best_fitness'] for r in generation_history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best, color=C_BEST, lw=2.5, marker='o', ms=5,
            label='Best individual')

    gbest = max(best)
    conv_gen = None
    for g, b in zip(gens, best):
        if abs(b - gbest) < 1e-9:
            conv_gen = g
            break

    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Objective (A3)', fontsize=14)
    ax.set_title('Optimization Convergence', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(loc='upper right', fontsize=12,
              title=f'Best: {gbest:.3f} @ gen {conv_gen if conv_gen is not None else "-"}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _extract_metric(generation_history, key):
    """从每代的 individual_metrics 提取指标（缺失时返回全空列表）。"""
    out = []
    for r in generation_history:
        vals = [m.get(key) for m in r.get('individual_metrics', [])
                if m and m.get(key) is not None]
        out.append(vals)
    return out


def plot_population_boxplots(generation_history, hyst_comparable=True,
                             save_path='pop_boxplot.png'):
    """每代种群箱线图：obj / 平均Cl / 平均滞回面积（可选）。"""
    gens = [r['generation'] for r in generation_history]
    obj_data = [r['all_fitness'] for r in generation_history]
    cl_data = _extract_metric(generation_history, 'cycle_mean_cl')
    hyst_data = _extract_metric(generation_history, 'hysteresis_mean_area')

    rows = [('Objective (A3)', obj_data, C_BEST),
            ('Mean Cl', cl_data, C_MEAN)]
    if hyst_comparable:
        rows.append(('Hysteresis Area (raw)', hyst_data, C_HYST))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3.1 * n_rows), sharex=True)

    for ax, (title, data, color) in zip(axes, rows):
        positions = [g for g, d in zip(gens, data) if d]
        data_ok = [d for d in data if d]
        if not data_ok:
            ax.set_ylabel(title, fontsize=11)
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            continue
        bp = ax.boxplot(data_ok, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')
        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, alpha=0.3, ls=':')

    axes[-1].set_xlabel('Generation', fontsize=13)
    fig.suptitle('Population Distribution by Generation',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_population_distribution(generation_record, save_path='population_dist.png'):
    """末代种群参数分布（A_u / A_l 前端与中部维度）。"""
    dims = [0, 4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    items = [(f'A_u[{d}]', 'A_u', d) for d in dims] + [(f'A_l[{d}]', 'A_l', d) for d in dims]

    for ax, (title, name, dim) in zip(axes.ravel(), items):
        vals = [ind[name][dim] for ind in generation_record['population']]
        ax.hist(vals, bins=8, color=C_BEST, alpha=0.75, edgecolor='white')
        m = generation_record['mean'][name][dim]
        ax.axvline(m, color=C_CONV, ls='--', lw=2, label=f'mean={m:.3f}')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, ls=':')
        ax.legend(fontsize=9)

    fig.suptitle(f'Population Parameter Distribution (Generation '
                 f'{generation_record["generation"]})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def analyze_pso_results(result_file='pso_final_result.json'):
    """分析优化结果并生成三张可视化图。"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    gen_history = results['generation_history']
    if not gen_history:
        print("[WARN] 无世代历史，跳过可视化")
        return

    best_fitness = results['best_fitness']
    initial = gen_history[0]['best_fitness']
    improvement = best_fitness - initial
    rate = (improvement / initial * 100) if initial else 0.0

    conv_gen = None
    for r in gen_history:
        if abs(r['best_fitness'] - best_fitness) < 1e-6:
            conv_gen = r['generation']
            break

    # 滞回面积可比性：顶层字段优先，其次第一代记录
    hyst_comparable = bool(results.get(
        'hyst_comparable', gen_history[0].get('hyst_comparable', False)))

    print("\n=== Optimization Result Analysis ===")
    print(f"Initial Fitness: {initial:.2f}")
    print(f"Optimal Fitness: {best_fitness:.2f}")
    print(f"Absolute Improvement: {improvement:.2f}")
    print(f"Relative Improvement: {rate:.1f}%")
    print(f"Convergence Generation: {conv_gen if conv_gen else 'Unknown'}")

    plot_pso_convergence(gen_history)
    plot_population_boxplots(gen_history, hyst_comparable=hyst_comparable)
    plot_population_distribution(gen_history[-1])

    print("\nVisualization charts generated:")
    print("  - convergence.png       (convergence curve)")
    print("  - pop_boxplot.png       (per-generation boxplots)"
          + ("" if hyst_comparable else "  [hysteresis skipped: mixed aoa/amp]"))
    print("  - population_dist.png   (final population distribution)")
