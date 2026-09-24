from airfoil_optimizer import AirfoilOptimizer
from visualization import analyze_pso_results
import airfoil_optimizer
import argparse
import json
import os

RESULT_FILE = 'pso_final_result.json'


def _load_previous_result():
    """加载上次运行结果，不存在则返回 None。"""
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULT_FILE)
    if not os.path.exists(result_path):
        return None
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='翼型优化工具（自动断点续跑）')
    parser.add_argument('--generations', type=int, default=100, help='进化世代数')
    parser.add_argument('--population', type=int, default=50, help='种群规模')
    parser.add_argument('--sigma', type=float, default=0.02, help='高斯采样方差')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--analyze', action='store_true', help='仅分析结果文件')
    parser.add_argument('--obj', choices=['Ld', 'cl', 'hybrid'], default='hybrid',
                        help='优化目标（续跑时从结果文件自动恢复）')
    parser.add_argument('--prompt-mode', choices=['few_shot', 'standard'], default='few_shot',
                        help='提示词结构（续跑时从结果文件自动恢复）')
    parser.add_argument('--method', choices=['LLM', 'GA', 'PSO', 'CMA-ES'], default='LLM',
                        help='优化方法（续跑时从结果文件自动恢复）')
    parser.add_argument('--fresh', action='store_true',
                        help='强制全新运行（忽略已有 pso_final_result.json）')
    parser.add_argument('--test', action='store_true',
                        help='测试模式：仅D1工况、7-10°攻角升降、2个体、2代，快速跑通功能')

    args = parser.parse_args()

    if args.analyze:
        analyze_pso_results()
        return

    # 加载配置
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

    # 测试模式：覆盖种群/世代，注入 test_mode 配置（优先用 --config 中的 test_mode）
    test_conditions = None
    if args.test:
        cfg_tm = (config or {}).get('test_mode', {})
        tm_conditions = list(cfg_tm.get('conditions') or ['D1'])
        tm_angles = [float(a) for a in (cfg_tm.get('angles') or [7.0, 8.0, 9.0, 10.0])]
        args.population = int(cfg_tm.get('population', 2))
        args.generations = int(cfg_tm.get('generations', 5))
        test_conditions = {
            "test_mode": {"enabled": True,
                          "conditions": tm_conditions,
                          "angles": tm_angles}
        }
        print(f"🧪 测试模式：工况 {tm_conditions}、攻角 {tm_angles}°升降、"
              f"种群 {args.population}、世代 {args.generations}")

    # ================================================================
    #  自动检测续跑
    # ================================================================
    prev = None if args.fresh else _load_previous_result()

    if prev and prev.get('generation_history'):
        history = prev['generation_history']
        last_gen = history[-1]['generation']          # 1-indexed 最后完成的世代
        total_gens = prev.get('total_generations', args.generations)

        # 允许 --generations 覆盖（扩展或截断）
        if args.generations != parser.get_default('generations'):
            print(f"⚠️  检测到 --generations={args.generations}，覆盖结果文件中的 {total_gens} 代")
            total_gens = args.generations

        remaining = total_gens - last_gen
        if remaining <= 0:
            print(f"✅ 所有 {total_gens} 世代已完成，无需续跑。")
            print(f"   全局最优适应度：{prev['best_fitness']:.6f}")
            analyze_pso_results(RESULT_FILE)
            return

        # ---- 恢复优化参数（缺失时用 CLI 默认值兜底） ----
        airfoil_optimizer.OPT_OBJ = prev.get('optimization_objective', args.obj)
        airfoil_optimizer.PROMPT_MODE = prev.get('prompt_mode', args.prompt_mode)
        airfoil_optimizer.METHOD = prev.get('method', args.method)

        # ---- 恢复 current_mean（优先用顶层字段，兜底用最后一代的 mean） ----
        if 'current_mean' in prev:
            resume_mean = prev['current_mean']
        else:
            resume_mean = history[-1].get('mean', None)

        # ---- 创建优化器并恢复状态 ----
        pop_size = prev.get('population_size', args.population)
        optimizer = AirfoilOptimizer(population_size=pop_size, sigma=args.sigma)

        optimizer.best_global = prev.get('best_individual')
        optimizer.best_fitness = prev.get('best_fitness', -float('inf'))
        optimizer.sampler.generation_history = history
        if resume_mean:
            optimizer.sampler.current_mean = resume_mean

        print(f"📂 自动检测到上次运行结果，从第 {last_gen + 1} 代续跑")
        print(f"   已完成: {last_gen}/{total_gens}  剩余: {remaining}")
        print(f"   优化目标: {airfoil_optimizer.OPT_OBJ}")
        print(f"   优化方法: {airfoil_optimizer.METHOD}")
        print(f"   提示词模式: {airfoil_optimizer.PROMPT_MODE}")
        print(f"   当前最优: {optimizer.best_fitness:.6f}\n")

        results = optimizer.optimize(
            generations=total_gens,
            resume=True,
            start_gen=last_gen,      # 0-indexed：已完成 last_gen 代，从它开始
            operating_conditions=config or None
        )
    else:
        # ---- 全新运行 ----
        airfoil_optimizer.OPT_OBJ = args.obj
        airfoil_optimizer.PROMPT_MODE = args.prompt_mode
        airfoil_optimizer.METHOD = args.method

        optimizer = AirfoilOptimizer(
            population_size=args.population,
            sigma=args.sigma
        )
        # --config 指定的配置文件作为工况基础；--test 时叠加测试模式
        operating_conditions = config or None
        if test_conditions:
            if operating_conditions is None:
                operating_conditions = test_conditions
            else:
                tm = dict(test_conditions['test_mode'])
                # 测试工况名在 --config 中不存在时，回退到第一个工况
                avail = list(operating_conditions.get('conditions', {}).keys())
                if avail and tm.get('conditions') and tm['conditions'][0] not in avail:
                    tm['conditions'] = [avail[0]]
                operating_conditions['test_mode'] = tm

        results = optimizer.optimize(generations=args.generations,
                                     operating_conditions=operating_conditions)

    # 输出最终结果
    analyze_pso_results(RESULT_FILE)

    print("\n=== 最终优化结果 ===")
    best_params = results['best_individual']
    print(f"最优CST参数：")
    print(f"  A_u: {[f'{x:.4f}' for x in best_params['A_u']]}")
    print(f"  A_l: {[f'{x:.4f}' for x in best_params['A_l']]}")

    print(f"\n输出文件：")
    print(f"  - pso_generations.jsonl: 世代详细记录")
    print(f"  - pso_final_result.json: 最终优化结果（含续跑状态）")
    print(f"  - convergence.png: 收敛曲线图")


if __name__ == "__main__":
    main()