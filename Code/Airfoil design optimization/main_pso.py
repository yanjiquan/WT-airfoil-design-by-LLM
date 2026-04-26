from airfoil_pso_optimizer import AirfoilPSOOptimizer
from utils_pso import analyze_pso_results
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='LLM-PSO翼型优化工具（论文方法）')
    parser.add_argument('--generations', type=int, default=100, help='进化世代数')
    parser.add_argument('--population', type=int, default=50, help='种群规模')
    parser.add_argument('--sigma', type=float, default=0.02, help='高斯采样方差')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--analyze', action='store_true', help='仅分析结果文件')
    
    args = parser.parse_args()
    
    if args.analyze:
        # 仅分析结果
        analyze_pso_results()
        return
    
    # 加载配置
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 创建PSO优化器
    optimizer = AirfoilPSOOptimizer(
        population_size=args.population,
        sigma=args.sigma
    )
    
    # 执行优化
    results = optimizer.optimize(
        generations=args.generations,
    )
    
    # 生成分析报告和可视化
    analyze_pso_results('pso_final_result.json')
    
    # 输出最终结果
    print("\n=== 最终优化结果 ===")
    print(f"最优CST参数：")
    best_params = results['best_individual']
    print(f"  A_u: {[f'{x:.4f}' for x in best_params['A_u']]}")
    print(f"  A_l: {[f'{x:.4f}' for x in best_params['A_l']]}")
    
    print(f"\n输出文件：")
    print(f"  - pso_generations.jsonl: 世代详细记录")
    print(f"  - pso_final_result.json: 最终优化结果")
    print(f"  - pso_convergence.png: 收敛曲线图")

if __name__ == "__main__":
    main()