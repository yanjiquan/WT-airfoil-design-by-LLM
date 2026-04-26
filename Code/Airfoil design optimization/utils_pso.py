import matplotlib.pyplot as plt
import numpy as np
import json

def plot_pso_convergence(generation_history, save_path='pso_convergence.png'):
    """Plot PSO convergence curve"""
    generations = [record['generation'] for record in generation_history]
    best_fitness = [record['best_fitness'] for record in generation_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'b-', linewidth=2, markersize=8, label='Best Individual')
    
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Objective', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_population_distribution(generation_record, save_path='population_dist.png'):
    """Plot population parameter distribution"""
    # Extract the first dimension of A_u for visualization
    A_u_dim0 = [ind['A_u'][0] for ind in generation_record['population']]
    
    plt.figure(figsize=(8, 5))
    plt.hist(A_u_dim0, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(generation_record['mean']['A_u'][0], color='red', linestyle='--', linewidth=2, label='Distribution Mean')
    
    plt.xlabel('A_u[0] Parameter Value', fontsize=12)
    plt.ylabel('Number of Individuals', fontsize=12)
    plt.title(f'Population Parameter Distribution (Generation {generation_record["generation"]})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_evolution(generation_history, param_idx=0, save_path='param_evolution.png'):
    """Plot parameter evolution trajectory"""
    generations = [record['generation'] for record in generation_history]
    mean_values = [record['mean']['A_u'][param_idx] for record in generation_history]
    best_values = [record['best_individual']['A_u'][param_idx] for record in generation_history]
    
    plt.figure(figsize=(10, 5))
    plt.plot(generations, mean_values, 'g-o', linewidth=2, label='Distribution Mean')
    plt.plot(generations, best_values, color='orange', marker='s', linewidth=2, label='Best Individual')
    
    plt.xlabel('Evolution Generation', fontsize=12)
    plt.ylabel(f'A_u[{param_idx}] Parameter Value', fontsize=12)
    plt.title('Parameter Evolution Trajectory', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_pso_results(result_file='pso_final_result.json'):
    """Analyze PSO optimization results"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 提取数据
    gen_history = results['generation_history']
    best_fitness = results['best_fitness']
    initial_fitness = gen_history[0]['best_fitness']
    
    # 计算提升
    improvement = best_fitness - initial_fitness
    improvement_rate = (improvement / initial_fitness) * 100

    # 推导收敛世代（找到最优适应度对应的世代）
    convergence_gen = None
    for record in gen_history:
        if abs(record['best_fitness'] - best_fitness) < 1e-6:  # 浮点精度容错
            convergence_gen = record['generation']
            break
    
    print("\n=== LLM-PSO Optimization Result Analysis ===")
    print(f"Initial Fitness: {initial_fitness:.2f}")
    print(f"Optimal Fitness: {best_fitness:.2f}")
    print(f"Absolute Improvement: {improvement:.2f}")
    print(f"Relative Improvement: {improvement_rate:.1f}%")
    print(f"Convergence Generation: {convergence_gen if convergence_gen else 'Unknown'}")  # 修改这行
    
    # 生成可视化
    plot_pso_convergence(gen_history)
    plot_population_distribution(gen_history[-1])
    plot_parameter_evolution(gen_history)
    
    print("\nVisualization charts have been generated:")
    print("  - pso_convergence.png: Convergence curve")
    print("  - population_dist.png: Population distribution")
    print("  - param_evolution.png: Parameter evolution trajectory")