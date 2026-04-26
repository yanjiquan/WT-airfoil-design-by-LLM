import numpy as np
from copy import deepcopy

class LLMPSO:
    def __init__(self, param_dim=19, population_size=8, sigma=0.05):
        """
        初始化LLM-PSO优化器（论文方法）
        param_dim: 参数维度
        population_size: 种群规模
        sigma: 高斯采样方差
        """
        self.param_dim = param_dim
        self.population_size = population_size
        self.sigma = sigma
        
        # 参数边界约束
        self.bounds = {
            'A_u': (0.0, 0.7),
            'A_l': (-0.7, 0.0),
        }
        
        # 世代记录
        self.generation_history = []
        self.current_mean = None  # 当前世代均值（由LLM输出）
        
    def initialize(self):
        """初始化第一代参数分布均值"""
        # 生成初始均值（符合CST参数约束）
        self.current_mean = {
            # 'A_u': np.random.uniform(0.0, 0.6, 9).tolist(),
            # 'A_l': np.random.uniform(-0.6, 0.0, 9).tolist(),
            'A_u': [0.250922, 0.304949, 0.255768, 0.359807, 0.245305, 0.312583, 0.323615, 0.274348, 0.372711],
            'A_l': [-0.167089, -0.125904, -0.119999, -0.016318, -0.171019, 0.027394, -0.133103, 0.010577, -0.104305],
        }
        return self.current_mean
    
    def sample_population(self):
        """基于当前均值进行高斯采样生成种群"""
        population = []
        
        # 采样A_u参数
        A_u_samples = np.random.normal(
            loc=self.current_mean['A_u'],
            scale=self.sigma,
            size=(self.population_size, 9)
        )
        A_u_samples = np.clip(A_u_samples, self.bounds['A_u'][0], self.bounds['A_u'][1])
        
        # 采样A_l参数
        A_l_samples = np.random.normal(
            loc=self.current_mean['A_l'],
            scale=self.sigma,
            size=(self.population_size, 9)
        )
        A_l_samples = np.clip(A_l_samples, self.bounds['A_l'][0], self.bounds['A_l'][1])
        
        # 构建种群
        for i in range(self.population_size):
            individual = {
                'A_u': A_u_samples[i].tolist(),
                'A_l': A_l_samples[i].tolist(),
                'index': i
            }
            population.append(individual)
        
        return population
    
    def update_mean_from_llm(self, llm_mean):
        """从LLM输出更新世代均值"""
        # 验证并修正LLM输出的均值
        self.current_mean = {
            'A_u': np.clip(llm_mean['A_u'], self.bounds['A_u'][0], self.bounds['A_u'][1]).tolist(),
            'A_l': np.clip(llm_mean['A_l'], self.bounds['A_l'][0], self.bounds['A_l'][1]).tolist(),
        }
        
        return self.current_mean
    
    # llm_pso.py中的record_generation方法需要修改
    def record_generation(self, generation_num, population, fitness_scores):
        """记录世代信息（精英保留）"""
        # 计算世代最优
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_individual = population[best_idx]

        # 记录世代数据
        generation_record = {
            'generation': generation_num,
            'mean': deepcopy(self.current_mean),
            'best_individual': deepcopy(best_individual),
            'best_fitness': best_fitness,
            'all_fitness': fitness_scores.tolist(),
            'population_size': self.population_size,
            'population': deepcopy(population)  # 添加种群记录
        }

        self.generation_history.append(generation_record)
        return generation_record