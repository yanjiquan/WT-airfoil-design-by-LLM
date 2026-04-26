import numpy as np


def get_population_mean(population):
    """计算种群的平均位置（A_u、A_l、z_TE的均值）"""
    au_list = [ind['A_u'] for ind in population]
    al_list = [ind['A_l'] for ind in population]
    
    mean_Au = np.mean(au_list, axis=0)
    mean_Al = np.mean(al_list, axis=0)
    return mean_Au, mean_Al

def pso_update(input_params):
    """
    遗传算法(GA)核心更新函数（原PSO函数名/输入/输出保持不变），生成下一代个体
    
    参数:
        input_params (list): 输入参数列表，格式为：
            [population，fitness_scores，bounds，gbest_fitness，gbest]
            - population: 种群列表，每个元素为individual格式字典
            - fitness_scores: 适应度分数列表（与种群大小一致）
            - bounds: 变量边界字典
            - gbest_fitness: 全局最优适应度（浮点数）
            - gbest: 全局最优个体（individual格式字典）
    
    返回:
        new_generation (dict): 下一代个体，格式与individual一致
    """
    # 解包输入参数（保持与原PSO一致）
    population, fitness_scores, bounds, gbest_fitness, gbest = input_params
    
    # -------------------------- 步骤1：GA参数配置（可根据需求调整）--------------------------
    cross_prob = 0.8       # 交叉概率
    mutate_prob = 0.1      # 变异概率
    elite_rate = 0.2       # 精英保留比例（确保全局最优个体参与繁殖）
    
    # -------------------------- 步骤2：适应度预处理（轮盘赌选择需非负适应度）--------------------------
    fitness_arr = np.array(fitness_scores)
    fitness_min = np.min(fitness_arr)
    # 处理负适应度：将适应度平移至非负区间
    fitness_non_neg = fitness_arr - fitness_min + 1e-6  # 加小值避免为0
    fitness_sum = np.sum(fitness_non_neg)
    select_prob = fitness_non_neg / fitness_sum  # 轮盘赌选择概率
    
    # -------------------------- 步骤3：选择操作（精英保留 + 轮盘赌选择）--------------------------
    # 1. 精英保留：选取前N个最优个体
    elite_num = max(1, int(len(population) * elite_rate))
    elite_idx = np.argsort(fitness_arr)[-elite_num:]  # 适应度降序索引
    elite_individuals = [population[i] for i in elite_idx]
    # 确保全局最优个体（gbest）加入精英池
    if not any(np.allclose(ind['A_u'], gbest['A_u']) and np.allclose(ind['A_l'], gbest['A_l']) 
               for ind in elite_individuals):
        elite_individuals.append(gbest)
    
    # 2. 轮盘赌选择：选2个父代（至少包含1个精英个体）
    def select_parent():
        idx = np.random.choice(len(population), p=select_prob)
        return population[idx]
    
    parent1 = elite_individuals[np.random.randint(len(elite_individuals))]  # 精英父代1
    parent2 = select_parent()  # 轮盘赌选父代2
    
    # 提取父代的A_u和A_l（转换为numpy数组）
    p1_Au = np.array(parent1['A_u'])
    p1_Al = np.array(parent1['A_l'])
    p2_Au = np.array(parent2['A_u'])
    p2_Al = np.array(parent2['A_l'])
    
    # -------------------------- 步骤4：交叉操作（算术交叉）--------------------------
    np.random.seed(None)
    if np.random.rand() < cross_prob:
        # 生成交叉权重（0-1之间随机数）
        cross_weight = np.random.rand(len(p1_Au))
        # 对A_u和A_l分别进行算术交叉
        child_Au = cross_weight * p1_Au + (1 - cross_weight) * p2_Au
        child_Al = cross_weight * p1_Al + (1 - cross_weight) * p2_Al
    else:
        # 不交叉，直接继承父代1（精英）
        child_Au = p1_Au.copy()
        child_Al = p1_Al.copy()
    
    # -------------------------- 步骤5：变异操作（高斯变异）--------------------------
    # 提取变量边界
    au_min, au_max = bounds['A_u']
    al_min, al_max = bounds['A_l']
    # 变异步长（基于变量边界范围）
    au_mutate_step = (au_max - au_min) * 0.1
    al_mutate_step = (al_max - al_min) * 0.1
    
    # 对A_u进行变异
    mutate_mask_au = np.random.rand(len(child_Au)) < mutate_prob  # 变异维度掩码
    child_Au[mutate_mask_au] += np.random.normal(0, au_mutate_step, size=np.sum(mutate_mask_au))
    
    # 对A_l进行变异
    mutate_mask_al = np.random.rand(len(child_Al)) < mutate_prob
    child_Al[mutate_mask_al] += np.random.normal(0, al_mutate_step, size=np.sum(mutate_mask_al))
    
    # -------------------------- 步骤6：边界裁剪（确保变量满足约束）--------------------------
    child_Au = np.clip(child_Au, au_min, au_max)
    child_Al = np.clip(child_Al, al_min, al_max)
    
    # -------------------------- 步骤7：构建输出格式（与原PSO输出一致）--------------------------
    new_generation = {
        'A_u': child_Au.tolist(),
        'A_l': child_Al.tolist(),
    }
    return new_generation