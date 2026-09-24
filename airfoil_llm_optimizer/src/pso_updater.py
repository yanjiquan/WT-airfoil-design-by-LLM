"""
PSO (Particle Swarm Optimization) updater — 经典粒子群优化均值更新器。

单次调用接口，与 ga_updater.py 的 ga_update() 保持一致：
    输入 [population, fitness_scores, bounds, gbest_fitness, gbest]
    输出 下一代均值 {'A_u': [...], 'A_l': [...]}，作为下一代表面采样中心。

每个粒子由外部传入的 population 位置进行同步（与 GaussianSampler 的逐代重采样配合），
粒子速度 v 与个体历史最优 pbest 以模块级单例跨代保存。

参数边界 bounds 由调用方传入，本模块内部不做硬编码。

References:
    [1] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
        Proceedings of IEEE ICNN'95 - International Conference on Neural
        Networks, 4, 1942-1948. doi:10.1109/ICNN.1995.488968
    [2] Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
        IEEE International Conference on Evolutionary Computation, 69-73.
        doi:10.1109/ICEC.1998.699146
"""

import numpy as np

# PSO 超参数（Shi & Eberhart 1998：惯性权重线性递减）
W_MAX = 0.9          # 惯性权重上界
W_MIN = 0.4          # 惯性权重下界
C1 = 2.0             # 认知学习因子（个体最优吸引）
C2 = 2.0             # 社会学习因子（全局最优吸引）
MAX_GENERATIONS = 100  # 惯性权重递减跨度（可按实际世代数调整）

# 跨代状态（按粒子索引保存；首次调用时初始化）
_state = {
    'v': None,             # (N, dim) 速度矩阵
    'pbest': None,         # (N, dim) 个体历史最优位置
    'pbest_fitness': None, # (N,)     个体历史最优适应度
    'N': 0,
    'dim': 18,
    'generation': 0,
}


def pso_update(input_params, num_particles=None):
    """
    PSO 单代更新 —— 输入种群与适应度，输出下一代粒子群均值。

    Args:
        input_params:
            [0] population     — list of dicts（每个个体含 'A_u' / 'A_l'）
            [1] fitness_scores — 1-D array（越高越好）
            [2] bounds         — dict {'A_u': (min,max), 'A_l': (min,max)}
            [3] gbest_fitness  — float（全局最优适应度，仅用于记录）
            [4] gbest          — dict（全局最优个体，用于社会吸引）
        num_particles (int, optional): 粒子数，默认取种群规模。

    Returns:
        dict {'A_u': [...], 'A_l': [...]} — 下一代粒子群均值
    """
    global _state

    population, fitness_scores, bounds, _gbest_fitness, gbest = input_params
    N = len(population)
    if num_particles is None:
        num_particles = N

    dim = 18
    X = _flatten(population, N, dim)

    # 边界（由调用方传入，不在此硬编码）
    au_min, au_max = bounds['A_u']
    al_min, al_max = bounds['A_l']
    low = np.array([au_min] * 9 + [al_min] * 9)
    high = np.array([au_max] * 9 + [al_max] * 9)

    # 首次调用（或种群规模变化）时初始化粒子状态
    if _state['v'] is None or _state['N'] != N or _state['dim'] != dim:
        _state['v'] = np.zeros((N, dim))
        _state['pbest'] = X.copy()
        _state['pbest_fitness'] = np.asarray(fitness_scores, dtype=float).copy()
        _state['N'] = N
        _state['dim'] = dim
        _state['generation'] = 0

    # ---- 个体历史最优 pbest 更新（Eq. 见 [1]） ----
    fitness = np.asarray(fitness_scores, dtype=float)
    improved = fitness > _state['pbest_fitness']
    _state['pbest'][improved] = X[improved]
    _state['pbest_fitness'][improved] = fitness[improved]

    # ---- 全局最优 gbest（优先取调用方传入，兜底取种群内最优 pbest） ----
    if gbest is not None:
        gbest_vec = np.array(gbest['A_u'] + gbest['A_l'])
    else:
        gbest_vec = _state['pbest'][np.argmax(_state['pbest_fitness'])]

    # ---- 惯性权重线性递减（[2]） ----
    gen_frac = min(_state['generation'] / MAX_GENERATIONS, 1.0)
    w = W_MAX - (W_MAX - W_MIN) * gen_frac

    # ---- 速度与位置更新（[1]）：v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x) ----
    r1 = np.random.rand(N, dim)
    r2 = np.random.rand(N, dim)
    _state['v'] = (w * _state['v']
                   + C1 * r1 * (_state['pbest'] - X)
                   + C2 * r2 * (gbest_vec - X))
    X_new = np.clip(X + _state['v'], low, high)

    _state['generation'] += 1

    # ---- 返回新粒子群均值作为下一代表面采样中心 ----
    return {
        'A_u': np.mean(X_new[:, :9], axis=0).tolist(),
        'A_l': np.mean(X_new[:, 9:], axis=0).tolist(),
    }


def _flatten(population, N, dim):
    """把种群个体列表展平成 (N, dim) 数组。"""
    X = np.zeros((N, dim))
    for i, ind in enumerate(population):
        X[i, :9] = ind['A_u']
        X[i, 9:] = ind['A_l']
    return X
