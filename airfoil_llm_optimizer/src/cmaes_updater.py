"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) updater.

单次调用接口，与 ga_updater.py 的 ga_update() 保持一致：
    输入 [population, fitness_scores, bounds, gbest_fitness, gbest]
    输出 下一代均值 {'A_u': [...], 'A_l': [...]}，作为下一代表面采样中心。

每代使用外部评估的 λ 个样本点（population）执行 CMA-ES 单代更新：
    1. 精英选择（取适应度最高 μ 个）
    2. 加权重组更新均值 m（Eq. 1）
    3. 步长路径 p_sigma 累积 + 累计步长自适应（CSA）更新 σ（Eq. 2-4）
    4. 协方差路径 p_c 累积（rank-one 更新）
    5. 协方差矩阵 C 的 rank-1 + rank-μ 自适应更新（Eq. 5-6）

均值 m、步长 σ、协方差 C 与路径向量等跨代状态以模块级单例保存。

参数边界 bounds 由调用方传入，本模块内部不做硬编码。

References:
    [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized
        self-adaptation in evolution strategies. Evolutionary Computation,
        9(2), 159-195. doi:10.1162/106365601750190398
    [2] Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.
        arXiv:1604.00772.
"""

import numpy as np

# 初始步长 = 参数边界跨度的该比例（取值范围约 1/6 ~ 1/3）
SIGMA_INIT_FRAC = 1 / 3


class _CMAESState:
    """CMA-ES 跨代内部状态（均值、步长、协方差、路径向量）。"""

    def __init__(self, dim, low, high, lam):
        self.dim = dim
        self.low = low
        self.high = high

        # 种群规模 λ 与精英数 μ（Hansen & Ostermeier 2001 默认取 λ/2）
        self.lam = lam
        self.mu = max(1, lam // 2)
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / np.sum(w)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        n = dim
        mueff = self.mueff
        # 策略参数（默认值见 [1][2]）
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1,
                       2.0 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        self.csigma = (mueff + 2) / (n + mueff + 5)
        self.dsigma = (1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1)
                       + self.csigma)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        # 初始分布参数
        self.m = (low + high) / 2.0
        span = np.maximum(high - low, 1e-8)
        self.sigma = float(np.mean(span) * SIGMA_INIT_FRAC)
        self.C = np.eye(n)
        self.p_sigma = np.zeros(n)   # 步长进化路径
        self.p_c = np.zeros(n)       # 协方差进化路径
        self.B = np.eye(n)           # 特征向量（特征分解缓存）
        self.D = np.ones(n)          # 特征值平方根（特征分解缓存）
        self.generation = 0

    def _eigendecompose(self):
        """对协方差矩阵做特征分解（带数值保护），供路径归一化使用。"""
        C = 0.5 * (self.C + self.C.T)
        try:
            D2, B = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            C = C + 1e-12 * np.eye(self.dim)
            D2, B = np.linalg.eigh(C)
        D2 = np.clip(D2, 1e-30, None)
        self.B = B
        self.D = np.sqrt(D2)


_state = None  # _CMAESState 单例


def cmaes_update(input_params):
    """
    CMA-ES 单代更新 —— 输入种群与适应度，输出更新后的分布均值。

    Args:
        input_params:
            [0] population     — list of dicts（λ 个评估样本点）
            [1] fitness_scores — 1-D array（越高越好）
            [2] bounds         — dict {'A_u': (min,max), 'A_l': (min,max)}
            [3] gbest_fitness  — float（未使用，保留接口一致）
            [4] gbest          — dict（未使用，保留接口一致）

    Returns:
        dict {'A_u': [...], 'A_l': [...]} — 更新后的分布均值 m
    """
    global _state

    population, fitness_scores, bounds, _gbest_fitness, _gbest = input_params
    N = len(population)
    dim = 18

    au_min, au_max = bounds['A_u']
    al_min, al_max = bounds['A_l']
    low = np.array([au_min] * 9 + [al_min] * 9)
    high = np.array([au_max] * 9 + [al_max] * 9)

    X = _flatten(population, N, dim)
    fitness = np.asarray(fitness_scores, dtype=float)

    # 首次调用（或种群规模变化）时初始化状态
    if _state is None or _state.lam != N:
        _state = _CMAESState(dim, low, high, N)
        _state.m = np.mean(X, axis=0)  # 以当前种群均值作为初始均值

    state = _state
    n = state.dim
    mu = state.mu

    # ---- 1. 精英选择（最大化：取适应度最高 μ 个样本） ----
    elite_idx = np.argsort(-fitness)[:mu]
    X_elite = X[elite_idx]
    m_old = state.m

    # ---- 2. 加权重组更新均值 m（[1] Eq. 1） ----
    state.m = np.sum(state.weights[:, None] * X_elite, axis=0)
    y = (state.m - m_old) / state.sigma

    # ---- 3. 步长路径累积 + 步长自适应 CSA（[1] Eq. 2-4） ----
    # z = C^{-1/2} y（利用 C = B D^2 B^T 分解，B 正交）
    z = state.B @ ((state.B.T @ y) / state.D)
    state.p_sigma = ((1 - state.csigma) * state.p_sigma
                     + np.sqrt(state.csigma * (2 - state.csigma) * state.mueff) * z)
    state.sigma = state.sigma * np.exp(
        (state.csigma / state.dsigma)
        * (np.linalg.norm(state.p_sigma) / state.chiN - 1))

    # ---- 4. 协方差路径累积（rank-one 更新，含 h_sigma 启发式） ----
    norm_psig = np.linalg.norm(state.p_sigma)
    denom = np.sqrt(1 - (1 - state.csigma) ** (2 * (state.generation + 1)))
    h_sigma = 1.0 if (norm_psig / denom < 1.4 + 2 / (n + 1)) else 0.0
    state.p_c = ((1 - state.cc) * state.p_c
                 + h_sigma * np.sqrt(state.cc * (2 - state.cc) * state.mueff) * y)

    # ---- 5. 协方差矩阵更新（rank-one + rank-μ，[1] Eq. 5-6） ----
    delta_h = (1 - h_sigma) * state.cc * (2 - state.cc)
    rank_one = state.p_c[:, None] @ state.p_c[None, :]
    y_elite = (X_elite - m_old) / state.sigma
    rank_mu = (state.weights[:, None] * y_elite).T @ y_elite
    state.C = ((1 - state.c1 - state.cmu) * state.C
               + state.c1 * (rank_one + delta_h * np.eye(n))
               + state.cmu * rank_mu)

    # ---- 6. 数值保护：对称化 + 特征分解（供下代路径归一化使用） ----
    state.C = 0.5 * (state.C + state.C.T)
    state._eigendecompose()

    # ---- 7. 均值裁剪到参数边界 ----
    state.m = np.clip(state.m, low, high)
    state.generation += 1

    return {'A_u': state.m[:9].tolist(), 'A_l': state.m[9:].tolist()}


def _flatten(population, N, dim):
    """把种群个体列表展平成 (N, dim) 数组。"""
    X = np.zeros((N, dim))
    for i, ind in enumerate(population):
        X[i, :9] = ind['A_u']
        X[i, 9:] = ind['A_l']
    return X
