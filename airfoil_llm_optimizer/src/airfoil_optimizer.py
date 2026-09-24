from ga_updater import ga_update
from pso_updater import pso_update
from cmaes_updater import cmaes_update
from population_manager import GaussianSampler
from prompts import get_llm_mean_prompt
from aerodynamic_prediction import AerodynamicPredictor
from llm_model import LLMModel
from visualization import analyze_pso_results
import numpy as np
import json
import os
import time
from copy import deepcopy

# 优化目标类型：
# "Ld" - 平均升阻比 | "cl" - 纯平均升力系数 | "hybrid" - 归一化平均升力系数/归一化滞回环面积
OPT_OBJ = "hybrid"

# 优化方法（非 LLM 路径接入的数值优化器，均通过单次调用接口输出下一代表面均值）：
# "LLM"    - LLM 引导更新均值（默认）
# "GA"     - 遗传算法（ga_updater.py）
# "PSO"    - 粒子群优化（pso_updater.py，Kennedy & Eberhart 1995）
# "CMA-ES" - 协方差矩阵自适应进化策略（cmaes_updater.py，Hansen & Ostermeier 2001）
METHOD = "LLM"

# 提示词结构模式：
# "few_shot" - 代际精英保留模式（默认）：按世代分组，筛选Top-T世代，每个世代保留Top-M精英个体
# "standard" - 标准扁平模式：所有历史设计扁平化排序，无代际区分，最多保留100条
PROMPT_MODE = "few_shot"

# 混合目标权重（可根据需求调整）
W1 = 0.5  # 平均升力系数权重
W2 = 0.5  # 滞回环面积权重


def round_float_values(val):
    """递归处理值：浮点数保留6位小数，列表递归处理，其他类型原样返回"""
    if isinstance(val, (list, tuple)):
        # 处理列表/元组中的每个元素
        return [round_float_values(item) for item in val]
    elif isinstance(val, float):
        # 浮点数保留6位小数
        return round(val, 6)
    else:
        # 非浮点数（如整数index）原样返回
        return val


class AirfoilOptimizer:
    def __init__(self, population_size=8, sigma=0.1):
        self.sampler = GaussianSampler(population_size=population_size, sigma=sigma)
        self.predictor = AerodynamicPredictor()
        if METHOD == "LLM":
            self.llm = LLMModel()
        self.best_global = None
        self.best_fitness = -float('inf')

        # 初始化输出文件
        self._init_output_files()
    
    def _init_output_files(self):
        """初始化输出文件路径（清空逻辑移到 optimize，续跑时保留历史追加）"""
        output_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(output_dir, 'pso_generations.jsonl')
    
    def evaluate_population(self, population, operating_conditions):
        """
        评估种群适应度（支持三种目标：平均升阻比、平均升力系数、混合目标）
        :param population: 种群列表（元素为19维数组/字典）
        :param operating_conditions: 气动计算工况
        :return: 适应度分数数组
        """
        fitness_scores = []
        individual_metrics = []   # 每个体指标（箱线图用）：obj / 平均Cl / 平均滞回面积

        for individual in population:
            # 统一转换为字典格式（适配气动预测函数）
            if isinstance(individual, (list, np.ndarray)):
                flat = np.asarray(individual).reshape(-1).tolist()
                ind_dict = {
                    'A_u': round_float_values(flat[:9]),       # 前9维：上表面CST系数（保留6位小数）
                    'A_l': round_float_values(flat[9:18]),     # 中间9维：下表面CST系数（保留6位小数）
                }
            elif isinstance(individual, dict):
                ind_dict = {k: round_float_values(v) for k, v in individual.items()}
            else:
                raise ValueError(f"个体格式不支持：{type(individual)}")
            
            # 气动性能预测（完整结果，包含aerodynamic_prediction计算的滞回环面积）
            try:
                results = self.predictor.predict_aerodynamic_performance(ind_dict)
            except Exception as e:
                # LLM 评估失败兜底：该个体降级为 0 适应度，不中断整个优化
                print(f"[WARN] 个体评估失败，适应度记为 0：{e}")
                fitness_scores.append(0.0)
                individual_metrics.append({'obj': 0.0, 'cycle_mean_cl': None,
                                           'hysteresis_mean_area': None})
                continue

            # 根据目标类型计算适应度（每工况独立指标，跨工况等权综合）
            if OPT_OBJ in ("Ld", "cl"):
                # 平均升力系数（各工况周期平均 Cl 的等权平均）。
                # Ld 严格需每工况升阻比，当前以周期平均 Cl 近似。
                fitness = results['summary']['cycle_mean_cl']
            elif OPT_OBJ == "hybrid":
                # 混合目标 A3：
                #   cl_norm    = clip((cl_i - cl_min)/(cl_max - cl_min), 0, 1)
                #   hyst_norm  = clip((hyst_i - hyst_min)/(hyst_max - hyst_min), 0, 1)
                #   obj_i      = cl_norm / (hyst_norm + c)   （c 为基线滞回容差，防除零）
                #   fitness    = mean(obj_i)   （各工况等权平均）
                obj_cfg = self.predictor.config.get('objective', {})
                cl_min = obj_cfg.get('cl_min', 0.4701)
                cl_max = obj_cfg.get('cl_max', 1.4088)
                hyst_min = obj_cfg.get('hyst_min', 0.0353)
                hyst_max = obj_cfg.get('hyst_max', 0.7705)
                c = obj_cfg.get('hyst_floor', 0.05)
                per_cond = results['summary'].get('per_condition', {})
                scores = []
                for pc in per_cond.values():
                    cl_i = pc['cycle_mean_cl']
                    hyst_i = pc.get('hysteresis_mean_area', 0.0)
                    cl_norm = np.clip((cl_i - cl_min) / (cl_max - cl_min + 1e-9), 0.0, 1.0)
                    hyst_norm = np.clip((hyst_i - hyst_min) / (hyst_max - hyst_min + 1e-9), 0.0, 1.0)
                    scores.append(cl_norm / (hyst_norm + c))
                fitness = float(np.mean(scores)) if scores else 0.0
            else:
                raise ValueError(f"不支持的优化目标类型：{OPT_OBJ}（可选：Ld/cl/hybrid）")

            fitness_scores.append(fitness)

            # 收集个体指标（供箱线图：平均Cl 与 原始滞回面积——尺度缩放前）
            per_cond = results['summary'].get('per_condition', {})
            hyst_vals = [pc.get('hysteresis_mean_area')
                         for pc in per_cond.values()
                         if pc.get('hysteresis_mean_area') is not None]
            individual_metrics.append({
                'obj': fitness,
                'cycle_mean_cl': results['summary'].get('cycle_mean_cl', 0.0),
                'hysteresis_mean_area': float(np.mean(hyst_vals)) if hyst_vals else None,
            })

            # 更新全局最优（深拷贝避免引用问题）
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_global = deepcopy(ind_dict)
        
        return np.array(fitness_scores), individual_metrics
    
    def get_llm_new_mean(self, generation_history):
        """
        获取LLM输出的新一代均值（移除冗余逻辑，增强鲁棒性）
        :param generation_history: 世代历史记录
        :return: 解析后的新均值字典
        """
        # 生成符合论文范式的提示词（仅传必要参数）
        prompt = get_llm_mean_prompt(
            generation_history,
            self.sampler.bounds,
            mode=PROMPT_MODE
        )
        
        # 获取LLM响应（确定性输出，temperature=0）
        # 优化引导开启 thinking，输出预算 8192 token（需服务端 max_model_len ≥ 16384）
        response = self.llm.generate_response(prompt, max_new_tokens=8192, temperature=0.0)
        # 确保响应为纯文本格式
        if isinstance(response, dict):
            response = response.get('response', '')
        response = str(response).strip()
        
        # 解析并截断边界的均值参数
        new_mean = self._parse_llm_mean(response)
        
        # 保存交互记录
        self._save_generation_prompt(
            gen_num=len(generation_history)+1,
            prompt=prompt,
            response=response,
            new_mean=new_mean
        )
        
        return new_mean
    
    def _parse_llm_mean(self, response):
        """
        解析LLM输出的均值参数（补充边界截断，符合提示词约束）
        :param response: LLM输出文本
        :return: 截断后的均值字典
        """
        import re

        # 剥离 thinking 内容（<think>...</think>），只解析最终输出，避免思考草稿中的 x_new/y_new 干扰
        response = re.sub(r'<think>.*?</think>', '', response or '', flags=re.DOTALL)

        # 当前均值兜底（缺维补齐用）
        cm_u = self.sampler.current_mean.get('A_u') or [0.0] * 9
        cm_l = self.sampler.current_mean.get('A_l') or [0.0] * 9

        # 提取A_u均值并截断边界（LLM 偶发输出不足 9 个时用当前均值补齐）
        au_match = re.search(r'x_new\s*[:,：]\s*\[(.*?)\]', response, re.DOTALL)
        if au_match:
            A_u = [float(x.strip()) for x in au_match.group(1).split(',')]
            # 截断到参数范围 + 确保9维
            au_min, au_max = self.sampler.bounds['A_u']
            A_u = [max(au_min, min(au_max, x)) for x in A_u[:9]]
            if len(A_u) < 9:
                A_u = A_u + cm_u[len(A_u):9]
        else:
            # 解析失败时使用当前均值兜底
            A_u = cm_u[:9]

        # 提取A_l均值并截断边界
        al_match = re.search(r'y_new\s*[:,：]\s*\[(.*?)\]', response, re.DOTALL)
        if al_match:
            A_l = [float(x.strip()) for x in al_match.group(1).split(',')]
            # 截断到参数范围 + 确保9维
            al_min, al_max = self.sampler.bounds['A_l']
            A_l = [max(al_min, min(al_max, x)) for x in A_l[:9]]
            if len(A_l) < 9:
                A_l = A_l + cm_l[len(A_l):9]
        else:
            A_l = cm_l[:9]

        return {'A_u': A_u, 'A_l': A_l}
    
    def _save_generation_prompt(self, gen_num, prompt, response, new_mean):
        """
        保存世代交互记录（标准化时间戳，增强可读性）
        :param gen_num: 世代编号
        :param prompt: 输入提示词
        :param response: LLM响应
        :param new_mean: 解析后的新均值
        """
        record = {
            'generation': gen_num,
            'prompt': prompt,
            'response': response,
            'new_mean': new_mean,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        with open(self.output_path, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    def _build_result_dict(self, generations):
        """
        构建结果字典（临时续跑状态与最终结果共用，避免重复定义）。
        """
        return {
            'best_individual': self.best_global,
            'best_fitness': self.best_fitness,
            'current_mean': self.sampler.current_mean,
            'optimization_objective': OPT_OBJ,
            'method': METHOD,
            'prompt_mode': PROMPT_MODE,
            'hybrid_weights': {"w1": W1, "w2": W2} if OPT_OBJ == "hybrid" else None,
            'generation_history': self.sampler.generation_history,
            'total_generations': generations,
            'population_size': self.sampler.population_size,
            'parameter_bounds': self.sampler.bounds,
            'hyst_comparable': getattr(self, 'hyst_comparable', False),
        }

    def _numerical_update(self, input_params):
        """按 METHOD 分发到数值优化器（GA / PSO / CMA-ES），返回下一代均值。"""
        if METHOD == "GA":
            return ga_update(input_params)
        elif METHOD == "PSO":
            return pso_update(input_params)
        elif METHOD == "CMA-ES":
            return cmaes_update(input_params)
        raise ValueError(f"不支持的优化方法：{METHOD}（可选：LLM/GA/PSO/CMA-ES）")

    def optimize(self, generations=10, operating_conditions=None,
                 resume=False, start_gen=0):
        """
        执行翼型优化（支持断点续跑）
        :param generations: 总优化世代数
        :param operating_conditions: 气动计算工况
        :param resume: 是否从 pso_final_result.json 续跑
        :param start_gen: 续跑起始世代（0-indexed，即已完成的1-indexed世代数）
        """
        # 设置气动计算工况
        if operating_conditions:
            self.predictor.set_operating_conditions(operating_conditions)

        # 工况 (平均攻角, 振幅) 是否唯一 → 决定箱线图是否绘制滞回面积子图
        conds = self.predictor.config.get('conditions', {})
        combos = {(c.get('mean_aoa'), c.get('amplitude')) for c in conds.values()}
        self.hyst_comparable = len(combos) == 1

        # 初始化PSO第一代均值（续跑时跳过）
        if not resume:
            self.sampler.initialize()
            # 全新运行才清空世代记录（续跑保留旧记录，追加新代）
            with open(self.output_path, 'w', encoding='utf-8') as f:
                pass

        # 打印优化配置信息
        opt_obj_desc = {
            "Ld": "平均升阻比",
            "cl": "纯平均升力系数",
            "hybrid": "混合目标（归一化Cl / 归一化滞回环面积）"
        }
        prompt_mode_desc = {
            "few_shot": "Few-Shot（代际精英保留）",
            "standard": "Standard（扁平化历史设计）"
        }
        if resume:
            print(f"=== 从断点续跑 (已完成 {start_gen}/{generations} 世代) ===")
        else:
            print(f"=== 翼型优化开始 (种群规模={self.sampler.population_size}) ===")
        print(f"优化目标：{opt_obj_desc[OPT_OBJ]}")
        print(f"优化方法：{METHOD}")
        print(f"提示词模式：{prompt_mode_desc.get(PROMPT_MODE, PROMPT_MODE)}")
        print(f"参数范围：A_u={self.sampler.bounds['A_u']}, A_l={self.sampler.bounds['A_l']}")
        if resume:
            print(f"当前全局最优：{self.best_fitness:.6f}")
        print()

        for gen in range(start_gen, generations):
            print(f"=== 世代 {gen+1}/{generations} ===")

            # 1. 采样生成种群
            population = self.sampler.sample_population()
            print(f"✅ 生成种群：{len(population)}个个体")

            # 2. 评估种群适应度
            fitness_scores, individual_metrics = self.evaluate_population(
                population, operating_conditions)
            gen_best_fitness = np.max(fitness_scores)
            print(f"📊 世代最优：{gen_best_fitness:.6f}")
            print(f"🌐 全局最优：{self.best_fitness:.6f}")

            # 3. 记录世代信息
            gen_record = self.sampler.record_generation(gen+1, population, fitness_scores)
            gen_record['individual_metrics'] = individual_metrics
            gen_record['hyst_comparable'] = self.hyst_comparable
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            top_3_individuals = []
            for ind, _ in sorted_pop[:3]:
                if isinstance(ind, dict):
                    flat_ind = ind['A_u'] + ind['A_l']
                elif isinstance(ind, (list, np.ndarray)):
                    flat_ind = list(ind)[:18]
                else:
                    flat_ind = []
                top_3_individuals.append(flat_ind)
            gen_record['top_individuals'] = top_3_individuals
            gen_record['best_fitness'] = gen_best_fitness

            # 4. 保存世代结果（数值优化方法才落盘世代记录）
            if METHOD != "LLM":
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    json.dump({'type': 'generation_result', **gen_record}, f, ensure_ascii=False)
                    f.write('\n')

            # 5. 更新均值（最后一代无需更新）
            if gen < generations - 1:
                try:
                    if METHOD == "LLM":
                        new_mean = self.get_llm_new_mean(self.sampler.generation_history)
                    else:
                        input_params = [population, fitness_scores, self.sampler.bounds, self.best_fitness, self.best_global]
                        new_mean = self._numerical_update(input_params)
                    self.sampler.update_mean(new_mean)
                    print(f"🔄 更新均值\n")
                except Exception as e:
                    print(f"⚠️  均值更新失败，使用当前均值：{e}\n")

                # 均值更新后保存临时结果（含 current_mean，用于续跑）
                temp_result = self._build_result_dict(generations)
                temp_path = os.path.join(os.path.dirname(__file__), 'pso_final_result.json')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_result, f, ensure_ascii=False, indent=2)

                analyze_pso_results()

        # 整理最终结果
        final_result = self._build_result_dict(generations)
        
        # 保存最终结果
        final_path = os.path.join(os.path.dirname(__file__), 'pso_final_result.json')
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        analyze_pso_results()
        
        # 打印最终结果
        print(f"\n=== 优化完成 ===")
        print(f"🏆 全局最优适应度：{self.best_fitness:.6f}")
        print(f"🎯 优化目标：{opt_obj_desc[OPT_OBJ]}")
        if OPT_OBJ == "hybrid":
            print(f"⚖️  混合目标权重：w1={W1}, w2={W2}")
        print(f"📝 最优参数：{json.dumps(self.best_global, ensure_ascii=False, indent=2)}")
        print(f"📂 结果文件：{final_path}")
        
        return final_result