from trad_pso import pso_update
from llm_pso import LLMPSO
from prompts import get_llm_mean_prompt
from aerodynamic_prediction import AerodynamicPredictor
from llm_model import LLMModel
from utils_pso import analyze_pso_results
import numpy as np
import json
import os
import time
from copy import deepcopy

# 优化目标类型：
# "Ld" - 平均升阻比 | "cl" - 平均升力系数 | "hybrid" - w1*平均升力系数 - w2*滞回环面积
OPT_OBJ = "hybrid"
LLM_ENABLE = True

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


class AirfoilPSOOptimizer:
    def __init__(self, population_size=8, sigma=0.1, w1=W1, w2=W2):
        self.pso = LLMPSO(population_size=population_size, sigma=sigma)
        self.predictor = AerodynamicPredictor()
        if LLM_ENABLE:
            self.llm = LLMModel()
        self.best_global = None
        self.best_fitness = -float('inf')
        
        # 混合目标权重
        self.w1 = w1
        self.w2 = w2
        
        # 初始化输出文件
        self._init_output_files()
    
    def _init_output_files(self):
        """初始化输出文件（确保目录可写）"""
        output_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(output_dir, 'pso_generations.jsonl')
        # 清空文件（若存在）
        with open(self.output_path, 'w', encoding='utf-8') as f:
            pass
    
    def evaluate_population(self, population, operating_conditions):
        """
        评估种群适应度（支持三种目标：平均升阻比、平均升力系数、混合目标）
        :param population: 种群列表（元素为19维数组/字典）
        :param operating_conditions: 气动计算工况
        :return: 适应度分数数组
        """
        fitness_scores = []
        
        for individual in population:
            # 统一转换为字典格式（适配气动预测函数）
            if isinstance(individual, (list, np.ndarray)):
                ind_dict = {
                    'A_u': [round(x, 6) for x in individual[:9]],       # 前9维：上表面CST系数（保留6位小数）
                    'A_l': [round(x, 6) for x in individual[9:18]],     # 中间9维：下表面CST系数（保留6位小数）
                }
            elif isinstance(individual, dict):
                ind_dict = {k: round_float_values(v) for k, v in individual.items()}
            else:
                raise ValueError(f"个体格式不支持：{type(individual)}")
            
            # 气动性能预测（完整结果，包含aerodynamic_prediction计算的滞回环面积）
            results = self.predictor.predict_aerodynamic_performance(ind_dict)
            
            # 根据目标类型计算适应度
            if OPT_OBJ == "Ld":
                # 原逻辑：平均升阻比
                fitness = results['summary']['cycle_mean']
            elif OPT_OBJ == "cl":
                # 原逻辑：平均升力系数
                fitness = results['summary']['cycle_mean_cl']
            elif OPT_OBJ == "hybrid":
                # 新目标：w1*平均升力系数 - w2*滞回环面积（直接使用aerodynamic_prediction的计算结果）
                mean_cl = results['summary']['cycle_mean_cl']
                print(mean_cl)
                hysteresis_area = results['hysteresis_mean_area']  # 调用aerodynamic_prediction的滞回环面积
                
                mean_cl_min, mean_cl_max = 0.8, 1.1
                norm_mean_cl = (mean_cl - mean_cl_min) / (mean_cl_max - mean_cl_min + 1e-8)  # 加小值避免除零
                
                if hysteresis_area < 2:
                    norm_hyst_area = 0
                else:
                    hyst_min, hyst_max = 2.5, 5
                    norm_hyst_area = (hysteresis_area - hyst_min) / (hyst_max - hyst_min + 1e-8)
                
                # 3. 归一化后计算适应度
                fitness =  norm_mean_cl / norm_hyst_area
            else:
                raise ValueError(f"不支持的优化目标类型：{OPT_OBJ}（可选：Ld/cl/hybrid）")
            
            fitness_scores.append(fitness)
            
            # 更新全局最优（深拷贝避免引用问题）
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_global = deepcopy(ind_dict)
        
        return np.array(fitness_scores)
    
    def get_llm_new_mean(self, generation_history):
        """
        获取LLM输出的新一代均值（移除冗余逻辑，增强鲁棒性）
        :param generation_history: 世代历史记录
        :return: 解析后的新均值字典
        """
        # 生成符合论文范式的提示词（仅传必要参数）
        prompt = get_llm_mean_prompt(
            generation_history, 
            self.pso.bounds
        )
        
        # 获取LLM响应（确定性输出，temperature=0）
        response = self.llm.generate_response(prompt, max_new_tokens=8192, temperature=0.1)
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
        
        # 提取A_u均值并截断边界
        au_match = re.search(r'x_new\s*[:,：]\s*\[(.*?)\]', response, re.DOTALL)
        if au_match:
            A_u = [float(x.strip()) for x in au_match.group(1).split(',')]
            # 截断到参数范围 + 确保9维
            au_min, au_max = self.pso.bounds['A_u']
            A_u = [max(au_min, min(au_max, x)) for x in A_u[:9]]
        else:
            # 解析失败时使用当前均值兜底
            A_u = self.pso.current_mean['A_u']
        
        # 提取A_l均值并截断边界
        al_match = re.search(r'y_new\s*[:,：]\s*\[(.*?)\]', response, re.DOTALL)
        if al_match:
            A_l = [float(x.strip()) for x in al_match.group(1).split(',')]
            # 截断到参数范围 + 确保9维
            al_min, al_max = self.pso.bounds['A_l']
            A_l = [max(al_min, min(al_max, x)) for x in A_l[:9]]
        else:
            A_l = self.pso.current_mean['A_l']
        
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
    
    def optimize(self, generations=10, operating_conditions=None):
        """
        执行LLM-PSO翼型优化（核心入口，支持三种目标类型）
        :param generations: 优化世代数
        :param operating_conditions: 气动计算工况
        :return: 最终优化结果
        """
        # 设置气动计算工况
        if operating_conditions:
            self.predictor.set_operating_conditions(operating_conditions)
        
        # 初始化PSO第一代均值
        self.pso.initialize()
        
        # 打印优化配置信息
        opt_obj_desc = {
            "Ld": "平均升阻比",
            "cl": "平均升力系数",
            "hybrid": f"混合目标（w1={self.w1}*平均升力系数 - w2={self.w2}*滞回环面积）"
        }
        print(f"=== LLM-PSO翼型优化开始 (种群规模={self.pso.population_size}) ===")
        print(f"优化目标：{opt_obj_desc[OPT_OBJ]}")
        print(f"参数范围：A_u={self.pso.bounds['A_u']}, A_l={self.pso.bounds['A_l']}\n")
        
        for gen in range(generations):
            print(f"=== 世代 {gen+1}/{generations} ===")
            
            # 1. 采样生成种群（基于当前均值+高斯噪声）
            population = self.pso.sample_population()
            print(f"✅ 生成种群：{len(population)}个个体")
            
            # 2. 评估种群适应度
            fitness_scores = self.evaluate_population(population, operating_conditions)
            gen_best_fitness = np.max(fitness_scores)
            print(f"📊 世代最优：{gen_best_fitness:.6f}")
            print(f"🌐 全局最优：{self.best_fitness:.6f}")
            
            # 3. 记录世代信息（补充Top-3个体，适配提示词要求）
            gen_record = self.pso.record_generation(gen+1, population, fitness_scores)
            # 补充Top-3个体（按适应度降序）
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            top_3_individuals = []
            for ind, _ in sorted_pop[:3]:
                # 统一转换为18维列表（A_u+A_l）
                if isinstance(ind, dict):
                    flat_ind = ind['A_u'] + ind['A_l']
                elif isinstance(ind, (list, np.ndarray)):
                    flat_ind = list(ind)[:18]
                else:
                    flat_ind = []
                top_3_individuals.append(flat_ind)
            gen_record['top_individuals'] = top_3_individuals
            gen_record['best_fitness'] = gen_best_fitness  # 确保存在最优适应度字段
            
            # 4. 保存世代结果
            if not LLM_ENABLE:
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    json.dump({'type': 'generation_result', **gen_record}, f, ensure_ascii=False)
                    f.write('\n')
            
            # 5. 更新均值（最后一代无需更新）
            if gen < generations - 1:
                # 整理临时结果（提前保存，避免异常丢失）
                temp_result = {
                    'best_individual': self.best_global,
                    'best_fitness': self.best_fitness,
                    'optimization_objective': OPT_OBJ,
                    'hybrid_weights': {"w1": self.w1, "w2": self.w2} if OPT_OBJ == "hybrid" else None,
                    'generation_history': self.pso.generation_history,
                    'total_generations': generations,
                    'population_size': self.pso.population_size,
                    'parameter_bounds': self.pso.bounds,
                }
                temp_path = os.path.join(os.path.dirname(__file__), 'pso_final_result.json')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_result, f, ensure_ascii=False, indent=2)
                    
                analyze_pso_results()

                try:
                    if LLM_ENABLE:
                        new_mean = self.get_llm_new_mean(self.pso.generation_history)
                        self.pso.update_mean_from_llm(new_mean)
                    else:
                        input_params = [population, fitness_scores, self.pso.bounds, self.best_fitness, self.best_global]
                        new_mean = pso_update(input_params)
                        self.pso.update_mean_from_llm(new_mean)
                    print(f"🔄 更新均值\n")
                except Exception as e:
                    print(f"⚠️  均值更新失败，使用当前均值：{e}\n")
        
        # 整理最终结果
        final_result = {
            'best_individual': self.best_global,
            'best_fitness': self.best_fitness,
            'optimization_objective': OPT_OBJ,
            'hybrid_weights': {"w1": self.w1, "w2": self.w2} if OPT_OBJ == "hybrid" else None,
            'generation_history': self.pso.generation_history,
            'total_generations': generations,
            'population_size': self.pso.population_size,
            'parameter_bounds': self.pso.bounds,
        }
        
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
            print(f"⚖️  混合目标权重：w1={self.w1}, w2={self.w2}")
        print(f"📝 最优参数：{json.dumps(self.best_global, ensure_ascii=False, indent=2)}")
        print(f"📂 结果文件：{final_path}")
        
        return final_result