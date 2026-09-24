from prompts import get_aerodynamic_prediction_prompt
from llm_model import LLMModel
import numpy as np
from tqdm import tqdm
import json
import os
import time
from collections import defaultdict

def load_config(config_path):
    """加载并验证配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    required_fields = ["sample_delta_by_amp", "roughness", "conditions", "cycle", "test_mode"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"config.json缺少必要字段：{field}")
    return config

def generate_attack_angles(angle_range, sample_delta):
    """生成攻角列表"""
    start_angle, end_angle = angle_range
    if sample_delta < 0.0001:
        return [round(start_angle, 1)]
    angles = np.arange(start_angle, end_angle + sample_delta, sample_delta)
    angles = np.round(angles, 1)
    angles = list(dict.fromkeys(angles))
    return angles

def format_reynolds(reynolds_value):
    """格式化雷诺数"""
    return f"{reynolds_value / 1e6:.2f}*10^6"

class AerodynamicPredictor:
    def __init__(self):
        # 仅保留LLM模型
        self.llm = LLMModel()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        self.config = load_config(config_path)
        self.output_path = os.path.join(script_dir, "generated_predictions.jsonl")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            pass
        # 攻角区间由每个工况的平均攻角±振幅推导，这里仅统计总预测攻角点数
        total_points = sum(
            len(self._attack_angles_for(cond))
            for cond in self._active_conditions().values()
        )
        print(f"加载配置完成 - 工况数量: {len(self.config['conditions'])} "
              f"({', '.join(self.config['conditions'].keys())}) × 粗糙度 "
              f"{len(self.config['roughness'])}档")
        print(f"攻角采样: 按振幅查表 {self.config['sample_delta_by_amp']}°，"
              f"每工况覆盖俯仰周期(平均攻角±振幅)，总攻角点数 {total_points}")
        print(f"当前使用模型：LLM气动预测模型")

    def _sample_delta_for(self, amp):
        """按振幅选择攻角采样间隔（如 5.5°→1°、10°→2°），未命中时回退全局默认 1°。"""
        table = self.config.get("sample_delta_by_amp", {})
        if table:
            key = min(table, key=lambda k: abs(float(k) - amp))
            return float(table[key])
        return float(self.config.get("sample_delta", 1.0))

    def _active_conditions(self):
        """测试模式下只返回 config.test_mode.conditions 指定的工况，否则返回全部工况。"""
        tm = self.config.get("test_mode", {})
        if tm.get("enabled") and tm.get("conditions"):
            conds = {c: self.config["conditions"][c] for c in tm["conditions"]
                     if c in self.config["conditions"]}
            if conds:
                return conds
        return self.config["conditions"]

    def _attack_angles_for(self, cond):
        """工况的攻角扫描区间。测试模式用固定攻角列表（config.test_mode.angles）。"""
        tm = self.config.get("test_mode", {})
        if tm.get("enabled") and tm.get("angles"):
            return [float(a) for a in tm["angles"]]
        return generate_attack_angles(
            [cond["mean_aoa"] - cond["amplitude"], cond["mean_aoa"] + cond["amplitude"]],
            self._sample_delta_for(cond["amplitude"])
        )

    def set_operating_conditions(self, conditions):
        """设置工况参数（--config 指定时，更新后打印实际生效工况）"""
        self.config.update(conditions)
        conds = self._active_conditions()
        if conds:
            print(f"实际生效工况: {len(conds)} ({', '.join(conds.keys())}) × 粗糙度 "
                  f"{len(self.config['roughness'])}档")

    def predict_aerodynamic_performance(self, cst_params):
        """
        预测气动性能（按 config 中 D1-D8 工况 × 粗糙度展开，仅保留升力系数Cl）。
        每个工况在自身的平均攻角±振幅俯仰周期内独立扫描攻角点。
        vLLM 后端并发执行（ThreadPoolExecutor，并发度 config.concurrency）；
        local 后端退回串行（LoRA 切换不线程安全）。
        """
        results = {
            'angle_results': {},
            'summary': {},
            'hysteresis_mean_area': 0.0
        }

        # ---- 1. 收集所有 (工况×粗糙度×攻角×周期) 预测任务 ----
        tasks = []
        for cond_code, cond in self._active_conditions().items():
            reynolds = cond["reynolds"]
            mean_aoa = cond["mean_aoa"]
            amplitude = cond["amplitude"]
            osc_freq = cond["osc_freq"]
            decay_freq = cond["decay_freq"]
            formatted_reyn = format_reynolds(reynolds)
            formatted_osc = f"{osc_freq:.2f}"
            formatted_decay = f"{decay_freq:.3f}"
            angles = self._attack_angles_for(cond)

            for roughness in self.config["roughness"]:
                for angle in angles:
                    for cycle in self.config["cycle"]:
                        prompt = get_aerodynamic_prediction_prompt(
                            cst_params, formatted_reyn, roughness, formatted_osc,
                            formatted_decay, mean_aoa, amplitude, angle, cycle
                        )
                        tasks.append({
                            'angle_key': f"{cond_code}_{angle}°_{cycle}",
                            'prompt': prompt,
                            'condition': {
                                'cond_code': cond_code,
                                'reynolds': reynolds,
                                'roughness': roughness,
                                'mean_aoa': mean_aoa,
                                'amplitude': amplitude,
                                'osc_freq': osc_freq,
                                'decay_freq': decay_freq,
                            },
                            'angle': angle,
                            'cycle': cycle,
                        })

        # ---- 2. 执行预测（vLLM 并发 / local 串行） ----
        self._run_prediction_tasks(tasks, results, sequential=(self.llm.backend != "vllm"))

        results['summary'] = self._calculate_statistics(results['angle_results'])
        results['hysteresis_mean_area'] = self._calculate_hysteresis_area(results['angle_results'])
        # 每工况滞回面积（原始 + 按攻角跨度归一化）并入 summary
        for code, area in self._per_cond_area.items():
            results['summary']['per_condition'].setdefault(code, {})['hysteresis_mean_area'] = area
        for code, area_n in self._per_cond_area_norm.items():
            results['summary']['per_condition'].setdefault(code, {})['hysteresis_mean_area_norm'] = area_n
        return results

    def _run_prediction_tasks(self, tasks, results, sequential=False):
        """并发/串行执行预测任务，结果归并到 results['angle_results']。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock

        if not tasks:
            return
        lock = Lock()

        def predict_one(task):
            response = self.llm.generate_response(
                task['prompt'], max_new_tokens=8192, temperature=0.0)
            aerodata = self.llm.parse_aerodynamic_coefficients(response)
            return task, response, aerodata

        if sequential:
            for task in tqdm(tasks, desc="气动预测"):
                try:
                    task_done, response, aerodata = predict_one(task)
                except Exception as e:
                    print(f"[WARN] 预测失败，跳过：{e}")
                    continue
                with lock:
                    self._absorb_prediction(task_done, response, aerodata, results)
        else:
            concurrency = self.config.get("concurrency", 8)
            workers = max(1, min(int(concurrency), len(tasks)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(predict_one, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="气动预测（并发）"):
                    try:
                        task_done, response, aerodata = fut.result()
                    except Exception as e:
                        print(f"[WARN] 预测失败，跳过：{e}")
                        continue
                    with lock:
                        self._absorb_prediction(task_done, response, aerodata, results)

    def _absorb_prediction(self, task, response, aerodata, results):
        """归并单次预测结果（写记录文件 + 更新 angle_results）。调用方须持 lock。"""
        angle_key = task['angle_key']
        if angle_key not in results['angle_results']:
            results['angle_results'][angle_key] = []
        if aerodata['Cl'] is not None:
            results['angle_results'][angle_key].append({
                'condition': task['condition'],
                'data': aerodata,  # 仅包含Cl
                'angle': task['angle'],
                'trend': task['cycle']
            })
        self._save_prediction(
            prompt=task['prompt'],
            response=response,
            angle=task['angle'],
            trend=task['cycle'],
            aerodata=aerodata
        )

    def _save_prediction(self, prompt, response, angle, trend, aerodata):
        """保存预测结果（仅保留升力系数）"""
        with open(self.output_path, 'a', encoding='utf-8') as f:
            # 仅保留Cl的标签，移除Cd/L/D
            label_text = f"攻角为{angle}°时的升力系数为：{aerodata['Cl']}。"
            result = {
                'prompt': prompt,
                'predict': response,
                'label': label_text,
                'angle': angle,
                'trend': trend,
                'calculated_cl': aerodata['Cl'],  # 仅保留Cl
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    def _generate_all_conditions(self):
        """生成所有工况组合（工况 × 粗糙度），供滞回环面积按条件过滤"""
        conditions = []
        for cond_code, cond in self._active_conditions().items():
            for roughness in self.config["roughness"]:
                conditions.append({
                    'cond_code': cond_code,
                    'reynolds': cond["reynolds"],
                    'roughness': roughness,
                    'osc_freq': cond["osc_freq"],
                    'decay_freq': cond["decay_freq"],
                    'mean_aoa': cond["mean_aoa"],
                    'amplitude': cond["amplitude"],
                })
        return conditions

    def _calculate_statistics(self, angle_results):
        """
        统计计算（仅保留升力系数Cl）。
        每工况先在其俯仰周期内独立计算平均 Cl，全局 = 各工况等权平均。
        """
        stats = {
            'angle_mean_cl': {},
            'cycle_mean_cl': 0.0,
            'angle_sequence': [],
            'per_condition': {},
        }

        per_cond_cl = defaultdict(list)
        for cond_code, cond in self._active_conditions().items():
            angles = self._attack_angles_for(cond)
            for angle in angles:
                for cycle in self.config["cycle"]:
                    angle_key = f"{cond_code}_{angle}°_{cycle}"
                    if angle_key in angle_results and angle_results[angle_key]:
                        cl_values = [r['data']['Cl'] for r in angle_results[angle_key]]
                        mean_cl = np.mean(cl_values)
                        stats['angle_mean_cl'][angle_key] = mean_cl
                        per_cond_cl[cond_code].append(mean_cl)
                        stats['angle_sequence'].append(angle_key)

        # 每工况周期平均 Cl
        for code, vals in per_cond_cl.items():
            stats['per_condition'][code] = {
                'cycle_mean_cl': float(np.mean(vals)) if vals else 0.0
            }

        # 全局 = 各工况等权平均（不按攻角点数加权）
        if stats['per_condition']:
            stats['cycle_mean_cl'] = float(np.mean(
                [pc['cycle_mean_cl'] for pc in stats['per_condition'].values()]
            ))

        return stats

    def _calculate_hysteresis_area(self, angle_results):
        """
        计算滞回环面积（梯形面积公式），按工况×粗糙度分别计算：
        - 每工况在其俯仰周期内计算原始梯形面积 A_i
        - 按攻角跨度归一化 A_i / (2·amplitude_i)（每度攻角的滞回程度）
        - 全局 = 各工况归一化面积的等权平均
        """
        per_cond_area = defaultdict(float)
        per_cond_area_norm = defaultdict(float)
        rough_count = len(self.config["roughness"]) if self.config["roughness"] else 1
        conditions = self._generate_all_conditions()

        for condition in conditions:
            cond_code = condition['cond_code']
            rough = condition['roughness']
            amplitude = condition['amplitude']
            angles = self._attack_angles_for(condition)
            up_angles = []
            up_cl = []
            down_angles = []
            down_cl = []

            for angle in sorted(angles):
                up_key = f"{cond_code}_{angle}°_上升"
                if up_key in angle_results and angle_results[up_key]:
                    cl_vals = [
                        r['data']['Cl'] for r in angle_results[up_key]
                        if r['condition']['roughness'] == rough
                        and r['data']['Cl'] is not None
                    ]
                    if cl_vals:
                        up_angles.append(angle)
                        up_cl.append(np.mean(cl_vals))

                down_key = f"{cond_code}_{angle}°_下降"
                if down_key in angle_results and angle_results[down_key]:
                    cl_vals = [
                        r['data']['Cl'] for r in angle_results[down_key]
                        if r['condition']['roughness'] == rough
                        and r['data']['Cl'] is not None
                    ]
                    if cl_vals:
                        down_angles.append(angle)
                        down_cl.append(np.mean(cl_vals))

            if len(up_angles) >= 2 and len(down_angles) >= 2:
                common_angles = sorted(list(set(up_angles) & set(down_angles)))
                if len(common_angles) >= 2:
                    up_cl_interp = np.interp(common_angles, up_angles, up_cl)
                    down_cl_interp = np.interp(common_angles, down_angles, down_cl)

                    # 梯形面积公式
                    area = 0.0
                    for i in range(1, len(common_angles)):
                        aoa_diff = common_angles[i] - common_angles[i-1]
                        area += (abs(up_cl_interp[i-1] - down_cl_interp[i-1]) + abs(up_cl_interp[i] - down_cl_interp[i])) * aoa_diff / 2

                    span = 2 * amplitude if amplitude > 0 else 1.0
                    norm_area = area / span
                    per_cond_area[cond_code] += area
                    per_cond_area_norm[cond_code] += norm_area

        # 按工况平均（跨粗糙度）
        for code in list(per_cond_area):
            per_cond_area[code] /= rough_count
            per_cond_area_norm[code] /= rough_count

        self._per_cond_area = dict(per_cond_area)
        self._per_cond_area_norm = dict(per_cond_area_norm)

        mean_area = float(np.mean(list(per_cond_area_norm.values()))) if per_cond_area_norm else 0.0
        print(f"平均滞回环面积（归一化，每度攻角）：{mean_area:.6f}")
        return mean_area
