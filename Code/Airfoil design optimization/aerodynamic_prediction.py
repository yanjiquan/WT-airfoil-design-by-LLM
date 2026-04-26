from prompts import get_aerodynamic_prediction_prompt
from llm_model import LLMModel
import numpy as np
from tqdm import tqdm
import json
import os
from scipy import integrate

def load_config(config_path):
    """加载并验证配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    required_fields = ["angle_range", "sample_delta", "roughness", "reynolds", 
                      "osc_freq", "decay_freq", "cycle"]
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
        self.attack_angles = generate_attack_angles(
            self.config["angle_range"],
            self.config["sample_delta"]
        )
        self.output_path = os.path.join(script_dir, "generated_predictions.jsonl")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            pass
        print(f"加载配置完成 - 攻角数量: {len(self.attack_angles)} "
              f"({self.attack_angles[0]}° - {self.attack_angles[-1]}°)")
        print(f"当前使用模型：LLM气动预测模型")
    
    def set_operating_conditions(self, conditions):
        """设置工况参数"""
        self.config.update(conditions)
        if "angle_range" in conditions or "sample_delta" in conditions:
            self.attack_angles = generate_attack_angles(
                self.config["angle_range"],
                self.config["sample_delta"]
            )
    
    def predict_aerodynamic_performance(self, cst_params):
        """预测气动性能（仅保留升力系数Cl）"""
        results = {
            'angle_results': {},
            'summary': {},
            'hysteresis_mean_area': 0.0
        }
        
        all_conditions = self._generate_all_conditions()
        
        for roughness_idx, roughness in enumerate(self.config["roughness"]):
            for reyn_idx, reynolds in enumerate(self.config["reynolds"]):
                osc_freq = self.config["osc_freq"][reyn_idx]
                decay_freq = self.config["decay_freq"][roughness_idx][reyn_idx]
                formatted_reyn = format_reynolds(reynolds)
                formatted_osc = f"{osc_freq:.2f}"
                formatted_decay = f"{decay_freq:.3f}"
                
                for angle in tqdm(self.attack_angles, desc="气动预测"):
                    for cycle in self.config["cycle"]:
                        angle_key = f"{angle}°_{cycle}"
                        if angle_key not in results['angle_results']:
                            results['angle_results'][angle_key] = []
                        
                        # 仅保留LLM模型预测逻辑
                        prompt = get_aerodynamic_prediction_prompt(
                            cst_params,
                            formatted_reyn,
                            roughness,
                            formatted_osc,
                            formatted_decay,
                            14,
                            10,
                            angle,
                            cycle
                        )
                        response = self.llm.generate_response(prompt, max_new_tokens=200, temperature=0.1)
                        # 仅解析升力系数Cl
                        aerodata = self.llm.parse_aerodynamic_coefficients(response)
                        
                        # 保存结果（仅保留Cl）
                        self._save_prediction(
                            prompt=prompt,
                            response=response,
                            angle=angle,
                            trend=cycle,
                            aerodata=aerodata
                        )
                        
                        if aerodata['Cl'] is not None:
                            results['angle_results'][angle_key].append({
                                'condition': {
                                    'reynolds': reynolds,
                                    'roughness': roughness,
                                    'osc_freq': osc_freq,
                                    'decay_freq': decay_freq
                                },
                                'data': aerodata,  # 仅包含Cl
                                'angle': angle,
                                'trend': cycle
                            })
        
        results['summary'] = self._calculate_statistics(results['angle_results'])
        results['hysteresis_mean_area'] = self._calculate_hysteresis_area(results['angle_results'])
        return results
    
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
                'timestamp': str(os.popen('date').read().strip())
            }
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    def _generate_all_conditions(self):
        """生成所有工况组合"""
        conditions = []
        for roughness_idx, roughness in enumerate(self.config["roughness"]):
            for reyn_idx, reynolds in enumerate(self.config["reynolds"]):
                osc_freq = self.config["osc_freq"][reyn_idx]
                decay_freq = self.config["decay_freq"][roughness_idx][reyn_idx]
                conditions.append({
                    'reynolds': reynolds,
                    'roughness': roughness,
                    'osc_freq': osc_freq,
                    'decay_freq': decay_freq
                })
        return conditions
    
    def _calculate_statistics(self, angle_results):
        """统计计算（仅保留升力系数Cl）"""
        stats = {
            'angle_mean_cl': {},
            'cycle_mean_cl': 0.0,
            'angle_sequence': []
        }
        
        angle_means_cl = []
        for angle in self.attack_angles:
            for cycle in self.config["cycle"]:
                angle_key = f"{angle}°_{cycle}"
                if angle_key in angle_results and angle_results[angle_key]:
                    results = angle_results[angle_key]
                    # 仅统计Cl
                    cl_values = [r['data']['Cl'] for r in results]
                    mean_cl = np.mean(cl_values)
                    stats['angle_mean_cl'][angle_key] = mean_cl
                    
                    angle_means_cl.append(mean_cl)
                    stats['angle_sequence'].append(angle_key)
        
        if angle_means_cl:        
            stats['cycle_mean_cl'] = np.mean(angle_means_cl)
        
        return stats
    
    def _calculate_hysteresis_area(self, angle_results):
        """计算滞回环面积（仅基于Cl，逻辑不变）"""
        valid_areas = []
        conditions = self._generate_all_conditions()

        for condition in conditions:
            up_angles = []
            up_cl = []
            down_angles = []
            down_cl = []

            for angle in sorted(self.attack_angles):
                up_key = f"{angle}°_上升"
                if up_key in angle_results and angle_results[up_key]:
                    cl_vals = [
                        r['data']['Cl'] for r in angle_results[up_key]
                        if r['condition']['reynolds'] == condition['reynolds'] 
                        and r['condition']['roughness'] == condition['roughness']
                        and r['data']['Cl'] is not None
                    ]
                    if cl_vals:
                        up_angles.append(angle)
                        up_cl.append(np.mean(cl_vals))

                down_key = f"{angle}°_下降"
                if down_key in angle_results and angle_results[down_key]:
                    cl_vals = [
                        r['data']['Cl'] for r in angle_results[down_key]
                        if r['condition']['reynolds'] == condition['reynolds'] 
                        and r['condition']['roughness'] == condition['roughness']
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

                    up_abs_sum = sum(abs(2 * np.pi * angle * np.pi / 180.0 - cl) for angle, cl in zip(common_angles, up_cl_interp))
                    down_abs_sum = sum(abs(2 * np.pi * angle * np.pi / 180.0 - cl) for angle, cl in zip(common_angles, down_cl_interp))
                    area = up_abs_sum + down_abs_sum

                    valid_areas.append(area)

        mean_area = np.mean(valid_areas) if valid_areas else 0.0
        print(f"平均滞回环面积：{mean_area:.6f}")
        return mean_area