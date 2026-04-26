import jsonlines
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
# 插值库导入异常处理
try:
    from scipy.interpolate import interp1d
except ImportError:
    raise ImportError("请先安装scipy库：pip install scipy")

# ===================== 工具函数：无冗余，双平滑算法独立保留 =====================
def sliding_window_smooth(aoa_array, cl_array, window_size=2.0, step=0.2, max_smooth_aoa=23.0):
    """攻角维度滑动窗口平滑（仅≤23°使用，回滞环绘图专用）"""
    aoa_array = np.array(aoa_array)
    cl_array = np.array(cl_array)
    min_aoa = np.min(aoa_array)
    window_centers = np.arange(min_aoa, max_smooth_aoa + step, step)
    
    smoothed_aoa, smoothed_cl = [], []
    for center in window_centers:
        window_left = center - window_size / 2
        window_right = center + window_size / 2
        mask = (aoa_array >= window_left) & (aoa_array <= window_right)
        if np.any(mask):
            smoothed_aoa.append(center)
            smoothed_cl.append(np.mean(cl_array[mask]))
    return np.array(smoothed_aoa), np.array(smoothed_cl)

def moving_average(y, window_size=25, step=5):
    """时间步维度滑动平均平滑（时间步绘图专用）"""
    if len(y) < window_size:
        return y
    smoothed = [np.mean(y[i:i+window_size]) for i in range(0, len(y)-window_size+1, step)]
    if len(smoothed) * step + window_size < len(y):
        smoothed.append(np.mean(y[-window_size:]))
    return np.array(smoothed)

# ===================== 公共核心函数：唯一实现，无重复 =====================
def load_cst_groups(cst_params_path="cst_params.txt"):
    """加载CST分组配置（公共函数）"""
    if not os.path.exists(cst_params_path):
        raise FileNotFoundError(f"CST参数文件不存在：{cst_params_path}")
    
    group_names, cst_param_map = [], {}
    with open(cst_params_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cleaned_cst = re.sub(r'\s+', '', line)
            group_name = f"Group {line_idx}"
            group_names.append(group_name)
            cst_param_map[group_name] = cleaned_cst
    
    if not group_names:
        raise ValueError("cst_params.txt中未找到有效CST参数行")
    print(f"成功加载{len(group_names)}个CST分组：{group_names}")
    return group_names, cst_param_map

def extract_sample_details(jsonl_file_path):
    """提取样本信息（兼容Cd，公共函数）"""
    samples = []
    # 正则表达式
    cst_pattern = r'CST拟合参数：([^。]+)'
    trend_pattern = r'攻角周期变化趋势为：([^，。]+)'
    aoa_response_pattern = r'攻角为([\d\.]+)°'
    cl_pattern = r'升力系数为：([-+]?\d+\.?\d*)'
    cd_pattern = r'阻力系数为：([-+]?\d+\.?\d*)'
    
    with jsonlines.open(jsonl_file_path, 'r') as reader:
        for global_idx, sample in enumerate(reader, start=1):
            prompt = sample.get('prompt', '').strip()
            response = sample.get('predict', '').strip()
            
            # 提取CST参数
            cst_match = re.search(cst_pattern, prompt)
            if not cst_match:
                print(f"警告：全局样本{global_idx}未提取到CST参数，已跳过")
                continue
            cleaned_cst = re.sub(r'\s+', '', cst_match.group(1))
            
            # 提取趋势
            trend_match = re.search(trend_pattern, prompt)
            if not trend_match or trend_match.group(1).strip() not in ['上升', '下降']:
                print(f"警告：全局样本{global_idx}趋势无效，已跳过")
                continue
            trend = trend_match.group(1).strip()
            
            # 提取攻角
            aoa_match = re.search(aoa_response_pattern, response)
            if not aoa_match:
                print(f"警告：全局样本{global_idx}未提取到攻角，已跳过")
                continue
            try:
                aoa = float(aoa_match.group(1))
            except ValueError:
                print(f"警告：全局样本{global_idx}攻角数值无效，已跳过")
                continue
            
            # 提取Cl/Cd
            cl_match = re.search(cl_pattern, response)
            cd_match = re.search(cd_pattern, response)
            if not cl_match:
                print(f"警告：全局样本{global_idx}未提取到升力系数，已跳过")
                continue
            try:
                cl = float(cl_match.group(1))
                cd = float(cd_match.group(1)) if cd_match else 0.0
            except ValueError:
                print(f"警告：全局样本{global_idx}系数数值无效，已跳过")
                continue
            
            samples.append({
                'global_index': global_idx, 'cleaned_cst': cleaned_cst,
                'trend': trend, 'aoa': aoa, 'cl': cl, 'cd': cd
            })
    print(f"成功提取{len(samples)}个有效样本详细信息")
    return samples

def group_samples(samples, cst_param_map):
    """样本分组（公共函数）"""
    cst_to_group = {cst: group for group, cst in cst_param_map.items()}
    grouped_samples = defaultdict(list)
    ungrouped_count = 0
    
    for sample in samples:
        if sample['cleaned_cst'] in cst_to_group:
            grouped_samples[cst_to_group[sample['cleaned_cst']]].append(sample)
        else:
            ungrouped_count += 1
    
    print(f"样本分组完成：")
    for k, v in grouped_samples.items(): print(f"  {k}：{len(v)}个样本")
    if ungrouped_count > 0: print(f"  未分组样本：{ungrouped_count}个")
    return grouped_samples

def process_group_data(group_samples):
    """处理分组数据（计算均值+标准差+95%置信区间+排序，公共函数）"""
    stats = defaultdict(lambda: {'cl_list': [], 'cd_list': []})
    for sample in group_samples:
        aoa_rounded = round(sample['aoa'], 1)
        key = (aoa_rounded, sample['trend'])
        stats[key]['cl_list'].append(sample['cl'])
        stats[key]['cd_list'].append(sample['cd'])
    
    mean_records = []
    for (aoa, trend), data in stats.items():
        if not data['cl_list']: continue
        # 计算均值、标准差
        cl_mean = np.mean(data['cl_list'])
        cl_std = np.std(data['cl_list'], ddof=1)  # 样本标准差（自由度=1）
        sample_count = len(data['cl_list'])
        
        # 计算95%置信区间：均值 ± 1.96 * (标准差/√样本数)
        if sample_count > 1:
            ci95_margin = 1.96 * (cl_std / np.sqrt(sample_count))
        else:
            ci95_margin = 0.0  # 样本数为1时置信区间为0
        cl_ci95_lower = round(cl_mean - ci95_margin, 4)
        cl_ci95_upper = round(cl_mean + ci95_margin, 4)
        
        mean_records.append({
            'trend': trend, 'AoA': aoa,
            'Cl_mean': round(cl_mean, 4),
            'Cl_std': round(cl_std, 4),
            'Cl_ci95_lower': cl_ci95_lower,
            'Cl_ci95_upper': cl_ci95_upper,
            'Cd_mean': round(np.mean(data['cd_list']), 4),
            'sample_count': sample_count
        })
    
    if not mean_records: return [], 0
    
    # 排序规则
    def sort_key(record):
        return (0, record['AoA']) if record['trend'] == '上升' else (1, -record['AoA'])
    sorted_records = sorted(mean_records, key=sort_key)
    
    for idx, record in enumerate(sorted_records, 1):
        record['index'] = idx
    return sorted_records, len(sorted_records)

# ===================== 功能1：回滞环面积 + 攻角-Cl曲线绘图 =====================
def calculate_hysteresis_area(mean_data_list, min_aoa, max_aoa):
    """计算升力系数回滞环面积（原始方法）"""
    filtered = [d for d in mean_data_list if min_aoa <= d['AoA'] <= max_aoa]
    if len(filtered) < 2: return 0.0
    
    up_data = sorted([d for d in filtered if d['trend'] == '上升'], key=lambda x: x['AoA'])
    down_data = sorted([d for d in filtered if d['trend'] == '下降'], key=lambda x: x['AoA'])
    common_aoas = sorted(set([d['AoA'] for d in up_data]) & set([d['AoA'] for d in down_data]))
    
    if len(common_aoas) < 2: return 0.0
    up_cl = [next(d['Cl_mean'] for d in up_data if d['AoA'] == a) for a in common_aoas]
    down_cl = [next(d['Cl_mean'] for d in down_data if d['AoA'] == a) for a in common_aoas]
    
    area = 0.0
    for i in range(1, len(common_aoas)):
        aoa_diff = common_aoas[i] - common_aoas[i-1]
        area += (abs(up_cl[i-1]-down_cl[i-1]) + abs(up_cl[i]-down_cl[i])) * aoa_diff / 2
    return round(area, 4)

def calculate_local_hysteresis_area(mean_data_list, min_aoa, max_aoa):
    """计算[3,10]°局部回滞面积（新方法）"""
    filtered = [d for d in mean_data_list if min_aoa <= d['AoA'] <= max_aoa]
    if len(filtered) < 2: return 0.0

    up_data = sorted([d for d in filtered if d['trend'] == '上升'], key=lambda x: x['AoA'])
    down_data = sorted([d for d in filtered if d['trend'] == '下降'], key=lambda x: x['AoA'])
    common_aoas = sorted(set([d['AoA'] for d in up_data]) & set([d['AoA'] for d in down_data]))

    if len(common_aoas) < 2: return 0.0
    up_cl = [next(d['Cl_mean'] for d in up_data if d['AoA'] == a) for a in common_aoas]
    down_cl = [next(d['Cl_mean'] for d in down_data if d['AoA'] == a) for a in common_aoas]

    common_angles = np.array(common_aoas)
    up_cl_interp = np.array(up_cl)
    down_cl_interp = np.array(down_cl)

    up_abs_sum = sum(abs(2 * np.pi * angle * np.pi / 180.0 - cl) for angle, cl in zip(common_angles, up_cl_interp))
    down_abs_sum = sum(abs(2 * np.pi * angle * np.pi / 180.0 - cl) for angle, cl in zip(common_angles, down_cl_interp))
    area = up_abs_sum + down_abs_sum
    return round(area, 4)

def get_complete_groups(all_processed_data, min_aoa, max_aoa):
    """获取指定攻角范围内的完整组数据"""
    if not all_processed_data: return {}
    
    filtered_data = {}
    for group, (data, _) in all_processed_data.items():
        rec = [d for d in data if min_aoa <= d['AoA'] <= max_aoa]
        if not rec: continue
        filtered_data[group] = (rec, len(rec))
    
    if not filtered_data: return {}
    
    max_idx = max([m for _, m in filtered_data.values()])
    complete_groups = {}
    for g, (d, m) in filtered_data.items():
        if m == max_idx:
            complete_groups[g] = d
    
    return complete_groups

def plot_aoa_cl_comparison(all_processed_data, min_aoa, max_aoa):
    """绘制攻角-Cl对比曲线（23°平滑+24°闭合）"""
    complete_groups = get_complete_groups(all_processed_data, min_aoa, max_aoa)
    if not complete_groups: return
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算各组回滞面积
    areas = {}
    for g, d in complete_groups.items():
        areas[g] = calculate_hysteresis_area(d, min_aoa, max_aoa)

    # 绘图配置
    colors = ['#2E86AB', '#E94F37', '#000000']
    linestyles = ['-', '-', '-.']
    fig, ax = plt.subplots(figsize=(10,6))

    for idx, (group, data) in enumerate(complete_groups.items()):
        up = sorted([d for d in data if d['trend']=='上升'], key=lambda x:x['AoA'])
        down = sorted([d for d in data if d['trend']=='下降'], key=lambda x:x['AoA'])
        color, ls = colors[idx%3], linestyles[idx%4]
        label = f"NACA4415 (Area: {areas[group]})" if idx==0 else f"NACA4415-LLM (Area: {areas[group]})"

        # 上升趋势处理
        if up:
            aoa, cl = [d['AoA'] for d in up], [d['Cl_mean'] for d in up]
            aoa_i = np.linspace(min(aoa), 24, 100)
            f = interp1d(aoa, cl, kind='cubic', fill_value='extrapolate')
            cl_i = f(aoa_i)
            aoa_s, cl_s = sliding_window_smooth(aoa_i, cl_i)
            mask = (aoa_i>23) & (aoa_i<=24)
            aoa_f = np.concatenate([aoa_s, aoa_i[mask]])
            cl_f = np.concatenate([cl_s, cl_i[mask]])
            ax.plot(aoa_f, cl_f, color=color, ls=ls, lw=1.5, label=label)

        # 下降趋势处理
        if down:
            aoa, cl = [d['AoA'] for d in down], [d['Cl_mean'] for d in down]
            aoa_i = np.linspace(min(aoa), 24, 100)
            f = interp1d(aoa, cl, kind='cubic', fill_value='extrapolate')
            cl_i = f(aoa_i)
            aoa_s, cl_s = sliding_window_smooth(aoa_i, cl_i)
            mask = (aoa_i>23) & (aoa_i<=24)
            aoa_f = np.concatenate([aoa_s, aoa_i[mask]])
            cl_f = np.concatenate([cl_s, cl_i[mask]])
            ax.plot(aoa_f, cl_f, color=color, ls=ls, lw=1.5)

        # 24°闭合
        if up and down:
            ax.plot([24,24], [f(24) for f in [interp1d([d['AoA'] for d in up], [d['Cl_mean'] for d in up]), interp1d([d['AoA'] for d in down], [d['Cl_mean'] for d in down])]], color=color, ls=ls, lw=1.5)

    ax.set_xlabel('Angle of Attack', fontsize=13)
    ax.set_ylabel('Mean Lift Coefficient', fontsize=13)
    ax.set_xlim(2.5, 24.5)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('aoa_cl_hysteresis_23smooth_24closed.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("攻角-Cl回滞环曲线已保存")

# ===================== 功能2：平均Cl + 时间步-Cl曲线绘图（新增95%置信区间） =====================
def calculate_cl_average(mean_data_list):
    """计算平均升力系数"""
    return round(np.mean([d['Cl_mean'] for d in mean_data_list]), 4) if mean_data_list else 0.0

def plot_timestep_cl_comparison(all_processed_data, min_aoa, max_aoa):
    """绘制时间步-Cl对比曲线（插值+滑动平均+95%置信区间）"""
    if not all_processed_data: return
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 过滤攻角范围
    filtered_data = {}
    for group, (data, _) in all_processed_data.items():
        rec = [d for d in data if min_aoa <= d['AoA'] <= max_aoa]
        if not rec: continue
        filtered_data[group] = (rec, len(rec))
    
    if not filtered_data:
        print("无有效数据，跳过时间步-Cl绘图")
        return

    # 筛选完整组
    max_idx = max([m for _, m in filtered_data.values()])
    complete_groups, cl_avg = {}, {}
    for g, (d, m) in filtered_data.items():
        if m == max_idx:
            complete_groups[g] = d
            cl_avg[g] = calculate_cl_average(d)

    if not complete_groups: return

    # 绘图配置
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#E94F37', '#000000']
    linestyles = ['-', '-', '-.']
    legend_handles = []  # 用于自定义图例

    for idx, (group, data) in enumerate(complete_groups.items()):
        # 提取核心数据：时间步、均值、95%置信区间上下限
        x = [d['index'] for d in data]
        y_mean = [d['Cl_mean'] for d in data]
        y_ci_lower = [d['Cl_ci95_lower'] for d in data]
        y_ci_upper = [d['Cl_ci95_upper'] for d in data]
        
        color, ls = colors[idx%3], linestyles[idx%4]
        airfoil_type = "NACA4415" if idx==0 else "NACA4415-LLM"
        curve_label = f"{airfoil_type} (Avg CL: {cl_avg[group]})"
        ci_label = f"{airfoil_type} 95% CI"

        # 对均值、置信区间进行插值（密集化）
        x_dense = np.linspace(min(x), max(x), 500)
        f_mean = interp1d(x, y_mean, kind='cubic', fill_value='extrapolate')
        f_ci_lower = interp1d(x, y_ci_lower, kind='cubic', fill_value='extrapolate')
        f_ci_upper = interp1d(x, y_ci_upper, kind='cubic', fill_value='extrapolate')
        
        y_mean_dense = f_mean(x_dense)
        y_ci_lower_dense = f_ci_lower(x_dense)
        y_ci_upper_dense = f_ci_upper(x_dense)

        # 滑动平均平滑
        y_mean_smooth = moving_average(y_mean_dense)
        y_ci_lower_smooth = moving_average(y_ci_lower_dense)
        y_ci_upper_smooth = moving_average(y_ci_upper_dense)
        x_smooth = np.linspace(min(x), max(x), len(y_mean_smooth))

        # 绘制均值曲线
        curve = ax.plot(x_smooth, y_mean_smooth, color=color, ls=ls, lw=1.5, label=curve_label)
        # 绘制95%置信区间（浅色填充）
        ci_fill = ax.fill_between(x_smooth, y_ci_lower_smooth, y_ci_upper_smooth, 
                                  color=color, alpha=0.2, label=ci_label)
        
        # 收集图例句柄
        legend_handles.append((curve[0], ci_fill))

    # 配置图表样式
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('Mean Lift Coefficient', fontsize=14)
    ax.set_xlim(0, max_idx+1)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('timestep_cl_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("时间步-Cl对比曲线（含95%置信区间）已保存")

# ===================== 主函数：统一执行 =====================
def main():
    # 统一配置
    JSONL_PATH = "generated_predictions.jsonl"
    CST_PATH = "cst_params.txt"
    MIN_AOA = 3.0
    MAX_AOA = 24.0

    try:
        print("="*60)
        group_names, cst_map = load_cst_groups(CST_PATH)
        samples = extract_sample_details(JSONL_PATH)
        if not samples: return
        
        grouped = group_samples(samples, cst_map)
        all_data = {}
        for group in group_names:
            if group not in grouped: continue
            data, max_idx = process_group_data(grouped[group])
            if data: all_data[group] = (data, max_idx)
        
        # 计算[3,10]°攻角区间平均回滞面积
        hysteresis_min_aoa = 3.0
        hysteresis_max_aoa = 10.0
        print(f"\n计算[{hysteresis_min_aoa},{hysteresis_max_aoa}]°攻角区间回滞面积...")
        complete_groups = get_complete_groups(all_data, MIN_AOA, MAX_AOA)
        hysteresis_areas = {}
        for group, data in complete_groups.items():
            area = calculate_local_hysteresis_area(data, hysteresis_min_aoa, hysteresis_max_aoa)
            hysteresis_areas[group] = area
            label = "基准翼型" if group == list(complete_groups.keys())[0] else "优化翼型"
            print(f"  {label} [{hysteresis_min_aoa},{hysteresis_max_aoa}]°回滞面积: {area}")
        if len(hysteresis_areas) == 2:
            avg_area = round(sum(hysteresis_areas.values()) / len(hysteresis_areas), 4)
            print(f"  平均回滞面积: {avg_area}")
        
        # 执行两个核心功能
        print("\n生成攻角-Cl回滞环曲线...")
        plot_aoa_cl_comparison(all_data, MIN_AOA, MAX_AOA)
        print("\n生成时间步-Cl对比曲线（含95%置信区间）...")
        plot_timestep_cl_comparison(all_data, MIN_AOA, MAX_AOA)
        
        print("\n" + "="*60)
        print("所有图表生成完成！")

    except Exception as e:
        print(f"执行错误：{str(e)}")

if __name__ == "__main__":
    main()