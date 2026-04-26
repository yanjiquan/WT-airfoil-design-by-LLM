def get_aerodynamic_prediction_prompt(cst_params, reynolds, roughness, osc_freq, decay_freq, mean_aoa, amplitude, angle, trend):
    """生成气动预测提示词"""
    prompt = f"""你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的升力系数。该翼型的雷诺数为：{reynolds}，粗糙度为：{roughness}，平均攻角为：{mean_aoa}°，攻角振幅为：{amplitude}°，振荡频率为：{osc_freq}Hz，衰减频率为：{decay_freq}Hz，攻角为：{angle}°，攻角周期变化趋势为：{trend}。CST拟合参数：N1=0.5, N2=1.0；上表面形函数系数 A_u = {cst_params['A_u']}，后缘厚度 z_u_TE = 0.0000；下表面形函数系数 A_l = {cst_params['A_l']}，后缘厚度 z_l_TE = 0.0000。
"""
    return prompt


def get_llm_mean_prompt(generation_history, bounds):
    """提示词生成函数"""
    # -------------------------- 1. 角色分配 --------------------------
    role_text = """你是一个优化器。

"""

    # -------------------------- 2. 任务定义 --------------------------
    task_text = """你需要基于历史记录最大化函数F[x1-x9, y1-y9]的值。

"""

    # -------------------------- 3. 参数约束 --------------------------
    constraint_text = f"""变量约束条件：
1. [x1-x9]：每个维度∈[{bounds['A_u'][0]}, {bounds['A_u'][1]}]；
2. [y1-y9]：每个维度∈[{bounds['A_l'][0]}, {bounds['A_l'][1]}]；
3. 下一代[x1-x9, y1-y9]基于历史记录微调。
"""

    # -------------------------- 4. 历史记录（按论文规则筛选） --------------------------
    # 筛选规则：Top-T=5个最优世代 + 最近R=3个世代（去重），每个世代保留Top-M=3个个体
    if not generation_history:
        history_text = "暂无有效记录，请基于取值范围生成初始[x1-x9, y1-y9]。\n"
    else:
        # 按目标值（攻角周期升阻比）升序排序所有世代（便于取最优）
        sorted_by_fitness = sorted(generation_history, key=lambda x: x['best_fitness'])
        # 取Top-5最优世代
        top_5 = sorted_by_fitness[-8:] if len(sorted_by_fitness)>=8 else sorted_by_fitness
        sort_len = 8 if len(sorted_by_fitness)>=8 else len(sorted_by_fitness)
        # 取最近3个世代（按世代号降序取前3）
        # sorted_by_generation = sorted(generation_history, key=lambda x: x['generation'], reverse=True)
        # latest_3 = sorted_by_generation[:3] if len(sorted_by_generation)>=3 else sorted_by_generation
        # 合并去重（按世代号）
        # unique_generations = {g['generation']: g for g in top_5 + latest_3}.values()
        unique_generations = {g['generation']: g for g in top_5}.values()
        # 按目标值升序排列最终候选世代
        candidate_generations = sorted(unique_generations, key=lambda x: x['best_fitness'])

        # 构建历史记录文本（每个世代保留Top-3浮点数个体）
        history_text = f"""以下是按F值升序排列的{sort_len}代历史记录。"""
        history_text += "每行记录代表一代，格式为「最高F值; 最高F值个体; 次高F值个体」；"
        history_text += "高F值个体的变量取值特征更优。\n"

        for gen in candidate_generations:
            # 假设每个世代记录包含'top_individuals'（Top-3浮点数编码个体列表）
            top_individuals = gen['top_individuals'][:2]  # 确保只取前3个
            # 转换个体为字符串（保留4位小数，如"[0.1234,0.2345,...,0.0012]"）
            ind_strs = []
            for ind in top_individuals:
                # 确保每个浮点数保留4位小数，提升可读性
                float_ind = [f"{num:.4f}" for num in ind]
                ind_strs.append(f"[{','.join(float_ind)}]")
            # 拼接该行记录（目标值为攻角周期升阻比）
            history_text += f"{gen['best_fitness']:.2f}: {'; '.join(ind_strs)}\n"

    # -------------------------- 5. 输出格式（严格按要求） --------------------------
    output_text = """确定下一代[x1-x9, y1-y9]。请将输出格式设置为（数值以及包含x_new，y_new）：
x_new: [x1, x2, x3, x4, x5, x6, x7, x8, x9]
y_new: [y1, y2, y3, y4, y5, y6, y7, y8, y9]
无需解释，仅输出上述两行内容，保留四位小数。"""

    # 整合完整提示词
    full_prompt = role_text + task_text + constraint_text + history_text + output_text
    return full_prompt
