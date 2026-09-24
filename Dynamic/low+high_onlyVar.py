import json
import os
import re

# =============================
# 设置参数
# =============================
FTypes = ['NACA']  # 翼型类型
base_path = r"D:\AirfoilDesign\PythonPram\Result\Qwen3\eval\0831_onkey"
folders = ['low', 'high']

# =============================
# 匿名化变量解析
# =============================
def extract_trend_from_prompt(prompt):
    """
    从匿名化 prompt 中提取 x8（trend）值。
    x8=1 → 上升，x8=-1 → 下降
    """
    m = re.search(r'"x8":\s*([-+]?\d+)', prompt)
    if m is None:
        return "未知"
    val = int(m.group(1))
    if val == 1:
        return "上升"
    elif val == -1:
        return "下降"
    else:
        return "未知"

# =============================
# 分段函数
# =============================
def segment_trend(tr_list, mode='low'):
    """
    根据tr列表进行分段
    low模式：'上升'->'下降'为分段
    high模式：'下降'->'上升'为分段
    返回列表: [(segment_name, [indices]), ...]
    """
    segments = []
    seg_name_num = 1
    seg_name_char = ord('a')
    current_segment = []

    for i in range(len(tr_list)-1):
        current_segment.append(i)
        tr_current = tr_list[i]
        tr_next = tr_list[i+1]

        if mode == 'low' and "上升" in tr_current and "下降" in tr_next:
            seg_name = str(seg_name_num)
            segments.append((seg_name, current_segment.copy()))
            current_segment = []
            seg_name_num += 1
        elif mode == 'high' and "下降" in tr_current and "上升" in tr_next:
            seg_name = chr(seg_name_char)
            segments.append((seg_name, current_segment.copy()))
            current_segment = []
            seg_name_char += 1

    # 添加最后一个索引，确保不丢行
    current_segment.append(len(tr_list)-1)
    if mode == 'low':
        segments.append((str(seg_name_num), current_segment.copy()))
    else:
        segments.append((chr(seg_name_char), current_segment.copy()))

    return segments

# =============================
# 主循环处理 FType
# =============================
for FType in FTypes:
    print(f"正在处理 FType: {FType}")

    # ---------- 读取 low 文件 ----------
    data_low = []
    low_path = os.path.join(base_path, folders[0], f"generated_predictions_{FType}.jsonl")
    with open(low_path, "r", encoding="utf-8") as f:
        for line in f:
            data_low.append(json.loads(line))

    # ---------- 读取 high 文件 ----------
    data_high = []
    high_path = os.path.join(base_path, folders[1], f"generated_predictions_{FType}.jsonl")
    with open(high_path, "r", encoding="utf-8") as f:
        for line in f:
            data_high.append(json.loads(line))

    # ---------- 提取 tr（从 x8 匿名变量） ----------
    tr_low = [extract_trend_from_prompt(item['prompt']) for item in data_low]
    tr_high = [extract_trend_from_prompt(item['prompt']) for item in data_high]

    # ---------- 分段 ----------
    segments_low = segment_trend(tr_low, mode='low')
    segments_high = segment_trend(tr_high, mode='high')

    # ---------- 合并段 1a2b3c... ----------
    combined_segments = []
    min_len = min(len(segments_low), len(segments_high))
    for i in range(min_len):
        # low 段
        seg_name_low, indices_low = segments_low[i]
        for idx in indices_low:
            data_low[idx]['segment'] = seg_name_low
            combined_segments.append(data_low[idx])
        # high 段
        seg_name_high, indices_high = segments_high[i]
        for idx in indices_high:
            data_high[idx]['segment'] = seg_name_high
            combined_segments.append(data_high[idx])

    # 处理 low/high 段数不一致的剩余段
    for i in range(min_len, len(segments_low)):
        seg_name_low, indices_low = segments_low[i]
        for idx in indices_low:
            data_low[idx]['segment'] = seg_name_low
            combined_segments.append(data_low[idx])

    for i in range(min_len, len(segments_high)):
        seg_name_high, indices_high = segments_high[i]
        for idx in indices_high:
            data_high[idx]['segment'] = seg_name_high
            combined_segments.append(data_high[idx])

    # ---------- 补充未分段数据（保险） ----------
    all_low_idx = set(range(len(data_low)))
    all_high_idx = set(range(len(data_high)))
    added_low_idx = set(idx for _, indices in segments_low for idx in indices)
    added_high_idx = set(idx for _, indices in segments_high for idx in indices)

    for idx in all_low_idx - added_low_idx:
        data_low[idx]['segment'] = 'last'
        combined_segments.append(data_low[idx])

    for idx in all_high_idx - added_high_idx:
        data_high[idx]['segment'] = 'last'
        combined_segments.append(data_high[idx])

    # ---------- 输出新 jsonl ----------
    output_path = os.path.join(base_path, f"generated_predictions_{FType}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in combined_segments:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"FType {FType} 处理完成，输出文件: {output_path}")
