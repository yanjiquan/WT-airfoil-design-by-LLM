import json
import os
import h5py
from scipy.io import loadmat
import numpy as np
def read_mat_file_no_name(mat_path):
    """
    读取没有变量名的 mat 文件（v7.3 或旧版）
    返回 numpy 数组
    """
    try:
        # 尝试用 scipy 读取 v7.2 以下版本
        data_dict = loadmat(mat_path)
        # 取第一个非 __ 开头的键对应的数组
        for k in data_dict:
            if not k.startswith("__"):
                return np.array(data_dict[k])
    except NotImplementedError:
        # v7.3 HDF5 文件
        with h5py.File(mat_path, 'r') as f:
            # 取第一个数据集
            for k in f.keys():
                return np.array(f[k])

def interp_and_classify(AOA, Cl, Cd):
    """
    输入：原始 AOA, Cl, Cd（一整个俯仰周期）
    输出：插值后的 AOA_lerp, Cl_lerp, Cd_lerp, Tr
    """

    n = len(AOA)

    # ========= 1. 按 AOA 跳变分段 =========
    AOA_seg_means, CL_seg_means, CD_seg_means = [], [], []
    start_idx = 0

    for i in range(1, n):
        if abs(AOA[i] - AOA[i - 1]) >= 0.2:
            AOA_seg_means.append(round(AOA[start_idx:i].mean(), 1))
            CL_seg_means.append(round(Cl[start_idx:i].mean(), 3))
            CD_seg_means.append(round(Cd[start_idx:i].mean(), 4))
            start_idx = i

    # 最后一段
    AOA_seg_means.append(round(AOA[start_idx:n].mean(), 1))
    CL_seg_means.append(round(Cl[start_idx:n].mean(), 3))
    CD_seg_means.append(round(Cd[start_idx:n].mean(), 4))

    AOA_seg_means = np.array(AOA_seg_means)
    CL_seg_means = np.array(CL_seg_means)
    CD_seg_means = np.array(CD_seg_means)

    # ========= 2. 构造目标 AOA 序列 =========
    AOA_lerp_up = np.arange(4, 25, 1)      # 4 → 24
    AOA_lerp_down = np.arange(23, 3, -1)   # 23 → 4
    AOA_lerp = np.concatenate([AOA_lerp_up, AOA_lerp_down])

    Cl_lerp = np.zeros_like(AOA_lerp, dtype=float)
    Cd_lerp = np.zeros_like(AOA_lerp, dtype=float)

    max_idx = np.argmax(AOA_seg_means)
    up_indices = np.arange(0, max_idx)
    down_indices = np.arange(max_idx, len(AOA_seg_means) - 1)

    # ========= 3. 上升段插值 =========
    for i, aoa in enumerate(AOA_lerp_up):
        for seg_idx in up_indices:
            a0, a1 = AOA_seg_means[seg_idx], AOA_seg_means[seg_idx + 1]
            if a0 <= aoa <= a1:
                r = (aoa - a0) / (a1 - a0)
                Cl_lerp[i] = round(CL_seg_means[seg_idx] + r * (CL_seg_means[seg_idx + 1] - CL_seg_means[seg_idx]), 3)
                Cd_lerp[i] = round(CD_seg_means[seg_idx] + r * (CD_seg_means[seg_idx + 1] - CD_seg_means[seg_idx]), 4)
                break

    # ========= 4. 下降段插值 =========
    offset = len(AOA_lerp_up)
    for i, aoa in enumerate(AOA_lerp_down):
        idx = offset + i
        for seg_idx in down_indices:
            a0, a1 = AOA_seg_means[seg_idx], AOA_seg_means[seg_idx + 1]
            if min(a0, a1) <= aoa <= max(a0, a1):
                r = (aoa - a0) / (a1 - a0)
                Cl_lerp[idx] = round(CL_seg_means[seg_idx] + r * (CL_seg_means[seg_idx + 1] - CL_seg_means[seg_idx]), 3)
                Cd_lerp[idx] = round(CD_seg_means[seg_idx] + r * (CD_seg_means[seg_idx + 1] - CD_seg_means[seg_idx]), 4)
                break

    # ========= 5. 趋势标签 =========
    Tr = ["上升"] * len(AOA_lerp)
    Tr[len(AOA_lerp_up):] = ["下降"] * len(AOA_lerp_down)

    return AOA_lerp, Cl_lerp, Cd_lerp, Tr

### 生成带有俯仰周期标识的，固定攻角范围的数据集 ###

file_names = ["LS-0417", "LS-0421", "NACA4415", "S801", "S809", "S810", "S812", "S813", "S814", "S815", "S825"]

#data_names = ["C_RE_75", "C_RE_100", "C_RE_125"]
R = []

for file_name in file_names:
    file_path = f"D:\AirfoilDesign\Data\{file_name}_CST.txt"
    folder_path = f"D:\AirfoilDesign\Data\Dynamic\{file_name}"  # e.g. D:\AirfoilDesign\PythonPram\Predict_Data\S814

    with open(file_path, 'r', encoding='utf-8') as file:
        CST_Cordinate = file.read()

    if os.path.isdir(folder_path):
        # 获取该文件夹下所有子文件夹（不包括文件）
        subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        # 获取所有子文件夹的昵称（文件夹的名称）
        data_names = [os.path.basename(subfolder) for subfolder in subfolders]

#打开mat数据文件
    for data_name in data_names:
        data_path = os.path.join("D:\\AirfoilDesign\\Data\\Dynamic", file_name, data_name)
        mat_files = [f for f in os.listdir(data_path) if f.endswith(".mat")]

        parts_folder = data_name.split("_")
        Re = float(parts_folder[2])/100
        if parts_folder[0] == "C":
            R = "光滑"
        elif parts_folder[0] == "G":
            R = "粗糙"

        for mat_file in mat_files:
            mat_path = os.path.join(data_path, mat_file)
            name_without_ext = os.path.splitext(mat_file)[0]
            parts = name_without_ext.split("_")
            number = (parts[0])
            AOA_mean = int(parts[1])
            AOA_amp = int(parts[2])
            f = float(parts[3][1:])
            f_red = float(parts[4][1:])

            if AOA_mean == 14 and AOA_amp == 10 and 1 <= f <= 1.5 :
                data_array = read_mat_file_no_name(mat_path)
                if data_array.shape[0] < data_array.shape[1]:
                    data_array = data_array.T

            # 假设 data_array 是 n x 3
                AOA = data_array[:, 0].flatten()  # 第一列
                Cl = data_array[:, 1].flatten()  # 第二列
                Cd = data_array[:, 2].flatten()  # 第三列

                AOA_lerp, Cl_lerp, Cd_lerp, Tr = interp_and_classify(AOA, Cl, Cd)

                AOA_high, Cl_high, Cd_high, Tr_high = [], [], [], []
                AOA_low, Cl_low, Cd_low, Tr_low = [], [], [], []

                for aoa, cl, cd, tr in zip(AOA_lerp, Cl_lerp, Cd_lerp, Tr):
                    if aoa > 14:
                        AOA_high.append(aoa)
                        Cl_high.append(cl)
                        Cd_high.append(cd)
                        Tr_high.append(tr)
                    else:
                        AOA_low.append(aoa)
                        Cl_low.append(cl)
                        Cd_low.append(cd)
                        Tr_low.append(tr)

                if np.any(AOA < 0):
                    print(f"文件{file_name}的{data_name}的{mat_file}中存在AOA < 0，跳过该文件")
                    continue
                Tr = [""]*(len(AOA))
                max_AOA_index = np.argmax(AOA)

                for i in range(len(AOA)):
                    if i <= max_AOA_index:
                        Tr[i] = "上升"
                    else:
                        Tr[i] = "下降"

                content1 = []
                content2 = []
                content3 = []
                content4 = []
                for i, value in enumerate(AOA_low):
                    content1.append({
                        "instruction":"你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                                "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。"
                                f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的升力系数",
                        "input":f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                            f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_low[i]}°，俯仰周期当前攻角变化趋势为：{Tr_low[i]}。"
                            f"翼型CST拟合参数为：{CST_Cordinate}",
                        "output":f"该翼型在攻角为{AOA_low[i]}°时的升力系数为：{Cl_low[i]}。"
                    })
                for i, value in enumerate(AOA_high):
                    content2.append({
                        "instruction":"你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                                "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。"
                                f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的升力系数。",
                        "input":f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                            f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_high[i]}°，俯仰周期当前攻角变化趋势为：{Tr_high[i]}。"
                            f"翼型CST拟合参数为：{CST_Cordinate}",
                        "output":f"该翼型在攻角为{AOA_high[i]}°时的升力系数为：{Cl_high[i]}。"
                    })

                    for i, value in enumerate(AOA_low):
                        content3.append({
                            "instruction": "你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                                           "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。"
                                           f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的升力系数",
                            "input": f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                                     f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_low[i]}°，俯仰周期当前攻角变化趋势为：{Tr_low[i]}。"
                                     f"翼型CST拟合参数为：{CST_Cordinate}",
                            "output": f"该翼型在攻角为{AOA_low[i]}°时的阻力系数为：{Cd_low[i]}。"
                        })
                    for i, value in enumerate(AOA_high):
                        content4.append({
                            "instruction": "你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                                           "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。"
                                           f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的阻力系数。",
                            "input": f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                                     f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_high[i]}°，俯仰周期当前攻角变化趋势为：{Tr_high[i]}。"
                                     f"翼型CST拟合参数为：{CST_Cordinate}",
                            "output": f"该翼型在攻角为{AOA_high[i]}°时的阻力系数为：{Cd_high[i]}。"
                        })

                json1_path = f"D:\\AirfoilDesign\\PythonPram\\Dataset\\Dyn_Multi_interp_ClCd\\Low_{number}_{file_name}_{data_name}_{AOA_mean}_{AOA_amp}_CL.json"
                json2_path = f"D:\\AirfoilDesign\\PythonPram\\Dataset\\Dyn_Multi_interp_ClCd\\High_{number}_{file_name}_{data_name}_{AOA_mean}_{AOA_amp}_CL.json"
                json3_path = f"D:\\AirfoilDesign\\PythonPram\\Dataset\\Dyn_Multi_interp_ClCd\\Low_{number}_{file_name}_{data_name}_{AOA_mean}_{AOA_amp}_CD.json"
                json4_path = f"D:\\AirfoilDesign\\PythonPram\\Dataset\\Dyn_Multi_interp_ClCd\\High_{number}_{file_name}_{data_name}_{AOA_mean}_{AOA_amp}_CD.json"
                with open(json1_path, "w", encoding="utf-8") as json_file:
                    json.dump(content1, json_file, ensure_ascii=False, indent=4)
                with open(json2_path, "w", encoding="utf-8") as json_file:
                    json.dump(content2, json_file, ensure_ascii=False, indent=4)
        
                with open(json3_path, "w", encoding="utf-8") as json_file:
                    json.dump(content3, json_file, ensure_ascii=False, indent=4)
                with open(json4_path, "w", encoding="utf-8") as json_file:
                    json.dump(content4, json_file, ensure_ascii=False, indent=4)