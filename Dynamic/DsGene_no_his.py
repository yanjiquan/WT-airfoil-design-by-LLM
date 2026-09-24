import json
import os
import numpy as np
import sys

# 打印Python环境信息
print('Python解释器路径:', sys.executable)
print('Python版本:', sys.version)
print('Python路径:', sys.path)
print('当前工作目录:', os.getcwd())

# 尝试导入scipy.io.loadmat，如果失败则尝试安装
scipy_available = False
try:
    from scipy.io import loadmat
    scipy_available = True
    print('scipy导入成功')
except ImportError as e:
    print(f"错误: 缺少scipy模块，导入失败: {e}")
    print("尝试使用pip安装scipy...")
    
    try:
        import subprocess
        import sys
        # 尝试使用pip安装scipy
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
        print("scipy安装成功，重新导入...")
        # 重新导入
        from scipy.io import loadmat
        scipy_available = True
        print('scipy重新导入成功')
    except Exception as install_error:
        print(f"安装scipy失败: {install_error}")
        print("请手动运行 'pip install scipy' 来安装")
        import sys
        sys.exit(1)

# 尝试导入h5py，如果失败则设置为None
try:
    import h5py
    h5py_available = True
except ImportError:
    h5py = None
    h5py_available = False
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
        if h5py_available:
            with h5py.File(mat_path, 'r') as f:
                # 取第一个数据集
                for k in f.keys():
                    return np.array(f[k])
        else:
            print(f"警告: 无法读取 HDF5 格式的 mat 文件 {mat_path}，因为 h5py 模块不可用")
            return None

### 生成带有俯仰周期标识的，固定攻角范围的数据集 ###

file_names = ["LS-0417", "LS-0421", "NACA4415", "S801", "S809", "S810", "S812", "S813", "S814", "S815", "S825"]
#data_names = ["C_RE_75", "C_RE_100", "C_RE_125"]
R = []
Dataset_Path = "D:\AirfoilDesign\PythonPram\Dataset_all\Dyn\CST_nohis"

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
        data_path = os.path.join("D:\AirfoilDesign\Data\Dynamic", file_name, data_name)
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

            # 读取所有mat文件，不再限制参数
            data_array = read_mat_file_no_name(mat_path)
            
            # 检查是否成功读取文件
            if data_array is None:
                continue
                
            if data_array.shape[0] < data_array.shape[1]:
                data_array = data_array.T

            # 假设 data_array 是 n x 3
            AOA = data_array[:, 0].flatten()  # 第一列
            Cl = data_array[:, 1].flatten()  # 第二列
            Cd = data_array[:, 2].flatten()  # 第三列

            AOA_high, Cl_high, Cd_high, Tr_high = [], [], [], []
            AOA_low, Cl_low, Cd_low, Tr_low = [], [], [], []

            max_AOA_index = np.argmax(AOA)

            for i, (aoa_val, cl_val, cd_val) in enumerate(zip(AOA, Cl, Cd)):
                trend = "上升" if i <= max_AOA_index else "下降"

                # 遍历每个数据点并分类
                # 上升段AOA>=15或下降段AOA>=6为高不确定性区high，区间外为low
                if (trend == "上升" and aoa_val >= 15) or (trend == "下降" and aoa_val >= 6):
                    AOA_high.append(aoa_val)
                    Cl_high.append(cl_val)
                    Cd_high.append(cd_val)
                    Tr_high.append(trend)
                else:
                    AOA_low.append(aoa_val)
                    Cl_low.append(cl_val)
                    Cd_low.append(cd_val)
                    Tr_low.append(trend)

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
                            f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的升力系数。",
                    "input":f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                        f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_low[i]}°，俯仰周期当前攻角变化趋势为：{Tr_low[i]}。"
                        f"翼型CST拟合参数为：{CST_Cordinate}。",
                    "output":f"该翼型在攻角为{AOA_low[i]}°时的升力系数为：{Cl_low[i]}。"
                })
            for i, value in enumerate(AOA_high):
                content2.append({
                    "instruction":"你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                            "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。"
                            f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的升力系数。",
                    "input":f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                        f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_high[i]}°，俯仰周期当前攻角变化趋势为：{Tr_high[i]}。"
                        f"翼型CST拟合参数为：{CST_Cordinate}。",
                    "output":f"该翼型在攻角为{AOA_high[i]}°时的升力系数为：{Cl_high[i]}。"
                })
            for i, value in enumerate(AOA_low):
                content3.append({
                    "instruction": "你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                               "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。"
                               f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的阻力系数。",
                    "input": f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                         f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_low[i]}°，俯仰周期当前攻角变化趋势为：{Tr_low[i]}。"
                         f"翼型CST拟合参数为：{CST_Cordinate}。",
                    "output": f"该翼型在攻角为{AOA_low[i]}°时的阻力系数为：{Cd_low[i]}。"
                })
            for i, value in enumerate(AOA_high):
                content4.append({
                    "instruction": "你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                               "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度、平均攻角、攻角振幅、振荡频率和衰减频率计算出它可能的升阻力系数。"
                               f"请你根据下面输入的各项参数计算该翼型在俯仰周期内特定攻角下的阻力系数。",
                    "input": f"该翼型的雷诺数为：{Re}*10^6，粗糙度为：{R}，平均攻角为：{AOA_mean}°，攻角振幅为：{AOA_amp}°，"
                         f"振荡频率为：{f}Hz，衰减频率为：{f_red}Hz，攻角为：{AOA_high[i]}°，俯仰周期当前攻角变化趋势为：{Tr_high[i]}。"
                         f"翼型CST拟合参数为：{CST_Cordinate}。",
                    "output": f"该翼型在攻角为{AOA_high[i]}°时的阻力系数为：{Cd_high[i]}。"
                })

            json1_path = f"{Dataset_Path}\{file_name}_{number}_{data_name}_{AOA_mean}_{AOA_amp}_{f}_Low_CL.json"
            json2_path = f"{Dataset_Path}\{file_name}_{number}_{data_name}_{AOA_mean}_{AOA_amp}_{f}_High_CL.json"
            json3_path = f"{Dataset_Path}\{file_name}_{number}_{data_name}_{AOA_mean}_{AOA_amp}_{f}_Low_CD.json"
            json4_path = f"{Dataset_Path}\{file_name}_{number}_{data_name}_{AOA_mean}_{AOA_amp}_{f}_High_CD.json"
            with open(json1_path, "w", encoding="utf-8") as json_file:
                json.dump(content1, json_file, ensure_ascii=False, indent=4)
            with open(json2_path, "w", encoding="utf-8") as json_file:
                json.dump(content2, json_file, ensure_ascii=False, indent=4)
            with open(json3_path, "w", encoding="utf-8") as json_file:
                json.dump(content3, json_file, ensure_ascii=False, indent=4)
            with open(json4_path, "w", encoding="utf-8") as json_file:
                json.dump(content4, json_file, ensure_ascii=False, indent=4)
