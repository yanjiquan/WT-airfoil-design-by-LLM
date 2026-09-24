import json
import os
import re
import sys
import numpy as np

# 打印Python环境信息
print('Python解释器路径:', sys.executable)
print('Python版本:', sys.version)
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
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
        print("scipy安装成功，重新导入...")
        from scipy.io import loadmat
        scipy_available = True
        print('scipy重新导入成功')
    except Exception as install_error:
        print(f"安装scipy失败: {install_error}")
        print("请手动运行 'pip install scipy' 来安装")
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
    except Exception as e:
        print(f"警告: 读取 mat 文件失败 {mat_path}: {e}")
        return None


def parse_cst(cst_text):
    """
    从 CST 文本中解析数值：
    N1, N2, A_u(9个), z_u_TE, A_l(9个), z_l_TE
    兼容中英文逗号/分号、负号、小数、科学计数法、空格
    返回 (dict, err) 其中 err 为 None 表示成功
    """
    num_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

    n1_match = re.search(r'N1\s*=\s*(' + num_pattern + r')', cst_text)
    n2_match = re.search(r'N2\s*=\s*(' + num_pattern + r')', cst_text)
    au_match = re.search(r'A_u\s*=\s*\[([^\]]*)\]', cst_text)
    al_match = re.search(r'A_l\s*=\s*\[([^\]]*)\]', cst_text)
    zu_match = re.search(r'z_u_TE\s*=\s*(' + num_pattern + r')', cst_text)
    zl_match = re.search(r'z_l_TE\s*=\s*(' + num_pattern + r')', cst_text)

    if not (n1_match and n2_match and au_match and al_match and zu_match and zl_match):
        return None, "CST 文本缺少必要字段（N1/N2/A_u/A_l/z_u_TE/z_l_TE）"

    try:
        N1 = float(n1_match.group(1))
        N2 = float(n2_match.group(1))
        z_u_TE = float(zu_match.group(1))
        z_l_TE = float(zl_match.group(1))
    except Exception as e:
        return None, f"CST 数值解析失败: {e}"

    # 数组内分隔符兼容中英文逗号
    au_vals = [float(x.strip()) for x in re.split(r'[,，]', au_match.group(1)) if x.strip() != '']
    al_vals = [float(x.strip()) for x in re.split(r'[,，]', al_match.group(1)) if x.strip() != '']

    if len(au_vals) != 9:
        return None, f"A_u 系数数量为 {len(au_vals)}，应为 9，跳过该翼型"
    if len(al_vals) != 9:
        return None, f"A_l 系数数量为 {len(al_vals)}，应为 9，跳过该翼型"

    return {
        "N1": N1,
        "N2": N2,
        "A_u": au_vals,
        "z_u_TE": z_u_TE,
        "A_l": al_vals,
        "z_l_TE": z_l_TE,
    }, None


def build_input_str(Re, roughness, AOA_mean, AOA_amp, f, f_red, aoa, trend, cst):
    """
    构造 x1~x14 固定顺序的紧凑 JSON input 字符串（匿名化变量）
    A_u / A_l 作为单个变量（含9个系数的数组）输入
    """
    input_dict = {
        "x1": float(Re),
        "x2": int(roughness),
        "x3": int(AOA_mean),
        "x4": int(AOA_amp),
        "x5": float(f),
        "x6": float(f_red),
        "x7": float(aoa),
        "x8": int(trend),
        "x9": cst["N1"],
        "x10": cst["N2"],
        "x11": cst["A_u"],
        "x12": cst["z_u_TE"],
        "x13": cst["A_l"],
        "x14": cst["z_l_TE"],
    }
    return json.dumps(input_dict, ensure_ascii=False, separators=(",", ":"))


### 生成匿名化变量（onlyVar）形式的动态 CL 数据集 ###

# Low / High 统一阈值
AOA_THRESHOLD = 14.0

file_names = ["LS-0417", "LS-0421", "NACA4415", "S801", "S809", "S810", "S812", "S813", "S814", "S815", "S825"]

Dataset_Path = r"D:\AirfoilDesign\PythonPram\Dataset_all\Dyn\onlyVar"
os.makedirs(Dataset_Path, exist_ok=True)

# 统计计数
airfoil_count = 0
mat_files_total = 0
low_samples_total = 0
high_samples_total = 0

for file_name in file_names:
    file_path = f"D:\\AirfoilDesign\\Data\\{file_name}_CST.txt"
    folder_path = f"D:\\AirfoilDesign\\Data\\Dynamic\\{file_name}"

    # 读取并解析 CST
    if not os.path.isfile(file_path):
        print(f"警告: CST 文件不存在 {file_path}，跳过该翼型")
        continue
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            CST_Cordinate = file.read()
    except Exception as e:
        print(f"警告: 读取 CST 文件失败 {file_path}: {e}，跳过该翼型")
        continue

    cst_dict, err = parse_cst(CST_Cordinate)
    if cst_dict is None:
        print(f"警告: 翼型 {file_name} 的 CST 解析失败: {err}，跳过该翼型")
        continue

    if not os.path.isdir(folder_path):
        print(f"警告: 动态数据目录不存在 {folder_path}，跳过该翼型")
        continue

    # 获取该翼型下所有工况子文件夹
    subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    if not subfolders:
        print(f"警告: 翼型 {file_name} 下无工况子文件夹，跳过该翼型")
        continue

    data_names = subfolders
    airfoil_count += 1
    airfoil_low = 0
    airfoil_high = 0
    airfoil_mat = 0

    for data_name in data_names:
        data_path = os.path.join("D:\\AirfoilDesign\\Data\\Dynamic", file_name, data_name)
        if not os.path.isdir(data_path):
            print(f"警告: 工况目录不存在 {data_path}，跳过该工况")
            continue

        # 解析工况文件夹名 C_RE_75 / G_RE_100
        parts_folder = data_name.split("_")
        if len(parts_folder) < 3:
            print(f"警告: 工况文件夹名格式非法 {data_name}，跳过该工况")
            continue
        try:
            Re = float(parts_folder[2]) / 100
        except Exception as e:
            print(f"警告: 工况文件夹名解析 Re 失败 {data_name}: {e}，跳过该工况")
            continue

        roughness_code = parts_folder[0]
        if roughness_code == "C":
            roughness = 0
        elif roughness_code == "G":
            roughness = 1
        else:
            print(f"警告: 未知粗糙度代码 {roughness_code}（工况 {data_name}），跳过该工况")
            continue

        mat_files = [f for f in os.listdir(data_path) if f.endswith(".mat")]
        if not mat_files:
            print(f"警告: 工况目录无 mat 文件 {data_path}，跳过该工况")
            continue

        for mat_file in mat_files:
            mat_path = os.path.join(data_path, mat_file)
            name_without_ext = os.path.splitext(mat_file)[0]
            parts = name_without_ext.split("_")
            if len(parts) < 5:
                print(f"警告: MAT 文件名格式非法 {mat_file}，跳过该文件")
                continue
            try:
                number = parts[0]
                AOA_mean = int(parts[1])
                AOA_amp = int(parts[2])
                f = float(parts[3][1:])
                f_red = float(parts[4][1:])
            except Exception as e:
                print(f"警告: MAT 文件名解析失败 {mat_file}: {e}，跳过该文件")
                continue

            # 读取所有 mat 文件，不再限制参数
            data_array = read_mat_file_no_name(mat_path)

            # 检查是否成功读取文件
            if data_array is None:
                continue

            data_array = np.array(data_array)
            if data_array.ndim != 2 or data_array.shape[1] < 3:
                print(f"警告: MAT 数据维度异常 {mat_file}（shape={data_array.shape}），跳过该文件")
                continue

            # 保证 n x 3
            if data_array.shape[0] < data_array.shape[1]:
                data_array = data_array.T

            if data_array.shape[1] < 3:
                print(f"警告: MAT 数据列数不足 {mat_file}（shape={data_array.shape}），跳过该文件")
                continue

            AOA = data_array[:, 0].flatten().astype(float)  # 第一列
            Cl = data_array[:, 1].flatten().astype(float)   # 第二列
            Cd = data_array[:, 2].flatten().astype(float)   # 第三列（读取但不生成 CD JSON）

            # 上升/下降标识：1=上升段，-1=下降段（保持原始时间顺序，禁止重排）
            max_AOA_index = int(np.argmax(AOA))

            content_low = []
            content_high = []

            for i, (aoa_val, cl_val, cd_val) in enumerate(zip(AOA, Cl, Cd)):
                trend = 1 if i <= max_AOA_index else -1

                input_str = build_input_str(Re, roughness, AOA_mean, AOA_amp, f, f_red, aoa_val, trend, cst_dict)

                sample = {
                    "instruction": "Predict CL",
                    "input": input_str,
                    "output": str(float(cl_val)),
                }

                # 统一 14° 阈值划分 Low / High
                if aoa_val >= AOA_THRESHOLD:
                    content_high.append(sample)
                else:
                    content_low.append(sample)

            # 文件命名保持原程序逻辑，仅生成 CL
            json_low_path = os.path.join(Dataset_Path, f"{file_name}_{number}_{data_name}_{AOA_mean}_{AOA_amp}_{f}_Low_CL.json")
            json_high_path = os.path.join(Dataset_Path, f"{file_name}_{number}_{data_name}_{AOA_mean}_{AOA_amp}_{f}_High_CL.json")

            with open(json_low_path, "w", encoding="utf-8") as json_file:
                json.dump(content_low, json_file, ensure_ascii=False, indent=4)
            with open(json_high_path, "w", encoding="utf-8") as json_file:
                json.dump(content_high, json_file, ensure_ascii=False, indent=4)

            mat_files_total += 1
            airfoil_mat += 1
            low_samples_total += len(content_low)
            high_samples_total += len(content_high)
            airfoil_low += len(content_low)
            airfoil_high += len(content_high)

    print(f"翼型 {file_name} 处理完成: MAT={airfoil_mat}, Low={airfoil_low}, High={airfoil_high}")

# 最终统计
print("\n========== 处理完成 ==========")
print(f"处理的翼型数: {airfoil_count}")
print(f"处理的 MAT 文件数: {mat_files_total}")
print(f"Low 样本总数: {low_samples_total}")
print(f"High 样本总数: {high_samples_total}")
print(f"输出目录: {Dataset_Path}")
