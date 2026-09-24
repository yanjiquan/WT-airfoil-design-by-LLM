import json
import os
import re
import sys

# 打印Python环境信息
print('Python解释器路径:', sys.executable)
print('Python版本:', sys.version)
print('当前工作目录:', os.getcwd())


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


def build_input_str(Re, roughness, aoa, cst):
    """
    构造静态 x1~x9 固定顺序的紧凑 JSON input 字符串（匿名化变量）
    A_u / A_l 作为单个变量（含9个系数的数组）输入
    静态无振荡参数，故不包含 AOA_mean/振幅/频率/衰减频率/trend
    """
    input_dict = {
        "x1": float(Re),
        "x2": int(roughness),
        "x3": float(aoa),
        "x4": cst["N1"],
        "x5": cst["N2"],
        "x6": cst["A_u"],
        "x7": cst["z_u_TE"],
        "x8": cst["A_l"],
        "x9": cst["z_l_TE"],
    }
    return json.dumps(input_dict, ensure_ascii=False, separators=(",", ":"))


### 生成匿名化变量（onlyVar）形式的静态 CL 数据集 ###

file_names = ["LS-0417", "LS-0421", "NACA4415", "S801", "S809", "S810", "S812", "S813", "S814", "S815", "S825"]

Dataset_Path = r"D:\AirfoilDesign\PythonPram\Dataset_all\Sta\onlyVar"
os.makedirs(Dataset_Path, exist_ok=True)

# 统计计数
airfoil_count = 0
data_files_total = 0
samples_total = 0

for file_name in file_names:
    file_path = f"D:\\AirfoilDesign\\Data\\{file_name}_CST.txt"
    folder_path = f"D:\\AirfoilDesign\\Data\\Static\\{file_name}"

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
        print(f"警告: 静态数据目录不存在 {folder_path}，跳过该翼型")
        continue

    airfoil_count += 1
    airfoil_files = 0
    airfoil_samples = 0

    # 遍历文件夹中以 C 或 U 开头的 txt 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if (file.startswith('C') or file.startswith('U')) and file.endswith('.txt'):
                # 提取文件名（不含扩展名）作为 data_name
                data_name = os.path.splitext(file)[0]
                Data_path = os.path.join(root, file)

                # 提取数字部分作为 Re 值（如 C0.75 -> 0.75）
                num_match = re.search(r'[0-9.]+', data_name)
                if num_match is None:
                    print(f"警告: 无法从文件名提取雷诺数 {data_name}，跳过该文件")
                    continue
                try:
                    Re = float(num_match.group())
                except Exception as e:
                    print(f"警告: 雷诺数解析失败 {data_name}: {e}，跳过该文件")
                    continue

                # 粗糙度：C=0，U=1（恢复文件名前缀区分）
                prefix = data_name[0]
                if prefix == "C":
                    roughness = 0
                elif prefix == "U":
                    roughness = 1
                else:
                    print(f"警告: 未知前缀 {prefix}（文件 {data_name}），跳过该文件")
                    continue

                AOA = []
                Cl = []
                Cd = []
                if os.path.isfile(Data_path):
                    try:
                        with open(Data_path, "r", encoding="utf-8") as f:
                            lines = [line.strip() for line in f.readlines()]
                    except Exception as e:
                        print(f"警告: 读取数据文件失败 {Data_path}: {e}，跳过该文件")
                        continue

                    for line in lines:
                        values = line.split()
                        if len(values) >= 3:
                            try:
                                AOA.append(float(values[0]))
                                Cl.append(float(values[1]))
                                Cd.append(float(values[2]))
                            except Exception as e:
                                print(f"警告: 数据行解析失败 {data_name} 行 '{line}': {e}，跳过该行")
                                continue

                if not AOA:
                    print(f"警告: 文件无有效数据 {data_name}，跳过该文件")
                    continue

                # 构造 onlyVar 形式的 CL Alpaca 数据（保持原始时间顺序，禁止重排/插值）
                content = []
                for i in range(len(AOA)):
                    input_str = build_input_str(Re, roughness, AOA[i], cst_dict)
                    content.append({
                        "instruction": "Predict CL",
                        "input": input_str,
                        "output": str(float(Cl[i])),
                    })

                # 文件命名保持原程序逻辑，仅生成 CL
                json_path = os.path.join(Dataset_Path, f"{file_name}_{data_name}_CL.json")
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(content, json_file, ensure_ascii=False, indent=4)

                data_files_total += 1
                airfoil_files += 1
                samples_total += len(content)
                airfoil_samples += len(content)

    print(f"翼型 {file_name} 处理完成: 数据文件={airfoil_files}, CL样本={airfoil_samples}")

# 最终统计
print("\n========== 处理完成 ==========")
print(f"处理的翼型数: {airfoil_count}")
print(f"处理的数据文件数: {data_files_total}")
print(f"CL 样本总数: {samples_total}")
print(f"输出目录: {Dataset_Path}")
