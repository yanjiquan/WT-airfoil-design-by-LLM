import os
import json

# 文件夹路径
folder_path = r"D:\AirfoilDesign\PythonPram\Dataset_all\Sta\onlyVar"
# 输出文件
output_train_cl = r"D:\AirfoilDesign\PythonPram\Dataset_all\Sta\onlyVar\train_CL.jsonl"
output_train_cd = r"D:\AirfoilDesign\PythonPram\Dataset_all\Sta\onlyVar\train_CD.jsonl"

merged_train_cl = []
merged_train_cd = []

# 为每个测试集翼型创建单独的数据列表
merged_test_data = {}

# 指定测试集翼型名称
test_airfoils = ["NACA4415"]

# 初始化测试集数据结构
for airfoil in test_airfoils:
    merged_test_data[airfoil] = {
        "CL": [],
        "CD": []
    }

# 遍历文件夹里的所有 JSON 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # 提取翼型名称和类型（CL/CD）
            airfoil_name = filename.split("_")[0]
            data_type = filename.split("_")[-1].split(".")[0]

            # 判断是测试集还是训练集
            is_test = any(airfoil in airfoil_name for airfoil in test_airfoils)

            # 根据类型选择目标列表
            if is_test:
                # 测试集：按翼型和类型分类
                if airfoil_name in merged_test_data:
                    target_list = merged_test_data[airfoil_name][data_type]
                else:
                    # 如果翼型不在测试集列表中，跳过
                    continue
            else:
                # 训练集：合并到统一列表
                if data_type == "CL":
                    target_list = merged_train_cl
                else:  # CD
                    target_list = merged_train_cd

            # 确保每个元素是字典
            if isinstance(data, dict):
                target_list.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        target_list.append(item)
                    elif isinstance(item, list):
                        # 如果是嵌套列表，包装成字典
                        target_list.append({"data": item})
                    else:
                        # 其他类型也包装成字典
                        target_list.append({"value": item})
            else:
                # 其他类型直接包装成字典
                target_list.append({"value": data})

# 写入 jsonlines 格式（每行一个 JSON 对象）
def write_jsonlines(file_path, data_list):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 写入训练集
write_jsonlines(output_train_cl, merged_train_cl)
write_jsonlines(output_train_cd, merged_train_cd)

# 写入测试集（按翼型单独保存）
test_files = []
for airfoil, data_types in merged_test_data.items():
    for data_type, data_list in data_types.items():
        output_file = os.path.join(folder_path, f"{airfoil}_{data_type}.jsonl")
        write_jsonlines(output_file, data_list)
        test_files.append(output_file)

print("合并完成，训练集和测试集已生成（jsonlines 格式）")
print(f"训练集 CL: {len(merged_train_cl)} 条数据")
print(f"训练集 CD: {len(merged_train_cd)} 条数据")
print("测试集文件:")
for file in test_files:
    print(f"- {os.path.basename(file)}")


