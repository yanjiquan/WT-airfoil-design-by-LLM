import os
import json

# 文件夹路径
folder_path = "D:\AirfoilDesign\PythonPram\Dataset_all\Dyn\onlyVar"
# 输出文件路径
output_low_cl = os.path.join(folder_path, "train_low_cl.json")
output_high_cl = os.path.join(folder_path, "train_high_cl.json")
output_low_cd = os.path.join(folder_path, "train_low_cd.json")
output_high_cd = os.path.join(folder_path, "train_high_cd.json")

# 测试集关键词（部分匹配即可）
test_files = ["NACA4415"]
Num = 1


# 初始化存储
merged_train_low_cl = []
merged_train_high_cl = []
merged_train_low_cd = []
merged_train_high_cd = []
merged_tests = {
    name: {
        "Low_CL": [],
        "High_CL": [],
        "Low_CD": [],
        "High_CD": []
    } for name in test_files
}



# 记录分类情况
train_file_list = []
test_file_list = {name: [] for name in test_files}

# 遍历文件夹里的所有 JSON 文件
for filename in os.listdir(folder_path):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(folder_path, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析失败，跳过: {filename}，错误: {e}")
        continue
    except Exception as e:
        print(f"⚠️ 读取文件失败，跳过: {filename}，错误: {e}")
        continue

    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")

    # 获取文件类型信息
    if len(parts) < 2:
        print(f"⚠️ 文件名格式不正确，跳过: {filename}")
        continue

    # 查找High或Low位置
    LH = None
    for i, part in enumerate(parts):
        if part in ("High", "Low"):
            LH = part
            # CD/CL在High/Low后面
            if i + 1 < len(parts):
                cd_cl = parts[i + 1]
            else:
                cd_cl = None
            break

    if not LH:
        print(f"⚠️ 未找到High/Low，跳过: {filename}")
        continue

    if not cd_cl or cd_cl not in ("CD", "CL"):
        print(f"⚠️ 非 CD/CL 文件，跳过: {filename}")
        continue

    if LH not in ("Low", "High"):
        print(f"⚠️ 非 Low/High 文件，跳过: {filename}")
        continue

    if cd_cl not in ("CD", "CL"):
        print(f"⚠️ 非 CD/CL 文件，跳过: {filename}")
        continue

    # ========= 1. 判断是否测试集 =========
    matched_test = None
    for test_name in test_files:
        if test_name in filename:
            matched_test = test_name
            break

    # ========= 2. 测试集 =========
    if matched_test:
        key = f"{LH}_{cd_cl}"
        if key in merged_tests[matched_test]:
            target_list = merged_tests[matched_test][key]
            test_file_list[matched_test].append(filename)
        else:
            print(f"⚠️ 测试集键不存在，跳过: {filename}")
            continue

    # ========= 3. 训练集 =========
    else:
        if LH == "Low" and cd_cl == "CL":
            target_list = merged_train_low_cl
        elif LH == "High" and cd_cl == "CL":
            target_list = merged_train_high_cl
        elif LH == "Low" and cd_cl == "CD":
            target_list = merged_train_low_cd
        elif LH == "High" and cd_cl == "CD":
            target_list = merged_train_high_cd
        else:
            print(f"⚠️ 未知文件类型，跳过: {filename}")
            continue

        train_file_list.append(filename)

    # ========= 4. 写入数据 =========
    if isinstance(data, list):
        for item in data:
            target_list.append(item)
    elif isinstance(data, dict):
        target_list.append(data)
    else:
        target_list.append({"value": data})


# 写入 jsonlines 格式
def write_jsonlines(file_path, data_list):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            for i in range(Num):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 写出训练集
write_jsonlines(output_low_cl, merged_train_low_cl)
write_jsonlines(output_high_cl, merged_train_high_cl)
write_jsonlines(output_low_cd, merged_train_low_cd)
write_jsonlines(output_high_cd, merged_train_high_cd)

# 写出每个测试集
for test_name, data_dict in merged_tests.items():
    for key, data_list in data_dict.items():
        if data_list:
            output_test = os.path.join(
                folder_path, f"test_{test_name}_{key.lower()}.json"
            )
            write_jsonlines(output_test, data_list)

print("\n合并完成！")
print(f"训练集文件数: {len(train_file_list)}")
for test_name, files in test_file_list.items():
    print(f"测试集 {test_name} 文件数: {len(files)}")
print(f"生成的训练集文件:")
print(f"  - {os.path.basename(output_low_cl)}")
print(f"  - {os.path.basename(output_high_cl)}")
print(f"  - {os.path.basename(output_low_cd)}")
print(f"  - {os.path.basename(output_high_cd)}")
