import json
import os
import pandas as pd
import re

# 指定文件路径
file_names = ["LS-0417", "LS-0421", "NACA4415", "S801", "S809", "S810", "S812", "S813", "S814", "S815", "S825"]

for file_name in file_names:
    file_path = f"D:\AirfoilDesign\Data\{file_name}_CST.txt"
    folder_path = f"D:\AirfoilDesign\Data\Static\{file_name}"   # e.g. D:\AirfoilDesign\PythonPram\Predict_Data\S814
    csv_arrays = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        CST_Cordinate = file.read()

    # 遍历文件夹中以C或U开头的txt文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否以C或U开头，并且是txt文件
            if (file.startswith('C') or file.startswith('U')) and file.endswith('.txt'):
                # 提取文件名（不含扩展名）作为data_name
                data_name = os.path.splitext(file)[0]
                json1_path = f"D:\\AirfoilDesign\\PythonPram\\Dataset_all\\Sta\\CST_nohis\\{file_name}_{data_name}_CL.json"
                json2_path = f"D:\\AirfoilDesign\\PythonPram\\Dataset_all\\Sta\\CST_nohis\\{file_name}_{data_name}_CD.json"
                Data_path = os.path.join(folder_path, file)

                AOA = []
                Cl = []
                Cd = []
                if os.path.isfile(Data_path):
                    with open(Data_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f.readlines()]

                        for line in lines:
                            values = line.split()
                            if len(values) >= 3:
                                AOA.append(float(values[0]))
                                Cl.append(float(values[1]))
                                Cd.append(float(values[2]))

                # 提取数字部分作为R值
                R = float(re.search(r'[0-9.]+', data_name).group())
                content1 = []
                content2 = []

                for i, value in enumerate(AOA):
                    content1.append({
                        "instruction":"你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                                    "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度和攻角计算出它可能的升阻力系数。"
                                    f"请你根据下面输入的翼型坐标参数，雷诺数、粗糙度和攻角计算该翼型的升力系数。",
                        "input":f"该翼型的雷诺数为{R}*10^6，粗糙度为光滑，攻角为{AOA[i]}°。"
                                f"翼型CST拟合参数为：{CST_Cordinate}。",
                        "output":f"该翼型在攻角为{AOA[i]}°时的升力系数为：{Cl[i]}。"
                    })
                    content2.append({
                        "instruction":"你是一位风力发电领域的风力发电机叶片翼型设计专家，你熟悉有关风力发电机叶片翼型设计领域的知识，流程和问题。"
                                    "你可以根据给出的风电机组叶片翼型CST拟合参数判断该叶片的翼型形状，并结合给出的雷诺数、粗糙度和攻角计算出它可能的升阻力系数。"
                                    f"请你根据下面输入的翼型坐标参数，雷诺数、粗糙度和攻角计算该翼型的阻力系数。",
                        "input":f"该翼型的雷诺数为{R}*10^6，粗糙度为光滑，攻角为{AOA[i]}°。"
                                f"翼型CST拟合参数为：{CST_Cordinate}。",
                        "output":f"该翼型在攻角为{AOA[i]}°时的阻力系数为：{Cd[i]}。"
                    })



                with open(json1_path, "w", encoding="utf-8") as json_file:
                    json.dump(content1, json_file, ensure_ascii=False, indent=4)
                with open(json2_path, "w", encoding="utf-8") as json_file:
                    json.dump(content2, json_file, ensure_ascii=False, indent=4)
