# import pandas as pd

# # 定义文件路径
# file_path = "/Users/jarrycf/Desktop/00 stage/BEtools_gpt/sample_data/source_data_sample/M2_ESM.xlsx"

# # 读取Excel文件中的所有表格
# excel_data = pd.read_excel(file_path, sheet_name=None)

# # 遍历每个表格并进行操作
# for sheet_name, sheet_data in excel_data.items():
#     # 在这里对每个表格进行处理，例如打印表格名和内容
#     print("表格名:", sheet_name)
#     print(sheet_data)

#     # 如果要进一步处理表格数据，可以使用pandas提供的各种功能和方法
#     # 例如，可以使用sheet_data中的列和行来访问和操作数据



# import pandas as pd

# # 定义文件路径
# file_path = "/Users/jarrycf/Desktop/BEtools2.0/sample_data/source_data_sample/M2_ESM.xlsx"

# # 读取名为"ABEmax_perbase"的表格
# df = pd.read_excel(file_path, sheet_name="ABEmax_perbase")
# # df = df[df["Purpose"].isin(["Validation/Test"])] # 2050
# df = df[df["Purpose"].isin(["Train"])] #10786 

# # 打印表格内容
# print(df)

# a = 1, 3
# print(a)

import torch
import torch.nn as nn

a = nn.Parameter(torch.randn(64, dtype=torch.float32), requires_grad=True)
print(a)