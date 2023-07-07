# https://blog.csdn.net/ztf312/article/details/102474190
# https://seaborn.pydata.org/generated/seaborn.FacetGrid.html

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rc('font',family='Arial Unicode MS',size=22)

# 读取csv文件
file_path = 'sample_data/source_data_sample/sns_plot.xls'
data = pd.read_excel(file_path, sheet_name="heatmap")

# 选择数据的范围
data = data.iloc[1:11, :]

# 使用seaborn的heatmap函数绘制热力图
fig, ax = plt.subplots(figsize=(20, 9))  # 创建图像和坐标轴对象
# heatmap = sns.heatmap(data, cmap='YlGnBu_r', cbar_kws={'label': '颜色变化'}, ax=ax)
heatmap = sns.heatmap(data, cmap='YlOrRd', cbar_kws={'label': '颜色变化'}, ax=ax)

# 在对角线位置显示字母'A'
for i in range(2, min(data.shape)+2): # 注意这里改为从 2 开始
    plt.text(i-0.5, i-1.5, 'A', ha='center', va='center', color='white', fontsize=20, fontweight='bold')

# 添加标题和坐标轴标签
plt.title('ABEmax', pad=20)
plt.xlabel('Position in spacer')
plt.ylabel('A-to-G conversion at position')

# 设置x轴的刻度为1到20
ax.set_xticks(np.arange(0.5, data.shape[1]+0.5))
ax.set_xticklabels(range(1, data.shape[1]+1))

# 设置y轴的刻度为2到11，并将标签显示在图的上方
ax.set_yticks(np.arange(0.5, data.shape[0]+0.5))
ax.set_yticklabels(range(2, data.shape[0]+2))
ax.xaxis.tick_top()  # 将x轴的标签移动到图的上方

plt.savefig('fig_plot/pic/heatmap_pic.png')

# 显示图像
plt.show()
