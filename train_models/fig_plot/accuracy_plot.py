import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
'''
pip install pandas==1.2.5  
'''

# 读取数据
#
file_path = 'perbase_train_predict/fig_plot/data/sns_plot.xls'
data = pd.read_excel(file_path, sheet_name="accuracy")
# data = data.to_numpy()[:, None]

# 筛选数据
df1 = data[data['Model Name'] == 'ABEmax'].reset_index(drop=True)
df2 = data[data['Model Name'] == 'Baseline_ABEmax'].reset_index(drop=True)

# 设置图形的大小
fig, ax = plt.subplots(figsize=(10, 6))

# 使用seaborn绘制图形，不同的数据使用不同的颜色，并且设置标签用于图例的显示，设置线宽使线更粗
# ax.plot(df1['Base position'], df1['Accuracy'])
sns.lineplot(data=df1, x='Base position', y='Accuracy', color='red', linewidth=2.5, label='ABE8e model', ax=ax)
# sns.lineplot(data=df2, x='Base position', y='Accuracy', color='orange', label='ABEmax', ax=ax)
sns.lineplot(data=df2, x='Base position', y='Accuracy', color='orange', label='ABEmax', ax=ax, ci=None)

# 设置坐标轴只显示一位小数
formatter = FuncFormatter(lambda y, _: '{:.1f}'.format(y))
ax.yaxis.set_major_formatter(formatter)

# 设置标题和轴标签为中文，移动标题到左边
# plt.title('基于位置的准确率', loc='left')
plt.xlabel('碱基位置', fontsize=22)
plt.ylabel('')

# 设置x轴和y轴的刻度
plt.xticks(ticks=range(1, 21))  # 设置x轴的刻度为1到20
plt.yticks(ticks=[i/10 for i in range(5, 11)])  # 设置y轴的刻度为0.5到1，分成6个刻度

# 显示图例，可以通过loc参数设置图例的位置
plt.legend(loc='lower right')

# # 添加倒立的文字，参数分别为：x坐标、y坐标、文本、旋转角度、字体大小
# plt.text(20, 0.5, 'ABE8e\n准确率', rotation=-90, fontsize=12)

# 获取x轴的最小值
x_min = df1['Base position'].min()

# 添加倒立的文字，参数分别为：x坐标、y坐标、文本、旋转角度、字体大小
fig.text(0.01, 0.45, 'ABE8e\n准确率', rotation=90, fontsize=22)

plt.savefig('perbase_train_predict/fig_plot/pic')

# 显示图形
plt.show()
