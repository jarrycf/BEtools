import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = 'perbase_train_predict/fig_plot/data/sns_plot.xls'
data = pd.read_excel(file_path, sheet_name='bar')


# 提取数据
motifs = data['Motif']
biorep_1 = data['Biorep_1']
biorep_2 = data['Biorep_2']
mean = data['mean']
std = data['SD']

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制第一组柱子（Biorep_1）
plt.bar(motifs, biorep_1, color='brown', label='Biorep_1')

# 绘制第二组柱子（Biorep_2）
# plt.bar(motifs, biorep_2, color='orange', label='Biorep_2')

# 绘制误差线
plt.errorbar(motifs, mean, yerr=[std, std], fmt='o', color='black', capsize=3)

# 设置y轴刻度
plt.yticks([0, 5, 10, 15])

# 添加标题和标签
# plt.title('Biorep_1 vs Biorep_2')
# plt.xlabel('Motif')
plt.ylabel('Proportion (%)')

# 移除图例
plt.legend().remove()

# 设置x轴刻度标签旋转角度和字体大小
plt.xticks(rotation='vertical', fontsize=8)

# 移除y轴刻度的辅助线
plt.gca().yaxis.grid(False)

# # 调整图形边距
# plt.tight_layout()

plt.savefig('perbase_train_predict/fig_plot/pic')

# 展示图形
plt.show()
