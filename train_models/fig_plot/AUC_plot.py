import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
# file_path = 'perbase_train_predict/fig_plot/data/sns_plot.xls'
file_path = '/Users/jarrycf/Desktop/BEtools2.0/perbase_train_predict/fig_plot/data/sns_plot.xls'
data = pd.read_excel(file_path, sheet_name='AUC')
# data = data.to_numpy()

# 提取数据
false_positive_rate = data['False-Pos.-Rate']
true_positive_rate = data['True-Pos.-Rate']

# 设置图形大小
plt.figure(figsize=(8, 6))

# 绘制Mean ROC曲线
plt.plot(false_positive_rate, true_positive_rate, marker='o', markersize=4, color='brown', label='Mean ROC')

# 绘制Optimal cutoff点
optimal_cutoff = data.iloc[39]
plt.scatter(optimal_cutoff['False-Pos.-Rate'], optimal_cutoff['True-Pos.-Rate'], marker='v', s=50, color='gray', label='Optimal cutoff')

# 绘制Chance线段
plt.plot([0, 1], [0, 1], linestyle='dashed', color='silver', label='Chance')

# 设置坐标轴范围和刻度
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 添加标题和标签
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('ABEmax True Positive Rate')

# 添加图例
plt.legend(loc='lower right')

# # 添加左侧文字
# plt.text(-0.12, 0.5, 'ABEmax True Positive Rate', rotation=90, fontsize=12, va='center')
# plt.text(0.5, -0.12, 'False Positive Rate', fontsize=12, ha='center')

# 显示边框
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# 添加十字辅助线和置信度标签
plt.axvline(optimal_cutoff['False-Pos.-Rate'], color='gray', linestyle='--')
plt.axhline(optimal_cutoff['True-Pos.-Rate'], color='gray', linestyle='--')
plt.text(optimal_cutoff['False-Pos.-Rate'], optimal_cutoff['True-Pos.-Rate'], f"AUC = {optimal_cutoff['True-Pos.-Rate']:.4f} ± {optimal_cutoff['False-Pos.-Rate']:.4f}", ha='right', va='top')

# 调整图形布局
# plt.tight_layout()

# plt.savefig('fig_plot/pic/AUC_pic.png')
plt.savefig('perbase_train_predict/fig_plot/pic')

# 展示图形
plt.show()
