import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

file_path = '/Users/jarrycf/Desktop/BEtools2.0/perbase_train_predict/00 stage/ABEmax_新版Transformer/train_val/run_2/predictions_validation.csv'

data = pd.read_csv(file_path)
true_class = data['true_class']
pred_class = data['pred_class']
print(true_class)


# 计算ROC曲线
fpr, tpr, _ = roc_curve(true_class, pred_class)
roc_auc = auc(fpr, tpr)

# 设置图形大小
plt.figure(figsize=(8, 6))

# 绘制ROC曲线
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Mean ROC')

# 绘制Chance线段
plt.plot([0, 1], [0, 1], linestyle='dashed', color='silver', label='Chance')

# 设置坐标轴范围和刻度
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 添加标题和标签
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# 添加图例
plt.legend(loc='lower right')

# 显示边框
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# 显示AUC值
plt.text(0.6, 0.2, f"AUC = {roc_auc:.2f}", ha='right', va='top')

# 调整图形布局
# plt.tight_layout()

# 保存图形
plt.savefig('perbase_train_predict/fig_plot/pic')

# 展示图形
plt.show()
