import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

file_path = '/Users/jarrycf/Desktop/BEtools2.0/perbase_train_predict/00 stage/ABEmax_新版Transformer/train_val/run_2/predictions_validation.csv'

data = pd.read_csv(file_path)
true_class = data['true_class']
pred_class = data['pred_class']

# 计算ROC曲线
fpr, tpr, _ = roc_curve(true_class, pred_class)

print(fpr)