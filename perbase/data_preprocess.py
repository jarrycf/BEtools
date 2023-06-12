import pandas as pd
import numpy as np
from scipy import stats

# 将每个连接序列扩展为其碱基的函数
def get_char(seq):
    """将字符串拆分为由字符组成的序列，并返回 pandas.Series"""
    chars = list(seq)
    return pd.Series(chars)
    
def process_perbase_df(df, target_base):
    """
    清理表示来自CRISPR实验的序列及其编辑信息的数据框
    参数:
        df: pandas.DataFrame
        cutoff_score: float，从编辑得分创建两个类别的阈值
        
    注意:
        假设数据框中的列为：[ID, seq, allCounts, V1, V2, ..., V20]

    """

    target_cols = ['ID', 'seq']
    df = df[target_cols].copy() # 0  CTRL_HEKsiteNO18  ACACACACACTTAGAATCTG
 

    # 统一序列的字符串表示形式为大写形式
    df['seq'] = df['seq'].str.upper()
    # 将序列分配为训练集和验证/测试集
    df['seq_type'] = 1 # 0   CTRL_HEKsiteNO18  ACACACACACTTAGAATCTG  1

    baseseq_df = df['seq'].apply(get_char)
    '''
       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
    0   A  C  A  C  A  C  A  C  A  C  T  T  A  G  A  A  T  C  T  G
    '''
    baseseq_df.columns = [f'B{i}' for  i in range(1, 21)]
    '''
       B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20
    0   A  C  A  C  A  C  A  C  A   C   T   T   A   G   A   A   T   C   T   G
    '''
    base_mask = (baseseq_df == target_base) * 1 # * 1是将布尔值转换为由0和1组成的整数型
    base_mask.columns = [f'M{i}' for  i in range(1, 21)]
    '''
    # 标记所有为A的位置
        M1  M2  M3  M4  M5  M6  M7  M8  M9  M10  M11  M12  M13  M14  M15  M16  M17  M18  M19  M20
    0    1   0   1   0   1   0   1   0   1    0    0    0    1    0    1    1    0    0    0    0
    '''
    baseseq_letters_df = baseseq_df.copy()
    baseseq_letters_df.columns = [f'L{i}' for  i in range(1, 21)]
    '''
       L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13 L14 L15 L16 L17 L18 L19 L20
    0   A  C  A  C  A  C  A  C  A   C   T   T   A   G   A   A   T   C   T   G
    '''

    # 用数字替换碱基字母
    baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
    '''
        B1  B2  B3  B4  B5  B6  B7  B8  B9  B10  B11  B12  B13  B14  B15  B16  B17  B18  B19  B20
    0    0   1   0   1   0   1   0   1   0    1    2    2    0    3    0    0    2    1    2    3
    '''
    base_df = pd.concat([base_mask,
                         df,
                         baseseq_letters_df,
                         baseseq_df], axis=1)
    '''
        M1	M20	ID	                seq	                seq_type    L1	L19	L20	B1	B19	B20
    0   1	0	CTRL_HEKsiteNO18	ACACACACACTTAGAATCTG	1	    A	T	G	0	2	3
    1   1	1	CTRL_RSF1NO2	    ACCCATTAAAGTTGAGGTGA	1	    A	G	A	0	3	0
    '''

    # 保留具有目标碱基的序列
    # M1..M20 是目标碱基出现的掩码
    base_df = base_df[(base_df[[f'M{i}' for  i in range(1, 21)]].sum(axis=1) != 0)].copy()
    # 过滤掉 M1 到 M20 的列中所有元素都为 0 的行
    base_df.reset_index(inplace=True) # 重置索引, 因为可能过滤一些行
    
    return base_df

def validate_df(df):
    mask_cols = [f'M{i}' for  i in range(1, 21)]
    print('NA的数量:', df.isna().any().sum())
    print('没有目标碱基的序列数量:', (df[mask_cols].sum(axis=1) == 0).sum())

def compute_rank(s, eff_scores=None):
    return stats.percentileofscore(eff_scores, s, kind='weak')
