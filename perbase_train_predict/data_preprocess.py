import pandas as pd

def get_char(seq):
    """将字符串拆分为字符序列，并以 pandas.Series 返回"""
    chars = list(seq)
    return pd.Series(chars)


def generate_clean_df(df):
    """
    清理CRISPR实验中的序列, 将数据添加以下3列: 
    efficiency_score (编辑效率), edited_seq_categ(0:会被编辑 1:未被编辑) seq_type(0:训练集 1:验证集)
    Args:
        df: pandas.DataFrame
    注意:
        传入的df中的字段需重命名为: [ID, seq, allCounts, V1, V2, ..., V20]
    """
    
    # 筛选csv中需要用到的列
    editbase_cols = [f'V{i}' for i in range(1,21)]
    target_cols = ['ID', 'seq', 'allCounts'] + editbase_cols
    df = df[target_cols].copy()

    # 移除 'allCounts' 为 NaN 的行
    df = df[df['allCounts'].notna()].copy() 

    # 将序列字符串表示标准化为大写形式
    df['seq'] = df['seq'].str.upper()

    # 计算效率得分 每个碱基的被编辑概率之和除以总编辑数 13.46/4704 = 0.002861
    df['efficiency_score'] = df[editbase_cols].sum(axis=1)/df['allCounts'] 

    # 效率得分>0的用1标记(被编辑了) 0表示未被编辑
    df['edited_seq_categ'] = 0
    df.loc[df['efficiency_score'] > 0, 'edited_seq_categ'] = 1 

    # 将'ID'列包含"RANDOM"字符的序列分配为训练集（seq_type为0）1表示测试集
    df['seq_type'] = 1
    df.loc[df['ID'].str.startswith("RANDOM"), 'seq_type'] = 0 # ID中包含RANDOM的是训练集 0 , 反正为验证集
    
    return df
    

def generate_perbase_df(df, target_base):
    """
    获取每个碱基对应的编辑信息
    Args:
        df: pandas.DataFrame (由`generate_clean_df`生成的数据)
        target_base: string 待编辑的目标碱基（如 'A' 或 'C'）
    """
    
    # 将字符串拆分为字符序列
    baseseq_df = df['seq'].apply(get_char)

    # 将拆分后的字符序列,列名L{i}来表示
    baseseq_letters_df = baseseq_df.copy()
    baseseq_letters_df.columns = [f'L{i}' for  i in range(1, 21)]

    # 用0来掩码不需要转换的基因序列, 并将用列名M{i}来表示
    base_mask = (baseseq_df == target_base) * 1
    base_mask.columns = [f'M{i}' for  i in range(1, 21)]
    
    # 对ACTG分别用0123编码,列名B{i}来表示
    baseseq_df.columns = [f'B{i}' for  i in range(1, 21)]
    baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)

    # 以 M{i}, 'ID', 'seq_type', 'efficiency_score', 'edited_seq_categ', V{i}, L{i}, B{i} 的顺序按列拼接
    editbase_cols = [f'V{i}' for i in range(1,21)]
    base_df = pd.concat([base_mask,
                         df[['ID', 'seq_type', 'efficiency_score', 'edited_seq_categ'] + editbase_cols],
                         baseseq_letters_df,
                         baseseq_df], axis=1)
        
    return base_df
    