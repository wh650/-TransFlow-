

import os
os.chdir('./data')
import pandas as pd

# 读取原始文件
df = pd.read_csv('selected_proteins5000_400.csv')

# 均匀打乱排序
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 重新输出为新文件
df_shuffled.to_csv('selected_proteins5000_400stuff.csv', index=False)

