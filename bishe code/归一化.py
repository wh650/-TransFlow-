import os
os.chdir('./data')


import pandas as pd

# 读取Excel文件
data = pd.read_excel('test2.xlsx', header=None)

# 定义数据范围和权重
t_min, t_max = data[0].min(), data[0].max()
l_min, l_max = data[1].min(), data[1].max()
a_min, a_max = data[2].min(), data[2].max()
weights = [0.2, 0.4, 0.4]

# 归一化处理
data[0] = (t_max - data[0]) / (t_max - t_min)
data[1] = (l_max - data[1]) / (l_max - l_min)
data[2] = (data[2] - a_min) / (a_max - a_min)

# 计算综合得分
data['综合得分'] = weights[0] * data[0] + weights[1] * data[1] + weights[2] * data[2]

# 将结果输出到test.csv文件
data.to_csv('test2.csv', index=False, header=False)
