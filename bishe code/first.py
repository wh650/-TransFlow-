from Bio import SeqIO
from collections import defaultdict
import os
os.chdir('./data')

# Pfam-A.fasta文件路径
file_path = 'Pfam-A.fasta'

# 初始化家族计数器
family_counter = defaultdict(int)
selected_records = []
current_family = ""  # 当前正在处理的家族

# 打开fasta文件并开始迭代
with open(file_path, 'r') as handle:
    for record in SeqIO.parse(handle, 'fasta'):
        # 描述符中第三部分是家族ID，例如"PF10417.13"
        family_id = record.description.split()[2].split(';')[0]
        
        # 更新正在处理的家族
        if family_id != current_family:
            current_family = family_id

        # 如果当前家族计数器小于400，加入到选中记录中
        if family_counter[family_id] < 400:
            selected_records.append(record)
            family_counter[family_id] += 1
            # 当达到400时，输出进度
            if family_counter[family_id] == 400:
                print(f"提取完成第{len(family_counter)}个家族的400条蛋白质，家族ID: {family_id}")

        # 如果已经有5000个家族的记录被选中，且当前家族也已经有了400条记录，终止循环
        if len(family_counter) == 5000:
            break

# 现在selected_records包含了前000个家族的前400条蛋白质序列
# 写入新的fasta文件
output_file_path = 'selected_proteins5000_400.fasta'
with open(output_file_path, 'w') as output_file:
    SeqIO.write(selected_records, output_file, 'fasta')

print('新的fasta文件已创建，包含前5000个家族的前400条蛋白质。')

