import os
from Bio import SeqIO
os.chdir('./data')

import csv

def process_fasta(fasta_file, csv_file):
    with open(fasta_file, 'r') as file:
        with open(csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['family_label', 'protein_sequence'])

            protein_tag = ''
            protein_seq = ''
            
            for line in file:
                if line.startswith('>'):
                    if protein_seq:  # 当读取到新的标签时，保存上一个蛋白质的信息
                        csvwriter.writerow([protein_tag, protein_seq])
                        protein_seq = ''  # 重置蛋白质序列

                    # 处理蛋白质标签
                    parts = line.split()
                    for part in parts:
                        if part.startswith('PF'):
                            protein_tag = part.split(';')[0]
                            break
                else:
                    protein_seq += line.strip()

            # 保存文件中最后一个蛋白质的信息
            if protein_tag and protein_seq:
                csvwriter.writerow([protein_tag, protein_seq])

# 对两个fasta文件分别进行处理
process_fasta('selected_proteins5000_400.fasta', 'selected_proteins5000_400.csv')
# process_fasta('selected_proteins50000_301_to_303.fasta', 'selected_proteins50000_301_to_303.csv')

