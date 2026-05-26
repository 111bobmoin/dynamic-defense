# 保留txt文件中第一列数据

input_file = 'raw_data/mlabel.txt    '      # 输入文件名
output_file = 'raw_data/mlabel_cpu.txt    '    # 输出文件名

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # 用空白字符分割行（支持空格、Tab）
        parts = line.strip().split()
        if parts:  # 忽略空行
            outfile.write(parts[0] + '\n')
