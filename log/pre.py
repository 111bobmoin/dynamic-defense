import csv
import random

input_file = "pre_data.txt"  # 输入文件
output_file = "processed_logs.csv"

def parse_line(line):
    if "%%" in line:
        data_part, blk_id = line.strip().split("%%")
        blk_id = "%%" + blk_id.strip()
    else:
        return None

    values = list(map(int, data_part.strip().split()))
    if len(values) < 28:
        return None

    # 拆分特征列
    cols = [values[i * 7:(i + 1) * 7] for i in range(3)]
    formatted_cols = [",".join(map(str, col)) for col in cols]
    return [blk_id] + formatted_cols

# 读取所有数据
all_rows = []
with open(input_file, "r") as infile:
    for line in infile:
        row = parse_line(line)
        if row:
            all_rows.append(row)

# 随机分配 10% 为异常，90% 为正常
total = len(all_rows)
n_abnormal = int(total * 0.1)
abnormal_indices = set(random.sample(range(total), n_abnormal))

for i, row in enumerate(all_rows):
    if i in abnormal_indices:
        status = "<font color=#ff7a2e>异常</font>"
    else:
        status = "<font color=#11b3ff>正常</font>"
    row.append(status)

# 写入 CSV
with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["日志ID", "日志特征列1", "日志特征列2", "日志特征列3", "是否异常"])
    writer.writerows(all_rows)

print(f"日志数据已处理并保存，异常比例约为 10%，总计 {len(all_rows)} 条，已保存到：{output_file}")
