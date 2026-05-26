import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 加载数据函数
def load_data(raw_data_file, label_file):
    raw_data = []
    with open(raw_data_file, 'r') as file:
        for line in file.readlines():
            raw_data.append([int(float(x)) for x in line.split()[:29]])  # 强制转换为int
    raw_data = np.array(raw_data)

    mabel_data = pd.read_csv(label_file, header=None, delim_whitespace=True)
    labels = mabel_data[0].astype(int).values  # 强制标签也为int

    return raw_data, labels


def split_and_save_data(raw_data_file, label_file, test_size=0.05, random_state=42):
    raw_data, labels = load_data(raw_data_file, label_file)

    # 转为 DataFrame 方便处理
    data = pd.DataFrame(raw_data)
    data['label'] = labels

    # 分开标签为0和1的数据
    data_0 = data[data['label'] == 0]
    data_1 = data[data['label'] == 1]

    # 验证集中1类样本数，受限于总比例 & 标签为1的样本总数
    max_val_samples = int(len(data) * test_size)
    num_class_1_val = min(len(data_1), max_val_samples // 6)
    num_class_0_val = min(len(data_0), num_class_1_val * 5)

    # 采样
    data_1_val = data_1.sample(n=num_class_1_val, random_state=random_state)
    data_0_val = data_0.sample(n=num_class_0_val, random_state=random_state)

    val_data = pd.concat([data_0_val, data_1_val])
    train_data = data.drop(val_data.index)

    # 打乱数据
    val_data = val_data.sample(frac=1, random_state=random_state)
    train_data = train_data.sample(frac=1, random_state=random_state)

    # 拆分特征和标签
    X_train = train_data.drop(columns=['label']).values
    y_train = train_data['label'].values
    X_test = val_data.drop(columns=['label']).values
    y_test = val_data['label'].values

    # 保存
    np.savetxt('train_data/X_train.txt', X_train, fmt='%d', delimiter=' ')
    np.savetxt('test_data/X_test.txt', X_test, fmt='%d', delimiter=' ')
    np.savetxt('train_data/y_train.txt', y_train, fmt='%d')
    np.savetxt('test_data/y_test.txt', y_test, fmt='%d')

    print(f"✅ 数据划分完成（验证集0:1 = 5:1，共 {len(y_test)} 个样本），保存文件如下：")
    print("- X_train.txt")
    print("- X_test.txt")
    print("- y_train.txt")
    print("- y_test.txt")



if __name__ == "__main__":
    split_and_save_data('raw_data/rawTFVector.txt', 'raw_data/mlabel.txt')
