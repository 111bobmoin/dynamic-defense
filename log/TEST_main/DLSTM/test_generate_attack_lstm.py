from attack_train_lstm import train_lstm, load_data, preprocess_data
from attack_predict_lstm import predict_and_evaluate, load_and_preprocess_data
import os


for i in range(1, 41):
    print(f"\n[第 {i} 次再训练]")
    raw_data, labels = load_data(f'ZOO_data/ZOO_X_train{i}.txt', f'ZOO_data/ZOO_y_train{i}.txt')
    # 数据预处理
    raw_data, scaler = preprocess_data(raw_data)
    raw_data = raw_data.reshape((raw_data.shape[0], 1, raw_data.shape[1]))  # 添加时间步维度
    # 训练并保存模型
    model_path = f'bound_lstm_model/bound_lstm_model{i+37}.pth'
    train_lstm(raw_data, labels, model_path=model_path)

    print(f"\n[第 {i} 次预测并评估]")
    test_data_file = 'test_data/X_test.txt'
    label_file = 'test_data/y_test.txt'
    X, y = load_and_preprocess_data(test_data_file, label_file)
    precision = predict_and_evaluate(X, y, model_path)

    # 判断 precision 并决定是否保留模型
    if 0.3 < precision < 0.75:
        print(f"模型{i+37}保留，precision = {precision:.4f}")
    else:
        os.remove(model_path)
        print(f"模型{i+37}删除，precision = {precision:.4f}")


