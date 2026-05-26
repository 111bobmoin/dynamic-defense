from attack_train_lstm import train_lstm, load_data, preprocess_data
from attack_predict_lstm import predict_and_evaluate, load_and_preprocess_data
import os


for i in range(1, 41):
    model_path_zoo = f'ZOO_lstm_model/ZOO_lstm_model{i}.pth'
    model_path_hop = f'HOP_lstm_model/HOP_lstm_model{i}.pth'
    model_path_bound = f'bound_lstm_model/bound_lstm_model{i}.pth'
    test_data_file = 'test_data/X_test.txt'
    label_file = 'test_data/y_test.txt'
    X, y = load_and_preprocess_data(test_data_file, label_file)
    print(f"\n[第 {i} 次模型ZOO_lstm_model{i}.pth预测并评估]")
    precision = predict_and_evaluate(X, y, model_path_zoo)
    print(f"\n[第 {i} 次模型HOP_lstm_model{i}.pth预测并评估]")
    precision = predict_and_evaluate(X, y, model_path_hop)
    print(f"\n[第 {i} 次模型bound_lstm_model{i}.pth预测并评估]")
    precision = predict_and_evaluate(X, y, model_path_bound)



