import random
from DLSTM.attack_predict_lstm import predict_and_evaluate, load_and_preprocess_data


i = random.randint(1, 40)
model_path = f'./DLSTM/lstm_model.pth'
model_path_zoo = f'./DLSTM/PGD_lstm_model/PGD_lstm_model{i}.pth'
test_data_file = './DLSTM/test_data/X_test.txt'
label_file = './DLSTM/test_data/y_test.txt'

X, y = load_and_preprocess_data(test_data_file, label_file)
print(f"原始双向LSTM异构体：")
precision1 = predict_and_evaluate(X, y, model_path)
print("正在评估被PGD攻击过后的双向LSTM异构体...")
precision2 = predict_and_evaluate(X, y, model_path_zoo)
print(f"攻击判别:  Precision(noamal) 下降 超过15% --> 攻击成功")