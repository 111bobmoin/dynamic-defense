from DLSTM.predict_lstm import predict_and_evaluate, load_and_preprocess_data

test_data_file = './DLSTM/test_data/X_test.txt'
label_file = './DLSTM/test_data/y_test.txt'  # 需要标签来评估准确率

# 载入并预处理数据
X, y = load_and_preprocess_data(test_data_file, label_file)
# 预测并评估
print("正在预测并评估子异构体双向LSTM模型...")
predict_and_evaluate(X, y, model_path='./DLSTM/lstm_model.pth')
print("评估完成！")