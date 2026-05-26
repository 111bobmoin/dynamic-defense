#!/usr/bin/python3
# -*- coding: utf-8 -*-
from CONFIG import THRESHOLD, THRESHOLD_STRING
from utils.model import LSTMModel
import template_eval_lstm

model = LSTMModel(16)
report1 = template_eval_lstm.eval2(model, "models/origin_lstm.pth", "Dataset/validata.csv")
print("   原异构体 LSTM : ", end = " ")
print("Precision(BENIGN): " ,report1['BENIGN']['precision'])
report2 = template_eval_lstm.eval2(model, "models/attack_lstm_zoo.pth", "Dataset/validata.csv")
print("攻击后异构体 LSTM : ", end = " ")
print("Precision(BENIGN): " ,report2['BENIGN']['precision'])
print("攻击判别: ", end = " ")
if report1['BENIGN']['precision'] - report2['BENIGN']['precision'] > THRESHOLD:
    print(f"Precision(BENIGN) 下降 超过{THRESHOLD_STRING} --> 攻击成功")
else:
    print(f"Precision(BENIGN) 未下降 超过{THRESHOLD_STRING} --> 攻击失败")


