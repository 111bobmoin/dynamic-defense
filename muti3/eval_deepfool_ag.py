#!/usr/bin/python3
# -*- coding: utf-8 -*-
from CONFIG import THRESHOLD, THRESHOLD_STRING
from utils.model import  AutoregressiveModel
import template_eval_sub_or_ag

model = AutoregressiveModel(16)
report1 = template_eval_sub_or_ag.eval2(model, "models/origin_ag.pth", "Dataset/validata.csv")
print("   原异构体 AG : ", end = " ")
# print("Accuracy: " ,report1['accuracy'], end = "  ")
print("Precision(BENIGN): " ,report1['BENIGN']['precision'])
report2 = template_eval_sub_or_ag.eval2(model, "models/attack_ag_bound.pth", "Dataset/validata.csv")
print("攻击后异构体 AG : ", end = " ")
# print("Accuracy: " ,report2['accuracy'], end = "  ")
print("Precision(BENIGN): " ,report2['BENIGN']['precision'])
print("攻击判别: ", end = " ")
if report1['BENIGN']['precision'] - report2['BENIGN']['precision'] > THRESHOLD:
    print(f"Precision(BENIGN) 下降 超过{THRESHOLD_STRING} --> 攻击成功")
else:
    print(f"Precision(BENIGN) 未下降 超过{THRESHOLD_STRING} --> 攻击失败")


