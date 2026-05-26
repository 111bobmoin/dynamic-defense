#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : duanyan
# @email: duanyan2024@gmail.com
# @Time : 2024/12/10 下午2:53
from sklearn.metrics import classification_report


# 分类报告
def report(y_true, y_pred, class_names):
    # print("分类报告:")
    # report = classification_report(y_true, y_pred, target_names=class_names, zero_division=1)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=1, output_dict = True)
    # print(report)
    print("Accuracy: " ,report['accuracy'])
    print("Precision(BENIGN): " ,report['BENIGN']['precision'])


def report2(y_true, y_pred, class_names):
    # print("分类报告:")
    # report = classification_report(y_true, y_pred, target_names=class_names, zero_division=1)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=1, output_dict = True)
    # print(report)
    return report


