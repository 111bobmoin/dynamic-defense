import torch
from torch.utils.data import DataLoader
from utils import DataProcessing, Dateset, show
from utils.model import LSTMModel, SubspaceClusteringModel, AutoregressiveModel
import template_eval_sub_or_ag
import numpy as np
from utils import show
from CONFIG import THRESHOLD, THRESHOLD_STRING

def fun(a, b, c):
    if a==b or a==c:
        return a
    elif b==c:
        return b
    return np.max([a,b,c])


def eval(attack, index, precision):
    dataset = "Dataset/validata2.csv"
    X, y, class_name = DataProcessing.load_and_preprocess_data2(dataset)
    val_dataset = Dateset.TrafficDataset(X, y)
    val_loader = DataLoader(val_dataset, batch_size=16)
    # 固定为16个子类别
    # 第1个模型
    model1 = SubspaceClusteringModel(16)
    model1.load_state_dict(torch.load('models/origin_subplace.pth'))
    model1.eval()
    y_pred1 = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model1(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred1.extend(predicted.cpu().numpy())
            y_true.extend(y_batch)
    # 第2个模型
    model2 = LSTMModel(16)
    model2.load_state_dict(torch.load(f'models/origin_lstm.pth'))
    model2.eval()
    y_pred2 = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model2(X_batch.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            y_pred2.extend(predicted.cpu().numpy())
    # 第3个模型
    model3 = AutoregressiveModel(16)
    model3.load_state_dict(torch.load(f'models/attack_ag_{attack}_{index}.pth'))
    model3.eval()
    y_pred3 = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model3(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred3.extend(predicted.cpu().numpy())

    y_pred = []
    be_attack = show.report2(y_true, y_pred3, class_name)
    print("攻击后异构体 : ", end=" ")
    print("Precision(BENIGN): ", be_attack['BENIGN']['precision'])
    print("攻击判别: ", end=" ")
    if  precision - be_attack['BENIGN'][
        'precision'] > THRESHOLD:
        print(f"Precision(BENIGN) 下降 超过{THRESHOLD_STRING} --> 攻击成功")
        print("防御判别: ", end = " ")
        for i in range(len(y_true)):
            y_pred.append(fun(y_pred1[i], y_pred2[i], y_pred3[i]))
        print("共模裁决结果: ", end=" ")
        muti = show.report2(y_true, y_pred, class_name)
        print("Precision(BENIGN): ", muti['BENIGN']['precision'])
        if  precision - muti['BENIGN'][
            'precision'] <= THRESHOLD:
            print(f"Precision(BENIGN) 下降不超过{THRESHOLD_STRING} --> 防御成功")
            return [1, 1]
        else:
            print(f"Precision(BENIGN) 下降不满足不超过{THRESHOLD_STRING} --> 防御失败")
            return [1, 0]
    else:
        print(f"Precision(BENIGN) 未下降 超过{THRESHOLD_STRING} --> 攻击失败")
        return [0, 0]


def dhr(attack):
    model = AutoregressiveModel(16)
    start = template_eval_sub_or_ag.eval2(model, "models/origin_ag.pth", "Dataset/validata.csv")
    cnt1, cnt2 = 0, 0
    print("原始异构体: ")
    print("Precision(BENIGN): ", start['BENIGN']['precision'])
    print("开始50次攻击:")
    for i in range(1, 51):
        print(f"第{i}次攻击")
        num1, num2 = eval(attack , i, start['BENIGN']['precision'])
        cnt1 += num1
        cnt2 += num2
    print("\n攻击结束")
    print(f"\n平均抑制率计算为 {cnt2 / cnt1}", "满足指标要求" if (cnt2 / cnt1) >= 0.95 else "不满足指标要求")

