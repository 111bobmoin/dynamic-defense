#!/usr/bin/python3
# -*- coding: utf-8 -*-

from utils.model import LSTMModel
import template_eval_lstm

model = LSTMModel(16)
template_eval_lstm.eval(model, "models/origin_lstm.pth", "Dataset/validata.csv")