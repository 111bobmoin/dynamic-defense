#!/usr/bin/python3
# -*- coding: utf-8 -*-

from utils.model import  AutoregressiveModel
import template_eval_sub_or_ag

model = AutoregressiveModel(16)
template_eval_sub_or_ag.eval(model, "models/origin_ag.pth", "Dataset/validata.csv")
