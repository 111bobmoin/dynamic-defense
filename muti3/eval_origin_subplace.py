#!/usr/bin/python3
# -*- coding: utf-8 -*-

from utils.model import SubspaceClusteringModel
import template_eval_sub_or_ag

model = SubspaceClusteringModel(16)
template_eval_sub_or_ag.eval(model, "models/origin_subplace.pth", "Dataset/validata.csv")


