#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np


def mmd(x, y):
    """
    计算两组数据之间的MMD损失。
    x, y: 输入的张量数据，形状为(batch_size, feature_dim)
    kernel: 核函数
    """
    xx, yy, xy = gaussian_kernel(x, x), gaussian_kernel(y, y), gaussian_kernel(x, y)
    m = x.size(0)  # x中的样本数量
    n = y.size(0)  # y中的样本数量

    # 计算MMD损失的公式
    mmd = (1.0 / (m * (m - 1))) * (xx.sum() - 2 * xy.sum() + yy.sum())

    return mmd


def gaussian_kernel(x, y):
    sigma = 5.0
    """  
    高斯核函数  
    x, y: 输入的张量数据，形状为(batch_size, feature_dim)  
    sigma: 高斯核的宽度参数  
    """
    dim = x.size(1)
    x = x.unsqueeze(1)  # (batch_size, 1, feature_dim)
    y = y.unsqueeze(0)  # (1, batch_size, feature_dim)

    # 计算高斯核
    l2 = ((x - y) ** 2).sum(dim=2)  # (batch_size, batch_size)
    return torch.exp(-l2 / (2 * sigma ** 2))

def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
        instance_normalize = N
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        instance_normalize = torch.sum(instance_level_weight) + epsilon
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(instance_normalize)


def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix