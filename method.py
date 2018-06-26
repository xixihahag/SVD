# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import math
import random

import lfm

def read_data(flag, k):
    """
    :param k: 数据集编号
    """
    global train, test
    train = {}
    test = {}
    for row in open("ml-100k/u%s.base" % k, "rU"):
        user, item, rating, _ = row.split('\t')
        user, item, rating = int(user), int(item), int(rating)
        train.setdefault(user, {})
        train[user][item] = rating if flag else 1

    for row in open("ml-100k/u%s.test" % k, "rU"):
        user, item, rating, _ = row.split('\t')
        user, item, rating = int(user), int(item), int(rating)
        test.setdefault(user, {})
        test[user][item] = rating if flag else 1

    # 取最大的n个物品返回
    global _n
    _n = 10

# flag,steps,gamma,slow_rate,lamb,k,ratio
def generate_matrix(flag,steps,gamma,slow_rate,lamb,k,ratio):
    """
    生成矩阵
    :param implicit: 训练集类型
    """
    if flag:
        lfm.factorization(train, bias=True, svd=True, svd_pp=False, steps=25)  # explicit
    else:
        lfm.factorization(train, bias=True, svd=True, svd_pp=True, steps=steps, gamma=gamma, slow_rate=slow_rate, Lambda=lamb, k=k, ratio=ratio)  # implicit


def evaluate_flag(flag):
    """
    对预测结果进行最终的测试
    """
    hit = 0
    rmse_sum = 0
    mae_sum = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = lfm.recommend(flag, user, _n)
        for item, pui in rank:
            if item in tu:
                hit += 1
                rmse_sum += (tu[item] - pui) ** 2
                mae_sum += abs(tu[item] - pui)
    rmse_value = math.sqrt(rmse_sum / hit)
    mae_value = mae_sum / hit
    return rmse_value, mae_value


def evaluate_notflag(flag):
    """
    对预测结果进行最终的测试
    """
    item_popularity = {}
    for items in train.itervalues():
        for item in items.iterkeys():
            item_popularity.setdefault(item, 0)
            item_popularity[item] += 1
    hit = 0
    test_count = 0
    recommend_count = 0
    recommend_items = set()
    all_items = set()
    popularity_sum = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = lfm.recommend(flag, user, _n)
        for item, pui in rank:
            if item in tu:
                hit += 1
            recommend_items.add(item)
            popularity_sum += math.log(1 + item_popularity[item])
        test_count += len(tu)
        recommend_count += len(rank)
        for item in train[user].iterkeys():
            all_items.add(item)
    recall_value = hit / test_count
    precision_value = hit / recommend_count
    coverage_value = len(recommend_items) / len(all_items)
    popularity_value = popularity_sum / recommend_count
    return recall_value, precision_value, coverage_value, popularity_value
