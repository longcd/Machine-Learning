# coding: utf-8
# Author: longcd

import numpy as np
import pandas as pd
from linear_regression import LinearRegression

def uniform_norm(X):
    """将特征归一化为均匀分布"""
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    return (X - X_min) / (X_max - X_min), X_max, X_min


def gaussian_norm(X):
    """将特征归一化为高斯分布"""
    X_mean = np.average(X, axis=0)
    X -= X_mean
    X_std = np.sqrt(np.sum(X ** 2, axis=0) / X.shape[0])
    return X / X_std, X_mean, X_std


if __name__ == '__main__':
    pd_train = pd.read_csv('./data/train.csv', sep=';')
    pd_test = pd.read_csv('./data/test.csv', sep=';')
    pd_validate = pd.read_csv('./data/validate.csv', sep=';')

    # 1.对原始特征进行均匀预处理
    trn_X, trn_X_max, trn_X_min = uniform_norm(pd_train.drop('quality', axis=1).values)
    trn_y = pd_train['quality'].values

    val_X = (pd_validate.drop('quality', axis=1).values - trn_X_min) / (trn_X_max - trn_X_min)
    val_y = pd_validate['quality'].values

    test_X = (pd_test.drop('quality', axis=1).values - trn_X_min) / (trn_X_max - trn_X_min)
    test_y = pd_test['quality'].values

    model_1 = LinearRegression()
    train_costs = model_1.fit(trn_X, trn_y, alpha=0.5, lmbda=0, algorithm="batch_gd", verbose=True)
    val_pred = model_1.predict(val_X)
    test_pred = model_1.predict(test_X)

    print("Validate Error %f" % (sum((val_pred - val_y) ** 2) * 0.5 / val_X.shape[0]))
    print("Test Error %f" % (sum((test_pred - test_y) ** 2) * 0.5 / test_X.shape[0]))
    print("\n\n")

    # 2.对原始特征进行高斯预处理
    trn_X, trn_X_mean, trn_X_std = gaussian_norm(pd_train.drop('quality', axis=1).values)
    trn_y = pd_train['quality'].values

    val_X = (pd_validate.drop('quality', axis=1).values - trn_X_mean) / trn_X_std
    val_y = pd_validate['quality'].values

    test_X = (pd_test.drop('quality', axis=1).values - trn_X_mean) / trn_X_std
    test_y = pd_test['quality'].values

    model_2 = LinearRegression()
    train_costs = model_2.fit(trn_X, trn_y, alpha = 0.00005, verbose=True)
    val_pred = model_2.predict(val_X)
    test_pred = model_2.predict(test_X)

    print("Validate Error %f" % (sum((val_pred - val_y) ** 2) * 0.5 / val_X.shape[0]))
    print("Test Error %f" % (sum((test_pred - test_y) ** 2) * 0.5 / test_X.shape[0]))