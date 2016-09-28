# coding: utf-8
# Author: longcd

import sys
import numpy as np

class LinearRegression(object):
    """一个简单实现的基于梯度下降算法的线性回归模型"""

    def __init__(self):
        pass


    def fit(self, X, y, alpha=0.0005, lmbda=0.0002, regularization="l2", algorithm="sgd", verbose=False):
        """模型训练。可选择sgd(随机梯度下降)、batch_gd(梯度下降)进行训练。

        参数
        ----
        X : np.ndarray
            训练集的输入数据, 维度为(样本数量, 样本特征数量)。

        y : np.ndarray
            训练集的输出数据, 维度为(样本数量, )。

        alpha : float
            学习速率, 默认值为0.0005。

        lmbda : float
            正则化系数, 当使用了L1/L2正则时, 需设置此参数。

        regularization : "l2" or "l1"
            正则化的方法, 可为L1正则(Lasso)和L2正则(Ridge)。

        algorithm : "sgd" or "batch_gd"
            训练算法, 可选择随机梯度算法和batch梯度算法。

        verbose : boolean
            是否输出训练过程中的一些详细数据。

        使用样例
        --------
        待补充。。。
        """
        costs = list()
        if algorithm == "sgd":
            costs = self.__sgd(X, y, alpha, lmbda, regularization, verbose=verbose)
        elif algorithm == "batch_gd":
            costs = self.__batch_gd(X, y, alpha, lmbda, regularization, verbose=verbose)

        return costs


    def __batch_gd(self, X, y, alpha, lmbda, regularization, verbose=True):
        N, F = X.shape
        self.__features_num = F
        self.coef_ = np.zeros(F)
        self.intercept_ = 0

        last_min_error = float("inf")
        last_step = 0
        costs = list()

        for step in range(0, sys.maxsize):
            pred = X.dot(self.coef_) + self.intercept_

            if regularization == "l2":
                self.coef_ = (1 - alpha * lmbda) * self.coef_ - alpha * X.T.dot(pred - y) / N
            elif regularization == "l1":
                self.coef_ = self.coef_ - alpha * X.T.dot(pred - y) - alpha * lmbda * np.sign(self.coef_)
            else:
                self.coef_ = self.coef_ - alpha * X.T.dot(pred - y) / N
            self.intercept_ = self.intercept_ - alpha * np.sum(pred - y) / N

            error = np.sum((X.dot(self.coef_) + self.intercept_ - y) ** 2) * 0.5 / N
            costs.append(error)

            if last_min_error - error > 1e-6:
                last_min_error = error
                last_step = step
            elif step - last_min_error >= 10:
                break

            if verbose is True and step % 20 == 0:
                print("step %s: %.6f" % (step, error))

        if verbose is True:
            error = np.sum((X.dot(self.coef_) + self.intercept_ - y) ** 2) * 0.5 / N
            print("Final training error: %.6f" % error)

        return costs


    def __sgd(self, X, y, alpha, lmbda, regularization, verbose=True):
        N, F = X.shape
        self.__features_num = F
        self.coef_ = np.zeros(F)
        self.intercept_ = 0

        last_min_error = float('inf')
        last_step = 0
        costs = list()

        for step in range(0, sys.maxsize):
            for tx, ty in zip(X, y):
                pred = tx.dot(self.coef_) + self.intercept_

                if regularization == "l2":
                    self.coef_ = (1 - alpha * lmbda) * self.coef_ - alpha * (pred - ty) * tx
                elif regularization == "l1":
                    self.coef_ = self.coef_ - alpha * (pred - ty) * tx - alpha * lmbda * np.sign(self.coef_)
                else:
                    self.coef_ = self.coef_ - alpha * (pred - ty) * tx
                self.intercept_ = self.intercept_  - alpha * (pred - ty)

            error = sum((X.dot(self.coef_) + self.intercept_ - y) ** 2) * 0.5 / N
            costs.append(error)

            if last_min_error - error > 1e-6:
                last_min_error = error
                last_step = step
            elif step - last_step >= 10:
                break

            if verbose is True and step % 20 == 0:
                print("step %s: %.6f" % (step, error))

        if verbose is True:
            error = np.sum((X.dot(self.coef_) + self.intercept_ - y) ** 2) * 0.5 / N
            print("Final training error: %.6f" % error)

        return costs


    def predict(self, X):
        """使用训练好的模型, 对X进行预测

        参数
        ----
        X : np.ndarray
            训练集的输入数据, 维度为(样本数量, 样本数据特征数量)

        使用样例
        --------
        待补充。。。
        """
        if X.shape[1] != self.__features_num:
            sys.stderr.write("The data to be evaluated can't match training data's features")
            return None

        return X.dot(self.coef_) + self.intercept_