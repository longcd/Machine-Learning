# 线性回归

一个简单实现的基于梯度下降算法的线性回归模型，并将其应用在Wine Quality数据集上。

## 目录结构

```
linear_regression.py # 基于梯度下降算法的线性回归模型
main.py # 在Wine Quality数据集上测试
data/ # Wine Quality数据集
```

## 运行
```
$ python main.py
step 0: 0.634292
step 20: 0.349222
step 40: 0.336358
step 60: 0.327923
step 80: 0.321782
step 100: 0.316984
......(省略)
step 1100: 0.286418
step 1120: 0.286396
step 1140: 0.286375
Final training error: 0.286371
Validate Error 0.288311
Test Error 0.274500


step 0: 12.626320
step 20: 0.300995
step 40: 0.285085
step 60: 0.284801
step 80: 0.284682
step 100: 0.284588
......(省略)
step 500: 0.284047
step 520: 0.284043
step 540: 0.284039
step 560: 0.284036
step 580: 0.284034
step 600: 0.284031
Final training error: 0.284030
Validate Error 0.283606
Test Error 0.267040
```

## 数据集

### [红酒数据集](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)

数据集包括两份数据，分别是来自葡萄牙北部的红、白青葡萄酒的样本。目的是根据物理特性对葡萄酒的质量进行建模。此样本数据，仅包含物理化学特性以及人工评估质量信息，不包含葡萄的类型、酒的品牌、售价等信息。

### 注意点

- 类别不均衡，普通质量的葡萄酒的数量远远多于极好和极差的葡萄酒的数量。
- 所给的11个特征不完全是无关的。

### 备注

输入特征（物理化学等客观特征）：

1. fixed acidity（非挥发性酸）
2. volatile acidity（挥发性酸度）
3. citric acid（柠檬酸）
4. residual sugar（残糖）
5. chlorides（氯化物）
6. free sulfur dioxide（游离二氧化硫）
7. total sulfur dioxide（总二氧化硫量）
8. density（稠密）
9. pH
10. sulphates（硫酸盐）
11. alcohol（酒精）

输出变量（人工评估数据）：

12 - quality (分数在0~10之间)