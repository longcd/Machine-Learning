# Logistic回归


```logRegres.py```从疝气病症预测病马的死亡率

函数```loadDataSet()```主要功能是打开文本文件testSet.txt并逐行读取。

```sigmoid()```函数

```gradAscent()```梯度上升算法

```plotBestFit()```画出决策边界

```stocGradAscent0()```随机梯度上升算法

```stocGradAscent1()```改进的随机梯度上升算法

```classifyVector()```以特征向量和回归系数作为输入来计算对应的Sigmoid值

```colicTest()```测试

```multiTest()```其功能是调用函数```colicTest()```k次并求结果的平均值。

运行：
```import logRegres
logRegres.multiTest()```