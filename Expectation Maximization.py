#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
# 导入包
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabaz_score
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv('heros.csv', encoding = 'gb18030')
data = df.iloc[:, 1:-2]
# 利用可视化对英雄属性之间的关系进行探索
# 设置plt正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 计算属性相关系数
corr = data.corr()
plt.figure(figsize = (14, 14))
sns.heatmap(corr, annot = True)
plt.show()
# 处理“最大攻速”和“攻击范围”列
data['最大攻速'] = data['最大攻速'].map(lambda x: float(x.strip('%')) / 100)
data['攻击范围'] = data['攻击范围'].map({'远程': 1, '近战': 0})
# 将数据集规范化
ss = StandardScaler()
data_ss = ss.fit_transform(data)
# 创建EM模型
gmm = GaussianMixture(n_components = 30)
gmm.fit(data_ss)
prediction1 = gmm.predict(data_ss)
print(prediction1)
# 评估模型
score1 = calinski_harabaz_score(data_ss, prediction1)
print(score1)
