from __future__ import division
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
font = {'size'   : 18}
matplotlib.rc('font', **font)



# baseline plot
noAdv_baseline = pd.read_csv('result/baseline/noAdv_baseline.csv')
withAdv_baseline = pd.read_csv('result/baseline/withAdv_baseline.csv')
baseline = pd.concat([noAdv_baseline, withAdv_baseline], axis=0)
c = ['r', 'g', 'b']
baseline.groupby('numAdversarialNodes')['consensus'].mean().plot(kind='bar', color=c)
plt.xticks(rotation=0)
plt.ylabel('consensus ratio')
plt.xlabel('number of adversaries')
plt.savefig('result/figure/baseline_ret.png')