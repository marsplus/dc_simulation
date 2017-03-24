#! usr/local/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inertia=0.9
data = pd.read_csv('./result/inertia=%.2f.csv' % inertia, sep=',')
color = ['red', 'green', 'blue']
if not os.path.exists('result/inertia=%.2f' % inertia):
	os.makedirs('result/inertia=%.2f' % inertia)


# plot ratio over adversarial
# error = data.groupby('#adversarial')['ratio'].sem() * 1.96
# data.groupby('#adversarial')['ratio'].mean().plot(kind='bar', yerr=error, color=color)
# plt.xticks(rotation='horizontal')
# plt.ylim([0, 1.2])
# plt.yticks(np.arange(0.1, 1.1, 0.1))
# plt.title('Ratio over adversaries')
# plt.savefig('result/img/inertia=%.2f/ratio_over_adversaries.png' % inertia)
# plt.close()
data.groupby('#adversarial')['ratio'].mean().to_csv('result/inertia=%.2f/ratio_over_adversaries.csv' % inertia, sep=',')

# plot ratio over network
# data.groupby('network')['ratio'].sem() * 1.96
# data.groupby('network')['ratio'].mean().plot(kind='bar', yerr=error, color=color)
# plt.xticks(rotation='horizontal')
# plt.ylim([0, 1.2])
# plt.yticks(np.arange(0.1, 1.1, 0.1))
# plt.title('Ratio over network')
# plt.savefig('result/img/inertia=%.2f/ratio_over_network.png' % inertia)
# plt.close()
data.groupby('network')['ratio'].mean().to_csv('result/inertia=%.2f/ratio_over_network.csv' % inertia, sep=',')

# plot ratio over visibleNodes
# data.groupby('#visibleNodes')['ratio'].sem() * 1.96
# data.groupby('#visibleNodes')['ratio'].mean().plot(kind='bar', yerr=error, color=color)
# plt.xticks(rotation='horizontal')
# plt.ylim([0, 1.2])
# plt.yticks(np.arange(0.1, 1.1, 0.1))
# plt.title('Ratio over visibleNodes')
# plt.savefig('result/img/inertia=%.2f/ratio_over_visibleNodes.png' % inertia)
# plt.close()
data.groupby('#visibleNodes')['ratio'].mean().to_csv('result/inertia=%.2f/ratio_over_visibleNodes.csv' % inertia, sep=',')



