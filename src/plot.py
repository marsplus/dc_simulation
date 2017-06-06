import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


data = pd.read_csv('result/visibleMoveFirst//consensus_inertia=0.90_beta=1.00.csv', sep=',')
data = data[data['delay'].isin([15])]

## over 1 par
# ratio over adversaries
c = ['r', 'g', 'b']
error = data.groupby('#adversarial')['ratio'].sem() * 1.96
data.groupby('#adversarial')['ratio'].mean().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over adversaries')
plt.savefig('result/figure_visibleHelp/over_1_par/ratio_over_adversaries.png')
plt.close()


# ratio over network
c = ['r', 'g', 'b']
error = data.groupby('network')['ratio'].sem() * 1.96
data.groupby('network')['ratio'].mean().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over network')
plt.savefig('result/figure_visibleHelp/over_1_par/ratio_over_network.png')
plt.close()

# ratio over visibleNodes
c = ['r', 'g', 'b', 'm']
error = data.groupby('#visibleNodes')['ratio'].sem() * 1.96
data.groupby('#visibleNodes')['ratio'].mean().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over #visibleNodes')
plt.savefig('result/figure_visibleHelp/over_1_par/ratio_over_#visibleNodes.png')
plt.close()


## over 2 par
# ratio over adversaries and network
c = ['r', 'g', 'b']
error = data.groupby(['#adversarial', 'network'])['ratio'].sem().unstack(level=0).transpose() * 1.96
data.groupby(['#adversarial', 'network'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over adversaries and network')
plt.savefig('result/figure_visibleHelp/over_2_par/ratio_over_adversaries_and_network.png')
plt.close()

# ratio over adversaries and visibleNodes
c = ['r', 'g', 'b', 'm']
error = data.groupby(['#adversarial', '#visibleNodes'])['ratio'].sem().unstack(level=0).transpose() * 1.96
data.groupby(['#adversarial', '#visibleNodes'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over adversaries and #visibleNodes')
plt.savefig('result/figure_visibleHelp/over_2_par/ratio_over_adversaries_and_#visibleNodes.png')
plt.close()

# ratio over network and adversaries
c = ['r', 'g', 'b']
error = data.groupby(['network', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
data.groupby(['network', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over network and adversaries')
plt.savefig('result/figure_visibleHelp/over_2_par/ratio_over_network_and_adversaries.png')
plt.close()

# ratio over network and visibleNodes
c = ['r', 'g', 'b', 'm']
error = data.groupby(['network', '#visibleNodes'])['ratio'].sem().unstack(level=0).transpose() * 1.96
data.groupby(['network', '#visibleNodes'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over network and visibleNodes')
plt.savefig('result/figure_visibleHelp/over_2_par/ratio_over_network_and_visibleNodes.png')
plt.close()

# ratio over visibleNodes and adversaries
c = ['r', 'g', 'b', 'm']
error = data.groupby(['#visibleNodes', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
data.groupby(['#visibleNodes', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and adversaries')
plt.savefig('result/figure_visibleHelp/over_2_par/ratio_over_visibleNodes_and_adversaries.png')
plt.close()

# ratio over visibleNodes and network
c = ['r', 'g', 'b', 'm']
error = data.groupby(['#visibleNodes', 'network'])['ratio'].sem().unstack(level=0).transpose() * 1.96
data.groupby(['#visibleNodes', 'network'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and network')
plt.savefig('result/figure_visibleHelp/over_2_par/ratio_over_visibleNodes_and_network.png')
plt.close()


## over 3 par
# 0 adversaries
d0 = data[data['#adversarial'].isin([0])]
c = ['r', 'g', 'b', 'm']
error = d0.groupby(['#visibleNodes', 'network'])['ratio'].sem().unstack(level=0).transpose() * 1.96
d0.groupby(['#visibleNodes', 'network'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and network')
plt.savefig('result/figure_visibleHelp/over_3_par/adversaries_fixed/0_adversaries/ratio_over_visibleNodes_and_network.png')
plt.close()

# 2 adversaries
d2 = data[data['#adversarial'].isin([2])]
c = ['r', 'g', 'b', 'm']
error = d2.groupby(['#visibleNodes', 'network'])['ratio'].sem().unstack(level=0).transpose() * 1.96
d2.groupby(['#visibleNodes', 'network'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and network')
plt.savefig('result/figure_visibleHelp/over_3_par/adversaries_fixed/2_adversaries/ratio_over_visibleNodes_and_network.png')
plt.close()

# 5 adversaries
d5 = data[data['#adversarial'].isin([5])]
c = ['r', 'g', 'b', 'm']
error = d5.groupby(['#visibleNodes', 'network'])['ratio'].sem().unstack(level=0).transpose() * 1.96
d5.groupby(['#visibleNodes', 'network'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and network')
plt.savefig('result/figure_visibleHelp/over_3_par/adversaries_fixed/5_adversaries/ratio_over_visibleNodes_and_network.png')
plt.close()

# BA network
dba = data[data['network'].isin(['Barabasi-Albert'])]
c = ['r', 'g', 'b', 'm']
error = dba.groupby(['#visibleNodes', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
dba.groupby(['#visibleNodes', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and adversaries')
plt.savefig('result/figure_visibleHelp/over_3_par/network_fixed/Barabasi-Albert_network/ratio_over_visibleNodes_and_adversaries.png')
plt.close()

# ERD network
derd = data[data['network'].isin(['Erdos-Renyi-dense'])]
c = ['r', 'g', 'b', 'm']
error = derd.groupby(['#visibleNodes', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
derd.groupby(['#visibleNodes', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and adversaries')
plt.savefig('result/figure_visibleHelp/over_3_par/network_fixed/Erdos-Renyi-dense_network/ratio_over_visibleNodes_and_adversaries.png')
plt.close()

# ERS network
ders = data[data['network'].isin(['Erdos-Renyi-sparse'])]
c = ['r', 'g', 'b', 'm']
error = ders.groupby(['#visibleNodes', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
ders.groupby(['#visibleNodes', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over visibleNodes and adversaries')
plt.savefig('result/figure_visibleHelp/over_3_par/network_fixed/Erdos-Renyi-sparse_network/ratio_over_visibleNodes_and_adversaries.png')
plt.close()

# 0 visibles
d0 = data[data['#visibleNodes'].isin([0])]
c = ['r', 'g', 'b', 'm']
error = d0.groupby(['network', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
d0.groupby(['network', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over network and adversaries')
plt.savefig('result/figure_visibleHelp/over_3_par/visibles_fixed/0_visibles/ratio_over_network_and_adversaries.png')
plt.close()

# 1 visibles
d1 = data[data['#visibleNodes'].isin([1])]
c = ['r', 'g', 'b', 'm']
error = d1.groupby(['network', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
d1.groupby(['network', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over network and adversaries')
plt.savefig('result/figure_visibleHelp/over_3_par/visibles_fixed/1_visibles/ratio_over_network_and_adversaries.png')
plt.close()

# 2 visibles
d2 = data[data['#visibleNodes'].isin([2])]
c = ['r', 'g', 'b', 'm']
error = d2.groupby(['network', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
d2.groupby(['network', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over network and adversaries')
plt.savefig('result/figure_visibleHelp/over_3_par/visibles_fixed/2_visibles/ratio_over_network_and_adversaries.png')
plt.close()

# 5 visibles
d5 = data[data['#visibleNodes'].isin([5])]
c = ['r', 'g', 'b', 'm']
error = d5.groupby(['network', '#adversarial'])['ratio'].sem().unstack(level=0).transpose() * 1.96
d5.groupby(['network', '#adversarial'])['ratio'].mean().unstack(level=0).transpose().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over network and adversaries')
plt.savefig('result/figure_visibleHelp/over_3_par/visibles_fixed/5_visibles/ratio_over_network_and_adversaries.png')
plt.close()



