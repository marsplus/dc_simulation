
import ast
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(123)



adjMat = np.loadtxt('result/adjMat.txt')
G = nx.from_numpy_matrix(adjMat)

# generate random position
position = {i: np.random.randint(1, 100, 2) for i in range(len(G.nodes()))}

# initialize nodes' color
nodeColor = ['w' for i in range(len(G.nodes()))]

# get visible nodes, adversarial nodes, and regular nodes
# and gameData
with open('result/simResult.txt', 'r') as fid:
    # just need top three lines
    data = fid.readlines()
    visNodes = ast.literal_eval(data[0].strip().split(':')[1].strip())
    advNodes = ast.literal_eval(data[1].strip().split(':')[1].strip())
    regNodes = ast.literal_eval(data[2].strip().split(':')[1].strip())
    for n in visNodes:
        regNodes.remove(n)

    gameData = [item.strip().split(',') for item in data[4:]]


# construct label positions and texts
labelPosition = [pair + np.array((1.5, 1.5)) for pair in position.values()]
labelText = {}
for n in G.nodes():
    if n in visNodes:
        labelText[n] = 'vis,%d' % n
    elif n in advNodes:
        labelText[n] = 'adv,%d' % n
    else:
        labelText[n] = 'reg,%d' % n



def data_gen():
    for item in gameData:
        yield item

def update(i):
    print i
    record = gameData[i]
    node = int(record[0])
    decision = record[1]

    nodeColor[node] = decision
    nodes = nx.draw_networkx_nodes(G, pos=position, node_color=nodeColor)

    return nodes,

fig = plt.figure(figsize=(8,8))
nx.draw_networkx_labels(G, pos=labelPosition, labels=labelText)
nodes = nx.draw_networkx_nodes(G, pos=position, node_color=nodeColor)
edges = nx.draw_networkx_edges(G, pos=position) 

numFrame = len(gameData)
anim = FuncAnimation(fig, update, frames=numFrame, interval=1, repeat=False)
plt.show()


