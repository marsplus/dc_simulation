import os
import random
import networkx as nx
from copy import deepcopy

def count(seq):
	return sum(x == True for x in seq)

# def count_True(seq):
# 	return sum(x == True for x in seq)

def shuffled(iterable):
    # randomly shuffle a copy of iterable
    items = list(iterable)
    random.shuffle(items)
    return items


# extract adjacency matrix for experiment expDate
def createAdjMat(expDate):
	adjMatPath = os.path.join("data", '_'.join([expDate, "Experiment"]), "Input_data", "adjacency_matrix.txt")
	with open(adjMatPath) as fid:
		allMat = fid.readlines()
	allMat = [item.strip() for item in allMat]

	ret = []
	tmpMat = []
	for item in allMat:
		if item != '#':
			tmpMat.append(item)
		else:
			ret.append(tmpMat)
			tmpMat = []

	for idx, mat in enumerate(ret):
		for i in range(len(mat)):
			mat[i] = [True if element == "True" else False for element in mat[i].split()]
		ret[idx] = mat

	return ret


def getBatchConfig(expDate):
    batchConfigPath = os.path.join("data", '_'.join([expDate, "Experiment"]), "Input_data", "batch_configuration.txt")
    with open(batchConfigPath) as fid:
    	batch_config = fid.readlines()
    batch_config = [item.strip() for item in batch_config]
    return batch_config


def getNetworkConfig(expDate):
    networkConfigPath = os.path.join("data", '_'.join([expDate, "Experiment"]), "Input_data", "network_configuration.txt")
    with open(networkConfigPath) as fid:
    	network_config = fid.readlines()
    network_config = [item.strip() for item in network_config]
    return network_config


def expSummary(expDate):
	batchConfig = getBatchConfig(expDate)
	network_config = getNetworkConfig(expDate)
	adjMat = createAdjMat(expDate)
	numExp = len(batchConfig)
	summary = {}
	for exp in range(numExp):
		numAdversarial = int(batchConfig[exp].split()[3])
		numVisibleNods = int(batchConfig[exp].split()[-2])
		communication = network_config[exp].split()[2]
		network = network_config[exp].split()[0]
		numAgents = numAdversarial + 20
		summary[exp] = {"numAdv": numAdversarial, "numVisible": numVisibleNods, \
					    "communication": communication, "network": network, \
						"numAgents": numAgents, "adjMat": adjMat[exp]}
	return summary


def checkAdjacencyMatrix(mat):
	nodes = len(mat)
	for node in range(nodes):
		if mat[node][node] == True:
			errMsg = "Node should not be connected to itself"
			return False

	if len(mat) != len(mat[0]):
		errMsg = "Wrong dimensions of adjacency matrix"
		return False
	return True


def generateAdjacencyMatrix(graph):
	n = nx.number_of_nodes(graph)
	graph = nx.convert.to_dict_of_lists(graph)
	adjacencyMatrix = [[False]*n for i in range(n)]

	for node, neighbors in graph.items():
		for neighbor in neighbors:
			adjacencyMatrix[node][neighbor] = True

	if not checkAdjacencyMatrix(adjacencyMatrix):
		return "See error message for more info"
	else:
		return adjacencyMatrix


# Generate Albert-Barabasi graph
def AlbertBarabasi(n, m0, m, d, display=None, seed=None):
	"""
	n: Albert-Barabasi graph on n vertices
	m: number of edges to attach from a new node to existing nodes
	"""
	if m > m0:
		print("Error: m must be less or equal to m0")
		return

	# counter = 0
	while True:
		G = nx.barabasi_albert_graph(n, m, seed=None)
		maxDegree = max([item for item in G.degree().values()])
		if maxDegree <= d:
			break
		# counter += 1

	# avgDegree = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
	# print avgDegree

	if display:
		nx.draw(G)
		img_name = 'barabasi_albert_%i_%i.png' % (n, m)
		plt.savefig(img_name)
		sh.open(img_name)

	adjacencyMatrix = generateAdjacencyMatrix(G)
	return adjacencyMatrix


# Generate Erdos-Renyi graph
def ErdosRenyi(n, m, d, display=None, seed=None):
	"""
	n: the number of nodes
	m: the number of edges
	"""
	# counter = 0
	# naive way to generate connected graph
	while True:
		if m <= 30:
			G = nx.gnm_random_graph(n, m, seed=None, directed=False)
		else:
			G = nx.dense_gnm_random_graph(n, m, seed=None)
		maxDegree = max([item for item in G.degree().values()])
		if nx.is_connected(G) and maxDegree <= d:
			break
		# if maxDegree > d:
			# counter += 1

	# avgDegree = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
	# print avgDegree

	if display:
		nx.draw(G)
		img_name = 'erdos_renyi_%i_%i.png' % (n, m)
		plt.savefig(img_name)
		sh.open(img_name)

	adjacencyMatrix = generateAdjacencyMatrix(G)
	return adjacencyMatrix