#-------------------------------------------------------------------------------
# Name:        adversarial
# Purpose:     Generate input files for adversarial experiments
#
# Author:      Zlatko
#
# Created:     07.06.2016
#-------------------------------------------------------------------------------

from networkx import nx
import random
import os
import time
import argparse
import copy
import math


import numpy as np
from scipy import stats

from graphs import graph

# Random sampling
# -----------------------------------------------------------
def _random_element(distrib):
    """ Returns an element sampled from distrib """
    return distrib.rvs(size=1)[0]

def _random_subset(distrib, m):
    """ Returns m unique elements sampled from distrib """
    subset = set()
    while len(subset)<m:
        x = _random_element(distrib)
        subset.add(x)
    return subset

def _normalized_probabilities(p):
    """ Returns an array with values proportional to the values of p,
    but scaled so that they sum up to 1 """

    return [float(x) / float(sum(p)) for x in p]

# Network generators and helper methods
# -----------------------------------------------------------
def checkAdjacencyMatrix(mat):
	nodes = len(mat)
	for node in range(nodes):
		if mat[node][node] == True:
			errMsg = "node should not be connected to itself"
			return False

	if len(mat) != len(mat[0]):
		errMsg = "wrong dimensions of adjacency matrix"
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

def ErdosRenyi(n, m, d, display=None, seed=None):
    """
    n: the number of nodes
    m: the number of edges
    d: maximum node degree
    """

	# naive way to generate connected graph
    while True:
        G = nx.gnm_random_graph(n, m, seed=None, directed=False)

    	maxDegree = max([item for item in G.degree().values()])
        if nx.is_connected(G) and maxDegree <= d:
            break

    adjacencyMatrix = generateAdjacencyMatrix(G)
    return adjacencyMatrix

def AlbertBarabasi(n, m, d, display=None, seed=None):
	"""
	n: Albert-Barabasi graph on n vertices
	m: number of edges to attach from a new node to existing nodes
    d: maximum node degree
	"""
	while True:
		G = nx.barabasi_albert_graph(n, m, seed=None)
		maxDegree = max([item for item in G.degree().values()])
		if maxDegree <= d:
			break

	adjacencyMatrix = generateAdjacencyMatrix(G)
	return adjacencyMatrix

def main():
	## set parameters here
    # total number of nodes in the Consensus team
    consensus_nodes = 20

    # possible values for the number of nodes of the No-Consensus team
    # current range: 0-5
    no_consensus_nodes_range = range(11)

    # maximum allowed vertex degree
    max_degree = 17

    # each new node is connected to m new nodes (BA)
    m = 3

    # number of edges for each network size under the different network generating models
    BA_edges = [(consensus_nodes + no_consensus_nodes -3) * m for no_consensus_nodes in no_consensus_nodes_range]
    ERD_edges = [edges_no for edges_no in BA_edges]
    ERS_edges = [int(math.ceil(edges_no/2.0)) for edges_no in ERD_edges]

    networkConfigurationFile = 'input_files/network_configuration.txt'
    adjacencyMatrixFile      = 'input_files/adjacency_matrix.txt'
    batchConfigurationFile   = 'input_files/batch_configuration.txt'

    practiceNetworkConfigurationFile = 'input_files/network_configuration_practice.txt'
    practiceAdjacencyMatrixFile      = 'input_files/adjacency_matrix_practice.txt'
    practiceBatchConfigurationFile   = 'input_files/batch_configuration_practice.txt'

    # specify the treatments and the number of games per each treatment

    games_per_treatment = 3
    graphNames = ['Barabasi-Albert', 'Erdos-Renyi-dense', 'Erdos-Renyi-sparse'] * games_per_treatment
    incentiveLevels = ['none']
    communicationScope = ['local']
    communicationTypes = ['unstructured']
    messageOptions = ['NA']

    gamesInBatch = ['1']
    batchTreatmentType = ['adversarial']
    adversaryRoleAssignment = ['random']
    numberOfAdversaries = ['0', '2', '5']
    regularPlayersPayout = ['balanced']
    adversariesPayout = ['noConsensusOnly']

    visibilityMode = ['visibleConsensusNodes']
    visibleConsensusNodes = ['1', '2']
    consensusVisibilityAssignment = ['random']

    """
    Here, since we're dealing with one-game batches, we keep track of full game configurations,
    but in general we would need to deal with batch and network configurations separately, as one
    batch configuration line may correspond to several network configuration lines in the input files.
    """

    # specify the regular games configurations of 1-game treatments
    configurations = []
    for graN in graphNames:
        for incL in incentiveLevels:
            for comS in communicationScope:
                for comT in communicationTypes:
                    for msgO in messageOptions:
                        for gamB in gamesInBatch:
                            for batT in batchTreatmentType:
                                for advA in adversaryRoleAssignment:
                                    for numA in numberOfAdversaries:
                                        for regP in regularPlayersPayout:
                                            for advP in adversariesPayout:
                                                for visM in visibilityMode:
                                                    for visC in visibleConsensusNodes:
                                                        for conA in consensusVisibilityAssignment:
                                                            configurations.append((graN, incL, comS, comT, msgO, gamB, batT, advA, numA, regP, advP, visM, visC, conA))


    random.shuffle(configurations)
    regular_games = len(configurations)
    print(len(configurations))

    adjacencyMatrixList = []
    for config in configurations:
        if config[0] == 'Erdos-Renyi-dense':
            adjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERD_edges[int(config[8])], max_degree))
        elif config[0] == 'Erdos-Renyi-sparse':
            adjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERS_edges[int(config[8])], max_degree))
        elif config[0] == 'Barabasi-Albert':
            adjacencyMatrixList.append(AlbertBarabasi(consensus_nodes + int(config[8]), m, max_degree))

    print(len(adjacencyMatrixList))

    # write network configurations to file
    with open(networkConfigurationFile, 'w') as netConfig:
    	for item in configurations[:regular_games]:
    		netConfig.write(' '.join(item[0:5]) + '\n')

    # write batch configurations to file
    with open(batchConfigurationFile, 'w') as batchConfig:
    	for item in configurations[:regular_games]:
    		batchConfig.write(' '.join(item[5:14]) + '\n')

    # write adjacency matrices to file
    with open(adjacencyMatrixFile, 'w') as adjMat:
    	for mat in adjacencyMatrixList[:regular_games]:
    		for row in mat:
    			row = [str(item) for item in row]
    			adjMat.write(' '.join(row) + '\n')
    		adjMat.write('#\n')

    # ---------------------------------------------------------
    # deal with practice input data

    practice_games = 5
    practiceConfigurations = random.sample(configurations, practice_games)

    random.shuffle(practiceConfigurations)
    practice_games = len(practiceConfigurations)
    print(len(practiceConfigurations))

    practiceAdjacencyMatrixList = []
    for config in practiceConfigurations:
    	if config[0] == 'Erdos-Renyi-dense':
            practiceAdjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERD_edges[int(config[8])], max_degree))
        elif config[0] == 'Erdos-Renyi-sparse':
            practiceAdjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERS_edges[int(config[8])], max_degree))
        elif config[0] == 'Barabasi-Albert':
            practiceAdjacencyMatrixList.append(AlbertBarabasi(consensus_nodes + int(config[8]), m, max_degree))

    # print(len(practiceAdjacencyMatrixList))

    # write network configurations to file
    with open(practiceNetworkConfigurationFile, 'w') as netConfig:
    	for item in practiceConfigurations[:practice_games]:
    		netConfig.write(' '.join(item[0:5]) + '\n')

    # write batch configurations to file
    with open(practiceBatchConfigurationFile, 'w') as batchConfig:
    	for item in practiceConfigurations[:practice_games]:
    		batchConfig.write(' '.join(item[5:14]) + '\n')

    # write adjacency matrices to file
    with open(practiceAdjacencyMatrixFile, 'w') as adjMat:
    	for mat in practiceAdjacencyMatrixList[:practice_games]:
    		for row in mat:
    			row = [str(item) for item in row]
    			adjMat.write(' '.join(row) + '\n')
    		adjMat.write('#\n')

    # ---------------------------------------------------------

if __name__ == '__main__':
    main()
