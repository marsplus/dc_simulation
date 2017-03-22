import os
import random
from copy import deepcopy

def count(seq):
	return sum(x == 'True' for x in seq)

def count_True(seq):
	return sum(x == True for x in seq)

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
		ret[idx] = [item.split() for item in mat]

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
