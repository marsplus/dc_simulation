from __future__ import division
import random
import time
import math
import pandas as pd
import numpy as np
from utils import *
from mesa import Agent, Model
from mesa.time import RandomActivation
from multiprocessing import Pool
from mesa.datacollection import DataCollector

# random.seed(123)

class GameAgent(Agent):
    def __init__(self, unique_id, isVisibleNode, isAdversarial, neighbors, visibleColorNodes, inertia, model):
        super().__init__(unique_id, model)
        # whether this node is a visible node
        self.isVisibleNode = isVisibleNode
        # whether this node is an adversarial
        self.isAdversarial = isAdversarial
        self.neighbors = [agent for agent in model.schedule.agents if
                                                 agent.unique_id in neighbors]
        # for each agent initial color is white
        self.color = "white"
        self.visibleColorNodes = [agent for agent in model.schedule.agents if
                                                 agent.unique_id in visibleColorNodes]
        # probability to make a change
        self.p = inertia

        # statistics
        self.colorChanges = 0


    # determine if there is any visible color node in the neighborhood
    def hasVisibleColorNode(self):
        return len(self.visibleColorNodes) > 0


    # return current majority color
    def majorityColor(self):
        # regular node
        if not self.isAdversarial and not self.isVisibleNode:
            # if there is any visible color node in the neighbor
            if self.hasVisibleColorNode():
                visibleColor = [agent.color for agent in self.visibleColorNodes if agent.color != "white"]
                # if no visible node makes choice
                if len(visibleColor) == 0:

                    # if there is indeed visible color node, but none of them
                    # makes a decision, then the agent doesn't make any decision
                    # either
                    return self.color
                else:
                    # print("hello")
                    red = len([color for color in visibleColor if color == "red"])
                    green = len(visibleColor) - red
                    if red > green:
                        return red
                    elif green > red:
                        return green
                    else:
                        # if #red == #green, randomly pick one
                        random.choice(["red", "green"])

            # if no visible color node, follow majority
            else:
                red = green = 0
                for agent in self.neighbors:
                    if agent.color == 'red':
                        red += 1
                    elif agent.color == 'green':
                        green += 1
                    else:
                        pass
                if red > green:
                    return "red"
                elif green > red:
                    return "green"
                else:
                    # if #red == #green, randomly pick on
                    random.choice(["red", "green"])

        # visible nodes choose majority color, whereas adversarial
        # nodes choose the opposite
        else:
            red = green = 0
            for agent in self.neighbors:
                if agent.color == "red":
                    red += 1
                elif agent.color == "green":
                    green += 1
                else:
                    pass
            if red > green:
                if self.isVisibleNode:
                    return "red"
                else:
                    return "green"
            elif green > red:
                if self.isVisibleNode:
                    return "green"
                else:
                    return "red"
            else:
                return random.choice(["red", "green"])

    # make a decision
    def step(self):
        major_color = self.majorityColor()
        if major_color == "white":
            # agents cannot go back to white once they
            # choosed certain color
            pass
        else:
            if random.random() < self.p:
                if major_color == "red":
                    self.color = "red"
                else:
                    self.color = "green"
            # each agent has a small probability to not make
            # any decision
            else:
                # do nothing
                pass

    def degree(self):
        return len(self.neighbors)


# # this function is used to retrieve color information
# at each time step
def getCurrentColor(model):
    ret = {"red": 0, "green": 0, "white": 0}
    current_color = [(a.color, a.unique_id) for a in model.schedule.agents]
    for item in current_color:
        ret[item[0]] += 1
    return ret


# get the number of nodes selecting red in each time step
def getRed(model):
	red = 0
	current_color = [a.color for a in model.schedule.agents]
	for color in current_color:
		if color == "red":
			red += 1
	return red


def getGreen(model):
	green = 0
	current_color = [a.color for a in model.schedule.agents]
	for color in current_color:
		if color == "green":
			green += 1
	return green



class DCGame(Model):
	def __init__(self, adjMat, numVisibleColorNodes, numAdversarialNodes, inertia):
		self.adjMat = adjMat
		self.numVisibleColorNodes = numVisibleColorNodes
		self.numAdversarialNodes = numAdversarialNodes
		self.adversarialNodes = []
		self.visibleColorNodes = []
		self.schedule = RandomActivation(self)
		self.numAgents = len(adjMat)
		self.inertia = inertia
        # if there are 20 consensus colors then a 
        # terminal state is reached
		self.terminate = False
        

        # convert adjMat to adjList
		def getAdjList(adjMat):
			adjList = {key: [] for key in range(self.numAgents)}
			for node in range(self.numAgents):
				adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == True]
			return adjList

		self.adjList = getAdjList(self.adjMat)


        ############# designate adversarial #############
        # (node, degree)
		node_deg = [(idx, count(adjMat[idx])) for idx in range(self.numAgents)]
        # select the top-k nodes with largest degrees as adversarial
		node_deg.sort(key=lambda x: x[1], reverse=True)
		self.adversarialNodes = [item[0] for item in node_deg[:self.numAdversarialNodes]]


        ############# designate visible nodes #############
		availableNodes = shuffled(node_deg[numAdversarialNodes:])
		self.visibleColorNodes = [item[0] for item in availableNodes[:self.numVisibleColorNodes]]

		self.regularNodes = [n for n in range(self.numAgents) if n not in self.visibleColorNodes
                            and n not in self.adversarialNodes]


        ############# initialize all agents #############
		for i in range(self.numAgents):
			# if i is a visible node
			isVisibleNode = i in self.visibleColorNodes
            # if i is an adversarial
			isAdversarial = i in self.adversarialNodes

			neighbors = self.adjList[i]
            # visible color nodes in i's neighbors
			visibleColorNodes = list(set(neighbors) & set(self.visibleColorNodes))
			inertia = self.inertia

            # def __init__(self, unique_id, visibleNode, adversarial, neighbors, visibleColorNodes, inertia, model):

			# print("Add agent:", (i, visibleNode, adversarial, neighbors, visibleColorNodes))
			a = GameAgent(i, isVisibleNode, isAdversarial, neighbors, visibleColorNodes, inertia, self)
			self.schedule.add(a)

		self.datacollector = DataCollector(
				model_reporters = {"red": getRed, "green": getGreen},
				agent_reporters = {"agent_color": lambda a: a.color}
				) 

    # simulate the whole model for one step
	def step(self):
		# # if either red or green reaches consensus, terminates!
		# in terminal state we do not collect data
		test = getCurrentColor(self)
		if test['red'] >= 20 or test['green'] >= 20:
			pass
		else:
			self.datacollector.collect(self)
		self.schedule.step()


#define a wrapper function for multi-processing
def simulationFunc(args):
	# dispatch arguments
	numSimulation, gameTime, numRegularPlayers, numVisibleNodes, numAdversarialNodes, net, inertia = args

	# calculate how many players we have
	numPlayers = numRegularPlayers + numAdversarialNodes

	# network parameters
	################################
	### CODE FROM Zlatko ###
	# each new node is connected to m new nodes
	m = 3
	no_consensus_nodes_range = range(11)
    # max degrees
	maxDegree = 17
	BA_edges = [(numRegularPlayers + no_consensus_nodes - 3) * m for \
    				no_consensus_nodes in no_consensus_nodes_range]
	ERD_edges = [edges_no for edges_no in BA_edges]
	ERS_edges = [int(math.ceil(edges_no/2.0)) for edges_no in ERD_edges]
	################################

	# ret contains simulated results
	ret = []
	for j in range(numSimulation):
		if j % 1000 == 0:
			print("Current number of simulations: ", j)

		# generate adjMat according to network type
		if net == 'Erdos-Renyi-dense':
			adjMat = ErdosRenyi(numPlayers, ERD_edges[numAdversarialNodes], maxDegree)
		elif net == 'Erdos-Renyi-sparse':
			adjMat = ErdosRenyi(numPlayers, ERS_edges[numAdversarialNodes], maxDegree)
		else:
			adjMat = AlbertBarabasi(numPlayers, m, maxDegree)

		
		model = DCGame(adjMat, numVisibleNodes, numAdversarialNodes, inertia)
		for i in range(gameTime):
			model.step()

		ret.append(model.datacollector.get_model_vars_dataframe())

	# determine success ratio
	# if a game reaches consensus under 60s, then it's successful
	ratio = count([len(item) < gameTime for item in ret]) / numSimulation
	# result.append([numVisibleNodes, numAdversarialNodes, networkType, ratio])
	return ratio



if __name__ =="__main__":
	# iterate over all inertia values
	# for inertia in np.arange(0.1, 1.0, 0.1):
	inertia = 0.9
	print("Current inertia: ", inertia)

	# allExpDate = ['2017_03_10']

	# experimental parameters
	################################
	numSimulation = 50000
	gameTime = 60
	# inertia = 0.5
	numRegularPlayers = 20
	################################


	args = []
	networks = ['Erdos-Renyi-dense', 'Erdos-Renyi-sparse', 'Barabasi-Albert']
	numVisibleNodes_ = [0, 1, 2, 5]
	numAdversarialNodes_ = [0, 2, 5]


	# get all combinations of parameters
	for net in networks:
		for numVisible in numVisibleNodes_:
			for numAdv in numAdversarialNodes_:
				print("Generate parameters combinations: ", (net, numVisible, numAdv))
				args.append((numSimulation, gameTime, numRegularPlayers, numVisible, 
						 numAdv, net, inertia))

	# initialize processes pool
	pool = Pool(processes=50)
	result = pool.map(simulationFunc, args)

	# match results with parameters
	for i in range(len(result)):
		result[i] = list(args[i][3:6]) + [result[i]]

	result = pd.DataFrame(result)
	result.columns = ['#visibleNodes', '#adversarial', 'network', 'ratio']
	result.to_csv('./result/inertia=%.2f.csv' % inertia, index=None)

	pool.close()
	pool.join()
	






