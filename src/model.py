from __future__ import division
import random
import pickle
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
        self.game = model
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


    def getNeighborMajorColor(self):
        neighbor_color = {"red": 0, "green": 0}
        for a in self.neighbors:
            if a.color != "white":
                neighbor_color[a.color] += 1

        if neighbor_color["red"] > neighbor_color["green"]:
            return "red"
        elif neighbor_color["red"] < neighbor_color["green"]:
            return "green"
        else:
            return random.choice(["red", "green"])


    # return current majority color
    # this actually corresponds to different players' strategies
    def decision(self):

        # regular node
        if not self.isAdversarial and not self.isVisibleNode:
            # if there is any visible color node in the neighbor
            if self.hasVisibleColorNode():

                # if random.random() < 0.9:

                visibleColor = [agent.color for agent in self.visibleColorNodes if agent.color != "white"]
                # if no visible node makes choice
                if len(visibleColor) == 0:

                    # if there is indeed visible color node, but none of them
                    # makes a decision, then the agent doesn't make any decision
                    # either
                    return self.color
                else:
                    numRed = len([color for color in visibleColor if color == "red"])
                    numGreen = len(visibleColor) - numRed
                    if numRed > numGreen:
                        return "red"
                    elif numRed < numGreen:
                        return "green"
                    else:
                        # if #red == #green, randomly pick one
                        # return random.choice(["red", "green"])
                        return self.getNeighborMajorColor()

                # else:
                #     return self.getNeighborMajorColor()

            # if no visible color node, follow majority
            else:
                return self.getNeighborMajorColor() 

        # visible nodes choose majority color, whereas adversarial
        # nodes choose the opposite
        else:
            # either visible node or adversarial node
            assert self.isVisibleNode | self.isAdversarial == True

            color = self.getNeighborMajorColor()
            if self.isVisibleNode:
                # print(self.isVisibleNode)
                return color
            else:
                # print(self.isAdversarial)
                # return "red" if color == "green" else "green"
                return "red" if color == "green" else "green"



    # make a decision
    def step(self):
        # check game state
        current_color = getCurrentColor(self.game)
        if current_color["red"] == 20 or current_color["green"] == 20:
            self.game.setTerminal()
        else:
            decision_color = self.decision()

            # ###
            # if self.hasVisibleColorNode() and  self.isAdversarial:
            #     print('*'*80)
            #     if self.hasVisibleColorNode():
            #         vNodeColor = [(a.unique_id, a.color) for a in self.neighbors if a.unique_id in self.game.visibleColorNodes]
            #         print("visible color: %s" % vNodeColor[0][1])
            #         print("%i: %s" % (self.unique_id, decision_color))
            #     print('*'*80)
            # ###
            
            if decision_color == "white":
                # agents cannot go back to white once they
                # choosed certain color
                pass
            else:
                if random.random() < self.p:
                    self.color = decision_color
                # each agent has a small probability to not make
                # any decision
                else:
                    # do nothing
                    pass

    def degree(self):
        return len(self.neighbors)


## this function is used to retrieve color information
##  of regular nodes at each time steop
def getCurrentColor(model):
    ret = {"red": 0, "green": 0}
    current_color = [a.color for a in model.schedule.agents\
                if a.unique_id in model.regularNodes]
    # a  = set(model.visibleColorNodes) & set(model.regularNodes) == set(model.visibleColorNodes)
    # print(a)
    for item in current_color:
        if item != "white":
            ret[item] += 1
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
        # self.adversarialNodes = []
        self.visibleColorNodes = []
        self.regularNodes = []
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
        availableNodes = shuffled(node_deg[self.numAdversarialNodes:])
        self.visibleColorNodes = [item[0] for item in availableNodes[:self.numVisibleColorNodes]]

        self.regularNodes = [n for n in range(self.numAgents) if n not in self.adversarialNodes]
        # make sure we have 20 regular nodes
        assert len(self.regularNodes) == 20

        # adversarial nodes and regular nodes should not overlap
        assert set(self.adversarialNodes) & set(self.regularNodes) == set()
        # visible nodes should belong to regular nodes
        assert set(self.visibleColorNodes) & set(self.regularNodes) == set(self.visibleColorNodes)

        ############# initialize all agents #############
        for i in range(self.numAgents):
            # if i is a visible node
            isVisibleNode = i in self.visibleColorNodes
            # if i is an adversarial
            isAdversarial = i in self.adversarialNodes
            # make sure adversarial nodes are not intersected with visible nodes
            assert isVisibleNode & isAdversarial == False

            neighbors = self.adjList[i]
            # visible color nodes in i's neighbors
            vNode = list(set(neighbors) & set(self.visibleColorNodes))
            inertia = self.inertia

            # print("Add agent:", (i, visibleNode, adversarial, neighbors, visibleColorNodes))
            a = GameAgent(i, isVisibleNode, isAdversarial, neighbors, vNode, inertia, self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
                        model_reporters = {"red": getRed, "green": getGreen},
                        agent_reporters = {"agent_color": lambda a: a.color}
                        )

    # simulate the whole model for one step
    def step(self):
        # # # if either red or green reaches consensus, terminates!
        # # in terminal state we do not collect data

        if not self.terminate:
            self.datacollector.collect(self)
            self.schedule.step()
        return self.terminate


    def setTerminal(self):
        assert self.terminate == False
        self.terminate = True


class BatchResult(object):
    def __init__(self, data, args, arg_id):
        # used to uniquely pair BatchResult and args
        self.ret_id = arg_id
        self.data = data
        self.args = args
        self.gameTime = args[1]
        self.numVisibleNodes = args[3]
        self.numAdversarialNodes = args[4]
        self.network = args[5]
        self.consensus_ret = None
        self.dynamics_ret = None
        self.time_ret = None

    def generateResult(self):
        # generate a DataFrame where each row corresponds
        # to a simulation
        consensus_ret = []
        for i in range(len(self.data)):
            if_consensus = 1 if len(self.data[i]) < self.gameTime else 0
            consensus_ret.append((self.numVisibleNodes, self.numAdversarialNodes, self.network, if_consensus))
        consensus_ret = pd.DataFrame(consensus_ret)
        self.consensus_ret = consensus_ret

        # generate detailed dynamics for each simulation
        dynamics_ret = {}
        for i in range(len(self.data)):
            dynamics_ret[i] = self.data[i]
        self.dynamics_ret = dynamics_ret

        # generate time to consensus
        time_ret = []
        for i in range(len(self.data)):
            t = len(self.data[i])
            time_ret.append((self.numVisibleNodes, self.numAdversarialNodes, self.network, t))
        time_ret = pd.DataFrame(time_ret)
        self.time_ret = time_ret

    def getConsensusResult(self):
        return self.consensus_ret

    def getDynamicsResult(self):
        return self.dynamics_ret

    def getTimeResult(self):
        return self.time_ret


def getAdjMat(net, numPlayers, numRegularPlayers, numAdversarialNodes):
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

    # generate adjMat according to network type
    if net == 'Erdos-Renyi-dense':
        adjMat = ErdosRenyi(numPlayers, ERD_edges[numAdversarialNodes], maxDegree)
    elif net == 'Erdos-Renyi-sparse':
        adjMat = ErdosRenyi(numPlayers, ERS_edges[numAdversarialNodes], maxDegree)
    else:
        adjMat = AlbertBarabasi(numPlayers, m, maxDegree)

    return adjMat



#define a wrapper function for multi-processing
def simulationFunc(args):
    # dispatch arguments
    numSimulation, gameTime, numRegularPlayers, numVisibleNodes, numAdversarialNodes, net, inertia, arg_id = args

    # calculate how many players we have
    numPlayers = numRegularPlayers + numAdversarialNodes

    # ret contains simulated results
    ret = []
    for j in range(numSimulation):
        if j % 1000 == 0:
            print("Current number of simulations: ", j)

        adjMat = getAdjMat(net, numPlayers, numRegularPlayers, numAdversarialNodes)
        model = DCGame(adjMat, numVisibleNodes, numAdversarialNodes, inertia)
        for i in range(gameTime):
            # check whether a game terminates
            terminate = model.step()
            if terminate:
                break
        ret.append(model.datacollector.get_model_vars_dataframe())

    # the collected data is actually an object
    result = BatchResult(ret, args, arg_id)
    return result



def combineResults(result, args, folder=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    inertia = args[0][-2]
    # result is returned from multi-processing 
    for ret in result:
        ret.generateResult()

    consensus_ret = pd.concat([item.getConsensusResult() for item in result])
    consensus_ret.columns = ['#visibleNodes', '#adversarial', 'network', 'ratio']
    p = os.path.join(folder, 'consensus_inertia=%.2f.csv' % inertia)
    consensus_ret.to_csv(p, index=None)


    time_ret = pd.concat([item.getTimeResult() for item in result])
    time_ret.columns = ['#visibleNodes', '#adversarial', 'network', 'time']
    p = os.path.join(folder, 'time_inertia=%.2f.csv' % inertia)
    time_ret.to_csv(p, index=None)

    dynamics_ret = {args[idx]: item.getDynamicsResult() for idx, item in enumerate(result)}
    p = os.path.join(folder, 'dynamics_inertia=%.2f.p' % inertia)
    with open(p, 'wb') as fid:
        pickle.dump(dynamics_ret, fid)



if __name__ =="__main__":
    # iterate over all inertia values
    for inertia in np.arange(0.9, 0.8, -0.1):
        print("Current inertia: ", inertia)

        # experimental parameters
        ################################
        numSimulation = 1
        gameTime = 60
        # inertia = 0.5
        numRegularPlayers = 20
        ################################


        args = []
        # networks = ['Erdos-Renyi-dense', 'Erdos-Renyi-sparse', 'Barabasi-Albert']
        networks = ['Erdos-Renyi-dense']
        numVisibleNodes_ = [1]
        numAdversarialNodes_ = [2]


        # get all combinations of parameters
        counter = 0
        for net in networks:
            for numVisible in numVisibleNodes_:
                for numAdv in numAdversarialNodes_:
                    print("Generate parameters combinations: ", (net, numVisible, numAdv))
                    args.append((numSimulation, gameTime, numRegularPlayers, numVisible,
                                     numAdv, net, inertia, counter))
                    counter += 1

        result = simulationFunc(args[0])
        # result.generateResult()
        # a = result.getConsensusResult()
        # a.columns = ['#visibleNodes', '#adversarial', 'network', 'ratio']


        # # initialize processes pool
        # pool = Pool(processes=36)
        # result = pool.map(simulationFunc, args)
        # combineResults(result, args, 'result/newStrategy_regularNodes')


        # pool.close()
        # pool.join()


