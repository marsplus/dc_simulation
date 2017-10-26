from __future__ import division
import random
import pickle
import time
import math
import ast
import pandas as pd
import numpy as np
from log import Log
from utils import *
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.time import FollowVisibleActivation
from mesa.time import SimultaneousActivation
from multiprocessing import Pool
from collections import defaultdict
from mesa.datacollection import DataCollector
import numpy.linalg as LA
import itertools

random.seed(0)

class GameAgent(Agent):
    def __init__(self, unique_id, isVisibleNode, isAdversarial, neighbors, visibleColorNodes, inertia, beta, model):
        super().__init__(unique_id, model)
        self.game = model
        # whether this node is a visible node
        self.isVisibleNode = isVisibleNode
        # whether this node is an adversarial
        self.isAdversarial = isAdversarial
        self.neighbors = neighbors

        # for each agent initial color is white
        self.new_color = "white"
        self.color = "white"            # still used by neighbors in the same time step

        self.visibleColorNodes = visibleColorNodes

        # probability to make a change
        self.p = inertia
        self.regular_p = 0.84

        # randomize regular players' (excluding visibles)
        # decision
        self.beta = beta

        self.numPlayers = self.game.numAgents

        #added by Yifan
        self.beta_majority = 0.77
        self.beta_tie = 0.9
        self.unique_id = unique_id

        self.colorChanges = 0


    def __hash__(self):
        return hash(self.unique_id)

    def __eq__(self, other):
        return self.unique_id == other.unique_id

    def instantiateNeighbors(self, model):
        self.neighbors = [agent for agent in model.schedule.agents if
                            agent.unique_id in self.neighbors]

    def instantiateVisibleColorNodes(self, model):
        self.visibleColorNodes = [agent for agent in model.schedule.agents if
                            agent.unique_id in self.visibleColorNodes]


    # determine if there is any visible color node in the neighborhood
    def hasVisibleColorNode(self):
        return len(self.visibleColorNodes) > 0

    # if anybody in the neighbor makes decision
    def hasNeighborDecision(self):
        return [agent.color for agent in self.neighbors if agent.color != "white"]


    def getNeighborMajorColor(self):
        neighbor_color = {"red": 0, "green": 0}
        for a in self.neighbors:
            if a.color != "white":
                neighbor_color[a.color] += 1

        # take one's own decision into account
        if self.color != "white":
            neighbor_color[self.color] += 1

        if neighbor_color["red"] > neighbor_color["green"]:
            # dominant = True if and only if red > green
            dominant = True
            return ("red", dominant)
        elif neighbor_color["red"] < neighbor_color["green"]:
            # dominant = True if and only if red < green
            dominant = True
            return ("green", dominant)
        else:
            # dominant != True if and only if red == green
            dominant = False
            return (random.choice(["red", "green"]), dominant)

    def getNonAdversarialNeighborMajorColor(self):
        neighbor_color = {"red": 0, "green": 0}
        nonAdversarialNeighbors = [neighbor for neighbor in self.neighbors if not neighbor.isAdversarial]   #regular neighbors
        for a in nonAdversarialNeighbors:
            if a.color != "white":
                neighbor_color[a.color] += 1

        # take one's own decision into account
        if self.color != "white":
            neighbor_color[self.color] += 1

        if neighbor_color["red"] > neighbor_color["green"]:
            # dominant = True if and only if red > green
            dominant = True
            return ("red", dominant)
        elif neighbor_color["red"] < neighbor_color["green"]:
            # dominant = True if and only if red < green
            dominant = True
            return ("green", dominant)
        else:
            # dominant != True if and only if red == green
            dominant = False
            return (random.choice(["red", "green"]), dominant)


    # return current majority color
    # this actually corresponds to different players' strategies
    def pickInitialColor(self):
        mid_game = 0
        end_game = 0
        if self.game.time >= 45:
            end_game = 1
        elif self.game.time >= 30 and self.game.time <= 45:
            mid_game = 1
        neighbors = self.degree()

        vis_neighbors = [neighbor for neighbor in self.neighbors if neighbor.isVisibleNode]
        neighbors_vis = float(len(vis_neighbors)) / float(neighbors)
        if len(vis_neighbors) != 0:
            green_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color == "green"])) / float(len(vis_neighbors))
            red_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color == "red"])) / float(len(vis_neighbors))
        else:
            green_local_vis = 0
            red_local_vis = 0
        inv_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode]
        neighbors_inv = float(len(inv_neighbors)) / float(neighbors)
        if len(inv_neighbors) != 0:
            green_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color == "green"])) / float(len(inv_neighbors))
            red_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color == "red"])) / float(len(inv_neighbors))
        else:
            green_local_inv = 0
            red_local_inv = 0
        reg_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode and not neighbor.isAdversarial]
        neighbors_reg = float(len(reg_neighbors)) / float(neighbors)
        if len(reg_neighbors) != 0:
            green_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color == "green"])) / float(len(reg_neighbors))
            red_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color == "red"])) / float(len(reg_neighbors))
        else:
            green_local_reg = 0
            red_local_reg = 0

        diff_vis = abs(red_local_vis - green_local_vis)
        diff_inv = abs(red_local_inv - green_local_inv)
        diff_reg = abs(red_local_reg - green_local_reg)

        if self.isAdversarial:
            # adversarial node
            if self.hasVisibleColorNode():
                y = -2.68 - 0.04 * self.game.time + 1.03 * diff_vis + 1.01 * diff_inv + 0.21 * neighbors_vis    #trained on all games (time only)
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    y2 = -0.37 + 0.83 * green_local_vis     #trained on all games (time only)
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
            else:
                y = -2.18 - 0.016 * self.game.time + 1.45 * diff_inv    #trained on all games (time only)
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    y2 = -0.15 + 1.07 * green_local_inv - 0.69 * red_local_inv  #trained on all games (time only)
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"

        elif self.isVisibleNode:
            # visible node
            if self.hasVisibleColorNode():
                y = -1.95 + 0.86 * diff_vis + 0.61 * diff_inv       #trained on all games (time only)
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    y2 = 0.14 - 3.86 * green_local_inv - 1.6 * green_local_vis + 2.63 * red_local_inv + 2.41 * red_local_vis    #trained on all games (time only)
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"

            else:
                y = -1.93 + 1.77 * diff_inv     #trained on all games (time only)
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    y2 = 0.01 - 4.32 * green_local_inv + 4.32 * red_local_inv   #trained on all games (time only)
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
        else:
            # regular player
            if self.hasVisibleColorNode():
                y = -2.2 - 0.04 * self.game.time + 1.1 * diff_vis + 0.82 * diff_inv + 0.08 * neighbors_vis  #trained on all games (time only)
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    y2 = -0.08 - 2.88 * green_local_inv - 2.07 * green_local_vis + 3.41 * red_local_inv + 1.76 * red_local_vis  #trained on all games (time only)
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
            else:
                y = -1.94 - 0.03 * self.game.time + 1.63 * diff_inv + 0.01 * neighbors_inv  #trained on all games (time only)
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    y2 = -0.003 - 4.95 * green_local_inv + 5.11 * red_local_inv     #trained on all games (time only)
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"

    def pickSubsequentColor(self):
        mid_game = 0
        end_game = 0
        if self.game.time >= 45:
            end_game = 1
        elif self.game.time >= 30 and self.game.time <= 45:
            mid_game = 1
        neighbors = self.degree()

        vis_neighbors = [neighbor for neighbor in self.neighbors if neighbor.isVisibleNode]
        neighbors_vis = float(len(vis_neighbors)) / float(neighbors)
        if len(vis_neighbors) != 0:
            opposite_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color != "white" and neighbor.color != self.color])) / float(len(vis_neighbors))
            current_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color != "white" and neighbor.color == self.color])) / float(len(vis_neighbors))
        else:
            opposite_local_vis = 0
            current_local_vis = 0
        inv_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode]
        neighbors_inv = float(len(inv_neighbors)) / float(neighbors)
        if len(inv_neighbors) != 0:
            opposite_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color != "white" and neighbor.color != self.color])) / float(len(inv_neighbors))
            current_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color != "white" and neighbor.color == self.color])) / float(len(inv_neighbors))
        else:
            opposite_local_inv = 0
            current_local_inv = 0
        reg_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode and not neighbor.isAdversarial]
        neighbors_reg = float(len(reg_neighbors)) / float(neighbors)
        if len(reg_neighbors) != 0:
            opposite_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color != "white" and neighbor.color != self.color])) / float(len(reg_neighbors))
            current_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color != "white" and neighbor.color == self.color])) / float(len(reg_neighbors))
        else:
            opposite_local_reg = 0
            current_local_reg = 0

        features = [1, self.game.time, opposite_local_inv, opposite_local_vis, current_local_inv, current_local_vis, neighbors_inv, neighbors_vis]
        numFeatures = len(features)
        feature_vector = np.asmatrix(features).reshape(numFeatures, 1)


        if not self.isAdversarial and not self.isVisibleNode:
            # regular node
            if self.hasVisibleColorNode():
                w = [-3.75, 0, 1.12, 1.4, -0.85, 0, 0, 0]
                assert(len(w) == numFeatures)
                w = np.asmatrix(w).reshape(numFeatures, 1)
                # amplifier certain weights in the logistic regression
                assert(len(self.game.regularNodeAmplifier) == numFeatures)
                w += self.game.regularNodeAmplifier
                y = w.T * feature_vector
                y = y[0, 0]
                # y = -3.75 + 1.12 * opposite_local_inv + 1.4 * opposite_local_vis - 0.85 * current_local_inv    
                prob_of_change = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_change:
                    return "red" if self.color == "green" else "green"
                else:
                    return self.color
            else:
                w = [-3.94, 0.004, 2.47, 0, -0.51, 0, 0, 0]
                assert(len(w) == numFeatures)
                w = np.asmatrix(w).reshape(numFeatures, 1)
                # amplifier certain weights in the logistic regression
                assert(len(self.game.regularNodeAmplifier) == numFeatures)
                w += self.game.regularNodeAmplifier
                y = w.T * feature_vector
                y = y[0, 0]
                # y = -3.94 + 0.004 * self.game.time + 2.47 * opposite_local_inv - 0.51 * current_local_inv   #trained on all games (time only)
                prob_of_change = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_change:
                    return "red" if self.color == "green" else "green"
                else:
                    return self.color

        else:

            if self.isVisibleNode:
                #visible node
                if self.hasVisibleColorNode():
                    w = [-4.06, 0, 1.36, 1.55, 0, 0, -0.07, 0]
                    assert(len(w) == numFeatures)
                    w = np.asmatrix(w).reshape(numFeatures, 1)
                    y = w.T * feature_vector
                    y = y[0, 0]
                    # y = -4.06 + 1.36 * opposite_local_inv + 1.55 * opposite_local_vis - 0.07 * neighbors_inv    #trained on all games (time only)
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
                else:
                    w = [-4.31, 0, 2.85, 0, 0, 0, 0, 0]
                    assert(len(w) == numFeatures)
                    w = np.asmatrix(w).reshape(numFeatures, 1)
                    y = w.T * feature_vector
                    y = y[0, 0]
                    # y = -4.31 + 2.85 * opposite_local_inv   #trained on all games (time only)
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
            else:
                #adversary node
                if self.hasVisibleColorNode():
                    w = [-3.08, 0, 0, 0, 0, 0.9, 0, -0.15]
                    assert(len(w) == numFeatures)
                    w = np.asmatrix(w).reshape(numFeatures, 1)
                    y = w.T * feature_vector
                    y = y[0, 0]
                    # y = -3.08 + 0.9 * current_local_vis - 0.15 * neighbors_vis    #trained on all games (time only)
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
                else:
                    w = [-2.79, 0, -1.1, 0, 1.21, 0, 0, 0]
                    assert(len(w) == numFeatures)
                    w = np.asmatrix(w).reshape(numFeatures, 1)
                    y = w.T * feature_vector
                    y = y[0, 0]
                    # y = -2.79 - 1.1 * opposite_local_inv + 1.21 * current_local_inv     #trained on al games (time only)
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color


    # make a decision
    def step(self):
        # check game state
        current_color = getCurrentColor(self.game)
        if current_color["red"] == (self.numPlayers - self.game.numAdversarialNodes) or current_color["green"] == (self.numPlayers - self.game.numAdversarialNodes):
            self.game.setTerminal()
        else:
            if self.color != "white":
                #if node already picked a color
                decision_color = self.pickSubsequentColor()
                # if random.random() > self.changing_p:
                #     decision_color, dominant = self.getNeighborMajorColor()
                if self.color != decision_color:
                    self.game.colorChanges += 1
                    if not self.isAdversarial:
                        self.colorChanges += 1
                    # print("self: " + str(self.unique_id) + " new_color: " + str(decision_color) + " at time: " + str(self.game.time))
                self.new_color = decision_color

            else:
                # node has not yet picked a color
                decision_color = self.pickInitialColor()
                # if random.random() > self.choosing_p:
                #     decision_color, dominant = self.getNeighborMajorColor()
                if self.color != decision_color:
                    self.game.colorChanges += 1
                    self.game.hasSomeAgentChosenColor = True
                    if not self.isAdversarial:
                        self.colorChanges += 1
                    # print("self: " + str(self.unique_id) + " new_color: " + str(decision_color) + " at time: " + str(self.game.time))
                self.new_color = decision_color

    def advance(self):
        self.color = self.new_color

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
    counter = 0
    for item in current_color:
        if item != "white":
            if item == None:
                print("unique id: " + str(model.schedule.agents[counter].unique_id))
            ret[item] += 1
        counter += 1
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
    def __init__(self, adjMat, G, numVisibleColorNodes, numAdversarialNodes, inertia, beta, delay, \
            visibles, adversaries):
        self.adjMat = adjMat
        self.numVisibleColorNodes = numVisibleColorNodes
        self.numAdversarialNodes = numAdversarialNodes
        # self.adversarialNodes = []
        self.visibleColorNodes = []
        self.regularNodes = []
        self.schedule = SimultaneousActivation(self)
        self.numAgents = len(adjMat)
        self.inertia = inertia
        # if there are 20 consensus colors then a
        # terminal state is reached
        self.terminate = False
        self.time = 0
        # logging information
        self.log = Log()

        ##  temporarily added this for figuring out 
        ##  why visible nodes have no help
        self.hasConflict = False

        # randomize regular players (exclude visibles)
        # decision
        self.beta = beta

        # a amount of time to delay ordinary players' decision
        # ordinary players = players who are neither visibles
        # nor has any visibles in their neighbor
        self.delay = delay

        # total number of color changes in a game
        self.colorChanges = 0

        # game elapsed time
        self.elapsedTime = 0

        # addded by Yifan
        self.reach_of_adversaries = 0
        self.reach_of_visibles = 0
        self.total_cnt_of_adversaries = 0
        self.total_cnt_of_visibles = 0
        self.graph = G

        # tune some parameters of the logistic regression
        self.regularNodeAmplifier = None
        self.visibleNodeAmplifier = None


        # convert adjMat to adjList
        def getAdjList(adjMat):
            adjList = {key: [] for key in range(self.numAgents)}
            for node in range(self.numAgents):
                #adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == True]
                adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == 'True']
            return adjList

        self.adjList = getAdjList(self.adjMat)

        #return the subset of L availableNodes in G with the largest number of distinct neighbors
        def getSubsetWithMaxDistinctNeighbors(availableNodes, G, L):
            acc = []
            max_cnt = 0
            local_cnt = 0
            hasBeenConsidered = [False for i in range(self.numAgents)]
            graph = nx.convert.to_dict_of_lists(G)
            for subset in itertools.combinations(availableNodes, L):
                upper_bound = 0
                for agent in subset:
                    upper_bound += len(graph[agent])
                if upper_bound < max_cnt:
                    continue
                # compute reach
                for agent in subset:
                    for neighbor in G.neighbors(agent):
                        if neighbor not in subset and hasBeenConsidered[neighbor] == False:
                            local_cnt += 1
                            hasBeenConsidered[neighbor] = True
                if local_cnt > max_cnt:
                    max_cnt = local_cnt
                    acc.clear()
                    for agent in subset:
                        acc.append(agent)
                local_cnt = 0
                hasBeenConsidered = [False for i in range(self.numAgents)]
            return acc

        #returns the connected component of visible players in descending order of degree
        #recursive BFS
        def getRecursiveConnectedNeighborhood(visibleCandidate, count):
            if self.numVisibleColorNodes == 0:
                return True
            self.visibleColorNodes.append(visibleCandidate)
            count += 1      #problem!?!?!?
            if count == self.numVisibleColorNodes:
                #base case
                return True
            else:
                availableNeighbors = []
                for neighbor in G.neighbors(visibleCandidate):
                    if not neighbor in self.visibleColorNodes:
                        availableNeighbors.append((neighbor, self.graph.degree(neighbor)))

                if availableNeighbors == []:
                    self.visibleColorNodes.remove(visibleCandidate)
                    count -= 1
                    return False        #reached a dead end

                availableNeighbors.sort(key=lambda x : x[1], reverse=True)  #highest degree nodes first
                
                for pair in availableNeighbors:
                    candidate = pair[0]
                    if getRecursiveConnectedNeighborhood(candidate, count):
                        return True
                self.visibleColorNodes.remove(visibleCandidate)
                return False

        ############# designate visible #############
        node_deg = [(idx, count(adjMat[idx])) for idx in range(self.numAgents)]
        node_deg.sort(key=lambda x : x[1], reverse=True)       #highest degree nodes first
        availableNodes = [item[0] for item in node_deg]
        
        # availableNodes.sort(key=lambda x : x)
        # self.visibleColorNodes = getSubsetWithMaxDistinctNeighbors(availableNodes, G, numVisibleColorNodes)
        # self.visibleColorNodes = [item for item in availableNodes[:self.numVisibleColorNodes]]
        # tmpVisibleNode = availableNodes[0]
        # getRecursiveConnectedNeighborhood(tmpVisibleNode, 0)
        self.visibleColorNodes = visibles
        for visibleNode in self.visibleColorNodes:
            availableNodes.remove(visibleNode)

        ############# designate adversarial ###############
        #self.adversarialNodes = getSubsetWithMaxDistinctNeighbors(availableNodes, G, numAdversarialNodes)
        #random.shuffle(availableNodes)
        # self.adversarialNodes = [item for item in availableNodes[:self.numAdversarialNodes]]
        self.adversarialNodes = adversaries


        # ================ prev version: designate adversarial and visible nodes ===========
        # node_deg = [(idx, count(adjMat[idx])) for idx in range(self.numAgents)]
        # all_nodes = [item[0] for item in node_deg]
        # random.shuffle(node_deg)
        # self.adversarialNodes = [item[0] for item in node_deg[:self.numAdversarialNodes]]

        # reach_of_adversaries = 0
        # total_cnt_of_adversaries = 0
        # hasBeenReached = dict.fromkeys(all_nodes, False)                
        # for adversarialNode in self.adversarialNodes:
        #     for neighbor in G.neighbors(adversarialNode):
        #         if neighbor not in self.adversarialNodes:
        #             total_cnt_of_adversaries += 1
        #         if neighbor not in self.adversarialNodes and hasBeenReached[neighbor] == False:
        #             reach_of_adversaries += 1
        #             hasBeenReached[neighbor] = True
        # self.reach_of_adversaries = reach_of_adversaries
        # self.total_cnt_of_adversaries = total_cnt_of_adversaries

        # ############# designate visible nodes #############
        # availableNodes = shuffled(node_deg[self.numAdversarialNodes:])
        # self.visibleColorNodes = [item[0] for item in availableNodes[:self.numVisibleColorNodes]]

        # reach_of_visibles = 0
        # total_cnt_of_visibles = 0
        # hasBeenReached = dict.fromkeys(all_nodes, False)
        # for visibleColorNode in self.visibleColorNodes:
        #     for neighbor in G.neighbors(visibleColorNode):
        #         if neighbor not in self.adversarialNodes and neighbor not in self.visibleColorNodes:
        #             total_cnt_of_visibles += 1
        #         if neighbor not in self.adversarialNodes and neighbor not in self.visibleColorNodes and hasBeenReached[neighbor] == False:
        #             reach_of_visibles += 1
        #             hasBeenReached[neighbor] = True
        # self.reach_of_visibles = reach_of_visibles
        # self.total_cnt_of_visibles = total_cnt_of_visibles

        # ===============================

        self.regularNodes = [n for n in range(self.numAgents) if n not in self.adversarialNodes]
        # make sure we have 20 regular nodes
        # assert len(self.regularNodes) ==20

        # adversarial nodes and regular nodes should not overlap
        assert set(self.adversarialNodes) & set(self.regularNodes) == set()
        # visible nodes should belong to regular nodes
        assert set(self.visibleColorNodes) & set(self.regularNodes) == set(self.visibleColorNodes)

        # logging simulation configuration
        self.log.add("#visible nodes: " + str(self.visibleColorNodes))
        self.log.add("#adversarial nodes: " + str(self.adversarialNodes))
        self.log.add("#regular nodes: " + str(self.regularNodes) + '\n')

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
            beta = self.beta

            a = GameAgent(i, isVisibleNode, isAdversarial, neighbors, vNode, inertia, beta, self)
            self.schedule.add(a)

        # instantiate all nodes' neighbors and visibleColorNodes
        for agent in self.schedule.agents:
            agent.instantiateNeighbors(self)
            agent.instantiateVisibleColorNodes(self)

        self.datacollector = DataCollector(
                        model_reporters = {"red": getRed, "green": getGreen},
                        agent_reporters = {"agent_color": lambda a: a.color}
                        )

    def getReachOfAdversaries(self):
        return self.reach_of_adversaries

    def getReachOfVisibles(self):
        return self.reach_of_visibles

    def getTotalCntOfAdversaries(self):
        return self.total_cnt_of_adversaries

    def getTotalCntOfVisibles(self):
        return self.total_cnt_of_visibles

    # simulate the whole model for one step
    def step(self):
        # # # if either red or green reaches consensus, terminates!
        # # in terminal state we do not collect data
        if not self.terminate:
            self.datacollector.collect(self)
            self.schedule.step()
        return self.terminate

    def simulate(self, simulationTimes):
        for i in range(simulationTimes):
            # update model's time
            self.updateTime(i)
            terminate = self.step()
            if terminate:
                break

        #added by Yifan
        isRegWhite = False
        # output log file to disk
        if not terminate:
            # did not reach consensus
            for agent in self.schedule.agents:
                if not agent.isAdversarial and not agent.isVisibleNode and agent.color == "white":
                    #at least one regular player remained white
                    isRegWhite = True

        self.log.outputLog('result/simResult.txt')
        simulatedResult = self.datacollector.get_model_vars_dataframe()
        # record how long does this game take
        self.elapsedTime = len(simulatedResult)
        return (simulatedResult, isRegWhite)

    # update model's clock
    def updateTime(self, t):
        self.time = t

    def setTerminal(self):
        assert self.terminate == False
        self.terminate = True

    def addRecord(self, msg):
        self.log.add(msg)

    # for degub purpose only
    def outputAdjMat(self, path):
        with open(path, 'w') as fid:
            for line in self.adjMat:
                # convert list of boolean values to string values
                tline = ["1" if item else "0" for item in line]
                fid.write(' '.join(tline) + '\n') 

    def setRegularNodeAmplifier(self, amplifier):
        self.regularNodeAmplifier = amplifier

    def setVisibleNodeAmplifier(self, amplifier):
        self.visibleNodeAmplifier = amplifier    

class BatchResult(object):
    def __init__(self, data, dataOnGameLevel, args):
        # self.data records data at each time step
        self.data = data
        self.args = args

        # self.dataOnGameLevel records data on 
        # each game level
        self.dataOnGameLevel = dataOnGameLevel
        ###
        self.consensus_ret = None
        self.dynamics_ret = None
        self.time_ret = None
        self.columnNames = None

    def generateResult(self):
        # generate a DataFrame where each row corresponds
        # to a simulation
        consensus_ret = []
        results_column = ['expDate', 'expNum', 'numVisibleNodes', 'numAdversarialNodes',\
                          'network', 'elapsedTime', 'colorChange', 'whiteNode', 'consensus']
        for i in range(len(self.data)):
            if_consensus = 1 if len(self.data[i]) < self.args['gameTime'] else 0
            consensus_ret.append((self.args['expDate'], self.args['expNum'], self.args['numVisibleNodes'], self.args['numAdversarialNodes'],\
                                  self.args['network'], self.dataOnGameLevel['elapsedTime'][i], self.dataOnGameLevel['colorChanges'][i], \
                                  self.dataOnGameLevel['whiteNode'][i], if_consensus))
        consensus_ret = pd.DataFrame(consensus_ret)
        consensus_ret.columns = results_column
        self.columnNames = results_column
        self.consensus_ret = consensus_ret

        # # generate detailed dynamics for each simulation
        # dynamics_ret = {}
        # for i in range(len(self.data)):
        #     dynamics_ret[i] = self.data[i]
        # self.dynamics_ret = dynamics_ret

        # # generate time to consensus
        # time_ret = []
        # for i in range(len(self.data)):
        #     t = len(self.data[i])
        #     time_ret.append((self.numvisibles[i], self.numadversaries[i], self.networks[i], t))
        # time_ret = pd.DataFrame(time_ret)
        # self.time_ret = time_ret

    def getConsensusResult(self):
        return self.consensus_ret

    def getColumnNames(self):
        return self.columnNames

    def getDynamicsResult(self):
        return self.dynamics_ret

    def getTimeResult(self):
        return self.time_ret
    


# def getAdjMat(net, numPlayers, numRegularPlayers, numAdversarialNodes):
#     # network parameters
#     ################################
#     ### CODE FROM Zlatko ###
#     # each new node is connected to m new nodes
#     m = 3
#     no_consensus_nodes_range = range(11)
#     # max degrees
#     maxDegree = 17
#     BA_edges = [(numRegularPlayers + no_consensus_nodes - 3) * m for \
#                             no_consensus_nodes in no_consensus_nodes_range]
#     ERD_edges = [edges_no for edges_no in BA_edges]
#     ERS_edges = [int(math.ceil(edges_no/2.0)) for edges_no in ERD_edges]
#     ################################ 

#     # generate adjMat according to network type
#     if net == 'Erdos-Renyi-dense':
#         adjMat, G = ErdosRenyi(numPlayers, ERD_edges[numAdversarialNodes], maxDegree)
#     elif net == 'Erdos-Renyi-sparse':
#         adjMat, G = ErdosRenyi(numPlayers, ERS_edges[numAdversarialNodes], maxDegree)
#     else:
#         adjMat, G = AlbertBarabasi(numPlayers, m, maxDegree)

#     return (adjMat, G)

# def getAdjMat(net, numPlayers, numEdges):
#     m = 3
#     maxDegree = 17
#     adjMat, G = ErdosRenyi(numPlayers, numEdges, maxDegree)
#     return (adjMat, G)

def getAdjMat(network_path, i):
    matrices = []   #all matrices for this game
    with open(network_path, "r") as f:
        for row in f:
            row = row.strip('\n')
            row = row.split(' ')
            matrices.append(row)

    adjMat = [] #get the one you want
    occ = [w for w, n in enumerate(matrices) if n[0] == '#']
    assert(len(occ) != 0)
    up = occ[i-1]   #index of first tag
    if i == 1:
        for x in range(up):
            adjMat.append(matrices[x])
    else:
        down = occ[i-2]
        for x in range(down + 1, up):
            adjMat.append(matrices[x])

    fladjMat = [[0 for n in range(len(adjMat))] for k in range(len(adjMat))]  #create a binary eq of adjMat
    for a in range(len(adjMat)):
        for b in range(len(adjMat)):
            if adjMat[a][b] == 'True':
                fladjMat[a][b] = 1.
            else:
                fladjMat[a][b] = 0.

    numadjMat = np.matrix(fladjMat)
    G =nx.from_numpy_matrix(numadjMat)  #reconstruct the graph

    return (adjMat, G)

# read configurations from text files, and return
# a list of combinations of parameters
def readConfigurationFromFile(file_path):
    config = pd.read_csv(file_path, sep='\t')
    config.index = range(len(config))

    params = []
    for idx in config.index:
        line = config.iloc[idx]

        visibleNodes = ast.literal_eval(line["list_of_visibles"])
        adversaryNodes = ast.literal_eval(line["list_of_adversaries"])

        numVisibles = len(visibleNodes)
        numAdversaries = len(adversaryNodes)

        expDate = line['experiment']
        expNum = line['game']

        network = line['network']

        adjMat_path = os.path.join('data/adjacency_matrix/', "%s_adjacency_matrix.txt" % expDate)
        adjMat, G = getAdjMat(adjMat_path, expNum)

        params.append({
            'visibleNodes': visibleNodes,
            'adversarialNodes': adversaryNodes,
            'numVisibleNodes': numVisibles,
            'numAdversarialNodes': numAdversaries,
            'network': network,
            'adjMat': adjMat,
            'G': G,
            'expDate': expDate,
            'expNum': expNum
            })
    return params


#define a wrapper function for multi-processing
def simulationFunc(args):
    # dispatch arguments
    numSimulation = args['numSimulation']
    gameTime      = args['gameTime']
    numVisibleNodes = args['numVisibleNodes']
    numAdversarialNodes = args['numAdversarialNodes']
    visibleNodes = args['visibleNodes']
    adversarialNodes = args['adversarialNodes']
    inertia = args['inertia']
    beta = args['beta']
    delay = args['delay']
    regularNodeAmplifier = args['regularNodeAmplifier']
    visibleNodeAmplifier = args['visibleNodeAmplifier']
    network = args['network']
    adjMat = args['adjMat']
    G = args['G']
    expDate = args['expDate']
    expNum = args['expNum']
    outputPath = args['outputPath']
    arg_id = args['arg_id']

    # ret contains simulated results
    ret = []
    retOnGameLevel = defaultdict(list)

    for j in range(numSimulation):
        # if j % 10 == 0:
        #     print("Current number of simulations: ", j)

        model = DCGame(adjMat, G, numVisibleNodes, numAdversarialNodes, inertia, beta, \
                delay, visibleNodes, adversarialNodes)

        # set amplifier
        model.setRegularNodeAmplifier(regularNodeAmplifier)
        model.setVisibleNodeAmplifier(visibleNodeAmplifier)

        simulatedResult, isRegWhite = model.simulate(gameTime)
        ret.append(simulatedResult)
        ### a game-level data collector
        # retOnGameLevel['hasConflict'].append(model.hasConflict)
        # retOnGameLevel['delay'].append(model.delay)
        retOnGameLevel['elapsedTime'].append(model.elapsedTime)
        retOnGameLevel['colorChanges'].append(model.colorChanges)
        retOnGameLevel['whiteNode'].append(isRegWhite)
        ###
        model.outputAdjMat('result/adjMat.txt')

    # the collected data is actually an object
    result = BatchResult(ret, retOnGameLevel, args)
    result.generateResult()

    # calculate and return consensus ratio
    result = result.getConsensusResult()
    consensus_ratio = result['consensus'].mean()
    return consensus_ratio
    # result.getConsensusResult().to_csv(os.path.join(outputPath, '%d.csv' % arg_id), index=None, sep=',')
    # return result.getConsensusResult()


def combineResults(result, args, folder=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # result is returned from multi-processing 
    for ret in result:
        ret.generateResult()

    consensus_ret = pd.concat([item.getConsensusResult() for item in result])
    consensus_ret.columns = result[0].getColumnNames()
    p = os.path.join(folder, 'consensus_inertia=%.2f_beta=%.2f.csv' % (inertia, beta))
    consensus_ret.to_csv(p, index=None)


    # time_ret = pd.concat([item.getTimeResult() for item in result])
    # time_ret.columns = ['#visibleNodes', '#adversarial', 'network', 'time']
    # p = os.path.join(folder, 'time_inertia=%.2f.csv' % inertia)
    # time_ret.to_csv(p, index=None)

    # dynamics_ret = {args[idx]: item.getDynamicsResult() for idx, item in enumerate(result)}
    # p = os.path.join(folder, 'dynamics_inertia=%.2f.p' % inertia)
    # with open(p, 'wb') as fid:
    #     pickle.dump(dynamics_ret, fid)



if __name__ =="__main__":

    inertia = 0
    beta = 0
    # experimental parameters
    ################################
    numSimulation = 30
    gameTime = 60
    # inertia = 0.5
    numRegularPlayers = 20
    # numRegularPlayersList = [17 + (n * 3) for n in range(5)]
    ################################

    networks = ['Erdos-Renyi-dense', 'Erdos-Renyi-sparse', 'Barabasi-Albert']
    # networks = ['Erdos-Renyi']
    numVisibleNodes_ = [0]
    numAdversarialNodes_ = [0]
    delayTime_ = [0]
    # ER_edges = [23, 26, 45, 51]
    ER_edges = [25 + 5 * i for i in range(15)]

    args_from_file = readConfigurationFromFile('data/nocomm.csv')
    args = []
    cnt = 0
    outputPath = 'result/noAdv'
    for item in args_from_file:
        if item['numAdversarialNodes'] != 0:
            args.append({
                'numSimulation': numSimulation,
                'gameTime': gameTime,
                'numVisibleNodes': item['numVisibleNodes'],
                'numAdversarialNodes': item['numAdversarialNodes'],
                'visibleNodes': item['visibleNodes'],
                'adversarialNodes': item['adversarialNodes'],
                'inertia': inertia,
                'beta': beta,
                'delay': 0,
                'network': item['network'],
                'adjMat': item['adjMat'],
                'G': item['G'],
                'expDate': item['expDate'],
                'expNum': item['expNum'],
                'outputPath': outputPath, 
                'arg_id': cnt
                })
            cnt += 1

    # split configurations into training and testing 
    train_test_ratio = 0.5
    np.random.shuffle(args)
    train_args = args[:np.int(np.floor(len(args) * train_test_ratio))]
    test_args = args[np.int(np.floor(len(args) * train_test_ratio)):]
    assert(len(train_args) + len(test_args) == len(args))

    # coordinate descent
    # we first consider infinity norm
    result = []
    numFeatures = 8
    coord_iter = 1
    pool = Pool(processes=70)
    regularNodeAmplifier = np.asmatrix(np.zeros(numFeatures)).reshape(numFeatures, 1)
    visibleNodeAmplifier = np.asmatrix(np.zeros(numFeatures)).reshape(numFeatures, 1)
    for item in train_args:
        item['regularNodeAmplifier'] = regularNodeAmplifier
        item['visibleNodeAmplifier'] = visibleNodeAmplifier
    baseline_consensus_ratio = pool.map(simulationFunc, train_args)
    train_consensus_ratio = np.mean(baseline_consensus_ratio)

    for budget in np.arange(0.1, 1.1, 0.1):
        search_space = np.linspace(-budget, budget, int(budget * 100) + 1)
        for j in range(coord_iter):
            used_budget = 0
            # find optimal amplifiers for regular nodes
            for i in range(numFeatures):
                print("Current #feature: %i" % i)
                for delta_i in search_space:
                    tmp_amplifier = regularNodeAmplifier.copy()
                    tmp_amplifier[i] = delta_i
                    if LA.norm(tmp_amplifier, 1) > budget:
                        continue
                    else:
                        for item in train_args:
                            item['regularNodeAmplifier'] = tmp_amplifier
                            item['visibleNodeAmplifier'] = visibleNodeAmplifier
                        ratio = pool.map(simulationFunc, train_args)
                        if np.mean(ratio) > train_consensus_ratio:
                            train_consensus_ratio = np.mean(ratio)
                            regularNodeAmplifier[i] = delta_i
                            print("regular nodes        budget: %.2f        #feature: %i        ratio: %.5f" % (budget, i, train_consensus_ratio) )
            # budget used by regular nodes' amplifier
            used_budget += LA.norm(regularNodeAmplifier, 1)

            # find optimal amplifiers for visible nodes
            left_budget = budget - used_budget
            for i in range(numFeatures):
                print("Current #feature: %i" % i)
                for delta_i in search_space:
                    tmp_amplifier = visibleNodeAmplifier.copy()
                    tmp_amplifier[i] = delta_i
                    if LA.norm(tmp_amplifier, 1) > left_budget:
                        continue
                    else:
                        for item in train_args:
                            item['regularNodeAmplifier'] = regularNodeAmplifier
                            item['visibleNodeAmplifier'] = tmp_amplifier
                        ratio = pool.map(simulationFunc, train_args)
                        if np.mean(ratio) > train_consensus_ratio:
                            train_consensus_ratio = np.mean(ratio)
                            visibleNodeAmplifier[i] = delta_i
                            print("visible nodes        budget: %.2f        #feature: %i        ratio: %.5f" % (budget, i, train_consensus_ratio) )           

        # test the optimal amplifier
        for item in test_args:
            item['regularNodeAmplifier'] = regularNodeAmplifier
            item['visibleNodeAmplifier'] = visibleNodeAmplifier
        test_ratio = pool.map(simulationFunc, test_args)
        test_consensus_ratio = np.mean(test_ratio)
        result.append((budget, baseline_consensus_ratio, train_consensus_ratio, test_consensus_ratio, regularNodeAmplifier, visibleNodeAmplifier))

    pool.close()
    pool.join()


