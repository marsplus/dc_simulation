import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from utils import *

# random.seed(123)

class GameAgent(Agent):
    def __init__(self, unique_id, visibleNode, adversarial, neighbors, visibleColorNodes, inertia, model):
        super().__init__(unique_id, model)
        # whether this node is a visible node
        self.visibleNode = visibleNode
        # whether this node is an adversarial
        self.adversarial = adversarial
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


    # determine if there is any visible color node in the neighbor
    def hasVisibleColorNode(self):
        return len(self.visibleColorNodes) > 0


    # return current majority color
    def majorityColor(self):
        # regular node
        if not self.adversarial and not self.visibleNode:
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
                if self.visibleNode:
                    return "red"
                else:
                    return "green"
            elif green > red:
                if self.visibleNode:
                    return "green"
                else:
                    return "red"
            else:
                return random.choice(["red", "green"])

    # make a decision
    def step(self):
        # if self.visibleNode:
        #     print((self.visibleNode, self.color))
        major_color = self.majorityColor()
        if major_color == "white":
            # agents cannot go back to white once they
            # choosed certain color
            pass
        else:
            # each agent has a small probability to not make
            # any decision
            if random.random() < self.p:
                if major_color == "red":
                    self.color = "red"
                else:
                    self.color = "green"
            else:
                # do nothing
                pass

    def degree(self):
        return len(self.neighbors)



def getCurrentColor(model):
    ret = {"red": 0, "green": 0, "white": 0}
    current_color = [(a.color, a.unique_id) for a in model.schedule.agents]
    for item in current_color:
        ret[item[0]] += 1
    return ret


class DCGame(Model):
    def __init__(self, adjMat, numVisibleNodes, numAdversarialNodes, inertia):
        self.adjMat = adjMat
        self.numVisibleNodes = numVisibleNodes
        self.numAdversarialNodes = numAdversarialNodes
        self.adversarialNodes = []
        self.visibleNodes = []
        self.schedule = RandomActivation(self)
        self.numAgents = len(adjMat)
        self.inertia = inertia
        

        # convert adjMat to adjList
        def getAdjList(adjMat):
            adjList = {key: [] for key in range(self.numAgents)}
            for node in range(self.numAgents):
                adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == "True"]
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
        self.visibleNodes = [item[0] for item in availableNodes[:self.numVisibleNodes]]

        self.regularNodes = [n for n in range(self.numAgents) if n not in self.visibleNodes
                            and n not in self.adversarialNodes]


        ############# initialize all agents #############
        for i in range(self.numAgents):
            # if i is a visible node
            visibleNode = i in self.visibleNodes
            # if i is an adversarial
            adversarial = i in self.adversarialNodes

            neighbors = self.adjList[i]
            # visible color nodes in i's neighbors
            visibleColorNodes = list(set(neighbors) & set(self.visibleNodes))
            inertia = self.inertia

            # def __init__(self, unique_id, visibleNode, adversarial, neighbors, visibleColorNodes, inertia, model):

            # print("Add agent:", (i, visibleNode, adversarial, neighbors, visibleColorNodes))
            a = GameAgent(i, visibleNode, adversarial, neighbors, visibleColorNodes, inertia, self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
                model_reporters = {"color": getCurrentColor},
                agent_reporters = {"agent_color": lambda a: a.color}
            )


    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


def terminate(model):
    curr_color = getCurrentColor(model)
    if curr_color["red"] == 20 or curr_color["green"] == 20:
        return True
    else:
        return False


expDate = '2017_03_10'
a = expSummary(expDate)
numVisibleNodes = a[0]['numVisible']
numAdversarialNodes = a[0]['numAdv']
adjMat = a[0]['adjMat']

ret = []
for j in range(100):
    m = DCGame(adjMat, numVisibleNodes, numAdversarialNodes, 0.9)
    for i in range(60):
        m.step()
    ret.append(m.datacollector.get_model_vars_dataframe())

# m.datacollector.get_model_vars_dataframe()

