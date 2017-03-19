import csv
import re
import random

from adversarial import *

def main():
    ### Initialize the game counts for each treatment
    # all possible treatments
    graphNames = ['Barabasi-Albert', 'Erdos-Renyi-dense', 'Erdos-Renyi-sparse']
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
    visibleConsensusNodes = ['0', '1', '2', '5']
    consensusVisibilityAssignment = ['random']

    # initialize the game counts for each treatment
    treatment_count = {}
    delta_count = {}
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
                                                            treatment = (graN, incL, comS, comT, msgO, gamB, batT, advA, numA, regP, advP, visM, visC, conA)
                                                            treatment_count[treatment] = 0
                                                            delta_count[treatment] = 0

    #######################################################################################

    ### Load the valid games data from previous sessions
    expLabels = []
    validGames = {}

    with open('index_of_all_valid_games.csv', 'r') as csvfile:
        indexReader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in indexReader:
            if row[0] != "Date":
                expLabels.append(row[0])
                validGames[row[0]] = map(int, re.split(' ', row[1]))
    #######################################################################################

    ### Explore each individual session and  count the number of covered games for each treatment
    valid_games_count = 0
    new_games_count = 0

    for label in expLabels:
        # open the network configuration .txt file for the corresponding experiment and read the lines into a list
        with open("./config_files/"+label+"_net_config.txt") as f:
            net_config = f.readlines()

        # open the batch configuration .txt file for the corresponding experiment and read the lines into a list
        with open("./config_files/"+label+"_batch_config.txt") as f:
            batch_config = f.readlines()

        # remove trailing '\n' characters from lines
        net_config = [x.strip('\n') for x in net_config]
        batch_config = [x.strip('\n') for x in batch_config]


        # count the number of valid games for each treatment
        for k in range(len(net_config)):
            game = k+1
            if game in validGames[label]:
                net_row = net_config[k]
                net_config_row = re.split(' ', net_row)
                graN = net_config_row[0]
                incL = net_config_row[1]
                comS = net_config_row[2]
                comT = net_config_row[3]
                if len(net_config_row) > 4: msgO = net_config_row[4]
                else: msgO = 'NA'

                batch_row = batch_config[k]
                batch_config_row = re.split(' ', batch_row)
                gamB = batch_config_row[0]
                batT = batch_config_row[1]
                advA = batch_config_row[2]
                numA = batch_config_row[3]
                regP = batch_config_row[4]
                advP = batch_config_row[5]
                if len(batch_config_row) > 6:
                    visM = batch_config_row[6]
                    visC = batch_config_row[7]
                    conA = batch_config_row[8]
                else:
                    visM = 'visibleConsensusNodes'
                    visC = '0'
                    conA = 'random'

                treatment = (graN, incL, comS, comT, msgO, gamB, batT, advA, numA, regP, advP, visM, visC, conA)
                treatment_count[treatment] += 1

    """
    # print the number of valid games for each treatment
    for treatment, count in treatment_count.items():
        print (treatment, count)
        valid_games_count += count

    print valid_games_count
    """
    
    
    #######################################################################################

    ### After we know all covered games, calculate the numbers of games per treatment that
    ### still need to be covered
    target_count = 13
    fullConfig = []

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
                                                            treatment = (graN, incL, comS, comT, msgO, gamB, batT, advA, numA, regP, advP, visM, visC, conA)
                                                            valid_games_count += treatment_count[treatment]

                                                            tre_delta = target_count - treatment_count[treatment]
                                                            delta_count[treatment] = tre_delta
                                                            new_games_count += tre_delta

                                                            # print (treatment, tre_delta)

                                                            for i in range(tre_delta):
                                                                fullConfig.append(treatment)

    print ("previous valid games", valid_games_count)
    print ("games to cover", new_games_count)
    print ("configurations to cover", len(fullConfig))
    # print fullConfig

    
    # sort the treatments by the count of missing games, in order to add an additional game in
    # the session for those that have the biggest number of missing games
    import operator
    sorted_delta_counts = sorted(delta_count.items(), key=operator.itemgetter(1), reverse=True)
    # print sorted_delta_counts

    """
    #######################################################################################

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

    networkConfigurationFile = 'generated_input_files/network_configuration.txt'
    adjacencyMatrixFile      = 'generated_input_files/adjacency_matrix.txt'
    batchConfigurationFile   = 'generated_input_files/batch_configuration.txt'

    practiceNetworkConfigurationFile = 'generated_input_files/network_configuration_practice.txt'
    practiceAdjacencyMatrixFile      = 'generated_input_files/adjacency_matrix_practice.txt'
    practiceBatchConfigurationFile   = 'generated_input_files/batch_configuration_practice.txt'

    ### Generate new input files for an experiment session of specified length L, where
    ### the L games are drawn randomly from the full list of uncovered games (fullConfig)

    regular_games = 65
    practice_games = 5


    # print "Missing games: " + str(len(fullConfig))

    # if the session has room for extra games
    if new_games_count < regular_games:
        print "Extra games added!!!"
        # identify the regular_games-new_games_count treatments with the most missing games
        treatment_count_pairs = sorted_delta_counts[:regular_games-new_games_count]
        top_treatments = [pair[0] for pair in treatment_count_pairs]
        # print treatment_count_pairs
        # print top_treatments

        # add one game per treatment to the configurations list
        fullConfig.extend(top_treatments)

    # print "New session games: " + str(len(fullConfig))


    random.shuffle(fullConfig)
    # print fullConfig[0]
    random.shuffle(fullConfig)
    # print fullConfig[0]
    random.shuffle(fullConfig)
    # print fullConfig[0]

    configurations = fullConfig[0:regular_games]
    random.shuffle(configurations)
    print ("configurations", len(configurations))

    adjacencyMatrixList = []
    for config in configurations:
        if config[0] == 'Erdos-Renyi-dense':
            adjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERD_edges[int(config[8])], max_degree))
        elif config[0] == 'Erdos-Renyi-sparse':
            adjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERS_edges[int(config[8])], max_degree))
        elif config[0] == 'Barabasi-Albert':
            adjacencyMatrixList.append(AlbertBarabasi(consensus_nodes + int(config[8]), m, max_degree))

    print ("adjacency matrices", len(adjacencyMatrixList))

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

    practiceConfigurations = random.sample(configurations, practice_games)

    random.shuffle(practiceConfigurations)
    print("practice configurations", len(practiceConfigurations))

    practiceAdjacencyMatrixList = []
    for config in practiceConfigurations:
    	if config[0] == 'Erdos-Renyi-dense':
            practiceAdjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERD_edges[int(config[8])], max_degree))
        elif config[0] == 'Erdos-Renyi-sparse':
            practiceAdjacencyMatrixList.append(ErdosRenyi(consensus_nodes + int(config[8]), ERS_edges[int(config[8])], max_degree))
        elif config[0] == 'Barabasi-Albert':
            practiceAdjacencyMatrixList.append(AlbertBarabasi(consensus_nodes + int(config[8]), m, max_degree))

    print("practice adjacency matrices", len(practiceAdjacencyMatrixList))

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
    """

if __name__ == "__main__":
	main()
