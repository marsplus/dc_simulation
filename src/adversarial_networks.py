from networkx import nx
import math

# Network generators and helper methods
def checkAdjacencyMatrix(mat):
    nodes = len(mat)
    for node in range(nodes):
        if mat[node][node] == True:
            errMsg = "node should not be connected to itself"
            print errMsg
            return False

    if len(mat) != len(mat[0]):
        errMsg = "wrong dimensions of adjacency matrix"
        print errMsg
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
    ## parameters are set below
    # total number of nodes in the Consensus team
    consensus_nodes = 20

    # possible values for the number of nodes of the No-Consensus team
    # the actual current range is {0, 2, 5}
    no_consensus_nodes_range = range(11)

    # maximum allowed vertex degree
    max_degree = 17

    # each new node is connected to m new nodes (applies to BA networks)
    m = 3

    # number of edges for each network size under the different network 
    # generating models
    """ The goal here is to make sure BA and ER-dense networks are of the same 
    density, and ER-sparse networks are of half their density. Note that even 
    though the m parameter is fixed at 3, when we are generating BA networks 
    of different size (20, 22 or 25), the netwok density slightly increases 
    when the network size is increased. That is why we tie the densities/# 
    edges of ER-dense/sparse networks to those of the BA networks of the 
    corresponding size.
    """
    BA_edges = [(consensus_nodes + no_consensus_nodes -3) * m for
                no_consensus_nodes in no_consensus_nodes_range]
    ERD_edges = [edges_no for edges_no in BA_edges]
    ERS_edges = [int(math.ceil(edges_no/2.0)) for edges_no in ERD_edges]


    # BA_edges[x] specifies the number of edges for a BA network of size 20+x (x is the number of No-Consensus players
    # ERD_edges[x], ERS_edges[x] are analogous

    ### Network/adjacency matrix generating examples ###
	
    ### BA network of size 22 (20 Consensus and 2 Non-Consensus players/nodes)
    adj1 = AlbertBarabasi(22, m, max_degree)		# no need to specify # edges/density here
    
    # print the full adjacency matrix
    # print adj1

    # print the degree sequence
    print [sum(adj1[i]) for i in range(len(adj1))]

    ### ER-dense network of size 20 (20 Consensus and 0 Non-consensus players/nodes)
    adj2 = ErdosRenyi(20, ERD_edges[0], max_degree)
    
    # print the full adjacency matrix
    # print adj2

    # print the degree sequence
    print [sum(adj2[i]) for i in range(len(adj2))]
    
    ### ER-sparse network of size 25 (20 Consensus and 5 Non-consensus players/nodes)
    adj3 = ErdosRenyi(25, ERS_edges[5], max_degree)
    
    # print the full adjacency matrix
    # print adj3
    
    # print the degree sequence
    print [sum(adj3[i]) for i in range(len(adj3))]

if __name__ == '__main__':
    main()
