#-------------------------------------------------------------------------------
# Name:        graph
# Purpose:
#
# Author:      Zlatko
#
# Created:     27.02.2016
#-------------------------------------------------------------------------------

class graph():
    """A graph represented through an adjacency matrix. Graphs have the folloing
    attributes:

    Attributes:
        adjacency_matrix: A 2D boolean array storing the node adjacencies.
        n: An integer storing the number of nodes.
        [Nodes are labeled {0,1,2, ..., n-1}.]
        degs: An array storing the degree sequence of the graph.
    """

    def __init__(self, n):
        """Return a graph object with a zero adjacency_matrix with n rows and n
        columns(corresponding to an edgeless graph on n nodes)."""
        self.n = n
        self.adj = [[False for i in range(n)] for j in range(n)]
        self.degs = [0 for i in range(n)]

    def add_edge(self, i, j):
        if (i in range(self.n) and j in range(self.n) and i != j):
            self.adj[i][j] = True
            self.adj[j][i] = True

            # make sure to also update the node degrees
            self.degs[i] += 1
            self.degs[j] += 1

    def remove_edge(self, i, j):
        if (i in range(self.n) and j in range(self.n) and i != j):
            self.adj[i][j] = False
            self.adj[j][i] = False

            # make sure to also update the node degrees
            self.degs[i] -= 1
            self.degs[j] -= 1

    def number_of_vertices(self):
        """ Returns the number of vertices of the graph n. """
        return self.n

    def adjacency_matrix(self):
        """ Returns the adjacency matrix of the graph. """
        return self.adj

    def degree_sequence(self):
        """ Returns the original degree sequence of the graph. """
        return self.degs

    def reverse_degree_sequence(self):
        """ Returns the degree sequence of the graph sorted in non-increasing order. """
        return sorted(self.degs, reverse = True)

    def max_degree(self):
        """ Returns the maximum degree of the graph. """
        return max(self.degs)