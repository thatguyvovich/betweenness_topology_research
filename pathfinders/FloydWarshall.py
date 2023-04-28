import numpy as np
from typing import List

from pathfinders.AbstractPathfinder import AbstractPathfinder

class FloydWarshallPathfinder(AbstractPathfinder):

    def __init__(self, MA: List[List[int]]):
        self.MA = MA

    def find_paths(self) -> List[List[int]]:
        """
        Finds the shortest path between two nodes in a topology using Floyd's algorithm
        :param MA: Adjacency matrix
        :return: A matrix of the shortest path between two nodes in the topology
        """
        self.__path_reconstructor()
        return self.MD, self.MN
    
    def __path_reconstructor(self):
        """
        Reconstructs the shortest path between two nodes in a graph using Floyd's algorithm
        :param MA: Adjacency matrix
        :return: A matrix of the shortest path between two nodes in the graph
        """
        distance = np.zeros(shape=(len(self.MA), len(self.MA)), dtype=np.int64)
        distance = np.copy(self.MA)
        next = np.zeros(shape=(len(self.MA), len(self.MA)), dtype=np.int64)
        for i in range(len(self.MA)):
            for j in range(len(self.MA)):
                if distance[i, j] == 0:
                    distance[i, j] = 9000000
                elif distance[i, j] == 1:
                    next[i, j] = j
                elif i == j:
                    distance[i, i] = 0
                    next[i, i] = i
        
        for k in range(len(self.MA)):
            for i in range(len(self.MA)):
                for j in range(len(self.MA)):
                    if distance[i, j] > distance[i, k] + distance[k, j]:
                        distance[i, j] = distance[i, k] + distance[k, j]
                        next[i][j] = next[i][k]

            
        for i in range(len(self.MA)):
            next[i, i] = i
            for j in range(len(self.MA)):
                next[j, i] = next[i, j]
        
        self.MD = distance
        self.MN = next