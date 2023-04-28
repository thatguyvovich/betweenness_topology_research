import numpy as np
from typing import List

from pathfinders.AbstractPathfinder import AbstractPathfinder

class FloydWarshallMemoryPathfinder(AbstractPathfinder):
    
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
        Reconstructs the shortest path between two nodes in a topology using Floyd's algorithm
        :param MA: Adjacency matrix
        :return: A matrix of the shortest path between two nodes in the topology
        """
        distance = np.zeros(shape=(len(self.MA), len(self.MA), len(self.MA)), dtype=np.int64)
        next = np.zeros(shape=(len(self.MA), len(self.MA), len(self.MA)), dtype=np.int64)

        for i in range(len(self.MA)):
            for j in range(len(self.MA)):
                if i == j:
                    distance[i][j][0] = 0
                elif self.MA[i][j] != 0:
                    distance[i][j][0] = self.MA[i][j]
                    next[i][j][0] = j
                else:
                    distance[i][j][0] = 9000000
                    next[i][j][0] = -1

        for k in range(1, len(self.MA)):
            for i in range(len(self.MA)):
                for j in range(len(self.MA)):
                    distance[i][j][k] = min(distance[i][j][k-1], distance[i][k][k-1] + distance[k][j][k-1])
                    if distance[i][j][k] == distance[i][j][k-1]:
                        next[i][j][k] = next[i][j][k-1]
                    else:
                        next[i][j][k] = next[k][j][k-1]

        self.MD = distance
        self.MN = next