# -*- coding: utf-8 -*-
import math
import argparse
from typing import List
# LLVM compiler
from numba import njit
import numpy as np

from pathfinders.FloydWarshall import FloydWarshallPathfinder
from pathfinders.FloydWarshallMemory import FloydWarshallMemoryPathfinder
from pathfinders.AbstractPathfinder import AbstractPathfinder


class PathfinderFactory():
    """
    Factory class for pathfinding algorithms
    """
    @staticmethod
    def create_pathfinder(patfinder_type: str, topology: List[List[int]]) -> AbstractPathfinder:
        """
        Factory method that returns an instance of a pathfinding algorithm
        :return: An instance of a pathfinding algorithm
        """
        if patfinder_type == "FloydWarshallMemory":
            return FloydWarshallMemoryPathfinder(topology)
        elif patfinder_type == "FloydWarshall":
            return FloydWarshallPathfinder(topology)
        else:
            raise Exception("Pathfinder not found")


def Betweeness(MA):
    """
    A function that calculates the betweeness of a topology
    :param MA: Adjacency matrix
    :return: A vector of the betweeness of each node
    """
    betweeness = [np.count_nonzero(MA == i) for i in range(len(MA))]
    return betweeness

def ToBetweenessVector(MA):
    """
    A function that calculates the betweeness of a topology
    param MA: Adjacency matrix
    return: A vector of the betweeness of each node
    """
    distance = np.zeros(shape=(len(MA), len(MA)), dtype=np.int64)
    distance = np.copy(MA)
    betweeness = np.zeros(shape=(len(MA)), dtype=np.int64)
    for i in range(len(MA)):
        for j in range(len(MA)):
            if distance[i, j] == 0:
                distance[i, j] = 9000000

    for k in range(len(MA)):
        for i in range(len(MA)):
            for j in range(len(MA)):
                if distance[i, j] > distance[i, k] + distance[k, j]:
                    distance[i, j] = distance[i, k] + distance[k, j]
                    betweeness[k] = betweeness[k] + 1

    for i in range(len(MA)):
        distance[i, i] = 0
        for j in range(len(MA)):
            distance[j, i] = distance[i, j]

    return betweeness


def CopyDiagonalMatrix(MF):
    """
    Copy the diagonal of a matrix to the other diagonal
    """
    size = len(MF)
    for i in range(size):
        for j in range(i, size):
            MF[j][i] = MF[i][j]
    return MF


def JavaMatrixToPython(MF: str):
    """
    A function that takes in a string representation of a matrix as in Java 
    and returns a python matrix
    """
    MF = MF.replace("{", "")
    MF = MF.replace("}", "")
    MF = MF.replace(" ", "")
    MF = MF.split(",")
    MF = [int(i) for i in MF]
    MF = np.array(MF)
    MF = MF.reshape((int(len(MF) ** (1 / 2)), int(len(MF) ** (1 / 2))))
    return MF

def get_betweenness(topology):
    """
    Returns a list of betwennness coefficients for the example graph
    """
    FWP = FloydWarshallPathfinder(topology)
    FWM = FloydWarshallMemoryPathfinder(topology)
    MA, MN = FWM.find_paths()
    print(MN)
    print(Betweeness(MN))

    return Betweeness(MN)

def convert_string_to_matrix(topology: str):
    """
    A function that converts a string representation of a matrix to a python matrix
    """
    topology = JavaMatrixToPython(topology)
    topology = CopyDiagonalMatrix(topology)
    return topology


def model_faults(betweenness, topology, pathfinder_type: str, fault_rate: float=0.3):
    """
    A function that models the faults in the network
    """
    indices = []

    for _ in range(math.floor(len(betweenness) * fault_rate)):
        max_index = betweenness.index(max(betweenness))
        print("Removing node: ", max_index + len([i for i in indices if i <= max_index]))
        indices.append(max_index)
        topology = np.delete(topology, max_index, axis=0)
        topology = np.delete(topology, max_index, axis=1)
        _, MN = PathfinderFactory.create_pathfinder(pathfinder_type, topology).find_paths()
        betweenness = Betweeness(MN)
        print(betweenness)


def get_topology_from_input(path: str=None):
    """
    A function that takes in the path of file and retrieves the topology described as a java array 
    (ex. {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}})
    """
    if path is None:
        path = input("Enter the path of the file: ")
    with open(path, 'r') as file:
        topology = file.read()

    convert_string_to_matrix(topology)
    return topology


def main(args):
    """
    Entry point for tests during development
    """
    dragon_tree = get_topology_from_input(args.topology)
    dragon_tree = convert_string_to_matrix(dragon_tree)

    betweeness_results = get_betweenness(dragon_tree)

    betweeness_results.index(max(betweeness_results))

    dragon_tree_new = np.array(dragon_tree)

    model_faults(betweeness_results, dragon_tree_new, "FloydWarshallMemory",
                 fault_rate=float(args.fault_rate))

def read_cmd_arguments():
    """
    A function that reads the command line arguments
    """
    parser = argparse.ArgumentParser(description='A program that models the faults in a network')
    parser.add_argument('-t', '--topology', help='The topology of the network', required=True)
    parser.add_argument('-f', '--fault_rate', help='The fault rate of the network', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_cmd_arguments()
    print(args)
    main(args)
