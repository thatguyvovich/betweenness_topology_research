from abc import ABC, abstractmethod
from typing import List

class AbstractPathfinder:
    # Adjacency matrix
    MA: List[List[int]]
    # Distance matrix
    MD: List[List[int]]
    # Next matrix
    MN: List[List[int]]

    # Method that returns a matrix of paths
    @abstractmethod
    def find_paths() -> List[List[int]]: ...

    # Internal function that calculates paths
    @abstractmethod
    def __path_reconstructor(): ...
