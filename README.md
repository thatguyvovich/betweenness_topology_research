# Betweenness topology research
This is a part of collaborative research idea, that works with topological characteristics, in this instance, betweenness.

We define betweenness as a coefficient, that shows us which nodes are passed through the most. As there is a direct correlation 
between fault tolerance and workload being spread unevenly, it is important to predict which nodes are the most "central" to see
if any potential faults will impact the overall stability of the system and its other characteristics.

## Current version
Current version of the algorithm uses Floyd-Warshall pathfinding algorithm, thanks to its ability to be quickly deployed and
maintained. 

## How to use
Just enter into the console the following command:
_python3 prs.py -t {path_to_topology} -f {fault_rate}_

*-t* takes in a path to the file where topology is stored as a java-like matrix (_ex. {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}_)

*-f* takes in the fault rate of the topology (_from 0.0 to 1.0_).