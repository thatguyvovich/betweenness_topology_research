# Betweenness topology research
This is a part of collaborative research idea, that works with topological characteristics, in this instance, betweenness.

We define betweenness as a coefficient, that shows us which nodes are passed through the most. As there is a direct correlation 
between fault tolerance and workload being spread unevenly, it is important to predict which nodes are the most "central" to see
if any potential faults will impact the overall stability of the system and its other characteristics.

## Current version
Current version of the algorithm uses Floyd-Warshall pathfinding algorithm, thanks to its ability to be quickly deployed and
maintained. 
