# Parallel Graph Coloring with Cuda C++
## Introduction
In general, graph coloring can refer to conditionally labelling any component of a graph such as its vertices or edges. We deal with a special case of graph coloring called "<b>Vertex Coloring</b>". The problem statement is as follows:

An undirected graph G is a set of vertices V and a set of edges E. The edges are of the form ```(i, j)``` where i,j belong to V . A coloring of a graph G is a mapping c : ```V -> {1, 2,..., s}``` such that ```c(i) != c(j)``` for all edges ```(i, j)``` belonging to E. c(i) is referred to as the color of vertex i.

Finding the <b>optimal</b> graph coloring is an <i>NP-Complete</i> problem. The goal of this project is to obtain a balanced coloring of the graph i.e, the number of colors used is made as close as possible to the <b>chromatic number</b> (minimum number for that graph) to ensure some degree of load balancing.

## Graph representation
The <b>Compressed Sparse Row (CSR)</b> format has been used to represent the graphs and work with them in parallel. This format uses two arrays:

- Adjacency List array - To store the adjacency list of each vertex.
- Adjacency List Pointer array- To store the pointer to the first element of the adjacency list of each vertex.
Ex: Adjacency list of vertex v starts at ```adjList[adjListPointer[v]]``` and ends at ```adjList[adjListPointer[v+1]]-1```. This way 1-dimensional arrays can be used, thereby benefitting paralleliation.

## Approach
The serial implementation for the algorithm is an iterative greedy algorithm that is strictly sequential making parallelisation considerably difficult. However by assigning weights to each vertex based on certain constraints, parallelisation can be made possible. In the <b>Jones Plassman</b> approach, vertices are initially given random weights and each vertex is colored with the smallest available color if it has a greater weight than all its neighbors. In the <b>Largest Degree First </b>algorithm, the vertex is colored if it has the highest degree in the set containing itself and its neighbors. In the <b>Smallest Degree First</b> algorithm, the initial weighting process is done on the basis of degree. In each algorithm, the crucial portion is to resolve conflicts, that is to remove all the instances of wrong coloring (neighbors sharing the same color).

## Dependencies
CUDA (10.x or higher)

## Achieved Speedup : 
For random graphs of 10000 vertices and 1000000 edges, computed on an Nvidia 130MX GPU, the following speedups were achieved over the serial implementation: 
- Jones-Plassmann Algorithm : 3.75x
- Largest Degree First : 3.28x
- Smallest Degree Last : 3.07x 

## Contributors
- [Dhanwin Rao](https://github.com/dhanwin247)
- [Sai Krishna Anand](https://github.com/SaiKrishna1207)
