**Radial Observer Source Estimator (ROSE)**

ROSE algorithm is designed for estimating the rumor source node in social networks. The ROSE algorithm requires the following installation setup.
1. Python 3.10 are above
2. NDlib - Network Diffusion Library

The Python code (ROSE) requires the following packages to be imported
1. import networkx as nx
2. import csv
3. import random
4. import ndlib.models.ModelConfig as mc
5. import ndlib.models.epidemics as ep
6. import numpy as np
7. import timeit
8. from collections import deque

This code requires an Input file (.csv format) as shown in the following format: 
(It has two columns such as Source and Target)

Source,Target

1,2

2,1

1,3

3,1

2,3

3,2

The original source can be obtained from the simulation part (IC model).

The output of the ROSE algorithm code is the estimated source, i.e., any node belongs to the graph (input_file.csv).

Once the estimated source and the original source are available, we can generate Distance error, Execution time, Accuracy, Candidate sources, etc.


