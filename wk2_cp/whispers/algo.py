import math

import numpy as np
from collections import Counter
from work import *


def whisper(graph, node_list, run_time):
    for i in range(run_time):
        selected_node = node_list[np.random.randint(len(node_list))]
        nbrs = getNeighbors(selected_node, node_list)
        propagate_labels(selected_node, nbrs, graph)


def getNeighbors(cur, nodes):
    ret = []
    for i in cur.neighbors:
        ret.append(nodes[i])
    return ret


def propagate_labels(node, neighbors, adj_matrix):
    c = {}

    for n in neighbors:
        #print(n.label)
        if n not in c:
            c[n.label] = adj_matrix[node.id][n.id]
        else:
            c[n.label] += adj_matrix[node.id][n.id]

    count = Counter(c)
    node.label = count.most_common()[0][0]


def connected_components(graph):
    return ([graph[i]][i.neighbor_list] for i in range(len(graph)))

