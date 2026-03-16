from work import *
from algo import *
vecs = proc()
adj_matrix, node_list = generateGraph(vecs)

string = ""
for node in node_list:
    string += str(node.label)+" "
print(string)
whisper(adj_matrix, node_list, 1000)

string = ""
for node in node_list:
    string += str(node.label) +" "
print(string)