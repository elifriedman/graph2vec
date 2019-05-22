from __future__ import print_function
import os
import sys
from collections import defaultdict
import networkx as nx

f_edgelist="_A.txt"
f_graphlist="_graph_indicator.txt"
f_labellist="_node_labels.txt"
f_glabellist="_graph_labels.txt"

def read_graph_label(fname):
    graph2label = {}
    with open(fname) as f:
        for i, line in enumerate(f):
            line = line.strip()
            graph2label[str(i+1)] = line
    return graph2label

def read_node_label(fname):
    node2label = {}
    with open(fname) as f:
        for i, line in enumerate(f):
            line = line.strip()
            node2label[str(i+1)] = line
    return node2label

def read_graph_mapping(fname):
    with open(fname) as f:
        graph2nodes = defaultdict(list)
        node2graph = {}
        for i, line in enumerate(f):
            node = str(i+1)
            graph = line.strip()
            graph2nodes[graph].append(node)
            node2graph[node] = graph
    return graph2nodes, node2graph

def process_graph(path):
    _, folder = os.path.split(path)
    print("1: read node labels")
    n2l = read_node_label("{}/{}".format(path, folder)+f_labellist)
    print("2: read graph labels")
    g2l = read_graph_label("{}/{}".format(path, folder)+f_glabellist)
    print("3: read mapping")
    g2n, n2g = read_graph_mapping("{}/{}".format(path, folder)+f_graphlist)

    print("4: read edgelist")
    graphs = defaultdict(nx.Graph)
    with open("{}/{}".format(path, folder)+f_edgelist) as f:
        for line in f:
            n1, n2 = [n.strip() for n in line.strip().split(",")]
            graph = n2g[n1]
            graphs[graph].add_edge(n1, n2)
            graphs[graph].nodes[n1]['Label'] = n2l[n1]
            graphs[graph].nodes[n2]['Label'] = n2l[n2]

    print("5: write pickles")
    os.mkdir(path+"/graphs")
    for label in set(g2l.values()):
        os.mkdir(path+"/graphs/"+str(label))
    for graph in graphs:
        label = g2l[graph]
        nx.write_gpickle(graphs[graph], "{}/graphs/{}/{}.pickle".format(path, label, graph))

    print("done")

if __name__ == "__main__":
    process_graph(sys.argv[1])

