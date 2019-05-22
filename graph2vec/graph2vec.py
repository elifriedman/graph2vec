#!/usr/bin/python3
'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import os
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('input', help='Input graph folder')

    parser.add_argument('--output', default='graph2vec_output',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                    help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                  help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

def read_graph_pickle(fname):
    G = nx.read_gpickle(fname)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()

    return G


def read_graph_gexf(fname):
    G = nx.read_gexf(fname, node_type=int)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()

    return G

def read_graph(fname):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(fname, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(fname, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def learn_embeddings(docs):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    model = Doc2Vec(list(docs), vector_size=args.dimensions, window=args.window_size, min_count=0, workers=args.workers, epochs=args.iter)
    model.save(args.output+"_model")
    model.save_word2vec_format(args.output, doctag_vec=True)

    return

def load_graph_from_file(fname):
    if os.path.splitext(fname)[1] == ".gexf":
        nx_G = read_graph_gexf(fname)
    elif os.path.splitext(fname)[1] == '.pickle':
        nx_G = read_graph_pickle(fname)
    else:
        nx_G = read_graph(fname)
    return nx_G

def get_walk(nx_G):
    def convert_node(node):
        if 'Label' in nx_G.node[node]:
            return str(nx_G.node[node]['Label'])
        return node
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    ext = os.path.splitext(fname)[1]


    for i, walk in enumerate(walks):
        walks[i] = map(convert_node, walk)
    return walks


import time
def get_docs():
    folder = args.input
    dirs = os.listdir(args.input)
    files = []
    for dir in dirs:
        path = os.path.join(folder, dir)
        subs = os.listdir(path)
        files.extend([os.path.join(path, sub) for sub in subs])

    N = len(files)
    for i, fname in enumerate(files):
        base = os.path.basename(fname)
        t = time.time()
        graph = load_graph_from_file(fname)
        walks = get_walk(graph)
        for walk in walks:
            walk = map(str, walk)
            doc = TaggedDocument(walk, [int(base.split(".")[0])])
            yield doc
        print("{}/{} ({}) ({})".format(i, N, fname, time.time()-t))


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    docs = get_docs()
    learn_embeddings(docs)

if __name__ == "__main__":
    args = parse_args()
    main(args)
