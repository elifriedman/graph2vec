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
import pickle
import numpy as np
import networkx as nx
import graph2vec.node2vec as node2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from multiprocessing import Pool

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

    parser.add_argument('--num-iterations', default=1, type=int,
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

def learn_embeddings(docs, dimension=128, window_size=10, workers=8, num_iters=1, output='output'):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    model = Doc2Vec(docs, vector_size=dimension, window=window_size, min_count=0, workers=workers, epochs=num_iters)
    if output:
        model.save(output+"_model")
        model.save_word2vec_format(output, doctag_vec=True)

    return model

def load_graph_from_file(fname):
    if os.path.splitext(fname)[1] == ".gexf":
        nx_G = read_graph_gexf(fname)
    elif os.path.splitext(fname)[1] == '.pickle':
        nx_G = read_graph_pickle(fname)
    else:
        nx_G = read_graph(fname)
    return nx_G

def get_walks(nx_G, directed=True, p=1, q=1, walk_length=80, num_walks=10):
    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)

    for i, walk in enumerate(walks):
        for n, node in enumerate(walk):
            walk[n] = str(nx_G[node].get('Label', node))
    return walks

walk_ext = '_walks.pickle'
def load_and_get_walks(fname):
    if os.path.exists(fname+walk_ext):
        with open(fname+walk_ext, 'rb') as f:
            return pickle.load(f)
    t = time.time()
    graph = load_graph_from_file(fname)
    walks = get_walks(graph, p=args.p, q=args.q, num_walks=args.num_walks, walk_length=args.walk_length)
    for i in range(len(walks)):
        walks[i] = [str(walk) for walk in walks[i]]
    print('{} completed. ({})'.format(fname, time.time() - t))
    with open(fname+walk_ext, 'wb') as f:
        pickle.dump(walks, f)
    return walks

import time
def get_docs(input_dir, num_workers=8):
    folder = input_dir
    dirs = os.listdir(input_dir)
    files = []
    for fname in dirs:
        if 'walks' not in fname:
            path = os.path.join(folder, fname)
            files.append(path)

    N = len(files)

    with Pool(num_workers) as pool:
        res = pool.map(load_and_get_walks, files)
    print('Finished creating walks')

    docs = []
    for walks, fname  in zip(res, dirs):
        base = os.path.basename(fname)
        for walk in walks:
            doc = TaggedDocument(walk, [int(base.split(".")[0])])
            docs.append(doc)
    return docs


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    docs = get_docs(args.input, args.workers)
    learn_embeddings(docs, dimension=args.dimensions, window_size=args.window_size, num_iters=args.num_iterations, output=args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)
