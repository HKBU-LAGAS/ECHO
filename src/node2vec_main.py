'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
import time
from gensim.models import Word2Vec
from labelclassification import *
from EdgeCentrality import *

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--data', nargs='?', default='cora',
	                    help='Input graph path')

    parser.add_argument('--EC', type=str, default="ER",
	                    help='EC name.')
    
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

    parser.add_argument('--alpha', default=0.5, type=float, help='decay factor')
    return parser.parse_args()

def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    fpath = "../data/"+args.data+"/edgelist.txt"
    
    G = nx.read_edgelist(fpath, nodetype=int)
    G = G.to_undirected()
    nx.set_edge_attributes(G, values = 1, name = 'weight')

    print(G.number_of_nodes(), G.number_of_edges())
    
    maxu = G.number_of_nodes()

    return G, maxu

def learn_embeddings(data, maxu, walks, ptime):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    start = time.time()
    #walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
    end = time.time()
    ptime2 = end - start
    print("time: ", ptime2+ptime)
    #model.wv.save_word2vec_format(args.output)
    
    kv = model.wv

    X = kv.vectors
    
    print(X.shape)

    idx=[]
    o_idx=[]
    for i in range(maxu):
        if i in kv.key_to_index:
            j = kv.key_to_index[i] #kv.vocab[str(i)].index
            #jvec = X[j]
            idx.append(j)
            o_idx.append(i)

    print(len(idx), maxu)

    X = X[idx]


    print(X.shape)

    #print((kv.index2word))
    #print((kv.index_to_key))

    #print(X[kv.vocab['0'].index])


    #outfile = "emb/"+data+".node2vec.npz"
    #np.savez(outfile, X)

    return X

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    data = args.data
    nx_G, maxu = read_graph(args)

    A = nx.adjacency_matrix(nx_G, nodelist=[v for v in range(maxu)])
    if args.EC=='ER':
        EC = cal_ER(A)
    elif args.EC=='BDRC':
        EC = cal_BDRC(A)
    elif args.EC=='EB':
        EC = cal_EB(A)
    elif args.EC=='EP':
        EC = cal_EP(A)
    elif args.EC=='EK':
        EC = cal_EK(A)
    elif args.EC=='GTOM':
        EC = cal_GTOM(A)
    elif args.EC=='ERK':
        EC = cal_ERK(A, args.alpha)

    xs = sorted(EC.items(), key=lambda item: item[1], reverse=False)
    num = int(0.1*len(xs))
    for i in range(9):
        for (k,val) in xs[i*num:(i+1)*num]:
            #nx_G.remove_edge(*k)
            (u,v) = k
            A[u,v] = 0
            A[v,u] = 0

        A.eliminate_zeros()
        nx_G = nx.from_numpy_matrix(A)
        print("%d %d"%(i,nx_G.number_of_edges()))

        start = time.time()
        G = node2vec.Graph(nx_G, False, args.p, args.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        print("walsk size:", len(walks))
        end = time.time()
        ptime = end - start
        X = learn_embeddings(data, maxu, walks, ptime)
        label_path = "../data/"+args.data+"/labels.txt" 
        YY, Y = read_node_label(label_path)
        r = 0.5
        print('Training classifier using {:.2f}% nodes...'.format(r * 100))
        clf = Classifier(vectors=X, clf=LogisticRegression())
        results = clf.split_train_evaluate(YY, Y, r)
        print("XXX:"+str(results['micro']))

if __name__ == "__main__":
    args = parse_args()
    main(args)
