from EdgeCentrality import *
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import random
import math
import os
import numpy as np
from scipy.sparse.linalg import eigsh #Find k eigenvalues and eigenvectors of the real symmetric square matrix or complex Hermitian matrix A
from scipy.sparse.linalg import eigs  #Find k eigenvalues and eigenvectors of the square matrix A
from scipy.sparse import csgraph, csr_matrix, lil_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from munkres import Munkres
import time

# copied from https://github.com/Tiger101010/DAEGC/blob/main/DAEGC/evaluation.py
# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return 0, 0, 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)

    f1_macro = f1_score(y_true, new_predict, average="macro")
    f1_micro = f1_score(y_true, new_predict, average="micro")

    #auc = roc_auc_score(y_true, new_predict, multi_class='ovr')

    return acc, f1_macro, f1_micro #, auc

def load_matrix(data):
    filepath = '../data/'+data+'/edgelist.txt'
    print('loading '+filepath)

    IJV = np.fromfile(filepath,sep="\t").reshape(-1,2)
    data = [1]*2*len(IJV[:,0].astype(np.int))

    row = IJV[:,0].astype(np.int)
    col = IJV[:,1].astype(np.int)

    A = csr_matrix( (data,(np.concatenate([row,col]),np.concatenate([col,row]))) )
    print("Data size:", A.shape)

    return A


def run(args):
    A = load_matrix(args.data) 
    L, d = csgraph.laplacian(A, return_diag=True, normed=True)

    _, U = eigsh(L, args.k)
    clustering = KMeans(n_clusters=args.k, random_state=1024).fit(U)

    labels = clustering.labels_
   
    stime = time.time()
    if args.EC=='EB':
        EC = cal_EB(A)
    elif args.EC=='ER':
        EC = cal_ER(A)
    elif args.EC=='BDRC':
        EC = cal_BDRC(A)
    elif args.EC=='EP':
        EC = cal_EP(A)
    elif args.EC=='EK':
        EC = cal_EK(A)
    elif args.EC=='ERK':
        EC = cal_ECHO(A, args.alpha, args.eps)
    elif args.EC=='GTOM':
        EC = cal_GTOM(A)

    etime = time.time()
    print("elapsed time: %f"%(etime-stime))

    xs = sorted(EC.items(), key=lambda item: item[1], reverse=False)
    num = int(0.1*len(xs))
    for i in range(9):
        for (k,val) in xs[i*num:(i+1)*num]:
            (u,v) = k
            A[u,v]=0
            A[v,u]=0
        
        A.eliminate_zeros()

        L, d = csgraph.laplacian(A, return_diag=True, normed=True)
        _, U = eigsh(L, args.k)
        clustering = KMeans(n_clusters=args.k, random_state=1024).fit(U)
        pred_labels = clustering.labels_

        acc, f1_macro, f1_micro = cluster_acc(labels, pred_labels)
        print("%d %d"%((i+1)*10, A.nnz))
        #print(acc, f1_macro, f1_micro)
        print("XXX:%f"%(acc))

    

def main():
    parser = ArgumentParser("Our",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='default',
                        help='data name.')

    parser.add_argument('--EC', default='EP',
                        help='EC name.')

    parser.add_argument('--k', default=0, type=int, help='#cluster')
    parser.add_argument('--alpha', default=0.5, type=float, help='decay factor')
    parser.add_argument('--eps', default=1e-45, type=float, help='error threshold')

    args = parser.parse_args()

    print(args)

    print("data=%s, #clusters=%d"%(args.data, args.k))

    run(args)

if __name__ == "__main__":
    sys.exit(main())



