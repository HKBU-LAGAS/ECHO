import numpy as np
from scipy.sparse import csgraph, csr_matrix, eye, vstack
from scipy import linalg
import networkx as nx
from sklearn import preprocessing

def cal_ER(A):
    L, d = csgraph.laplacian(A, return_diag=True, normed=False)
    Linv = np.linalg.pinv(L.todense())#linalg.pinv(L.todense())

    row, col = A.nonzero()

    ER = {}
    for i in range(len(row)):
        u = row[i]
        v = col[i]
        er_uv = Linv[u,u]+Linv[v,v]-2*Linv[u,v]

        ER[(u,v)] = er_uv

    return ER

def cal_BDRC(A):
    L, d = csgraph.laplacian(A, return_diag=True, normed=False)
    Linv = np.linalg.pinv(L.todense()) #linalg.pinv(L.todense())
    
    Linv = Linv.dot(Linv)
    row, col = A.nonzero()
    ER = {}
    for i in range(len(row)):
        u = row[i]
        v = col[i]
        er_uv = Linv[u,u]+Linv[v,v]-2*Linv[u,v]
        ER[(u,v)] = er_uv

    return ER

def cal_EB(A):
    G=nx.from_scipy_sparse_matrix(A)
    dic_EB = nx.edge_betweenness_centrality(G)

    #print(dic_EB)
    EB={}
    row, col = A.nonzero()
    for i in range(len(row)):
        u = row[i]
        v = col[i]
        if u>v:
            x=u
            u=v
            v=x

        eb_uv = dic_EB[(u,v)] 
        EB[(u,v)] = eb_uv
        EB[(v,u)] = eb_uv

    return EB

def cal_EP(A):
    P = preprocessing.normalize(A, norm='l1', axis=1)
    
    T = 150
    n = P.shape[0]
    e = csr_matrix(np.ones(n)*1.0/n)
    alpha=0.85
    ppr = e.copy()
    for t in range(T):
        ppr = alpha*ppr.dot(P) + e

    ppr = (1-alpha)*ppr

    print(ppr.shape)

    EP ={}
    row, col = A.nonzero()
    for i in range(len(row)):
        u = row[i]
        v = col[i]
        ep_uv = ppr[0,u]*P[u,v]
        EP[(u,v)] = ep_uv

    return EP

def cal_EK(A):
    T = 150
    n = A.shape[0]
    e = csr_matrix(np.ones(n)*1.0/n)
    alpha=0.85
    katz = e.copy()
    for t in range(T):
        katz = alpha*katz.dot(A) + e

    katz = (1-alpha)*katz

    print(katz.shape)

    EK ={}
    row, col = A.nonzero()
    for i in range(len(row)):
        u = row[i]
        v = col[i]
        ek_uv = katz[0,u]*A[u,v]
        EK[(u,v)] = ek_uv

    return EK

def cal_ERK(A, alpha=0.5):
    n = A.shape[0]
    G=nx.from_scipy_sparse_matrix(A)
    G = G.to_undirected()
    E = nx.incidence_matrix(G, edgelist=G.edges()).T
    F = preprocessing.normalize(E, norm='l1', axis=1)
    B = preprocessing.normalize(E, norm='l1', axis=0)
    #P = F.dot(B.T)
    
    
    T = 150
    m = P.shape[0]
    #e = csr_matrix(np.ones(m)*1.0/m)
    e = csr_matrix(np.ones(m))
    i=0
    for (u,v) in G.edges():
        e[0,i] = 1.0/np.sqrt(G.degree[u]+G.degree[v])
        i+=1
        
    #alpha=0.5
    ppr = e.copy()
    for t in range(T):
        #ppr = alpha*ppr.dot(P) + e
        ppr = alpha*(ppr.dot(F)).dot(B.T) + e

    ppr = (1-alpha)*ppr

    print(ppr.shape,ppr)

    EPP ={}
    i=0
    for (u,v) in G.edges():
        ep_uv = ppr[0,i]#*np.sqrt(G.degree[u]*G.degree[v])
        EPP[(u,v)] = ep_uv
        EPP[(v,u)] = ep_uv
        i+=1

    return EPP


def cal_GTOM(A):
    row, col = A.nonzero()

    EC = {}
    A.eliminate_zeros()
    for i in range(len(row)):
        u = row[i]
        v = col[i]
        er_uv = ((A[u].dot(A[v].T))[0,0]*1.0+1.0)/min(A[u].nnz,A[v].nnz)

        EC[(u,v)] = er_uv

    return EC
