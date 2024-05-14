import torch
from torch_geometric.nn.models import GCN, GAT, GIN, LINKX
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor, DeezerEurope, WikiCS, LINKXDataset
# from ogb.nodeproppred import PygNodePropPredDataset 
from model import MyLinear, MyMLP, SGC, APPNP, GCNII, MGNN, pGNN, PointNet,MyGCN
import random
from scipy.sparse import csr_matrix, diags, identity, csgraph, hstack,csc_array,csr_array
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.cluster import KMeans
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from EdgeCentrality import *
import time

def get_model(model : str, dataset, args):
	if model == 'Linear':
		model = MyLinear(in_channels = dataset.data.num_features, out_channels = dataset.num_classes, 
					dropout = args['dropout'])
	elif model == 'MLP':
		model = MyMLP(channel_list = [dataset.data.num_features] + [args['hidden_dim']] * (args['num_layers'] - 1) + [dataset.num_classes],
					dropout = args['dropout'])
	elif model == 'GCN':
		model = GCN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					dropout = args['dropout'])
	elif model == 'MyGCN':
		model = MyGCN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					dropout = args['dropout'])
	elif model == 'SGC':
		model = SGC(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					K = args['num_layers'], 
					dropout = args['dropout'])
	elif model == 'GAT':
		model = GAT(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = dataset.num_classes,
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'APPNP':
		model = APPNP(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = dataset.num_classes,
					K = args['num_layers'],
					alpha = args['alpha'],
					dropout = args['dropout'])
	elif model == 'GCNII':
		model = GCNII(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					alpha = args['alpha'],
					theta = args['theta'],
					dropout = args['dropout'])
	elif model == 'MGNN':
		model = MGNN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					alpha = args['alpha'],
					beta = args['beta'],
					theta = args['theta'],
					dropout = args['dropout'],
					attention_method = args['attention_method'],
					initial = args['initial'])
	elif model == 'LINKX':
		model = LINKX(num_nodes = args['num_nodes'],
					in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes,
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'pGNN':
		model = pGNN(in_channels = dataset.data.num_features, 
                	out_channels = dataset.num_classes,
                	num_hid = args['hidden_dim'], 
                	mu = args['alpha'],
                 	p = args['theta'],
                 	K = args['num_layers'],
                	dropout = args['dropout'])
	elif model == 'GIN':
		model = GIN(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'PointNet':
		model = PointNet(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	return model

def get_dataset(root : str, name : str):
	if name in ['Cora', 'CiteSeer', 'PubMed']:
		dataset = Planetoid(root = root, name = name)
	elif name == 'CoraFull':
		dataset = CoraFull(root = root)
	elif name in ['Computers', 'Photo']:
		dataset = Amazon(root = root, name = name)
	elif name in ['CS', 'Physics']:
		dataset = Coauthor(root = root, name = name)
	elif name in ['Cornell', 'Texas', 'Wisconsin']:
		dataset = WebKB(root = root, name = name)
	elif name in ['Chameleon', 'Squirrel']:
		dataset = WikipediaNetwork(root = root, name = name.lower())
	elif name == 'Actor':
		dataset = Actor(root = root)
	elif name == 'DeezerEurope':
		dataset = DeezerEurope(root = root)
	elif name == 'WikiCS':
		dataset = WikiCS(root = root, is_undirected = True)
	elif name in ['genius']:
		dataset = LINKXDataset(root = root, name = name)
  
	elif name in ['ogbn-arxiv', 'ogbn-products']:
		pass 
  		# dataset = PygNodePropPredDataset(name = name, root = root)
  
	else:
		raise Exception('Unknown dataset.')

	return dataset

def cal_EC(dataset, stype, alpha, eps):
    data = dataset.data
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]
    m = rows.shape[0]
    A = csr_matrix( ([1]*2*m, (np.concatenate([rows, cols]) , np.concatenate([cols, rows]))) )
    
    stime = time.time()
    if stype=='ER':
        EC = cal_ER(A)
    elif stype=='BDRC':
        EC = cal_BDRC(A)
    elif stype=='EB':
        EC = cal_EB(A)
    elif stype=='EP':
        EC = cal_EP(A)
    elif stype=='EK':
        EC = cal_EK(A)
    elif stype=='ERK':
        EC = cal_ECHO(A, alpha, eps)
    elif stype=='GTOM':
        EC = cal_GTOM(A)

    etime = time.time()
    print("elapsed time: %f"%(etime-stime))
    
    arr = np.zeros(len(rows))
    for i in range(len(rows)):
        u = rows[i]
        v = cols[i]
        ec = EC[(u,v)]
        #print(type(ec),ec)
        arr[i]=ec

    EC = torch.tensor(arr)

    return EC
 


def del_edges(dataset, del_rate, EC, stype = 'ER'):
    data = dataset.data
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]
    m = rows.shape[0]
    n = dataset.data.x.size()[0]

    edge = dataset.data.edge_index
    row = edge[0]
    col = edge[1]
    edges_num = edge.shape[1]

    sample_edges_num = int(edges_num * del_rate)
    sorted_EC = torch.sort(EC,descending=False)
    idx =  sorted_EC.indices
    
    new_edge_idx = idx[sample_edges_num:]
    edge = edge[:, new_edge_idx]
    dataset.data.edge_index = edge

    return dataset

