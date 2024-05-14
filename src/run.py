import os
import random
from tqdm import tqdm
import numpy as np
import itertools
import torch
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.profile import count_parameters
from torch_geometric.transforms import RandomNodeSplit
from utils import get_model, get_dataset, cal_EC, del_edges #augment, add_edges, ER_approximation, graph_sparsifier,cal_SEC
import time as te 
# from ogb.nodeproppred import Evaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, graph, loss_fn, optimizer, weighted= False):
    # Training.
    model.train()
    optimizer.zero_grad()
    
    # Forward pass.
    if not weighted:
        pred = model.forward(graph.x, graph.edge_index)
    else:
        
        pred = model.forward(graph.x, graph.edge_index, graph.edge_attr)
        
    y = graph.y.squeeze()
    loss = loss_fn(pred[graph.train_mask], y[graph.train_mask])

    loss.backward()
    optimizer.step()

@torch.no_grad()
def infer(model, graph, mode : str = 'val', weighted: bool = False):
    # Inference.
    model.eval()
    # Get mask.
    assert mode in ['val', 'test']
    mask = graph.val_mask if mode == 'val' else graph.test_mask

    # Forward pass.
    if not weighted:
        pred = model.forward(graph.x, graph.edge_index)
    else:
        pred = model.forward(graph.x, graph.edge_index, graph.edge_attr)
        
    pred = pred.argmax(dim = 1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc

@torch.no_grad()
def ogb_infer(model, graph, evaluator, mode : str = 'val'):
    # Inference.
    model.eval()
    # Get mask.
    assert mode in ['val', 'test']
    mask = graph.val_mask if mode == 'val' else graph.test_mask

    # Forward pass.
    pred = model.forward(graph.x, graph.edge_index)
    pred = pred.argmax(dim = 1).unsqueeze(1)
    acc = evaluator.eval({'y_true': graph.y[mask], 'y_pred': pred[mask]})['acc']
    return acc

def run(args, verbose: bool = True):
    # torch.autograd.set_detect_anomaly(True)
    # Seed.

    if args['seed'] != -1:
        random.seed(args['seed'])
        os.environ['PYTHONHASHSEED'] = str(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device.
    device = (args['device'] if torch.cuda.is_available() else 'cpu')

    if verbose:
        print('Using device: {}'.format(device))
        print(args)

    times = args['times']
    epochs = args['epochs']

    val_accs = []
    test_accs = []
    dataset = get_dataset(root = '../data/{}'.format(args['dataset']), name = args['dataset'])

    n = dataset.data.x.size()[0]
    
    if args['del_rate']>0:
        print("deleting %f edges"%(args['del_rate']))
        EC = cal_EC(dataset,args['stype'],args['alpha'],args['eps'])
        dataset = del_edges(dataset, args['del_rate'], EC, args['stype'])
            
        
    for time in range(times):
        if device != 'cpu':
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                                
        model = get_model(model = args['model'], dataset = dataset, args = args).to(device)
        if verbose:
            print('Total parameters: {}'.format(count_parameters(model)))
        # Graph.
        graph = dataset.data

        if 'ogbn' in args['dataset']:
            split_idx = dataset.get_idx_split() 
            graph.train_mask = split_idx['train']
            graph.val_mask = split_idx['valid']
            graph.test_mask = split_idx['test']
            
        else:
            # Training node settings.
            if args['setting'] == 'GCN':
                split_gen = RandomNodeSplit(split = 'random')
            elif args['setting'] == 'semi':
                split_gen = RandomNodeSplit(split = 'train_rest', num_val = 0.025, num_test = 0.95)
            elif args['setting'] == '48/32/20':
                split_gen = RandomNodeSplit(split = 'train_rest', num_val = 0.32, num_test = 0.20)
            elif args['setting'] == '60/20/20':
                split_gen = RandomNodeSplit(split = 'train_rest', num_val = 0.20, num_test = 0.20)

            # Randomly pick a train-validation-test split.
            if args['setting'] != 'public':
                graph = split_gen(graph)

        # To undirected graph.
        
        if not is_undirected(graph.edge_index):
            graph.edge_index = to_undirected(graph.edge_index)

        graph = graph.to(device)
        
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args['lr'], weight_decay = args['weight_decay'])        
        loss_fn = torch.nn.CrossEntropyLoss()

        if verbose:
            print('Training time: {}'.format(time))
        best_val_acc = 0.0

        patience = 0
        
        for epoch in tqdm(range(epochs)) if verbose else range(epochs):
            
            a = te.time()
            train(model = model, graph = graph, loss_fn = loss_fn, optimizer = optimizer, weighted= args['weighted'])
            b = te.time() 
            train_time = b - a 
            
            if 'ogb' in args['dataset']:
                val_acc = ogb_infer(model = model, graph = graph, evaluator = evaluator, mode = 'val')
            else:
                c = te.time()
                val_acc = infer(model = model, graph = graph, mode = 'val', weighted= args['weighted'])
                d = te.time()
                infer_time = d - c 
                
            if val_acc > best_val_acc:
                patience = 0
                best_val_acc = val_acc
                if 'ogb' in args['dataset']:
                    test_acc = ogb_infer(model = model, graph = graph, evaluator = evaluator, mode = 'test' )
                else:
                    test_acc = infer(model = model, graph = graph, mode = 'test', weighted= args['weighted'])
            else:
                patience += 1
                if patience > args['early_stopping']:
                    break
        if verbose:
            print('Training done. Test_acc (on best valid_acc = {:.2f}%) = {:.2f}%\n'.format(100 * best_val_acc, 100 * test_acc))
        val_accs.append(100 * val_acc)
        test_accs.append(100 * test_acc)

    val_accs = np.array(val_accs)
    val_mean = np.mean(val_accs)
    val_stddev = np.std(val_accs)

    test_accs = np.array(test_accs)
    test_mean = np.mean(test_accs)
    test_stddev = np.std(test_accs)
    if verbose:
        print('Mean = {:.2f}%; Stddev = {:.2f}%'.format(test_mean, test_stddev))
        print('ACC: {:.2f}% {:.2f}$'.format(test_mean, test_stddev)+" by %s on %s"%(args['model'], args['dataset']))
        print('XXX:{:.4f}'.format(test_mean/100.0))
        print('Training time = {}'.format(train_time) )
        print('Inference time = {}'.format(infer_time) )

    return val_mean, val_stddev, test_mean, test_stddev
