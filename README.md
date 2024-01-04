# EdgeRAKE: An Effective Edge Centrality for Graph Analysis

### Requirements
- Python 3.6.8 or above
- See requirements.txt for the version requirements for other packages

### Datasets
- See data/
- Sources: [Email-EU](https://snap.stanford.edu/data/email-Eu-core.html) [Facebook](https://snap.stanford.edu/data/ego-Facebook.html) [PPI & BlogCatalog](https://snap.stanford.edu/node2vec/) [Cora](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) [Chameleon](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikipediaNetwork.html#torch_geometric.datasets.WikipediaNetwork)

### Usage
Node clustering using spectral clustering:
```
$ sh run_SC_Email.sh
$ sh run_SC_Facebook.sh
```

Node classification using node2vec:
```
$ sh run_n2v.sh PPI
$ sh run_n2v.sh Blogcatalog
```

Node classification using GCN:
```
$ sh run_GCN_Cora.sh ERK
$ sh run_GCN_Chameleon.sh ERK
```

### Parameter Analysis
```
$ sh vary_Email.sh
$ sh vary_Facebook.sh
$ sh vary_n2v.sh PPI
$ sh vary_n2v.sh Blogcatalog
$ sh vary_Cora.sh
$ sh vary_Chameleon.sh
```
