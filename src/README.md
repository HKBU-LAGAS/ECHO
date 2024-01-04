## Datasets & Models that We Need to Test
- Cora, CiteSeer, Chameleon, Squirrel, CS, Physics, WikiCS, PubMed, CoraFull, Ogbn-arxiv
- MLP, GCN, GAT, GIN, SGC, APPNP, GCNII

### Running without augmentation
Example
```shell
$ sh MLP.sh >> log/MLP.log
```
After finishing
```shell
$ cat log/MLP.log |grep ACC
```

### Running with augmentation
Example
```shell
$ sh Cora.sh >> log/Cora.log
```
After finishing
```shell
$ cat log/Cora.log |grep ACC
```
We can amend the "augment(dataset)" function in "utils.py" to try various augmentation strategies.


## Acknowledgements
- Code base from https://github.com/GuanyuCui/MGNN
