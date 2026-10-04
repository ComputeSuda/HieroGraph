import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import HeteroData


path = "./result"
cancers = ['LUAD', 'LSCC', 'HNSCC', 'COAD', 'PDAC', 'BRCA', 'OV', 'UCEC', 'CCRCC', 'GBM']
site_max_na_ratio, gene_max_na_ratio = 0.3, 0.2
node_types = ['PTM', 'Protein', 'Pathway']
edge_types = {
    1: "Pathway association Pathway",
    2: "Protein attr Pathway",
    3: "Protein interaction Protein", 32: "Protein association Protein",
    4: "Protein regulation PTM", 42: "PTM attr Protein", 43: "PTM association Protein",
    5: "PTM association PTM"
}


with open('./data/data.pkl', 'rb') as f:
    features = pickle.load(f)
anno = pd.read_table(f"{path}/annotation.txt")
edges = pd.read_table(f"{path}/raw_network.txt")


# 1. Node feature
features['mut'] = features['mut'].fillna(0)

global_mean = {}
for k, v in features.items():
    v = v.loc[v.index.isin(anno['name'])]
    features[k] = v
    global_mean[k] = v.mean(axis=1)

# PTM node
phyche = features.pop('phyche')
pho = features.pop('pho')
ptm_features, ptm_valid = [], []
for cancer in cancers:
    tmp = pho.filter(like=cancer)
    valid = tmp.isna().mean(axis=1) <= site_max_na_ratio
    local_mean = tmp.mean(axis=1)
    local_mean = local_mean.where(valid, global_mean['pho'])

    valid.name = local_mean.name = f"{cancer}-pho"
    ptm_features.append(local_mean)
    ptm_valid.append(valid)
ptm_features.append(phyche)
ptm_features = pd.concat(ptm_features, axis=1)
scaler = StandardScaler()
ptm_features.loc[:] = scaler.fit_transform(ptm_features)
# print(scaler.mean_, '\n\n', scaler.scale_)
ptm_valid = pd.concat(ptm_valid, axis=1)

# Protein node
pro_features, pro_valid = [], []
for cancer in cancers:
    for k, v in features.items():
        tmp = v.filter(like=cancer)
        valid = tmp.isna().mean(axis=1) <= gene_max_na_ratio
        local_mean = tmp.mean(axis=1)
        local_mean = local_mean.where(valid, global_mean[k])

        valid.name = local_mean.name = f"{cancer}-{k}"
        pro_features.append(local_mean)
        pro_valid.append(valid)
pro_features = pd.concat(pro_features, axis=1)
col = ~pro_features.columns.str.contains('mut')
scaler = StandardScaler()
pro_features.loc[:, col] = scaler.fit_transform(pro_features.loc[:, col])
pro_valid = pd.concat(pro_valid, axis=1)

# 2. Edge
edges = edges[edges['source'].isin(anno['name']) & edges['target'].isin(anno['name'])]
name_index = {}
for node_type in node_types:
    nodes = anno.loc[anno['type'] == node_type, 'name'].to_list()
    name_index[node_type] = {name: i for i, name in enumerate(nodes)}
print("\nNode name_index")
for node_type, d in name_index.items():
    print(f"{node_type}:{len(d)}  {min(d.values())}-{max(d.values())}")

all_edge_index = {}
for edge_type in edge_types.keys():
    edge_name = edge_types[edge_type]
    source, _, target = edge_name.split(' ')

    sub_edges = edges.loc[edges['edge_type'] == edge_type, ['source', 'target']]
    sub_edges['source'] = sub_edges['source'].map(name_index[source])
    sub_edges['target'] = sub_edges['target'].map(name_index[target])
    edge_index = torch.tensor(sub_edges.values.T, dtype=torch.long)
    # print(edge_index.is_contiguous())
    if source == target:
        # GraphConv propagate direction: source->target
        # add reverse edges
        edge_index = torch.cat((edge_index, edge_index[[1, 0], :]), dim=1)
    all_edge_index[edge_name] = edge_index.contiguous()

# 3. Graph
data = HeteroData()
data['PTM'].x = torch.tensor(ptm_features.values, dtype=torch.float32).contiguous()
data['PTM'].valid = torch.tensor(ptm_valid.values, dtype=torch.bool).contiguous()
data['Protein'].x = torch.tensor(pro_features.values, dtype=torch.float32).contiguous()
data['Protein'].valid = torch.tensor(pro_valid.values, dtype=torch.bool).contiguous()
data['Pathway'].x = torch.zeros(size=(sum(anno['type'] == 'Pathway'), 1), dtype=torch.float32)

for v in edge_types.values():
    src, rel, tgt = v.split()
    edge_index = all_edge_index[v]
    data[(src, rel, tgt)].edge_index = edge_index
    if src != tgt:
        data[tgt, f"rev_{rel}", src].edge_index = edge_index[[1, 0], :].contiguous()
print("\nGraph data\n", data)

torch.save(data, f"{path}/GNN_graphs_data.pt")
with open(f"{path}/GNN_node_index.pkl", 'wb') as f:
    pickle.dump(name_index, f)
