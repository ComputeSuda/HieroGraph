import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, confusion_matrix
from GNN_AE_model import Encoder
from GNN_node_classification import NodeClassifier, EncoderNodeClassification
from captum.attr import IntegratedGradients, LayerIntegratedGradients

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._utils.batching")

torch.set_default_device(torch.device('cuda'))


class CaptumHeteroWrapper(torch.nn.Module):
    def __init__(self, model, active_node_types, x_dict, edge_index_dict):
        super().__init__()
        self.model = model
        self.active_node_types = active_node_types
        self.edge_index_dict = edge_index_dict
        # freeze the raw pathway features that do not require differentiation
        self.static_x_dict = {k: v.clone().detach() for k, v in x_dict.items() if k not in active_node_types}

    def forward(self, *args):
        active_node_tensors = args[:-1]
        target_idx = args[-1]

        # dynamically assemble the active nodes for the current interpolation step
        x_dict = {
            node_type: tensor[0]
            for node_type, tensor in zip(self.active_node_types, active_node_tensors)
        }
        # add the static nodes (pathway) that do not participate in differentiation
        for k, v in self.static_x_dict.items():
            x_dict[k] = v

        y_logits = self.model(x_dict, self.edge_index_dict, target_idx)
        if y_logits.dim() == 1:
            y_logits = y_logits.unsqueeze(0)
        return y_logits


def evaluate(graph, model, loader, loader_type):
    model.eval()
    y_true, y_predict, all_y_logits = [], [], []
    results = {}

    with torch.no_grad():
        for batch_x_idx, batch_y in loader:
            y_logits = model(graph.x_dict, graph.edge_index_dict, batch_x_idx)
            y_true.extend(batch_y.tolist())
            y_predict.extend(y_logits.argmax(dim=1).tolist())
            all_y_logits.append(y_logits.cpu())

    y_true, y_predict = np.array(y_true), np.array(y_predict)
    all_y_logits = torch.cat(all_y_logits, dim=0)
    y_probs = F.softmax(all_y_logits, dim=1).numpy()
    subtypes = {0: 'Negative', 1: 'Positive'}

    cm = confusion_matrix(y_true, y_predict)
    cm = pd.DataFrame(data=cm,
                      index=[f"Actual-{s}" for s in subtypes.values()],
                      columns=[f"Predict-{s}" for s in subtypes.values()])
    # print(loader_type, cm, sep='\n')

    results['Accuracy'] = round(accuracy_score(y_true, y_predict), 3)
    if y_logits.shape[1] == 2:  # Binary evaluate
        results['Precision'] = precision_score(y_true, y_predict)
        results['Recall'] = recall_score(y_true, y_predict)
        results['F1'] = f1_score(y_true, y_predict)

        results['AUC'] = roc_auc_score(y_true, y_probs[:, 1])
        results['AUPRC'] = average_precision_score(y_true, y_probs[:, 1])
    else:   # Multi-class evaluate
        results['Weighted Precision'] = precision_score(y_true, y_predict, average='weighted')
        results['Weighted Recall'] = recall_score(y_true, y_predict, average='weighted')
        results['Weighted F1'] = f1_score(y_true, y_predict, average='weighted')

        results['Weighted AUC'] = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
        results['Weighted AUPRC'] = average_precision_score(y_true, y_probs, average='weighted')

        recall_in_subtypes = recall_score(y_true, y_predict, average=None)
        for cls in subtypes.keys():
            results[f"Recall in {subtypes[cls]}"] = recall_in_subtypes[cls]

    results = pd.DataFrame.from_dict(results, orient='index', columns=[loader_type]).round(3)
    return results


path = "./result"
graph = torch.load(f"{path}/GNN_graphs_data.pt", weights_only=False, map_location='cuda')
parameters = torch.load(f"{path}/GNN_best_Transformer_classifier.pt")
hyper_params, model_params = parameters['hyper_parameters'], parameters['model_parameters']
encoder = Encoder(graph.num_node_features, hyper_params['operator'],
                  hyper_params['hidden_channels'], hyper_params['activation_function'])
classifier = NodeClassifier(hyper_params['hidden_channels'], 64, 16, 2)
model = EncoderNodeClassification(encoder, classifier)
model.load_state_dict(model_params)
model.eval()

with open(f"{path}/GNN_node_index.pkl", 'rb') as f:
    name_index = pickle.load(f)
pathway = pd.read_csv('./knowledge/pathway.csv', index_col=0)
pathway_size = pathway.groupby("PATHWAY_ID")["GENE_ID"].apply(set).map(len)
pathway_name = pathway.loc[:, 'PATHWAY_NAME'].drop_duplicates(keep='first')
d = {}
for k in name_index.keys():
    df = pd.DataFrame(name_index[k].items(), columns=['name', 'idx']).set_index('idx')
    d[k] = df
name_index_ptm, name_index_protein, name_index_pathway = d['PTM'], d['Protein'], d['Pathway']
name_index_pathway['size'] = name_index_pathway['name'].map(pathway_size)
name_index_pathway['name'] = name_index_pathway['name'].map(pathway_name)

mapping = { 1: 'Canonical', -1: 'Candidate', 0: 'Unlabeled'}
with open(f"./knowledge/CG.pkl", 'rb') as f:
    cg = pickle.load(f)
canonical = cg['canonical']
candidate = cg['candidate']
label = pd.DataFrame.from_dict(name_index['Protein'], orient='index', columns=['idx'])
label = label.sort_values(by=['idx'], ascending=True)
label['label'] = 0
label.loc[label.index.isin(canonical), 'label'] = 1
label.loc[label.index.isin(candidate), 'label'] = -1
print(label['label'].value_counts())

pos_idx = label.loc[label['label'] == 1, 'idx'].values
_, test_pos_idx = train_test_split(pos_idx, test_size=0.2, random_state=42)
# Candidate
candidate_idx = label.loc[label['label'] == -1, 'idx'].values
# Unlabeled
unlabeled_idx = label.loc[label['label'] == 0, 'idx'].values
_, test_un_idx = train_test_split(unlabeled_idx, test_size=len(test_pos_idx)*3, random_state=42)
generator = torch.Generator(device='cuda')
test_x_idx = torch.tensor(np.hstack((test_pos_idx, test_un_idx)))
test_y = torch.cat([torch.ones(len(test_pos_idx)), torch.zeros(len(test_un_idx))]).long()
test_loader = DataLoader(TensorDataset(test_x_idx, test_y), batch_size=len(test_y), shuffle=True, generator=generator)
test_result = evaluate(graph, model, test_loader, 'Test')
print(test_result)


# Integrated Gradients
y_logits = model(graph.x_dict, graph.edge_index_dict, label['idx'].values)
label['probability'] = y_logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
target = label.loc[label['probability'] > 0.5]
target = target.sort_values(by=['probability'], ascending=False)
target['label'] = target['label'].map(mapping)
print(target['label'].value_counts())
target = target.iloc[:1000]
print('TOP1000\n', target['label'].value_counts())
# target = target.loc[target['label'] == 'Canonical']


if __name__ == '__main__':
    active_node_types = ['PTM', 'Protein']
    inputs = tuple(graph.x_dict[k].detach().unsqueeze(0).requires_grad_(True) for k in active_node_types)
    baselines = tuple(torch.zeros_like(inputs[i]) for i in range(len(inputs)))
    target_layer = model.encoder.norms[3]['Pathway']

    wrapper_model = CaptumHeteroWrapper(model, active_node_types, graph.x_dict, graph.edge_index_dict)
    wrapper_model.eval()
    lig = LayerIntegratedGradients(wrapper_model, target_layer)
    ig = IntegratedGradients(wrapper_model)

    all_ig_dict = {}
    target_names = target.index.values
    target_idxs = target['idx'].values
    for target_name, target_idx in zip(target_names, target_idxs):
        extra_args = (target_idx,)
        ig_dict = {}
        # interpret the embedding vectors obtained from Protein-to-Pathway convolution
        with torch.set_grad_enabled(True):
            layer_attributions = lig.attribute(inputs, baselines, target=1, additional_forward_args=extra_args,
                                               n_steps=50, internal_batch_size=1)
        ig_dict['Pathway'] = layer_attributions.detach().cpu().numpy()

        # interpret PTM and protein features
        with torch.set_grad_enabled(True):
            attributions = ig.attribute(inputs, baselines, target=1, additional_forward_args=extra_args, n_steps=50,
                                        internal_batch_size=1)
        for node_type, attr in zip(active_node_types, attributions):
            ig_dict[node_type] = attr[0].detach().cpu().numpy()

        all_ig_dict[target_name] = ig_dict
    with open('./result/IG.pkl', 'wb') as f:
        pickle.dump(all_ig_dict, f)

