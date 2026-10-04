import pickle
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
from GNN_AE_model import Encoder
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

torch.set_default_device(torch.device('cuda'))


class NodeClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, out_dim)
        self.dropout = nn.Dropout(0.2)
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.dropout(self.activation_fn(self.fc1(x)))
        x = self.dropout(self.activation_fn(self.fc2(x)))
        y_logit = self.fc3(x)
        return y_logit


class EncoderNodeClassification(nn.Module):
    def __init__(self, encoder, classifier, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()

    def forward(self, x_dict, edge_index_dict, target_idx):
        embedding = self.encoder(x_dict, edge_index_dict)['Protein'][target_idx]
        y_logits = self.classifier(embedding)
        return y_logits


class PolyLoss_1(nn.Module):
    def __init__(self, alpha1=2.0, weight=None, reduction='mean'):
        super(PolyLoss_1, self).__init__()
        self.alpha1 = alpha1
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Poly-1 : CE + alpha1 * (1 - pt)
        poly_loss = ce_loss + self.alpha1 * (1.0 - pt)

        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        return poly_loss


def evaluate(graph, model, criterion, loader, loader_type):
    model.eval()
    total_loss = 0.0
    y_true, y_predict, all_y_logits = [], [], []
    results = {}

    with torch.no_grad():
        for batch_x_idx, batch_y in loader:
            y_logits = model(graph.x_dict, graph.edge_index_dict, batch_x_idx)
            y_true.extend(batch_y.tolist())
            y_predict.extend(y_logits.argmax(dim=1).tolist())
            all_y_logits.append(y_logits.cpu())

            # torch.nn.CrossEntropyLoss() require data.y type is int64
            total_loss += criterion(y_logits, batch_y)

    y_true, y_predict = np.array(y_true), np.array(y_predict)
    all_y_logits = torch.cat(all_y_logits, dim=0)
    y_probs = F.softmax(all_y_logits, dim=1).numpy()
    subtypes = {0: 'Negative', 1: 'Positive'}

    cm = confusion_matrix(y_true, y_predict)
    cm = pd.DataFrame(data=cm,
                      index=[f"Actual-{s}" for s in subtypes.values()],
                      columns=[f"Predict-{s}" for s in subtypes.values()])
    # print(loader_type, cm, sep='\n')

    results['loss'] = round((total_loss.item() / len(loader)), 5)  # 仅batch_size=1时，等价于全局平均
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


if __name__ == '__main__':
    path = "./result"
    raw_graph = torch.load(f"{path}/GNN_graphs_data.pt", weights_only=False, map_location='cuda')
    for edge_type in raw_graph.edge_index_dict.keys():
        src, rel, dst = edge_type
        if (src  == 'Protein') and (dst == 'PTM'):
            del raw_graph[edge_type]

    with open(f"./knowledge/CG.pkl", 'rb') as f:
        cg = pickle.load(f)
    canonical = cg['canonical']
    candidate = cg['candidate']
    with open(f"{path}/GNN_node_index.pkl", 'rb') as f:
        name_index = pickle.load(f)
    label = pd.DataFrame.from_dict(name_index['Protein'], orient='index', columns=['idx'])
    label = label.sort_values(by=['idx'], ascending=True)
    label['label'] = 0
    label.loc[label.index.isin(canonical), 'label'] = 1
    label.loc[label.index.isin(candidate), 'label'] = -1
    print(label['label'].value_counts())

    configs = {
        'NPRatio': [3],
        'Encoder_type': ['Transformer'],
        'Pretrain': [True],
        'Freeze': [False],
        'Epochs': [5],
        'Repeats': list(range(1, 11))
    }


    all_parameters = {operator: torch.load(f"./result/GNN_best_{operator}_encoder.pt") for operator in configs['Encoder_type']}
    configs = pd.DataFrame(list(product(*configs.values())), columns=list(configs.keys()))

    batch_size = 64
    generator = torch.Generator(device='cuda')
    results = []
    best_auprc, best_hyper_params, best_model_params = 0, None, None
    Ablation = 'Ablation' in configs.columns
    for row in configs.itertuples(index=False, name='config'):
        graph = raw_graph.clone()
        if Ablation:
            if isinstance(row.Ablation, tuple):
                del graph[row.Ablation]
            else:
                for edge_type in row.Ablation:
                    del graph[edge_type]
            print(row.Ablation)

        operator = row.Encoder_type
        parameters = all_parameters[operator]
        hyper_params, model_params = parameters['hyper_parameters'], parameters['model_parameters']

        encoder = Encoder(graph.num_node_features, operator,
                          hyper_params['hidden_channels'], hyper_params['activation_function'])
        encoder.load_state_dict(model_params)
        encoder.eval()
        with torch.no_grad():
            embedding = encoder(graph.x_dict, graph.edge_index_dict)['Protein']
            embedding = embedding.cpu().numpy()[label['idx'].values]


        # PU Learning - Iterative Two-Step Approach
        # Positive
        pos_idx = label.loc[label['label'] == 1, 'idx'].values
        train_pos_idx, test_pos_idx = train_test_split(pos_idx, test_size=0.2, random_state=42)
        # Candidate
        candidate_idx = label.loc[label['label'] == -1, 'idx'].values
        # Unlabeled
        unlabeled_idx = label.loc[label['label'] == 0, 'idx'].values
        train_un_idx, test_un_idx = train_test_split(unlabeled_idx, test_size=len(test_pos_idx)*row.NPRatio, random_state=42)

        # Independent test set
        test_x_idx = torch.tensor(np.hstack((test_pos_idx, test_un_idx)))
        test_y = torch.cat([torch.ones(len(test_pos_idx)), torch.zeros(len(test_un_idx))]).long()
        test_loader = DataLoader(TensorDataset(test_x_idx, test_y), batch_size=len(test_y), shuffle=True, generator=generator)

        # Reliable Negative
        # train isolation forest based on the train-positive
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        iso_forest.fit(embedding[train_pos_idx])
        scores = iso_forest.score_samples(embedding[train_un_idx])
        rn_num = int(len(train_un_idx) * 0.15)
        rn_idx = train_un_idx[np.argsort(scores)[:rn_num]]
        remaining_un_idx = train_un_idx[np.argsort(scores)[rn_num:]]

        # Subsequent iterations will continue to draw from the remaining pool of unlabeled
        max_iteration = 5
        threshold = 0.1  # only predicted positive probability less than threshold, defined as new RN
        current_rn_idx, current_un_idx = np.array([], dtype=np.int64), remaining_un_idx.copy()
        for i in range(1, max_iteration+1):
            current_rn_idx = np.hstack((rn_idx, current_rn_idx))
            print(f"Iteration{i}: {len(train_pos_idx)}P {len(current_rn_idx)}RN {len(current_un_idx)}U", end='')

            train_x_idx = torch.tensor(np.hstack((train_pos_idx, current_rn_idx)))
            train_y = torch.cat([torch.ones(len(train_pos_idx)), torch.zeros(len(current_rn_idx))]).long()
            train_loader = DataLoader(TensorDataset(train_x_idx, train_y), batch_size=batch_size, shuffle=True, generator=generator)

            # The classifier is re-initialized at each iteration to eliminate biases from the previous model
            encoder = Encoder(graph.num_node_features, operator,
                              hyper_params['hidden_channels'], hyper_params['activation_function'])
            row.Pretrain and encoder.load_state_dict(model_params)
            classifier = NodeClassifier(hyper_params['hidden_channels'], 64, 16, 2)
            model = EncoderNodeClassification(encoder, classifier, row.Freeze)
            # print(id(encoder) == id(model.encoder))

            class_weights = [len(current_rn_idx), len(train_pos_idx)]
            class_weights = torch.tensor(np.sqrt(np.sum(class_weights) / class_weights), dtype=torch.float32)
            # criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            criterion = PolyLoss_1(alpha1=1.5, weight=class_weights, reduction='mean')
            epochs = row.Epochs
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for batch_x_idx, batch_y in train_loader:
                    # print(batch_x_idx, batch_y)
                    optimizer.zero_grad()
                    y_logits = model(graph.x_dict, graph.edge_index_dict, batch_x_idx)
                    loss = criterion(y_logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()

            model.eval()
            with torch.no_grad():
                y_logits = model(graph.x_dict, graph.edge_index_dict, torch.tensor(remaining_un_idx))
                y_prob = y_logits.softmax(dim=1).detach().cpu().numpy()
            rn_mask = y_prob[:, 1] <= threshold
            current_rn_idx = remaining_un_idx[rn_mask]
            current_un_idx = remaining_un_idx[~rn_mask]

        train_result = evaluate(graph, model, criterion, train_loader, 'Train')
        test_result = evaluate(graph, model, criterion, test_loader, 'Test')
        current_auprc = test_result.loc['AUPRC'].item()
        if current_auprc > best_auprc:
            hyper_params['operator'] = operator
            best_hyper_params = hyper_params
            best_model_params = model.state_dict()
            best_auprc = current_auprc
        results.append(train_result.iloc[1:, 0].tolist() + test_result.iloc[1:, 0].tolist())

    results = pd.DataFrame(
        results,
        columns=[f"{loader_type}-{metrics}" for loader_type in ['Train', 'Test'] for metrics in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AUPRC']]
    )
    results = pd.concat([configs, results], axis=1)
    results = results.sort_values(by=['Test-AUPRC'], ascending=False)
    print(results)
    results.to_excel(f"{path}/node_classification_{best_hyper_params['operator']}.xlsx", index=False)
    parameters = {'hyper_parameters': best_hyper_params, 'model_parameters': best_model_params}
    torch.save(parameters, f"{path}/GNN_best_{best_hyper_params['operator']}_classifier.pt")

