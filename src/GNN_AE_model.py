import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch_scatter import scatter
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GraphConv, SAGEConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import LayerNorm, GraphNorm
import optuna
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="There exist node types.*whose representations do not get updated during message passing.*")

torch.set_default_device(torch.device('cuda'))

class Encoder(nn.Module):
    def __init__(self, num_node_features, operator='Transformer', hidden_channels=256, activation_fn='relu', heads=4):
        super().__init__()

        # initial feature projection layer
        self.proj = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_channels)
            for node_type, in_dim in num_node_features.items()
        })

        conv_layer_configs = [
            [('PTM', 'association', 'PTM')],
            [('PTM', 'rev_regulation', 'Protein'), ('PTM', 'attr', 'Protein'), ('PTM', 'association', 'Protein')],
            [('Protein', 'interaction', 'Protein'), ('Protein', 'association', 'Protein')],
            [('Protein', 'attr', 'Pathway')],
            [('Pathway', 'association', 'Pathway')],
            [('Pathway', 'rev_attr', 'Protein')]
        ]
        if operator == 'SAGE':
            self.convs = nn.ModuleList([
                HeteroConv({
                    edge: SAGEConv(hidden_channels, hidden_channels, aggr='mean')
                    for edge in edges
                }, aggr='mean')
                for edges in conv_layer_configs
            ])
        elif operator == 'GAT':
            self.convs = nn.ModuleList([
                HeteroConv({
                    edge: GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, add_self_loops=False)
                    for edge in edges
                }, aggr='mean')
                for edges in conv_layer_configs
            ])
        elif operator == 'Transformer':
            self.convs = nn.ModuleList([
                HeteroConv({
                    edge: TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.2, beta=True)
                    for edge in edges
                }, aggr='mean')
                for edges in conv_layer_configs
            ])

        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: GraphNorm(hidden_channels)
                for node_type in num_node_features
            }) for _ in range(len(conv_layer_configs))
        ])

        if activation_fn == 'relu':
            self.activation = F.relu
        elif activation_fn == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation_fn == 'silu':
            self.activation = F.silu

    def update(self, h_in, h_out, layer_idx, batch_dict):
        h = {}
        for node_type, val in h_in.items():
            if node_type in h_out:
                batch = batch_dict and batch_dict[node_type]
                norm_layer = self.norms[layer_idx][node_type]
                # residual connection -> normalization -> activation
                res = h_out[node_type] + val
                h[node_type] = self.activation(norm_layer(res, batch))
            else:
                h[node_type] = val
        return h

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        # initial feature projection
        h = {}
        for k, x in x_dict.items():
            h[k] = self.activation(self.proj[k](x))

        for i, conv in enumerate(self.convs):
            h = self.update(h, conv(h, edge_index_dict), i, batch_dict)

        return h


class Decoder(nn.Module):
    def __init__(self, num_node_features, hidden_channels=256, activation_fn='relu'):
        super().__init__()

        conv_layer_configs = [
            [('Protein', 'attr', 'Pathway')],
            [('Pathway', 'association', 'Pathway')],
            [('Pathway', 'rev_attr', 'Protein')],
            [('Protein', 'interaction', 'Protein'), ('Protein', 'association', 'Protein')],
            [('Protein', 'regulation', 'PTM'), ('Protein', 'rev_attr', 'PTM'), ('Protein', 'rev_association', 'PTM')],
            [('PTM', 'association', 'PTM')]
        ]
        self.convs = nn.ModuleList([
            HeteroConv({
                edge: SAGEConv(hidden_channels, hidden_channels)
                for edge in edges
            }, aggr='mean')
            for edges in conv_layer_configs
        ])

        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: GraphNorm(hidden_channels)
                for node_type in num_node_features
            }) for _ in range(len(conv_layer_configs))
        ])

        # reconstruct output layer (only PTM and Protein, excluding the Pathway)
        self.recon_heads = nn.ModuleDict({
            'PTM': nn.Linear(hidden_channels, num_node_features['PTM']),
            'Protein': nn.Linear(hidden_channels, num_node_features['Protein'])
        })

        if activation_fn == 'relu':
            self.activation = F.relu
        elif activation_fn == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation_fn == 'silu':
            self.activation = F.silu

    def update(self, h_in, h_out, layer_idx, batch_dict):
        h = {}
        for node_type, val in h_in.items():
            if node_type in h_out:
                batch = batch_dict and batch_dict[node_type]
                norm_layer = self.norms[layer_idx][node_type]
                # residual connection + normalization + activation
                res = h_out[node_type] + val
                h[node_type] = self.activation(norm_layer(res, batch))
            else:
                h[node_type] = val
        return h

    def forward(self, h_dict, edge_index_dict, batch_dict=None):
        for i, conv in enumerate(self.convs):
            h_dict = self.update(h_dict, conv(h_dict, edge_index_dict), i, batch_dict)

        # feature reconstruct
        x_recon_dict = {
            node_type: self.recon_heads[node_type](h_dict[node_type])
            for node_type in self.recon_heads
        }

        return x_recon_dict


class HeteroAE(nn.Module):
    def __init__(self, num_node_features, encoder_type, hidden_channels=256, activation_fn='relu', mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = Encoder(num_node_features, encoder_type, hidden_channels, activation_fn)
        self.decoder = Decoder(num_node_features, hidden_channels, activation_fn)

    def mask_node(self, x_dict):
        x_masked_dict = {}
        mask_index_dict = {}

        for node_type, x in x_dict.items():
            x_masked = x.clone()

            if node_type == 'Pathway':
                x_masked_dict[node_type] = x_masked
                continue

            feature_dim = x.shape[1] - 8 if node_type == 'PTM' else x.shape[1]

            num_nodes = len(x)
            num_mask_nodes = int(self.mask_ratio * num_nodes)
            perm = torch.randperm(num_nodes)
            mask_nodes = perm[:num_mask_nodes]
            keep_nodes = perm[num_mask_nodes:]

            # mask node：80% zero + 10% noise + 10% keep
            num_zero = int(0.8 * num_mask_nodes)
            num_noise = int(0.1 * num_mask_nodes)
            mask_perm = torch.randperm(num_mask_nodes)
            zero_nodes = mask_nodes[mask_perm[:num_zero]]
            x_masked[zero_nodes, :feature_dim] = 0.0
            noise_nodes = mask_nodes[mask_perm[num_zero: num_zero + num_noise]]
            noise_to_be_chosen = torch.randperm(num_nodes)[:num_noise]
            x_masked[noise_nodes, :feature_dim] = x[noise_to_be_chosen, :feature_dim]

            x_masked_dict[node_type] = x_masked
            mask_index_dict[node_type] = mask_nodes

        return x_masked_dict, mask_index_dict

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        # random mask node features
        x_masked_dict, mask_index_dict = self.mask_node(x_dict)

        # Encoder：Raw feature -> Latent representation
        z_dict = self.encoder(x_masked_dict, edge_index_dict, batch_dict)

        # Decoder：Latent representation -> Raw feature
        x_recon_dict = self.decoder(z_dict, edge_index_dict, batch_dict)

        return z_dict, x_recon_dict, mask_index_dict


def scaled_cosine_error(x_dict, x_recon_dict, valid_dict, mask_index_dict, alpha=3):
    d = {}
    for k in x_recon_dict:
        feature_dim = x_dict[k].shape[1] - 8 if k == 'PTM' else x_dict[k].shape[1]
        mask_nodes = mask_index_dict[k]
        x = F.normalize(x_recon_dict[k][mask_nodes, :feature_dim], p=2, dim=-1)
        y = F.normalize(x_dict[k][mask_nodes, :feature_dim], p=2, dim=-1)
        valid = valid_dict[k][mask_nodes, :feature_dim]

        cos_sim = (x * y * valid.float()).sum(dim=-1)
        sce = (1 - cos_sim).pow(alpha).mean()
        d[k] = sce
    ptm_loss, protein_loss = d['PTM'], d['Protein']

    return ptm_loss, protein_loss


def functional_scaled_consistency_error(z_dict, edge_index_dict, alpha=3):
    edge_index = edge_index_dict[('Protein', 'attr', 'Pathway')]
    src_protein, dst_pathway = edge_index[0], edge_index[1]
    z_protein = z_dict['Protein']
    z_pathway = z_dict['Pathway']
    num_pathways = z_pathway.size(0)
    aggregated_protein = scatter(
        src=z_protein[src_protein],
        index=dst_pathway,
        dim=0,
        dim_size=num_pathways,
        reduce='mean'
    )
    active_mask = aggregated_protein.abs().sum(dim=-1) > 0

    x = F.normalize(z_pathway, p=2, dim=-1)
    y = F.normalize(aggregated_protein, p=2, dim=-1)
    cos_sim = (x * y).sum(dim=-1)
    fsce = ((1 - cos_sim).pow(alpha) * active_mask.float()).sum() / active_mask.sum()

    return fsce


def train_epoch(model, loader, criterion_dict, optimizer, beta=0.02):
    model.train()
    loss = torch.zeros(4)

    # data = next(iter(loader))
    for data in loader:
        optimizer.zero_grad()
        z_dict, x_recon_dict, mask_index_dict = model(data.x_dict, data.edge_index_dict, data.batch_dict)

        ptm_loss, protein_loss = criterion_dict['sce'](data.x_dict, x_recon_dict, data.valid_dict, mask_index_dict)
        pathway_loss = criterion_dict['fsce'](z_dict, data.edge_index_dict)
        total_loss = ptm_loss + protein_loss + beta * pathway_loss
        # total_loss = ptm_loss + protein_loss
        total_loss.backward()
        optimizer.step()

        loss += torch.stack([ptm_loss, protein_loss, pathway_loss, total_loss]).detach()
    loss /= len(loader)

    return loss.cpu().numpy()


def objective(trial, criterion_dict, num_node_features, encoder_type, epochs):
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3, 1e-2])
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu', 'silu'])
    mask_ratio = trial.suggest_float('mask_ratio', 0.3, 0.5, step=0.05)

    model = HeteroAE(num_node_features, encoder_type, hidden_channels, activation_function, mask_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loss_history = []
    try:
        for epoch in range(epochs):
            epoch_average_loss = train_epoch(model, loader, criterion_dict, optimizer)
            loss_history.append(epoch_average_loss)
            scheduler.step()

            final_loss = epoch_average_loss[-1]
            trial.report(final_loss, epoch)
            if trial.should_prune():
                trial.set_user_attr('loss_history', loss_history)
                raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("loss_history", loss_history)

        return final_loss
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Encoder Operator
    encoder_types = ['SAGE', 'GAT', 'Transformer']
    encoder_type = encoder_types[-1]

    path = "./result"
    # load graph dataset
    graph = torch.load(f"{path}/GNN_graphs_data.pt", weights_only=False, map_location='cuda')
    # for node_type, d in graph.node_items():
    #     print(node_type)
    #     for k, v in d.items():
    #         print(f"{k}: {v.shape}  {v.is_contiguous()}")
    # for edge_type, d in graph.edge_items():
    #     print(edge_type)
    #     for k, v in d.items():
    #         print(f"{k}: {v.shape}  {v.is_contiguous()}")
    num_node_features = graph.num_node_features

    dataset = InMemoryDataset(root=None, transform=None, pre_transform=None)
    dataset.data, dataset.slices = InMemoryDataset.collate([graph])
    generator = torch.Generator(device='cuda')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)
    # data = next(iter(loader))

    # Optuna hyperparameter optimization
    epochs = 200
    criterion_dict = {'sce': scaled_cosine_error, 'fsce': functional_scaled_consistency_error}
    study = optuna.create_study(
        study_name=f"Pretrain {encoder_type} AE",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=50
        ),
        direction='minimize'
    )
    # study.optimize(objective, n_trials=10)
    study.optimize(
        lambda t: objective(t, criterion_dict, num_node_features, encoder_type, epochs),
        n_trials=15
    )

    record = study.trials_dataframe()
    best_hyper_params = study.best_params
    best_loss_history = np.array(study.best_trial.user_attrs["loss_history"])
    loss_history = {t.number: np.array(t.user_attrs.get("loss_history"))[:, -1] for t in study.trials}

    plt.style.use('default')
    plt.rc('font', weight='bold')
    plt.rc('axes', labelweight='bold', titleweight='bold')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    axs = axs.flatten()
    ax = axs[0]
    for k, v in loss_history.items():
        ax.plot(np.arange(len(v)), v, label=f"Trial {k}")
    ax.set_title("Total Loss History Of 50 Trials", fontsize=20)
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("Scaled Cosine Error", fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    ax = axs[1]
    x = np.arange(len(best_loss_history))
    for col, label in enumerate(['PTM loss', 'Protein loss', 'Pathway loss', 'Total loss']):
        ax.plot(x, best_loss_history[:, col], label=label)
    ax.legend(fontsize=15)
    ax.set_title("Loss History Of Best Trials", fontsize=20)
    # ax.set_title("Loss History (Joint Optimization)", fontsize=20)
    # ax.set_title("Loss History (Pathway Loss Detached)", fontsize=20)
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("Scaled Cosine Error", fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    plt.savefig(f"{path}/plot/Pretrain {encoder_type} AE.png", dpi=300)
    plt.show()


    # best model train
    model = HeteroAE(
        num_node_features, encoder_type,
        best_hyper_params['hidden_channels'], best_hyper_params['activation_function'], best_hyper_params['mask_ratio']
    )
    # print(model)
    # print(model.state_dict())
    # for name, param in model.named_parameters():
    #     print(f"{name} : {param.shape}")
    # for i, conv in enumerate(model.encoder.convs):
    #     for edge_type, module in conv.convs.items():
    #         print(i, edge_type, module)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hyper_params['learning_rate'], weight_decay=best_hyper_params['weight_decay'])
    # Cosine annealing decay (CosineAnnealingLR)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loss_history = []
    for epoch in range(epochs):
        epoch_average_loss = train_epoch(model, loader, criterion_dict, optimizer)
        loss_history.append(epoch_average_loss)
        scheduler.step()
    encoder = model.encoder

    parameters = {'hyper_parameters': best_hyper_params, 'model_parameters': copy.deepcopy(encoder.state_dict())}
    torch.save(parameters, f"{path}/GNN_best_{encoder_type}_encoder.pt")

