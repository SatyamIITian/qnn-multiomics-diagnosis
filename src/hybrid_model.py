# src/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score

class QuantumInspiredLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.rotation = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        x = self.linear(x)
        x = x * torch.cos(self.rotation) + x * torch.sin(self.rotation)
        return torch.relu(x)

class QGAHybridModel(nn.Module):
    def __init__(self, input_dims=[50, 50, 50], hidden_dim=64, num_heads=4, dropout_rate=0.3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim

        # Quantum-inspired layers for each omics type
        self.omics_layers = nn.ModuleList([
            nn.Sequential(
                QuantumInspiredLayer(dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ) for dim in input_dims
        ])

        # Graph Attention Network
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout_rate)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.to(self.device)

    def construct_graph(self, omics_data, batch_size):
        # Concatenate omics data
        features = torch.cat([torch.tensor(data, dtype=torch.float32).to(self.device) for data in omics_data], dim=1)
        
        # Compute correlation matrix for edges
        corr_matrix = torch.corrcoef(features.T).fill_diagonal_(0)
        edge_index = torch.nonzero(corr_matrix > 0.3, as_tuple=False).T  # Threshold for edges
        edge_weight = corr_matrix[edge_index[0], edge_index[1]]

        # Process each omics type
        node_features = []
        for i, data in enumerate(omics_data):
            x = torch.tensor(data, dtype=torch.float32).to(self.device)
            x = self.omics_layers[i](x)
            node_features.append(x)
        
        node_features = torch.cat(node_features, dim=1)  # (batch_size, num_nodes * hidden_dim)
        
        # Create graph data
        graphs = []
        for i in range(batch_size):
            graph = Data(
                x=node_features[i].view(sum([50, 50, 50]), self.hidden_dim),
                edge_index=edge_index,
                edge_attr=edge_weight
            )
            graphs.append(graph)
        
        batch = Batch.from_data_list(graphs)
        return batch.to(self.device)

    def forward(self, omics_data, batch_size):
        # Construct graph
        batch = self.construct_graph(omics_data, batch_size)

        # GNN layers
        x = self.gat1(batch.x, batch.edge_index, batch.edge_attr)
        x = F.elu(x)
        x = self.gat2(x, batch.edge_index, batch.edge_attr)

        # Global pooling
        x = global_mean_pool(x, batch.batch)

        # Classification
        out = self.classifier(x)
        return out, x  # Return prediction and representation for contrastive loss

    def contrastive_loss(self, representations, labels, temperature=0.5):
        representations = F.normalize(representations, dim=1)
        similarity = torch.mm(representations, representations.T) / temperature
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(self.device)
        neg_mask = 1 - pos_mask
        exp_sim = torch.exp(similarity)
        pos_sim = exp_sim * pos_mask
        neg_sim = exp_sim * neg_mask
        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        return loss.mean()

    def fit(self, omics_data_train, y_train, trial=None, epochs=100, batch_size=32, lr=0.001, alpha=0.1):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(y_train))
            total_loss = 0
            for i in range(0, len(y_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_omics = [data[batch_idx] for data in omics_data_train]
                batch_y = y_train_tensor[batch_idx]
                batch_size_actual = len(batch_idx)

                optimizer.zero_grad()
                outputs, representations = self.forward(batch_omics, batch_size_actual)
                
                cls_loss = criterion(outputs, batch_y)
                cont_loss = self.contrastive_loss(representations, batch_y.view(-1))
                loss = cls_loss + alpha * cont_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(y_train) // batch_size + 1)
            if trial and epoch % 10 == 0:
                trial.report(avg_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    def predict(self, omics_data):
        self.eval()
        batch_size = omics_data[0].shape[0]
        with torch.no_grad():
            outputs, _ = self.forward(omics_data, batch_size)
            return (outputs.cpu().numpy() > 0.5).astype(int).flatten()

    def predict_proba(self, omics_data):
        self.eval()
        batch_size = omics_data[0].shape[0]
        with torch.no_grad():
            outputs, _ = self.forward(omics_data, batch_size)
            return outputs.cpu().numpy().flatten()

# Hyperparameter tuning with Optuna
def objective(trial, omics_data_train, y_train, omics_data_test, y_test):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_heads = trial.suggest_int("num_heads", 2, 8)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 50, 200)
    alpha = trial.suggest_float("alpha", 0.01, 1.0)

    model = QGAHybridModel(input_dims=[50, 50, 50], hidden_dim=hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate)
    model.fit(omics_data_train, y_train, trial=trial, epochs=epochs, batch_size=batch_size, lr=lr, alpha=alpha)
    
    y_pred_proba = model.predict_proba(omics_data_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc

def tune_hybrid_model(omics_data_train, y_train, omics_data_test, y_test):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, omics_data_train, y_train, omics_data_test, y_test), n_trials=30)
    return study.best_params