import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from collections import defaultdict
from pathlib import Path


# ============ Preprocessing: Extract all relation types ============
def extract_relation_types_and_validate(base_dir, domains):
    """Scan all data files, extract relation types, and validate data integrity"""
    relation_types = set()
    file_info = []
    missing_labels = []

    print("\n" + "=" * 70)
    print("Scanning data files...")
    print("=" * 70)

    for domain_folder in domains:
        domain_path = Path(base_dir) / domain_folder
        data_dir = domain_path / 'data'
        label_dir = domain_path / 'label'

        if not data_dir.exists():
            print(f"Warning: {domain_folder} data directory does not exist")
            continue

        domain_name = domain_folder
        data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_data.json')])

        print(f"\n{domain_folder}: {len(data_files)} files")

        for data_file in data_files:
            example_name = data_file.replace('_data.json', '')
            label_file = f"{example_name}_labels.json"

            data_path = data_dir / data_file
            label_path = label_dir / label_file

            has_label = label_path.exists()
            if not has_label:
                missing_labels.append(f"{domain_name}/{data_file}")

            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for rel in data.get('relation_elements', []):
                    rel_type = rel.get('relation', '').strip()
                    if rel_type:
                        relation_types.add(rel_type)

                file_info.append({
                    'domain': domain_name,
                    'file': data_file,
                    'has_label': has_label,
                    'num_nodes': len(data.get('matter_elements', [])) + len(data.get('action_elements', [])),
                    'num_edges': len(data.get('relation_elements', []))
                })

            except Exception as e:
                print(f"  Error: {data_file} - {e}")

    relation_types = sorted(list(relation_types))
    relation_mapping = {rt: i for i, rt in enumerate(relation_types)}
    relation_mapping['<UNK>'] = len(relation_mapping)
    print(f"\nFound {len(relation_types)} known relation types + 1 unknown type:")
    for i, rt in enumerate(relation_types):
        print(f"  {i}: {rt}")

    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} files are missing labels:")
        for f in missing_labels[:10]:
            print(f"  - {f}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    print(f"\nAvailable for training: {sum(1 for f in file_info if f['has_label'])}/{len(file_info)}")

    return relation_mapping, file_info


# ============ Multi-Domain Data Loader ============
class MultiDomainConflictDataset(Dataset):
    def __init__(self, base_dir, relation_type_mapping, domains=None):
        super().__init__()
        self.base_dir = Path(base_dir)

        if domains is None:
            self.domains = ['ecological', 'financial', 'medical', 'production']
        else:
            self.domains = domains

        self.domain_mapping = {
            'ecological': 0,
            'financial': 1,
            'medical': 2,
            'production': 3
        }

        self.relation_type_mapping = relation_type_mapping
        self.num_relation_types = len(relation_type_mapping)

        self.graphs = []
        self.failed_files = []
        self._load_all_data()

    def _load_all_data(self):
        """Load data from all domains"""
        print("\n" + "=" * 70)
        print("Loading dataset...")
        print("=" * 70)

        for domain_folder in self.domains:
            domain_path = self.base_dir / domain_folder
            data_dir = domain_path / 'data'
            label_dir = domain_path / 'label'

            if not data_dir.exists() or not label_dir.exists():
                continue

            domain_id = self.domain_mapping[domain_folder]
            domain_name = domain_folder

            data_files = sorted([f for f in os.listdir(data_dir)
                                 if f.endswith('_data.json')])

            loaded_count = 0
            for data_file in data_files:
                example_name = data_file.replace('_data.json', '')
                label_file = f"{example_name}_labels.json"

                data_path = data_dir / data_file
                label_path = label_dir / label_file

                if not label_path.exists():
                    self.failed_files.append({
                        'domain': domain_folder,
                        'file': data_file,
                        'reason': 'Label missing'
                    })
                    continue

                graph = self._process_single_example(
                    data_path, label_path, domain_id, domain_name
                )
                if graph is not None:
                    self.graphs.append(graph)
                    loaded_count += 1
                else:
                    self.failed_files.append({
                        'domain': domain_folder,
                        'file': data_file,
                        'reason': 'Processing failed'
                    })

            print(f"{domain_name}: {loaded_count}/{len(data_files)}")

    def _process_single_example(self, data_path, label_path, domain_id, domain_name):
        """Process a single sample"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        except Exception as e:
            return None

        node_ids = []
        node_types = []
        node_id_to_idx = {}
        node_names = []

        for m in data.get('matter_elements', []):
            idx = len(node_ids)
            node_ids.append(m['id'])
            node_types.append(0)
            node_id_to_idx[m['id']] = idx
            node_names.append(m.get('name', f"matter_{idx}"))

        for a in data.get('action_elements', []):
            idx = len(node_ids)
            node_ids.append(a['id'])
            node_types.append(1)
            node_id_to_idx[a['id']] = idx
            node_names.append(a.get('action', f"action_{idx}"))

        num_nodes = len(node_ids)
        if num_nodes == 0:
            return None

        # Node features: [one-hot type(2), domain encoding(4)]
        x = torch.zeros((num_nodes, 6))
        for i, node_type in enumerate(node_types):
            x[i, node_type] = 1
            x[i, 2 + domain_id] = 1

        node_importance = torch.zeros(num_nodes)
        node_relevance = torch.zeros(num_nodes)

        for node_id, idx in node_id_to_idx.items():
            if node_id in labels.get('node_scores', {}):
                node_importance[idx] = labels['node_scores'][node_id].get('importance', 0)
                node_relevance[idx] = labels['node_scores'][node_id].get('problem_relevance', 0)

        edge_index = []
        edge_attr = []
        edge_is_problem = []
        edge_severity = []
        edge_is_conflict = []

        for rel in data.get('relation_elements', []):
            src_idx = node_id_to_idx.get(rel['source'])
            tgt_idx = node_id_to_idx.get(rel['target'])

            if src_idx is not None and tgt_idx is not None:
                edge_index.append([src_idx, tgt_idx])

                rel_type = rel.get('relation', '').strip()
                edge_type_vec = [0] * (self.num_relation_types + 4)

                if rel_type in self.relation_type_mapping:
                    edge_type_vec[self.relation_type_mapping[rel_type]] = 1

                edge_type_vec[self.num_relation_types + domain_id] = 1
                edge_attr.append(edge_type_vec)

                rel_id = rel['id']
                if rel_id in labels.get('edge_scores', {}):
                    edge_is_problem.append(labels['edge_scores'][rel_id].get('is_problem', 0))
                    edge_severity.append(labels['edge_scores'][rel_id].get('severity', 0.0))
                    edge_is_conflict.append(labels['edge_scores'][rel_id].get('is_conflict', 0))
                else:
                    edge_is_problem.append(0)
                    edge_severity.append(0.0)
                    edge_is_conflict.append(0)

        if len(edge_index) == 0:
            return None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_is_problem = torch.tensor(edge_is_problem, dtype=torch.float)
        edge_severity = torch.tensor(edge_severity, dtype=torch.float)
        edge_is_conflict = torch.tensor(edge_is_conflict, dtype=torch.float)

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_importance=node_importance,
            node_relevance=node_relevance,
            edge_is_problem=edge_is_problem,
            edge_severity=edge_severity,
            edge_is_conflict=edge_is_conflict,
            num_nodes=num_nodes,
            domain_id=torch.tensor([domain_id], dtype=torch.long),
            domain_name=domain_name,
            node_ids=node_ids,
            node_names=node_names
        )

        return graph

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# ============ GNN Model ============
class HeterogeneousMultiTaskGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, num_domains=4):
        super().__init__()

        self.num_domains = num_domains
        self.domain_embedding = nn.Embedding(num_domains, hidden_channels // 4)
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=4,
                             edge_dim=edge_dim, concat=True, dropout=0.2)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4,
                             edge_dim=edge_dim, concat=True, dropout=0.2)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=2,
                             edge_dim=edge_dim, concat=True, dropout=0.2)
        self.conv4 = GATConv(hidden_channels * 2, hidden_channels, heads=1,
                             edge_dim=edge_dim, concat=False)

        self.shared_node_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.node_importance_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        self.node_relevance_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        edge_input_dim = hidden_channels * 2 + edge_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.edge_conflict_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        self.edge_severity_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        self.edge_problem_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        domain_id = data.domain_id

        x = self.input_proj(x)

        domain_emb = self.domain_embedding(domain_id)  # (batch_size, hidden//4)
        # Flatten to 2D if needed, then take mean across batch dim -> (1, hidden//4)
        if domain_emb.dim() == 3:
            domain_emb = domain_emb.squeeze(1)
        # Average across graphs in batch to get single domain embedding, broadcast to all nodes
        domain_emb = domain_emb.mean(dim=0, keepdim=True)  # (1, hidden//4)
        domain_emb = domain_emb.expand(x.size(0), -1)       # (num_nodes, hidden//4)

        # Pad to match x dimension
        domain_emb_expanded = torch.zeros_like(x)
        domain_emb_expanded[:, :domain_emb.size(1)] = domain_emb
        x = x + domain_emb_expanded

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)

        node_features = self.shared_node_encoder(x)
        node_importance = self.node_importance_head(node_features).squeeze(-1)
        node_relevance = self.node_relevance_head(node_features).squeeze(-1)

        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_features = self.edge_encoder(edge_features)

        edge_is_conflict = self.edge_conflict_head(edge_features).squeeze(-1)
        edge_severity = self.edge_severity_head(edge_features).squeeze(-1)
        edge_is_problem = self.edge_problem_head(edge_features).squeeze(-1)

        return {
            'node_importance': node_importance,
            'node_relevance': node_relevance,
            'edge_is_conflict': edge_is_conflict,
            'edge_severity': edge_severity,
            'edge_is_problem': edge_is_problem
        }


# ============ Training and Evaluation ============
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    eps = 1e-6  # Table 1: Numerical stability constant epsilon

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)

        # 1. Node-level Loss (MSE)
        loss_importance = F.mse_loss(outputs['node_importance'], data.node_importance)
        loss_relevance = F.mse_loss(outputs['node_relevance'], data.node_relevance)

        # 2. Edge-level Loss with Numerical Stability Protection
        # Clamp ensures probabilities stay within [eps, 1-eps] to prevent NaN
        loss_conflict = F.binary_cross_entropy(
            outputs['edge_is_conflict'].clamp(eps, 1-eps),
            data.edge_is_conflict
        )
        loss_severity = F.mse_loss(outputs['edge_severity'], data.edge_severity)
        loss_problem = F.binary_cross_entropy(
            outputs['edge_is_problem'].clamp(eps, 1-eps),
            data.edge_is_problem
        )

        # 3. Total Weighted Loss (Equation 15)
        # lambda_c = 2.0 (conflict), others = 1.0
        loss = (loss_importance + loss_relevance +
                2.0 * loss_conflict +
                loss_severity + loss_problem)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    metrics = defaultdict(list)
    domain_metrics = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)

            importance_mae = F.l1_loss(outputs['node_importance'], data.node_importance).item()
            relevance_mae = F.l1_loss(outputs['node_relevance'], data.node_relevance).item()
            severity_mae = F.l1_loss(outputs['edge_severity'], data.edge_severity).item()

            val_loss = importance_mae + relevance_mae + severity_mae

            metrics['importance_mae'].append(importance_mae)
            metrics['relevance_mae'].append(relevance_mae)
            metrics['severity_mae'].append(severity_mae)
            metrics['val_loss'].append(val_loss)

            if isinstance(data.domain_name, list):
                for domain in set(data.domain_name):
                    domain_metrics[domain]['importance_mae'].append(importance_mae)
            else:
                domain_metrics[data.domain_name]['importance_mae'].append(importance_mae)

    overall_metrics = {k: np.mean(v) for k, v in metrics.items()}
    domain_avg_metrics = {
        domain: {k: np.mean(v) for k, v in m.items()}
        for domain, m in domain_metrics.items()
    }

    return overall_metrics, domain_avg_metrics


# ============ 5-Fold Cross-Validation ============
def run_cross_validation(dataset, relation_mapping, edge_dim, device,
                         n_splits=5, epochs=500, batch_size=16,
                         hidden_dim=128, lr=0.0005, random_seed=42):
    """Run n-fold cross-validation and return summary metrics."""

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    indices = list(range(len(dataset)))
    fold_results = []

    print("\n" + "=" * 70)
    print(f"Running {n_splits}-Fold Cross-Validation...")
    print("=" * 70)

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Further split train_val into train / val (80/20 within fold)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2,
            random_state=random_seed + fold, shuffle=True
        )

        train_dataset = [dataset[i] for i in train_idx]
        val_dataset   = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        model = HeterogeneousMultiTaskGNN(
            in_channels=6,
            hidden_channels=hidden_dim,
            edge_dim=edge_dim,
            num_domains=4
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )

        best_val_loss = float('inf')
        best_metrics  = None
        patience_counter = 0
        max_patience = 30

        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, optimizer, device)
            val_metrics, _ = evaluate(model, val_loader, device)

            val_loss = val_metrics['val_loss']
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics  = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        fold_results.append(best_metrics)
        print(f"  Best Val Loss:    {best_val_loss:.4f}")
        print(f"  Importance MAE:   {best_metrics['importance_mae']:.4f}")
        print(f"  Severity MAE:     {best_metrics['severity_mae']:.4f}")

    # ---- Aggregate results ----
    print("\n" + "=" * 70)
    print(f"{n_splits}-Fold Cross-Validation Results Summary:")
    print("=" * 70)

    metric_keys = ['importance_mae', 'severity_mae', 'val_loss', 'relevance_mae']

    cv_summary = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results]
        mean_val = np.mean(values)
        std_val  = np.std(values)
        cv_summary[key] = {'mean': mean_val, 'std': std_val, 'values': values}
        print(f"  {key:25s}: {mean_val:.4f} ± {std_val:.4f}")

    # Save CV results to JSON
    cv_output = {
        'n_splits': n_splits,
        'node_feature_dim': 6,
        'summary': {k: {'mean': v['mean'], 'std': v['std']}
                    for k, v in cv_summary.items()},
        'per_fold': [{k: r[k] for k in metric_keys} for r in fold_results]
    }
    with open('cv_results_structural.json', 'w') as f:
        json.dump(cv_output, f, indent=2)
    print("\nCV results saved to: cv_results_structural.json")

    return cv_summary


# ============ Main Function ============
def main():
    BASE_DIR = r"  " # change to your path
    BATCH_SIZE = 16
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.0005
    EPOCHS = 500
    RANDOM_SEED = 42
    RUN_CV = True  # set False to skip cross-validation

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    domains = ['ecological', 'financial', 'medical', 'production']

    relation_mapping, file_info = extract_relation_types_and_validate(BASE_DIR, domains)

    dataset = MultiDomainConflictDataset(BASE_DIR, relation_mapping)
    print(f"\nTotal loaded: {len(dataset)} graphs")

    domain_counts = defaultdict(int)
    for graph in dataset.graphs:
        domain_counts[graph.domain_name] += 1
    print("\nDomain distribution:")
    for domain in ['ecological', 'financial', 'medical', 'production']:
        print(f"  {domain}: {domain_counts.get(domain, 0)}")

    edge_dim = len(relation_mapping) + 4

    # ---- Step 1: 5-Fold Cross-Validation ----
    if RUN_CV:
        cv_summary = run_cross_validation(
            dataset, relation_mapping, edge_dim, device,
            n_splits=5, epochs=EPOCHS, batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM, lr=LEARNING_RATE, random_seed=RANDOM_SEED
        )

    # ---- Step 2: Train final model on 60/20/20 split ----
    print("\n" + "=" * 70)
    print("Training final model (60% train / 20% val / 20% test)...")
    print("=" * 70)

    indices = list(range(len(dataset)))
    stratify_labels = [g.domain_id.item() for g in dataset.graphs]

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_SEED,
        stratify=stratify_labels, shuffle=True
    )
    train_val_stratify = [stratify_labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=RANDOM_SEED,
        stratify=train_val_stratify, shuffle=True
    )

    train_dataset = [dataset[i] for i in train_idx]
    val_dataset   = [dataset[i] for i in val_idx]
    test_dataset  = [dataset[i] for i in test_idx]

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model = HeterogeneousMultiTaskGNN(
        in_channels=6,
        hidden_channels=HIDDEN_DIM,
        edge_dim=edge_dim,
        num_domains=4
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Edge feature dim: {edge_dim} | Node feature dim: 6 (structural only)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 30

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics, val_domain_metrics = evaluate(model, val_loader, device)

        val_loss = val_metrics['val_loss']
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Importance MAE: {val_metrics['importance_mae']:.4f} | "
                  f"Severity MAE: {val_metrics['severity_mae']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'relation_type_mapping': relation_mapping,
                'edge_dim': edge_dim,
            }, 'best_model_structural.pt')
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ---- Test set evaluation ----
    print("\n" + "=" * 70)
    print("Test set evaluation...")
    print("=" * 70)

    checkpoint = torch.load('best_model_structural.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_domain_metrics = evaluate(model, test_loader, device)

    print("\nTest Set Results:")
    print(f"  Importance MAE: {test_metrics['importance_mae']:.4f}")
    print(f"  Relevance MAE:  {test_metrics['relevance_mae']:.4f}")
    print(f"  Severity MAE:   {test_metrics['severity_mae']:.4f}")
    print(f"  Val Loss:       {test_metrics['val_loss']:.4f}")

    print("\nTest Results by Domain:")
    for domain in sorted(test_domain_metrics.keys()):
        print(f"  {domain}: Importance MAE="
              f"{test_domain_metrics[domain].get('importance_mae', 0):.4f}")

    print("\n" + "=" * 70)
    print("Training finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Structural model saved to: best_model_structural.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()