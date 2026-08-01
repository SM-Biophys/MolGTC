import os
import sys
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
from torch_geometric.loader import DataLoader
from sklearn.metrics import (  
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, roc_curve, auc, confusion_matrix
)
from torch.utils.data import random_split, Subset
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

try:
    from Tool.Dataset import PyGMolecularDataset
except ImportError:
    sys.path.append(os.path.join(ROOT_DIR, 'Tool'))
    from Dataset import PyGMolecularDataset

try:
    from module.Downstream import GTCMolClassifier
except ImportError as e:
    print(f" Module import failed: {e}")
    exit(1)



def load_smiles_and_labels(data_path):
    """Read CSV, clean data, RDKit validation."""
    print(f" Loading raw CSV: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f" Error: file not found {data_path}")

    try:
        df = pd.read_csv(data_path)
        cols = [c.lower() for c in df.columns]
        if 'smiles' in cols:
            real_col = df.columns[cols.index('smiles')]
            df = df.rename(columns={real_col: 'Smiles'})

        if 'Smiles' not in df.columns or 'activity' not in df.columns:
            if 'label' in df.columns:
                df['activity'] = df['label']
            else:
                raise ValueError("CSV must contain 'Smiles'  and 'activity' (or 'label') columns")

        df = df[['Smiles', 'activity']]
    except Exception as e:
        print(f" CSV load failed: {e}")
        return [], []

    df_clean = df.dropna(subset=['Smiles', 'activity'])
    print(f"Original rows: {len(df)} -> after drop nulls: {len(df_clean)}")

    valid_smiles = []
    valid_labels = []

    print(" Running RDKit validation...")
    for smi, lbl in zip(df_clean['Smiles'], df_clean['activity']):
        if not isinstance(smi, str): continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_labels.append(lbl)

    print(f" Valid molecules: {len(valid_smiles)}")
    return valid_smiles, valid_labels


def scaffold_split(dataset, smiles_list, frac_train, frac_val, frac_test, seed):
    """scaffold split"""
    print(f" Performing scaffold split... (Seed={seed})")

    if not smiles_list or len(smiles_list) != len(dataset):
        print(" Warning: scaffold splitneeds original SMILES columns.")
        print("   Using cache, re-reading SMILES from CSV for split...")
        return None, None, None

    np.random.seed(seed)
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(tqdm(smiles_list, desc="Generating Scaffolds")):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[scaffold].append(ind)

    scaffold_sets = [scaffolds[k] for k in sorted(scaffolds, key=lambda x: (len(scaffolds[x]), x), reverse=True)]

    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = int(frac_train * len(dataset))
    val_cutoff = int((frac_train + frac_val) * len(dataset))

    for group in scaffold_sets:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    print(f" Scaffold split done: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def evaluate(model, loader, device, criterion):
    """Quick evaluation during training"""
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except:
        auc_score = 0.5

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return {'loss': avg_loss, 'auc': auc_score, 'acc': acc, 'f1': f1, 'recall': recall, 'precision': precision,
            'mcc': mcc}



def analyze_and_plot_results(model, loader, device, save_dir):
    """
    For test set analysis:
    1. Get all predictions
    2. Plot ROC curve and save
    3. Save ROC data (FPR, TPR) to CSV
    4. Plot confusion matrix and save
    """
    model.eval()
    all_probs = []  
    all_preds = []  
    all_labels = []  

    print(" Collecting test predictions...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    origin_data_path = os.path.join(save_dir, 'roc_data_for_origin.csv')
    df_origin = pd.DataFrame({
        'False Positive Rate (1-Specificity)': fpr,
        'True Positive Rate (Sensitivity)': tpr,
        'Thresholds': thresholds
    })
    df_origin.to_csv(origin_data_path, index=False)
    print(f" Plot data saved: {origin_data_path}")

    plt.figure(figsize=(8, 6), dpi=300)  
    plt.style.use('seaborn-v0_8-whitegrid')  

    plt.plot(fpr, tpr, color='#d62728', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    roc_img_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(roc_img_path, bbox_inches='tight')
    plt.close()
    print(f" ROC curve saved: {roc_img_path}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5), dpi=300)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f"{value}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]

    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                annot_kws={"size": 12, "weight": "bold"}, square=True, linewidths=1.5)

    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    cm_img_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_img_path, bbox_inches='tight')
    plt.close()
    print(f" Confusion matrix saved: {cm_img_path}")

    return all_labels, all_probs, all_preds


def main():
    config_path = "finetune_config.yaml"
    if not os.path.exists(config_path):
        print(" Config file finetune_config.yaml not found")
        exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config['model']
    config.update(yaml_config['training'])
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(yaml_config['save_dir'], exist_ok=True)

    data_path = yaml_config['data_path']
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    processed_root = os.path.join(ROOT_DIR, 'Tool', f'processed_{data_name}')

    is_cached = os.path.exists(os.path.join(processed_root, 'processed')) and \
                len(os.listdir(os.path.join(processed_root, 'processed'))) > 0

    smiles_list = None
    labels = None

    if is_cached:
        print(f" Cached data found: {processed_root}")
        print(" Skipping CSV loading, using cache...")
        dataset = PyGMolecularDataset(root=processed_root)

        if yaml_config['dataset'].get('splitting', 'random') == 'scaffold':
            print(" Need SMILES for scaffold split, reading CSV...")
            smiles_list, labels = load_smiles_and_labels(data_path)
            if len(dataset) != len(labels):
                min_len = min(len(dataset), len(labels))
                dataset = dataset[:min_len]
                smiles_list = smiles_list[:min_len]
                labels = labels[:min_len]
    else:
        print(f" No cache, running initial processing...")
        smiles_list, labels = load_smiles_and_labels(data_path)
        print(f"Init graph dataset (cache: {processed_root})...")
        dataset = PyGMolecularDataset(smiles_list=smiles_list, root=processed_root)
        if len(dataset) != len(labels):
            min_len = min(len(dataset), len(labels))
            dataset.data_list = dataset.data_list[:min_len]
            labels = labels[:min_len]
            smiles_list = smiles_list[:min_len]

    if labels is None:
        print(" Loading labels...")
        _, labels = load_smiles_and_labels(data_path)
        if len(dataset) != len(labels):
            dataset = dataset[:min(len(dataset), len(labels))]
            labels = labels[:min(len(dataset), len(labels))]

    print("Binding labels...")
    for i in range(len(dataset)):
        dataset.data_list[i].y = torch.tensor([int(labels[i])], dtype=torch.long)

    neg = labels.count(0)
    pos = labels.count(1)
    if pos == 0:
        print(" Error: no positive samples, cannot train!")
        exit(1)
    print(f" Sample stats: neg={neg}, pos={pos}, ratio={pos / neg:.2f}")
    class_weights = torch.tensor([1.0, neg / pos]).to(config['device'])

    total_len = len(dataset)
    valid_ratio = yaml_config['dataset']['valid_size']
    test_ratio = yaml_config['dataset']['test_size']
    train_ratio = 1.0 - valid_ratio - test_ratio

    split_method = yaml_config['dataset'].get('splitting', 'random')

    if split_method == 'scaffold':
        if smiles_list is None:
            smiles_list, _ = load_smiles_and_labels(data_path)
        train_set, val_set, test_set = scaffold_split(
            dataset, smiles_list, train_ratio, valid_ratio, test_ratio, seed=config['seed']
        )
    else:
        print(" Performing random split...")
        val_len = int(valid_ratio * total_len)
        test_len = int(test_ratio * total_len)
        train_len = total_len - val_len - test_len
        print(f"Dataset split: Train={train_len}, Val={val_len}, Test={test_len}")
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(config['seed'])
        )

    NUM_WORKERS = yaml_config['dataset']['num_workers']
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    model = GTCMolClassifier(config, n_classes=2).to(config['device'])
    print(f"Model built ({config['device']})")

    pretrain_path = yaml_config['pretrained_weights']
    if os.path.exists(pretrain_path):
        print(f"\n Pre-trained weights found: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=config['device'], weights_only=False)
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
        model_dict = model.state_dict()
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            elif ("encoder." + k) in model_dict and v.shape == model_dict["encoder." + k].shape:
                new_state_dict["encoder." + k] = v
            elif k.startswith("module."):
                k_fix = k[7:]
                if k_fix in model_dict and v.shape == model_dict[k_fix].shape:
                    new_state_dict[k_fix] = v
                elif ("encoder." + k_fix) in model_dict:
                    new_state_dict["encoder." + k_fix] = v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        print(f" Parameters loaded: {len(new_state_dict)}/{len(model_dict)}")
    else:
        print(f"\n Pre-trained weights not found: {pretrain_path}, using random init")

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['lr'] * 0.1},
        {'params': model.classifier.parameters(), 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n Starting fine-tuning (max {config['epochs']}  epochs)...")
    best_val_auc = 0.0
    save_path = os.path.join(yaml_config['save_dir'], 'best_finetuned.pth')

    pbar = tqdm(range(config['epochs']), desc=" Progress", unit="epoch")

    for epoch in pbar:
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(config['device'])
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            if torch.isnan(loss):
                pbar.write(" Error: Loss is NaN")
                exit(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, config['device'], criterion)
        scheduler.step(val_metrics['auc'])

        pbar.set_postfix({
            "Loss": f"{avg_train_loss:.4f}",
            "Val AUC": f"{val_metrics['auc']:.4f}",
            "Val F1": f"{val_metrics['f1']:.4f}"
        })

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), save_path)

    pbar.close()

    print("\n Testing best model and generating plots...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    analyze_and_plot_results(model, test_loader, config['device'], yaml_config['save_dir'])

    test_metrics = evaluate(model, test_loader, config['device'], criterion)

    print("=" * 40)
    print(f" TEST RESULTS (Best Model):")
    print(f"   AUC:       {test_metrics['auc']:.4f}")
    print(f"   Accuracy:  {test_metrics['acc']:.4f}")
    print(f"   F1-Score:  {test_metrics['f1']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   MCC:       {test_metrics['mcc']:.4f}")
    print("=" * 40)

    metrics_save_path = os.path.join(yaml_config['save_dir'], 'test_metrics.csv')
    df_res = pd.DataFrame([test_metrics])
    df_res.to_csv(metrics_save_path, index=False)
    print(f" Final test metrics saved to: {metrics_save_path}")


if __name__ == "__main__":
    main()