import os
import sys
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from rdkit import Chem
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
    """Read and clean data, ensure valid molecules"""
    print(f" [Full-Train] Loading raw CSV: {data_path}")

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

    valid_smiles = []
    valid_labels = []

    for smi, lbl in zip(df_clean['Smiles'], df_clean['activity']):
        if isinstance(smi, str) and len(smi) > 0:
            valid_smiles.append(smi)
            valid_labels.append(lbl)

    print(f" [Full-Train] Valid molecules: {len(valid_smiles)}")
    return valid_smiles, valid_labels


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

    prod_save_dir = os.path.join(yaml_config['save_dir'], "Production_Models")
    os.makedirs(prod_save_dir, exist_ok=True)

    print("=" * 60)
    print(" Starting full production model training")
    print("  Warning: using 100% data for training!")
    print("=" * 60)

    data_path = yaml_config['data_path']
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    processed_root = os.path.join(ROOT_DIR, 'Tool', f'processed_{data_name}_FULL')

    is_cached = os.path.exists(os.path.join(processed_root, 'processed')) and \
                len(os.listdir(os.path.join(processed_root, 'processed'))) > 0

    dataset = None
    labels = None

    if is_cached:
        print(f" Cached data found: {processed_root}")
        dataset = PyGMolecularDataset(root=processed_root)
        _, labels = load_smiles_and_labels(data_path)
    else:
        print(f" No cache, processing full data...")
        smiles_list, labels = load_smiles_and_labels(data_path)
        dataset = PyGMolecularDataset(smiles_list=smiles_list, root=processed_root)

    if len(dataset) != len(labels):
        min_len = min(len(dataset), len(labels))
        dataset = dataset[:min_len]
        labels = labels[:min_len]

    print("Binding full labels...")
    for i in range(len(dataset)):
        dataset.data_list[i].y = torch.tensor([int(labels[i])], dtype=torch.long)

    neg = labels.count(0)
    pos = labels.count(1)
    print(f" Full sample stats: total={len(dataset)}, neg={neg}, pos={pos}")

    class_weights = torch.tensor([1.0, neg / pos]).to(config['device'])

    NUM_WORKERS = yaml_config['dataset']['num_workers']
    full_train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=NUM_WORKERS)

    model = GTCMolClassifier(config, n_classes=2).to(config['device'])

    pretrain_path = yaml_config['pretrained_weights']
    if os.path.exists(pretrain_path):
        print(f"\n Loading pre-trained backbone: {pretrain_path}")
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
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        print(" Pre-trained backbone loaded")

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['lr'] * 0.1},
        {'params': model.classifier.parameters(), 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    epochs = config['epochs']
    print(f"\n Starting full fine-tuning (total {epochs}  epochs)...")

    model.train()
    pbar = tqdm(range(epochs), desc=" Full-Training", unit="epoch")

    loss_history = []

    for epoch in pbar:
        total_loss = 0

        for batch in full_train_loader:
            batch = batch.to(config['device'])
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(full_train_loader)
        loss_history.append(avg_loss)

        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[1]['lr']
        pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.6f}"})

    pbar.close()

    final_model_name = f"Final_Production_{data_name}.pth"
    final_save_path = os.path.join(prod_save_dir, final_model_name)

    torch.save(model.state_dict(), final_save_path)

    print("\n" + "=" * 60)
    print(f" Training completed! Production model saved.")
    print(f" Model path: {final_save_path}")
    print("=" * 60)
    print(" Hint: Use this model for virtual screening.")


if __name__ == "__main__":
    main()