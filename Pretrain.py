import os
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import math
import gc
import random
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

try:
    from Tool.Dataset import FusionChunkDataset
    from module.GTCMol import GTCMol
except ImportError as e:
    print(f" Import error:{e}")
    exit(1)

config = {
    'input_dim': 147,
    'edge_feature_dim': 5,
    'hidden_dim': 256,
    'gnn_layer_num': 3,
    't_n_layers': 4,
    't_num_heads': 8,
    't_dropout_rate': 0.1,
    't_attention_dropout_rate': 0.1,
    'tau': 0.1,

    'lr': 0.002,  
    'min_lr': 1e-5,  

    'weight_decay': 1e-4,
    'batch_size': 128,
    'accumulation_steps': 4,
    'epochs': 10,
    'warmup_epochs': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_root': os.path.join(ROOT_DIR, 'Tool', 'Pre-dataset2_285w'),
    'save_dir': os.path.join(ROOT_DIR, 'Pre-dataset2', 'checkpoints')
}


def adjust_learning_rate(optimizer, global_step, total_steps, config):
    """
    LR schedule: Cosine Annealing with Warmup
    Added min_lr protection
    """
    warmup_steps = int(total_steps * (config['warmup_epochs'] / config['epochs']))

    if global_step < warmup_steps:
        lr = config['lr'] * global_step / warmup_steps
    else:
        progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * cosine_decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_chunk_folders(root_dir):
    folders = []
    if os.path.exists(root_dir):
        for f in sorted(os.listdir(root_dir)):
            if f.startswith('mmap_fusion_data_part_'):
                full_path = os.path.join(root_dir, f)
                if os.path.isdir(full_path):
                    folders.append(full_path)
    return folders


def save_checkpoint(path, model, optimizer, scaler, epoch, chunk_idx, chunk_folders, global_step, best_val_loss,
                    config):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'finished_chunk_idx': chunk_idx,
        'chunk_folders': chunk_folders,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'config': config
    }
    torch.save(ckpt, path)


def main():
    chunk_folders = get_chunk_folders(config['data_root'])
    if not chunk_folders:
        print(" No chunk data found!")
        return

    print(f" Found {len(chunk_folders)}  chunks, starting training...")

    model = GTCMol(
        hidden_dim=config['hidden_dim'], input_dim=config['input_dim'],
        edge_feature_dim=config['edge_feature_dim'], gnn_layer_num=config['gnn_layer_num'],
        transformer_layers=config['t_n_layers'], transformer_heads=config['t_num_heads'],
        dropout_rate=config['t_dropout_rate'], att_dropout_rate=config['t_attention_dropout_rate'],
        tau=config['tau']
    ).to(config['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')

    if not os.path.exists(config['save_dir']): os.makedirs(config['save_dir'])

    approx_samples_per_chunk = 480000
    total_samples = len(chunk_folders) * approx_samples_per_chunk
    steps_per_epoch = total_samples // config['batch_size']
    total_training_steps = steps_per_epoch * config['epochs']

    start_epoch = 0
    resume_chunk_idx = 0
    resume_chunk_list = None
    global_step = 0
    best_val_loss = float('inf')

    ckpt_path = os.path.join(config['save_dir'], 'last_model.pth')

    if os.path.exists(ckpt_path):
        print(f" Checkpoint found: {ckpt_path}，Restoring state...")
        checkpoint = torch.load(ckpt_path, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        best_val_loss = checkpoint['best_val_loss']
        global_step = checkpoint['global_step']

        last_epoch = checkpoint['epoch']
        last_finished_chunk = checkpoint['finished_chunk_idx']

        if last_finished_chunk >= len(chunk_folders) - 1:
            start_epoch = last_epoch + 1
            resume_chunk_idx = 0
            resume_chunk_list = None
            print(f" Epoch {last_epoch + 1}  completed, starting from Epoch {start_epoch + 1}  start")
        else:
            start_epoch = last_epoch
            resume_chunk_idx = last_finished_chunk + 1
            resume_chunk_list = checkpoint['chunk_folders']
            print(f" Resuming Epoch {start_epoch + 1}，from chunk {resume_chunk_idx + 1} ")

    else:
        print(" No checkpoint, starting from scratch")

    for epoch in range(start_epoch, config['epochs']):

        if epoch == start_epoch and resume_chunk_list is not None:
            current_epoch_chunks = resume_chunk_list
            print(f" [Resume]using saved chunk order (skip first {resume_chunk_idx} )")
        else:
            current_epoch_chunks = chunk_folders.copy()
            random.shuffle(current_epoch_chunks)
            resume_chunk_idx = 0

        epoch_train_loss = 0
        epoch_val_loss = 0
        chunk_count = 0

        print(f"\n======== Epoch {epoch + 1}/{config['epochs']} ========")

        for chunk_idx, chunk_path in enumerate(current_epoch_chunks):

            if chunk_idx < resume_chunk_idx:
                continue

            print(f" Chunk ({chunk_idx + 1}/{len(current_epoch_chunks)}): {os.path.basename(chunk_path)}")

            gc.collect()
            torch.cuda.empty_cache()

            try:
                current_dataset = FusionChunkDataset(chunk_path)
            except Exception as e:
                print(f" Skip: {e}")
                continue

            c_len = len(current_dataset)
            t_len = int(c_len * 0.95)
            v_len = c_len - t_len
            gen = torch.Generator().manual_seed(42 + epoch)
            train_ds, val_ds = random_split(current_dataset, [t_len, v_len], generator=gen)

            train_loader = DataLoader(
                train_ds, batch_size=config['batch_size'], shuffle=True,
                num_workers=0, drop_last=True, pin_memory=True
            )
            val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

            model.train()
            pbar = tqdm(train_loader, desc=f"Train Chunk {chunk_idx + 1}", leave=False)

            for batch in pbar:
                curr_lr = adjust_learning_rate(optimizer, global_step, total_training_steps, config)
                global_step += 1

                try:
                    if batch.x.shape[0] != batch.batch.shape[0]: continue
                    if batch.edge_index.numel() > 0:
                        if batch.edge_index.max() >= batch.x.shape[0]: continue
                except Exception as e:
                    continue

                batch = batch.to(config['device'], non_blocking=True)

                with torch.amp.autocast('cuda'):
                    embeds, loss_val = model(batch)
                    loss = loss_val / config['accumulation_steps']

                scaler.scale(loss).backward()

                if (global_step + 1) % config['accumulation_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_train_loss += loss_val.item() * batch.num_graphs
                pbar.set_postfix({'loss': f"{loss_val.item():.4f}", 'lr': f"{curr_lr:.6f}"})
                del batch

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(config['device'], non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        _, loss = model(batch)
                    epoch_val_loss += loss.item() * batch.num_graphs
                    del batch

            chunk_count += c_len

            save_checkpoint(
                path=os.path.join(config['save_dir'], 'last_model.pth'),
                model=model, optimizer=optimizer, scaler=scaler,
                epoch=epoch, chunk_idx=chunk_idx, chunk_folders=current_epoch_chunks,
                global_step=global_step, best_val_loss=best_val_loss, config=config
            )

            del train_loader, val_loader, train_ds, val_ds, current_dataset

        if chunk_count > 0:
            avg_train_loss = epoch_train_loss / chunk_count
            avg_val_loss = epoch_val_loss / (chunk_count * 0.05)
            print(f" Epoch {epoch + 1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config
                }, os.path.join(config['save_dir'], 'best_model.pth'))
                print(f" Saved Best Model!")

            epoch_save_path = os.path.join(config['save_dir'], f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch + 1,
                'val_loss': avg_val_loss
            }, epoch_save_path)
            print(f" Checkpoint saved: {os.path.basename(epoch_save_path)}")

        resume_chunk_list = None

    print(" completed!")


if __name__ == "__main__":
    main()