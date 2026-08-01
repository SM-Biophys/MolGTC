import os
import gc
import shutil
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg as dense_linalg
import scipy.sparse.linalg as sp_linalg
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import torch.nn.functional as F
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_INPUT_PATH = os.path.join(ROOT_DIR, 'Tool', 'Pre-dataset', 'chembl_36_chemreagent.csv')

SAMPLE_NUM = 2000000  

base_name = 'Pre-dataset2'
if SAMPLE_NUM is not None:
    suffix = f"_{int(SAMPLE_NUM / 10000)}w"
    OUTPUT_PARENT_DIR = os.path.join(ROOT_DIR, 'Tool', base_name + suffix)
else:
    OUTPUT_PARENT_DIR = os.path.join(ROOT_DIR, 'Tool', base_name)

SAVE_INTERVAL = 5000
POS_ENC_DIM = 8
NUM_SPLITS = 30  


def compute_lap_pe(edge_index, num_nodes, k=POS_ENC_DIM):
    if num_nodes < 2:
        return torch.zeros((num_nodes, k), dtype=torch.float32)
    try:
        L_edge_index, L_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(L_edge_index, L_edge_weight, num_nodes)
        if k + 1 >= num_nodes:
            vals, vecs = dense_linalg.eigh(L.toarray())
        else:
            vals, vecs = sp_linalg.eigsh(L, k=k + 1, which='SM', tol=1e-3)
        pe = torch.from_numpy(vecs[:, 1:]).float()
        cur_k = pe.size(1)
        if cur_k < k:
            pe = torch.cat([pe, torch.zeros((num_nodes, k - cur_k))], dim=1)
        else:
            pe = pe[:, :k]
        sign_flip = torch.rand(1, k)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        return pe * sign_flip
    except:
        return torch.zeros((num_nodes, k), dtype=torch.float32)


def smiles_to_fusion_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    num_nodes = mol.GetNumAtoms()
    if num_nodes == 0: return None

    atom_features = []
    for atom in mol.GetAtoms():
        vec = [0.0] * 119
        atomic_num = atom.GetAtomicNum()
        if atomic_num < 119: vec[atomic_num] = 1.0
        vec.append(float(atom.GetChiralTag()))
        atom_features.append(vec)
    x_origin = torch.tensor(atom_features, dtype=torch.float32)

    src, dst, edge_attr_list = [], [], []
    edge_aggr = torch.zeros((num_nodes, 19), dtype=torch.float32)

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u >= num_nodes or v >= num_nodes: continue

        src.extend([u, v])
        dst.extend([v, u])
        bt = bond.GetBondTypeAsDouble()
        bond_feat = [bt, float(bond.GetIsConjugated()), float(bond.GetIsAromatic()),
                     float(bond.GetBondDir()), float(bond.GetStereo())]
        edge_attr_list.extend([bond_feat, bond_feat])
        bt_vec = [0.0] * 5
        if bt == 1.0:
            bt_vec[1] = 1.0
        elif bt == 1.5:
            bt_vec[2] = 1.0
        elif bt == 2.0:
            bt_vec[3] = 1.0
        elif bt == 3.0:
            bt_vec[4] = 1.0
        else:
            bt_vec[0] = 1.0

        full_ef = torch.tensor(bt_vec + [bond_feat[1], bond_feat[2]] + [0.0] * 12)
        edge_aggr[u] += full_ef
        edge_aggr[v] += full_ef

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32) if src else torch.empty((0, 5), dtype=torch.float32)

    pe = compute_lap_pe(edge_index, num_nodes)
    x_final = torch.cat([x_origin, pe, edge_aggr], dim=1).half()

    return x_final, edge_index, edge_attr.half(), num_nodes


def process_chunk_and_save(smiles_subset, part_id, output_dir):
    temp_dir = os.path.join(output_dir, 'temp')
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    current_batch = {
        'x': [], 'edge_index': [], 'edge_attr': [],
        'sizes_x': [], 'sizes_ei': [], 'num_nodes_list': []
    }
    chunk_idx = 0

    desc = f"Part {part_id + 1}/{NUM_SPLITS}"
    for i, smiles in enumerate(tqdm(smiles_subset, desc=desc)):
        try:
            res = smiles_to_fusion_data(smiles)
            if res is None: continue

            xf, ei, ea, n_nodes = res

            if xf.size(0) != n_nodes: continue

            current_batch['x'].append(xf)
            current_batch['edge_index'].append(ei)
            current_batch['edge_attr'].append(ea)
            current_batch['sizes_x'].append(xf.size(0))
            current_batch['sizes_ei'].append(ei.size(1))
            current_batch['num_nodes_list'].append(n_nodes)

            if len(current_batch['x']) >= SAVE_INTERVAL:
                torch.save({
                    'x': torch.cat(current_batch['x'], dim=0),
                    'edge_index': torch.cat(current_batch['edge_index'], dim=1),
                    'edge_attr': torch.cat(current_batch['edge_attr'], dim=0),
                    'sizes_x': current_batch['sizes_x'],
                    'sizes_ei': current_batch['sizes_ei'],
                    'num_nodes_list': current_batch['num_nodes_list']
                }, os.path.join(temp_dir, f'chunk_{chunk_idx}.pt'))

                current_batch = {
                    'x': [], 'edge_index': [], 'edge_attr': [],
                    'sizes_x': [], 'sizes_ei': [], 'num_nodes_list': []
                }
                chunk_idx += 1
                gc.collect()
        except:
            continue

    if current_batch['x']:
        torch.save({
            'x': torch.cat(current_batch['x'], dim=0),
            'edge_index': torch.cat(current_batch['edge_index'], dim=1),
            'edge_attr': torch.cat(current_batch['edge_attr'], dim=0),
            'sizes_x': current_batch['sizes_x'],
            'sizes_ei': current_batch['sizes_ei'],
            'num_nodes_list': current_batch['num_nodes_list']
        }, os.path.join(temp_dir, f'chunk_{chunk_idx}.pt'))

    merge_final(temp_dir, output_dir)
    print(f" Part {part_id + 1} completed!")


def merge_final(temp_dir, out_dir):
    files = sorted([f for f in os.listdir(temp_dir) if f.startswith('chunk_')],
                   key=lambda x: int(x.split('_')[1].split('.')[0]))

    slices = {'x': [], 'edge_index': [], 'edge_attr': []}
    cursors = {'x': 0, 'ei': 0}

    all_x, all_ei, all_ea = [], [], []
    all_num_nodes_values = []

    for f in tqdm(files, desc="Merging Chunks"):
        data = torch.load(os.path.join(temp_dir, f))
        all_x.append(data['x'])
        all_ei.append(data['edge_index'])
        all_ea.append(data['edge_attr'])

        all_num_nodes_values.extend(data['num_nodes_list'])

        for sx in data['sizes_x']:
            slices['x'].append((cursors['x'], sx))
            cursors['x'] += sx

        for sei in data['sizes_ei']:
            slices['edge_index'].append((cursors['ei'], sei))
            slices['edge_attr'].append((cursors['ei'], sei))
            cursors['ei'] += sei

    torch.save(torch.cat(all_x, dim=0), os.path.join(out_dir, 'x_all.pt'))
    torch.save(torch.cat(all_ei, dim=1), os.path.join(out_dir, 'edge_index_all.pt'))
    torch.save(torch.cat(all_ea, dim=0), os.path.join(out_dir, 'edge_attr_all.pt'))

    torch.save(torch.tensor(all_num_nodes_values, dtype=torch.long), os.path.join(out_dir, 'num_nodes_all.pt'))

    torch.save(slices, os.path.join(out_dir, 'slices.pt'))
    shutil.rmtree(temp_dir)


def main():
    print(f" Loading full CSV: {CSV_INPUT_PATH}")
    df = pd.read_csv(CSV_INPUT_PATH)

    all_smiles = df['canonical_smiles'].dropna().tolist()
    total_len = len(all_smiles)

    print(" Shuffling data...")
    random.shuffle(all_smiles)

    if SAMPLE_NUM is not None:
        if SAMPLE_NUM < total_len:
            print(f"  Sample mode: using {total_len} from {SAMPLE_NUM} samples")
            all_smiles = all_smiles[:SAMPLE_NUM]
        else:
            print(f" sample count ({SAMPLE_NUM}) >= total, using all data.")
    else:
        print(" Using full data mode")

    final_len = len(all_smiles)
    print(f" Final data count: {final_len} | Output dir: {OUTPUT_PARENT_DIR}")
    print(f"  Split count: {NUM_SPLITS}")


    smiles_chunks = np.array_split(all_smiles, NUM_SPLITS)

    for i, subset in enumerate(smiles_chunks):
        part_dir = os.path.join(OUTPUT_PARENT_DIR, f'mmap_fusion_data_part_{i}')
        if not os.path.exists(part_dir): os.makedirs(part_dir)

        print(f"\n Processing Part {i + 1}/{NUM_SPLITS} (count: {len(subset)}) -> {part_dir}")
        process_chunk_and_save(subset, i, part_dir)
        gc.collect()


if __name__ == "__main__":
    main()