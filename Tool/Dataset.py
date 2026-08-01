import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from tqdm import tqdm

import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

import scipy.sparse.linalg as sp_linalg  
import scipy.linalg as dense_linalg      

os.environ["OMP_NUM_THREADS"] = "4"
RDLogger.DisableLog('rdApp.*')  




def compute_lap_pe(edge_index, num_nodes, k=8):
    """Laplacian positional encoding (LapPE) - Final"""
    if num_nodes < 2:
        return torch.zeros((num_nodes, k), dtype=torch.float32)

    try:
        L_edge_index, L_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(L_edge_index, L_edge_weight, num_nodes)

        if k + 1 >= num_nodes:
            vals, vecs = dense_linalg.eigh(L.toarray())
        else:
            vals, vecs = sp_linalg.eigsh(L, k=k + 1, which='SM')

        pe = torch.from_numpy(vecs[:, 1:]).float()

        current_dim = pe.size(1)
        if current_dim > k:
            pe = pe[:, :k]
        elif current_dim < k:
            pad = torch.zeros((num_nodes, k - current_dim), dtype=torch.float32)
            pe = torch.cat([pe, pad], dim=1)

        sign_flip = torch.rand(1, k)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        pe = pe * sign_flip

        return pe

    except Exception as e:
        print(f" LapPE Calculation Error: {e}")
        return torch.zeros((num_nodes, k), dtype=torch.float32)


def expand_edge_features_online(edge_attr):
    """
    Expand 5-dim -> 19-dim (online)
    Input: [BondType, IsConjugated, IsAromatic, BondDir, Stereo]
    """
    bt_map = edge_attr[:, 0]
    bt_idx = torch.zeros_like(bt_map, dtype=torch.long)
    bt_idx[bt_map == 1.0] = 1
    bt_idx[bt_map == 1.5] = 2
    bt_idx[bt_map == 2.0] = 3
    bt_idx[bt_map == 3.0] = 4
    feat_type = F.one_hot(bt_idx, num_classes=5)

    feat_conj = edge_attr[:, 1].view(-1, 1).float()

    feat_arom = edge_attr[:, 2].view(-1, 1).float()

    bd_idx = edge_attr[:, 3].long().clamp(0, 5)
    feat_dir = F.one_hot(bd_idx, num_classes=6)

    stereo_idx = edge_attr[:, 4].long().clamp(0, 5)
    feat_stereo = F.one_hot(stereo_idx, num_classes=6)

    return torch.cat([feat_type, feat_conj, feat_arom, feat_dir, feat_stereo], dim=1).float()


class PyGMolecularDataset(Dataset):
    def __init__(self, smiles_list=None, root='data'):
        super().__init__(root)
        self.data_list = []
        if smiles_list is not None:
            self._process_from_smiles(smiles_list)

    def _process_from_smiles(self, smiles_list):
        """Convert SMILES to 147-dim graph data"""
        total = len(smiles_list)
        with tqdm(total=total, desc="Processing SMILES", unit="molecules") as pbar:
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    data = self._build_fusion_graph(mol)  
                    self.data_list.append(data)
                except Exception as e:
                    warnings.warn(f"Failed to process {smi}: {e}")
                pbar.update(1)

    def _build_fusion_graph(self, mol):
        """Build Fusion graph (147-dim)"""
        MAX_ATOMIC_NUM = 119
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num >= MAX_ATOMIC_NUM: atomic_num = MAX_ATOMIC_NUM - 1
            atom_vec = [0.0] * MAX_ATOMIC_NUM
            atom_vec[atomic_num] = 1.0
            atom_vec.append(int(atom.GetChiralTag()))  
            atom_features.append(atom_vec)
        x_origin = torch.tensor(atom_features, dtype=torch.float32)
        num_nodes = x_origin.size(0)

        src, dst, edge_attr_list = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([i, j])
            dst.extend([j, i])
            bond_feat = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.GetIsAromatic()),
                int(bond.GetBondDir()),
                int(bond.GetStereo())
            ]
            edge_attr_list.extend([bond_feat, bond_feat.copy()])

        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 5), dtype=torch.float32)

        pe = compute_lap_pe(edge_index, num_nodes)

        edge_aggr = torch.zeros((num_nodes, 19), dtype=torch.float32)
        if edge_index.size(1) > 0:
            row, _ = edge_index
            edge_feat_expanded = expand_edge_features_online(edge_attr)
            edge_aggr.index_add_(0, row, edge_feat_expanded)

        x_final = torch.cat([x_origin, pe, edge_aggr], dim=1)

        return Data(x=x_final, edge_index=edge_index, edge_attr=edge_attr)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class FusionChunkDataset(Dataset):
    def __init__(self, chunk_dir):
        super().__init__()

        slice_path = os.path.join(chunk_dir, 'slices.pt')
        if not os.path.exists(slice_path):
            raise FileNotFoundError(f" Data not found: {chunk_dir}")

        self.slices = torch.load(slice_path, weights_only=False)
        self.num_graphs = len(self.slices['x'])

        self.x_all = torch.load(os.path.join(chunk_dir, 'x_all.pt'), weights_only=False, mmap='r')
        self.edge_index_all = torch.load(os.path.join(chunk_dir, 'edge_index_all.pt'), weights_only=False, mmap='r')
        self.edge_attr_all = torch.load(os.path.join(chunk_dir, 'edge_attr_all.pt'), weights_only=False, mmap='r')

        num_nodes_path = os.path.join(chunk_dir, 'num_nodes_all.pt')
        if os.path.exists(num_nodes_path):
            self.num_nodes_all = torch.load(num_nodes_path, weights_only=False, mmap='r')
        else:
            self.num_nodes_all = None

    def len(self):
        return self.num_graphs

    def get(self, idx):
        x_s, x_l = self.slices['x'][idx]
        e_s, e_l = self.slices['edge_index'][idx]

        x = self.x_all[x_s: x_s + x_l].float()
        edge_index = self.edge_index_all[:, e_s: e_s + e_l]
        edge_attr = self.edge_attr_all[e_s: e_s + e_l].float()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if self.num_nodes_all is not None:
            data.num_nodes = self.num_nodes_all[idx].item()
        else:
            data.num_nodes = x.size(0)

        return data