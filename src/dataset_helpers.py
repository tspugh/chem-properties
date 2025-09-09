# https://github.com/mcunow/graph-matching/blob/main/README.md
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch, Dataset
from rdkit import Chem
from functools import cache
import os
from typing import List, Optional

BOND_TENSOR_DICT = {
    Chem.BondType.SINGLE:torch.tensor([1,0,0,0]),
    Chem.BondType.AROMATIC:torch.tensor([0,1,0,0]),
    Chem.BondType.DOUBLE:torch.tensor([0,0,1,0]),
    Chem.BondType.TRIPLE:torch.tensor([0,0,0,1])
}

ORBITAL_TYPE_ELECTRON_COUNT = {'s': 2, 'p': 6, 'd': 10, 'f': 14}
# (n, l) tuples in order of filling
ORBITAL_ORDER = [(1, 's'), (2, 's'), (2, 'p'), (3, 's'), (3, 'p'), (4, 's'), (3, 'd'), (4, 'p'), (5, 's'), (4, 'd'), (5, 'p'), (6, 's'), (4, 'f'), (5, 'd'), (6, 'p'), (7, 's'), (5, 'f'), (6, 'd'), (7, 'p')]
D_FROM_S_ATOMS = [24, 29, 41, 42, 44, 45, 46, 47, 78, 79, 103]
# Tuple of (starting index in ORBITAL_ORDER, number of electrons up to layer)
BASE_LAYER_INDEX_ELECTRON_TUPLES = [
    (0, 0),
    (ORBITAL_ORDER.index((2, 's')), 2),
    (ORBITAL_ORDER.index((3, 's')), 10),
    (ORBITAL_ORDER.index((4, 's')), 18),
    (ORBITAL_ORDER.index((5, 's')), 36),
    (ORBITAL_ORDER.index((6, 's')), 54),
    (ORBITAL_ORDER.index((7, 's')), 86),
]

ORBITAL_INDEX_LOOKUP = { 's': 1, 'p': 2, 'd': 3, 'f': 4 }

@cache
def orbital_token(atomic_num, dimension=11): #dimension up through 5th row
    orbitals = ORBITAL_ORDER[:dimension]
    token = torch.tensor([0] * dimension, dtype=torch.float32)

    if atomic_num <= 0:
        raise ValueError("Atomic number must be positive")

    electrons = 0
    for i, (n, l) in enumerate(orbitals):
        if electrons >= atomic_num:
            break
        shell_electron_count = ORBITAL_TYPE_ELECTRON_COUNT[l]
        if electrons + shell_electron_count <= atomic_num:
            token[i] = shell_electron_count
            electrons += shell_electron_count
        else:
            remaining_electrons = atomic_num - electrons
            # transition metal check:
            if l == 'd' and atomic_num == 46: # steal 2 electrons
                steal = 2
            elif l == 'd' and atomic_num in D_FROM_S_ATOMS: # steal 1 electron
                steal = 1
            # ignore f-block
            else:
                steal = 0
            token[i] = remaining_electrons + steal
            prev_s = orbitals.index((n-1, 's'))
            token[prev_s] -= steal

            electrons = atomic_num

    return token

# Token of [layer, s, p, d, f]
@cache
def minimal_orbital_token(atomic_num, include_star_dim = False):
    
    if atomic_num <= 0 and include_star_dim:
        return torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32)
    elif atomic_num <= 0:
        raise ValueError("Value must be > 0, or include_star_dim must be enabled")

    output_length = 6 if include_star_dim else 5
    token = torch.zeros(output_length, dtype=torch.float32)

    for layer, (start_index, num_electrons) in enumerate(BASE_LAYER_INDEX_ELECTRON_TUPLES):
        if atomic_num > num_electrons:
            continue
        else:
            token[0] = layer
            break
    else:
        layer += 1
        token[0] = layer

    start_index, num_electrons = BASE_LAYER_INDEX_ELECTRON_TUPLES[layer-1]
    
    for n, l in ORBITAL_ORDER[start_index:]:
        # Stop after filling the current layer
        if l == 's' and n > layer:
            break

        # Number of electrons held in an orbital of 'l' type
        orbital_electrons = ORBITAL_TYPE_ELECTRON_COUNT[l]
        out_index = ORBITAL_INDEX_LOOKUP[l]

        # Fill respective tensor entry with number of electrons
        if num_electrons + orbital_electrons <= atomic_num:
            token[out_index] = orbital_electrons
            num_electrons += orbital_electrons
        else:
            token[out_index] = atomic_num - num_electrons

        steal = 0
        # Handle transition metals
        if l == 'd' and atomic_num == 46: # steal 2 electrons
            steal = 2
        elif l == 'd' and atomic_num in D_FROM_S_ATOMS:  # steal 1 electron
            steal = 1

        if steal > 0:
            token[out_index] += steal
            token[ORBITAL_INDEX_LOOKUP['s']] -= steal
        


    return token

def create_node_features(mol, replace_with_hydrogen=False, use_position=True):
    from rdkit.Chem import rdDistGeom
    node_features = []
    m3d = Chem.Mol(mol)
    if use_position:
        rdDistGeom.EmbedMolecule(m3d,randomSeed=0xa100f)
    for atom in m3d.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if use_position:
            position = atom.GetAtomPosition()
            pos_vector = torch.tensor([position.x, position.y, position.z], dtype=torch.float32)
        if atomic_num == 0 and replace_with_hydrogen:
            atomic_num = 1
        feature_vector = minimal_orbital_token(atomic_num, include_star_dim=(not replace_with_hydrogen)) # dim = 6
        if use_position:
            feature_vector = torch.cat([feature_vector, pos_vector], dim=0)
        node_features.append(feature_vector)

    return torch.stack(node_features, dim=0).to(torch.float32)

def create_edge_features(mol):
    edge_attr_prelist = []
    edges = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        edges.append((start, end))
        edges.append((end, start))

        edge = BOND_TENSOR_DICT[bond.GetBondType()]

        edge_attr_prelist.append(edge)
        edge_attr_prelist.append(edge)

    if len(edges) <= 1:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if len(edge_attr_prelist) <= 1:
        edge_attr = torch.empty((0,4), dtype=torch.float)
    else:
        edge_attr = torch.stack(edge_attr_prelist, dim=0).to(torch.float32)
    
    return (edge_index, edge_attr)


def smiles_to_graph_data(smiles, output):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    
    node_features=create_node_features(mol)
    edge_index, edge_attr = create_edge_features(mol)
    data = Data(
        x = node_features,
        edge_index = edge_index,
        edge_attr = edge_attr,
        y=output
    )

    return data

def smiles_iter_to_graph_dataset(smiles_iter, y):
    dataset = []
    for smiles, output in zip(smiles_iter, y):
        dataset.append(smiles_to_graph_data(smiles, output))
    
    return dataset

def save_dataset(data_list, path):
    os.makedirs(path, exist_ok=True)
    for idx, data in enumerate(data_list):
        torch.save(data,os.path.join(path, f'data_{idx}.pt'))