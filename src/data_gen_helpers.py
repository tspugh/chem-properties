from typing import Optional, List, Set, AsyncGenerator, Union, Generator, Tuple
from pysmiles import read_smiles, write_smiles
from rdkit import Chem
import asyncio
import networkx as nx
import logging
import warnings
from datetime import datetime

logger = logging.getLogger(__name__)

# Suppress RDKit warnings
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')

# Additionally suppress specific SMILES writer warnings
warnings.filterwarnings('ignore', message='.*SMILES writer.*stereochemical.*')
warnings.filterwarnings('ignore', message='.*does not write stereochemical information.*')

# Suppress pysmiles warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pysmiles')

# Suppress all SMILES-related warnings
import sys
import io
from contextlib import redirect_stderr, redirect_stdout

class SuppressOutput:
    """Context manager for complete suppression of stdout and stderr"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def display_smiles(smiles: Union[str, List[str]]):
    from rdkit import Chem
    from rdkit.Chem import Draw
    from IPython.display import display

    if isinstance(smiles, str):
        smiles = [smiles]

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    for mol in mols:
        display(mol)

def get_symmetric_smiles_ending_at_asterick(smiles: str) -> list[str]:
    mol: nx.Graph = read_smiles(smiles)
    fragments = []
    for atom, element in mol.nodes(data='element'):
        if element:
            continue
        with SuppressOutput():
            fragments.append(write_smiles(mol, start=atom))
    return fragments

def connect_smiles_graphwise(frag_1_smiles: str, frag_2_smiles: str):
    # Read both fragments as graphs
    g1 = read_smiles(frag_1_smiles)
    g2 = read_smiles(frag_2_smiles)
    
    # Find the star (None) node in frag_1 (should be only one)
    star_nodes_1 = [n for n, d in g1.nodes(data=True) if d.get('element') is None]
    if not star_nodes_1:
        raise ValueError("No star node found in frag_1")
    star_1 = star_nodes_1[0]
    
    # Find all star (None) nodes in frag_2
    star_nodes_2 = [n for n, d in g2.nodes(data=True) if d.get('element') is None]
    if not star_nodes_2:
        raise ValueError("No star node found in frag_2")
    
    # Remove the star node from frag_1, keep its neighbors
    neighbors_1 = list(g1.neighbors(star_1))
    g1.remove_node(star_1)
    
    # Relabel frag_2 nodes to avoid collision
    offset = max(g1.nodes) + 1 if len(g1.nodes) > 0 else 0
    mapping = {n: n + offset for n in g2.nodes}
    g2 = nx.relabel_nodes(g2, mapping)
    star_nodes_2 = [mapping[n] for n in star_nodes_2]

    new_g2_graphs = []
    # Connect the two graphs
    for star_2 in star_nodes_2:
        new_graph = g2.copy()
        # For each neighbor of the removed star in frag_1, connect to all neighbors of the star in frag_2
        for n in neighbors_1:
            for end in g2.neighbors(star_2):
                new_graph.add_edge(n, end)
        new_graph.remove_node(star_2)
        new_g2_graphs.append(new_graph)

    # Merge graphs
    new_g2_graphs = [nx.compose(g1, g) for g in new_g2_graphs]
    
    # Write back to SMILES
    output = []
    for g_combined in new_g2_graphs:
        output.append(write_smiles(g_combined))

    return output


def safe_replace_asterisk_with_carbon(smiles: str) -> str:
    """Safely replace asterisk (*) with carbon (C) atoms where possible"""
    from rdkit import Chem
    
    # Try simple replacement first
    try:
        simple_replacement = smiles.replace('*', 'C')
        mol = Chem.MolFromSmiles(simple_replacement)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.error(f"Error cleaning SMILES {smiles}: {e}")
        pass
    
    # If that doesn't work, try other strategies
    # Strategy 1: Remove asterisks that lead to invalid bonds
    try:
        # Remove patterns like (*), =*, *=, etc.
        import re
        cleaned = re.sub(r'\(\*\)', '', smiles)  # Remove (*) 
        cleaned = re.sub(r'=\*', '', cleaned)    # Remove =*
        cleaned = re.sub(r'\*=', '', cleaned)    # Remove *=
        cleaned = re.sub(r'\*', 'C', cleaned)   # Replace remaining * with C
        
        mol = Chem.MolFromSmiles(cleaned)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.error(f"Error cleaning SMILES {smiles}: {e}")
        pass
    
    # Fallback: use original remove_asterisk
    from opt_helpers import remove_asterisk
    return remove_asterisk(smiles)

def count_non_hydrogen_atoms(smiles: str) -> int:
    """Return the number of non-hydrogen atoms in a SMILES"""
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        # Count only heavy atoms (no hydrogen)
        return mol.GetNumHeavyAtoms()
    except:
        return 0

def iterative_extend_smiles(
    smiles: str,
    min_length: int = 150,
    max_output: Optional[int] = None
) -> Generator[Tuple[str, int], None, None]:
    """Iteratively yield new SMILES strings by extending fragments at '*' positions.
       Avoids duplicates and handles fragment symmetries.
    """
    try:
        cleaned_smiles = safe_replace_asterisk_with_carbon(smiles)
        current_atoms = count_non_hydrogen_atoms(cleaned_smiles)
        
    except Exception as e:
        logger.error(f"Error cleaning SMILES {smiles}: {e}")
        # Fallback to original
        cleaned_smiles = smiles
        current_atoms = 0
    
    # Use current_atoms instead of number_of_nodes for the comparison
    if current_atoms >= min_length:
        logger.debug(f"SMILES {cleaned_smiles} already has {current_atoms} atoms, target: {min_length}")
        yield (cleaned_smiles, 1)
        return
    
    logger.debug(f"Starting extension with {current_atoms} atoms, target: {min_length}")
    
    # Deduplicate symmetric fragments (*-ABCD-* vs *-DCBA-*)
    frags = get_symmetric_smiles_ending_at_asterick(smiles)
    unique_frags = []
    seen = set()
    for f in frags:
        canonical = min(f, f[::-1])
        if canonical not in seen:
            seen.add(canonical)
            unique_frags.append(f)

    visited: Set[str] = set()
    max_chain_extend = min_length // current_atoms + 1

    counter = 0
    def update_counter():
        nonlocal counter
        counter += 1
        logger.debug(f"Extended {counter}/{len(visited)} structures")

    def helper(current_smiles: str, depth_left: int) -> Generator[Tuple[str, int], None, None]:
        if depth_left == 0:
            yield (safe_replace_asterisk_with_carbon(current_smiles), max_chain_extend)
            logger.debug(f"Yielding {safe_replace_asterisk_with_carbon(current_smiles)} with depth {max_chain_extend}, depth_left == 0")
            return

        if '*' not in current_smiles:
            yield (current_smiles, max_chain_extend)
            logger.debug(f"Yielding {current_smiles} with depth {max_chain_extend}, no * found")
            return

        for frag in unique_frags:
            try:
                structs = connect_smiles_graphwise(frag, current_smiles)
                for s in structs:
                    try:
                        clean_s = safe_replace_asterisk_with_carbon(s)
                        mol = Chem.MolFromSmiles(clean_s)
                        if mol is None:
                            continue
                        clean_smi_s = Chem.MolToSmiles(mol)

                        atoms_count = count_non_hydrogen_atoms(clean_smi_s)

                        if clean_smi_s not in visited:
                            visited.add(clean_smi_s)
                            yield (clean_smi_s, max_chain_extend - depth_left + 1)

                            if max_output and len(visited) >= max_output:
                                return

                            if atoms_count < min_length:
                                for res, dep in helper(s, depth_left - 1):
                                    yield res, dep
                                    if max_output and len(visited) >= max_output:
                                        logger.debug(f"Yielding {res} with depth {dep}, max_output reached")
                                        return
                    except Exception as e:
                        logger.error(f"Error processing structure {s}: {e}")
                        update_counter()
                        continue
            except Exception as e:
                # Specific error handling for missing star node in frag_2
                if "No star node found in frag_2" in str(e) or "No star node found in frag_2" in repr(e):
                    logger.warning(
                        f"No star node found in frag_2 for current_smiles='{current_smiles}'. "
                        "This can happen if the star was already removed in a previous step, "
                        "or if the SMILES string is malformed or does not contain a star at this point. "
                        "Check if current_smiles still contains a '*' before calling connect_smiles_graphwise."
                    )
                else:
                    logger.error(f"Error in graph connection: {e}")
                update_counter()
                continue

    # Safe handling of the start SMILES
    try:
        start = safe_replace_asterisk_with_carbon(smiles)
        mol = Chem.MolFromSmiles(start)
        if mol is not None:
            start = Chem.MolToSmiles(mol)
        else:
            start = smiles
    except:
        start = smiles
    if not start in visited:
        visited.add(start)

    for result in helper(smiles, max_chain_extend):
        yield (result[0], max_chain_extend - result[1] + 1)
    logger.debug(f"Extended {len(visited)-counter}/{len(visited)} structures")