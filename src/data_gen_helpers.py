from typing import Optional, List, Set, AsyncGenerator, Union, Generator, Tuple
from pysmiles import read_smiles, write_smiles
from rdkit import Chem
import asyncio
import networkx as nx

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
        fragments.append(write_smiles(mol, start=atom))
    return fragments

def connect_smiles_graphwise(frag_1_smiles: str, frag_2_smiles: str) -> str:
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


def iterative_extend_smiles(
    smiles: str,
    max_length: int = 150,
    max_output: Optional[int] = None
) -> Generator[Tuple[str, int], None, None]:
    """Asynchronously yield new SMILES strings by iteratively extending fragments at '*' positions.
       Avoids duplicates and handles fragment symmetries.
    """
    frag_length = read_smiles(smiles).number_of_nodes()
    if frag_length >= max_length:
        yield (smiles, 1)
        return
    
    print(frag_length)
    
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
    max_chain_extend = max_length // frag_length

    def helper(current_smiles: str, depth_left: int) -> Generator[str, None, None]:
        if depth_left == 0:
            yield (current_smiles, max_chain_extend)
            return

        for frag in unique_frags:
            structs = connect_smiles_graphwise(frag, current_smiles)
            for s in structs:
                can_s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                if can_s not in visited:
                    visited.add(can_s)
                    yield (can_s, max_chain_extend-depth_left+1)  # yield immediately instead of collecting in a list
                    if max_output and len(visited) >= max_output:
                        return
                    for res, _ in helper(can_s, depth_left - 1):
                        yield res
                        if max_output and len(visited) >= max_output:
                            return

    start = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    visited.add(start)
    
    for result in helper(start, max_chain_extend):
        yield result