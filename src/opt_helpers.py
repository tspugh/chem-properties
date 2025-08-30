from rdkit import Chem
from rdkit.Chem import rdDistGeom
from pyscf import gto, scf

def remove_asterisk(smiles):
    return smiles.replace('*', '').replace('()', '')

def optimize_ffv(smiles: str, basis: str = 'sto-3g', max_cycle: int = 50, conv: float = 1e-6, basis_set: str = 'sto-3g'):
    mol = Chem.rdmolfiles.MolFromSmiles(remove_asterisk(smiles))
    mol = Chem.AddHs(mol)
    
    geometry = rdDistGeom.ETKDGv3()
    Chem.rdDistGeom.EmbedMolecule(mol, geometry)
    mol_xyz = Chem.MolToXYZBlock(mol).splitlines()[2:]
    mol_xyz = '\n'.join(mol_xyz)
    
    mymol = gto.Mole()
    mymol.atom = mol_xyz
    mymol.basis = basis_set
    mymol.build()

    mf = scf.RHF(mymol)
    mf.conv_tol = conv
    mf.max_cycle = max_cycle
    mf.verbose = 4
    mf.kernel()
    return mf