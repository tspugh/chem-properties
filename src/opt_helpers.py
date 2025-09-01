from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from pyscf import gto, scf, semiempirical
from pyscf.geomopt import geometric_solver
from typing import Tuple, Optional, List
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

verbose=0

from data_gen_helpers import iterative_extend_smiles
import pandas as pd

def remove_asterisk(smiles):
    return smiles.replace('*', '').replace('()', '')

def create_gto_mol(smiles, basis_set='3-21g', add_hydrogen=True, opt=True):
    mol = Chem.MolFromSmiles(remove_asterisk(smiles))
    if mol is None:
        raise RuntimeError("Molecule failed to generate from smiles")
    
    if add_hydrogen:
        mol = Chem.AddHs(mol)
    
    # Try multiple geometry generation methods with more attempts
    success = False
    
    # Method 1: ETKDG with multiple attempts and different parameters
    for i in range(5):  # More attempts
        try:
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = i * 42
            params.useRandomCoords = True
            params.maxAttempts = 1000  # More attempts per embedding
            if rdDistGeom.EmbedMolecule(mol, params) == 0:
                success = True
                break
        except:
            continue
    
    # Method 2: Try basic ETKDG if v3 fails
    if not success:
        print("ETKDGv3 failed, trying basic ETKDG")
        for i in range(7):
            try:
                if rdDistGeom.EmbedMolecule(mol, randomSeed=i*42) == 0:
                    success = True
                    break
            except:
                continue
    
    # Method 3: Try with useRandomCoords
    if not success:
        print("ETKGDG failed, trying random coords")
        try:
            params = rdDistGeom.ETKDGv3()
            params.useRandomCoords = True
            params.randomSeed = 12345
            if rdDistGeom.EmbedMolecule(mol, params) == 0:
                success = True
        except:
            pass
    
    if not success:
        raise RuntimeError("Failed to embed molecule geometry after all attempts")
    
    # UFF optimization
    if opt:
        try:
            MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception as e:
            print("Pre-optimization failed: ", e)
            pass  # Continue even if UFF fails
    
    # Check if we have a valid conformer
    if mol.GetNumConformers() == 0:
        raise RuntimeError("No valid conformer")
    
    # Generate XYZ block
    try:
        xyz_block = Chem.MolToXYZBlock(mol)
        mol_xyz = xyz_block.splitlines()[2:]
        if len(mol_xyz) == 0:
            raise RuntimeError("mol xyz was empty")
        mol_xyz = '\n'.join(mol_xyz)
    except Exception as e:
        raise RuntimeError(f"XYZ generation failed: {e}")
    
    # Build PySCF molecule
    mymol = gto.Mole()
    mymol.atom = mol_xyz
    mymol.basis = basis_set
    try:
        mymol.build(verbose=0)
    except Exception as e:
        raise RuntimeError(f"PySCF build failed: {e}")
    
    global verbose
    if verbose:
        print("Sucessfully created molecule for", smiles)
    return mymol


def optimize_semiempirical(smiles: str, max_geom_steps: int = 25, method: str = 'MINDO', add_hydrogen=True) -> Tuple[float, int, str]:
    """
    Fast semi-empirical calculation
    Returns: (energy, cycles, method_used)
    """
    try:
        mymol = create_gto_mol(smiles, add_hydrogen=add_hydrogen)

        # Semi-empirical calculation
        if method == 'MINDO':
            mf = semiempirical.MINDO3(mymol)
            mf.build(mymol)
        else:
            raise ValueError(f"Unknown semi-empirical method: {method}")
            
        global verbose
        mf.verbose = verbose
        
        #mol_eq = geometric_solver.optimize(mf, maxsteps=max_geom_steps)
        mf.run()
        
        return mf.e_tot, mf.scf_cycles if hasattr(mf, 'scf_cycles') else 1, method
        
    except Exception as e:
        print(f"Semi-empirical failed for {smiles}: {str(e)}")
        return None, 0, f"{method}_failed"


def optimize_fast_dft(smiles: str, max_cycle: int = 10, max_geom_steps: int = 25, conv=1e-4, add_hydrogen=True) -> Tuple[float, int, str]:
    """
    Fast DFT calculation with LDA functional
    Returns: (energy, cycles, method_used)
    """
    try:
        mymol = create_gto_mol(smiles, add_hydrogen=add_hydrogen)

        mf = scf.RKS(mymol)
        mf.xc = 'lda,vwn'  # Fastest DFT functional
        mf.conv_tol = conv  # Looser convergence
        mf.max_cycle = 2

            # Key optimizations for large molecules:
        mf.init_guess = 'minao'  # Faster initial guess
        mf.diis_space = 12       # Larger DIIS space for better convergence
        mf.damping = 0.7         # Add damping for stability
        mf.level_shift = 0.1     # Level shifting can help convergence
        
        # For very large molecules, consider:
        mf.direct_scf = True     # Use direct SCF (less memory, sometimes faster)

        global verbose
        mf.verbose = verbose

        if verbose:
            print("Starting dft calculation")

        #mol_eq = geometric_solver.optimize(mf, maxsteps=max_geom_steps)
        mf.kernel()
        
        return mf.e_tot, mf.scf_cycles if hasattr(mf, 'scf_cycles') else max_cycle, "dft_lda"
        
    except Exception as e:
        print(f"DFT failed for {smiles}: {str(e)}")
        return None, 0, "dft_failed"
    

def optimize_hf(smiles: str, max_cycle: int = 10, max_geom_steps: int = 25, conv: float = 1e-4, add_hydrogen=True) -> Tuple[float, int, str]:
    """
    Hartree-Fock calculation
    Returns: (energy, cycles, method_used)
    """
    try:
        mymol = create_gto_mol(smiles, add_hydrogen=add_hydrogen)

        mf = scf.RHF(mymol)
        mf.conv_tol = conv
        mf.max_cycle = max_cycle

        global verbose
        mf.verbose = verbose
        
        #mol_eq = geometric_solver.optimize(mf, maxsteps=max_geom_steps)
        mf.kernel()
        
        return mf.e_tot, mf.scf_cycles if hasattr(mf, 'scf_cycles') else max_cycle, "hf"
    
    except Exception as e:
        print(f"HF failed for {smiles}: {str(e)}")
        return None, 0, "hf_failed"
    

def process_single_polymer(args) -> Tuple[str, str, float, str, int, float]:
    """
    Process a single polymer calculation
    Returns: (monomer_smiles, polymer_smiles, energy, method, cycles, calc_time)
    """
    monomer_smiles, polymer_smiles, method = args
    
    start_time = time.time()
    
    try:
        if method == 'semiempirical':
            energy, cycles, method_used = optimize_semiempirical(polymer_smiles, 'MINDO')
        elif method == 'dft':
            energy, cycles, method_used = optimize_fast_dft(polymer_smiles)
        elif method == 'hf':
            energy, cycles, method_used = optimize_hf(polymer_smiles)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        calc_time = time.time() - start_time
        global verbose
        if verbose:
            print("Sucessfully calculated energy for", polymer_smiles, "Completed in:", calc_time, "Energy:", energy, "Cycles:", cycles)

        return monomer_smiles, polymer_smiles, energy, method_used, cycles, calc_time
        
    except Exception as e:
        calc_time = time.time() - start_time
        print(f"Failed to process {polymer_smiles}: {str(e)}")
        return monomer_smiles, polymer_smiles, None, f"{method}_error", 0, calc_time


def calculate_polymer_energies(monomer_smiles_list: List[str], 
        max_chain_length: int = 60,
        method: str = 'semiempirical',
        max_polymers_per_monomer: Optional[int] = None,
        n_processes: Optional[int] = None
    ) -> pd.DataFrame:
    """
    Main function to calculate energies for all polymer configurations
    
    Parameters:
    -----------
    monomer_smiles_list : List[str]
        List of monomer SMILES strings
    max_chain_length : int
        Maximum number of monomers in a chain
    method : str
        Calculation method: 'semiempirical', 'dft', or 'hf'
    max_polymers_per_monomer : Optional[int]
        Limit number of polymers per monomer (for speed)
    n_processes : Optional[int]
        Number of parallel processes (default: number of CPUs)
    
    Returns:
    --------
    pd.DataFrame with columns: monomer_smiles, polymer_smiles, zero_energy, method, cycle_count, calc_time
    """
    
    if n_processes is None:
        n_processes = min(cpu_count(), 4)  # Limit to 4 for Raspberry Pi
    
    print(f"Generating polymer chains for {len(monomer_smiles_list)} monomers...")
    
    # Generate all polymer combinations
    all_tasks = []
    for monomer in monomer_smiles_list:
        print(f"Generating chains for monomer: {monomer}")
        polymers = list(iterative_extend_smiles(monomer, max_chain_length, max_polymers_per_monomer))
        print(f"  Generated {len(polymers)} polymer configurations")
        
        for polymer in polymers:
            all_tasks.append((monomer, polymer, method))
    
    print(f"Total calculations to perform: {len(all_tasks)}")
    print(f"Using {n_processes} parallel processes")
    dur = "1-10" if method=='semiempirical' else "10-60"
    print(f"Estimated time per calculation: {dur} seconds")
    
    # Process in parallel
    results = []
    batch_size = 100  # Process in batches to avoid memory issues
    
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_tasks)-1)//batch_size + 1} ({len(batch)} calculations)")
        
        with Pool(n_processes) as pool:
            batch_results = pool.map(process_single_polymer, batch)
            results.extend(batch_results)
        
        # Print progress
        completed = len(results)
        if completed > 0:
            avg_time = sum(r[5] for r in results[-len(batch):]) / len(batch)
            remaining = len(all_tasks) - completed
            eta_hours = (remaining * avg_time) / 3600 / n_processes
            print(f"  Batch completed. Average time: {avg_time:.1f}s. ETA: {eta_hours:.1f} hours")
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=[
        'monomer_smiles', 'polymer_smiles', 'zero_energy', 'method', 'cycle_count', 'calc_time'
    ])
    
    # Remove failed calculations
    df = df.dropna(subset=['zero_energy'])
    
    print(f"Completed calculations: {len(df)}")
    print(f"Failed calculations: {len(results) - len(df)}")
    
    return df
