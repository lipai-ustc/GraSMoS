import os
import platform
from typing import Tuple, List, Optional
from datetime import datetime
import multiprocessing
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

def get_version_info():
    # Get grasmos version
    try:
        from importlib.metadata import version as _pkg_version
        grasmos_version = _pkg_version('grasmos')
    except Exception:
        grasmos_version = 'unknown'
    
    # Get Python version
    python_ver = platform.python_version()
    
    # Get ASE version
    try:
        import ase
        ase_ver = getattr(ase, '__version__', 'unknown')
    except Exception:
        ase_ver = 'unknown'
    
    # Get dscribe version
    try:
        from importlib.metadata import version as _pkg_version
        dscribe_ver = _pkg_version('dscribe')
    except Exception:
        dscribe_ver = 'unknown'
    
    # Build header string
    os_name = platform.system()
    os_release = platform.release()
    now_str = datetime.now().strftime('%Y.%m.%d  %H:%M:%S')
    total_cores = multiprocessing.cpu_count()
    
    header=f"""grasmos {grasmos_version} ({os_name} {os_release})
executed on             {os_name} date {now_str}
running on    {total_cores} total cores
Python {python_ver}, ASE {ase_ver}, dscribe {dscribe_ver}"""
   
    return header
    
def load_potential(potential_config, custom_atomic=False):
    """
    Automatically load different types of potential calculators based on configuration
    
    Parameters:
        potential_config: Dictionary containing potential configuration with keys:
            - 'type': Potential type ('eam', 'chgnet', 'deepmd', 'lammps', 'python', 'nequip')
            - 'model': Model file path or name (unified parameter for all types)
            - For 'python' type: loads calculator from calculator.py in working directory
            - If empty/None: defaults to NequIP using NEQUIP_MODEL environment variable
    
    Returns:
        ASE Calculator object
    """
    
    try:
        pot_type = potential_config['type'].lower()
    except KeyError:
        raise ValueError("Potential configuration missing 'type' key")
    
    # Get current working directory to resolve relative paths
    cwd = os.getcwd()
    
    if pot_type == 'python':
        # Load custom calculator from calculator.py in current working directory
        import sys
        cwd = os.getcwd()
        calc_file = os.path.join(cwd, 'calculator.py')
        if not os.path.exists(calc_file):
            raise FileNotFoundError(
                f"calculator.py not found in {cwd}\n"
                f"When using type='python', you must provide a calculator.py file "
                f"that defines a 'calculator' variable with an ASE Calculator object."
            )
        # Temporarily add cwd to path
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            import calculator as calc_module
            # Force reload in case it was previously imported
            import importlib
            importlib.reload(calc_module)
            if not hasattr(calc_module, 'calculator'):
                raise AttributeError(
                    f"calculator.py must define a 'calculator' variable.\n"
                    f"Example: calculator = Tersoff()\n"
                )
            return calc_module.calculator
        except ImportError as e:
            raise ImportError(
                f"Failed to import calculator.py: {e}\n"
                f"Make sure calculator.py is valid Python code."
            )
        finally:
            # Clean up sys.path
            if cwd in sys.path:
                sys.path.remove(cwd)
    elif pot_type == 'eam':
        from ase.calculators.eam import EAM
        model_path = potential_config.get('model')
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(cwd, model_path)
        return EAM(potential=model_path)
    elif pot_type == 'chgnet':
        from chgnet.model import CHGNet
        model_name = potential_config.get('model', 'pretrained')
        if model_name == 'pretrained':
            model = CHGNet.load()
        else:
            if not os.path.isabs(model_name):
                model_name = os.path.join(cwd, model_name)
            model = CHGNet.load(model_name)
        return model.get_calculator()
    elif pot_type == 'deepmd':
        model_path = potential_config.get('model')
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(cwd, model_path)
        if custom_atomic:    # custom 
            return DeepMDCalculatorWithAtomicEnergy(model=model_path)
        else:    
            from deepmd.calculator import DP
            return DP(model=model_path)
    elif pot_type == 'nep-cpu' or pot_type == 'nep-gpu' or pot_type == 'nep':
        if pot_type== 'nep-cpu' or pot_type == 'nep':
            from calorine.calculators import CPUNEP
        elif pot_type == 'nep-gpu':
            from calorine.calculators import GPUNEP
        model_path = potential_config.get('model')
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(cwd, model_path)
        return CPUNEP(model_path) if pot_type == 'nep-cpu' or pot_type == 'nep' else GPUNEP(model_path)
    elif pot_type == 'lammps':
        from ase.calculators.lammpslib import LAMMPSlib
        # Parse LAMMPS potential configuration
        lammps_commands = potential_config['commands']
        return LAMMPSlib(lmpcmds=lammps_commands)
    elif pot_type == 'fairchem':
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        # Parse FAIRChem configuration
        model_path = potential_config.get('model', 'EquiformerV2-31M-S2EF-OC20-All+MD')
        if model_path != 'EquiformerV2-31M-S2EF-OC20-All+MD' and not os.path.isabs(model_path):
            model_path = os.path.join(cwd, model_path)
        device = potential_config.get('device', 'cpu')
        task_name = potential_config.get('task_name', 'oc20')
        # Load pretrained model
        predictor = pretrained_mlip.load_predict_unit(model_path, device=device)
        # Create FAIRChem calculator
        return FAIRChemCalculator(predictor, task_name=task_name)
    elif pot_type == 'nequip':
        from nequip.ase import NequIPCalculator
        model_path = potential_config.get('model')
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(cwd, model_path)
        device = potential_config.get('device', 'cpu')
        return NequIPCalculator.from_compiled_model(compile_path=model_path, device=device)
    elif pot_type == 'vasp':
        from ase.calculators.vasp import Vasp
        incar_file = potential_config.get('model', 'INCAR')
        # Try to read INCAR parameters if file exists
        vasp_params = {}
        if os.path.exists(incar_file):
            # Parse INCAR file to extract parameters
            with open(incar_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Try to convert to appropriate type
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass  # Keep as string
                        vasp_params[key.lower()] = value
        else:
            # Use default PBE parameters
            print(f"Warning: INCAR file '{incar_file}' not found. Using default PBE parameters.")
            vasp_params = {
                'xc': 'PBE',
                'prec': 'Accurate',
                'encut': 520,
                'ediff': 1e-5,
                'ismear': 0,
                'sigma': 0.05
            }
        
        return Vasp(**vasp_params)

# Structure analysis and I/O utilities

def compute_sorted_structure_descriptor(atoms: Atoms, mobile_atoms: Optional[List[int]] = None, rcut: float = 4.0, nmax: int = 5, lmax: int = 5) -> np.ndarray:
    """Permutation-invariant structure descriptor with element-wise grouping.

    - Mobile atoms (indices in `mobile_atoms`) are included; all others are ignored.
    - For each chemical element present in atoms, collect SOAP rows of that element,
      sort them by L2 norm, then flatten and concatenate across elements.
    """
    # Get unique species from atoms
    symbols = np.array(atoms.get_chemical_symbols())
    species = sorted(set(symbols))
    
    # Build SOAP descriptor for all atoms
    from dscribe.descriptors import SOAP
    soap = SOAP(species=species, periodic=True, r_cut=rcut, n_max=nmax, l_max=lmax)
    per_atom = soap.create(atoms)

    # Determine mobile mask from mobile_atoms
    n_atoms = len(atoms)
    if mobile_atoms is not None:
        mobile_mask = np.zeros(n_atoms, dtype=bool)
        mobile_mask[np.array(mobile_atoms, dtype=int)] = True
    else:
        mobile_mask = np.ones(n_atoms, dtype=bool)

    # Build descriptor grouped by element type
    blocks = []
    for elem in species:
        elem_mask = (symbols == elem) & mobile_mask
        if not np.any(elem_mask):
            continue
        elem_desc = per_atom[elem_mask]
        norms = np.linalg.norm(elem_desc, axis=1)
        order = np.argsort(norms)
        sorted_rows = elem_desc[order]
        blocks.append(sorted_rows.flatten())

    if not blocks:
        # No mobile atoms; return empty descriptor
        return np.array([])

    return np.concatenate(blocks)

def is_duplicate_by_desc_and_energy(new_atoms: Atoms,
                                    pool: List[Atoms],
                                    #species: List[str],
                                    tol: float = 0.1,
                                    energy: Optional[float] = None,
                                    pool_energies: Optional[List[float]] = None,
                                    energy_tol: float = 1,
                                    mobile_atoms: Optional[List[int]] = None) -> bool:
    """
    Duplicate check combining permutation-invariant descriptor and energy gating.
    - Find closest structure in pool by descriptor distance.
    - If closest distance < tol, then require |ΔE| <= energy_tol (if energies provided).
    """
    if not pool:
        return False
    desc_new = compute_sorted_structure_descriptor(new_atoms, mobile_atoms=mobile_atoms)
    best_idx = -1
    best_dist = float('inf')
    for i, atoms in enumerate(pool):
        # Same mobile_atoms set applies to all structures in this run
        desc_old = compute_sorted_structure_descriptor(atoms, mobile_atoms=mobile_atoms)
        d = np.linalg.norm(desc_new - desc_old)
        if d < best_dist:
            best_dist = d
            best_idx = i
    if best_dist >= tol:
        return False
    if energy is not None and pool_energies is not None and 0 <= best_idx < len(pool_energies):
        return abs(energy - pool_energies[best_idx]) <= energy_tol
    return True

# --- Geometry and Mobile Utilities ---

def infer_geometry_type(atoms: Atoms, vacuum_threshold_angstrom: float = 3.0):
    """
    Infer geometry type based on absolute vacuum margins (Å) along each lattice axis.
    Returns a tuple: (geometry_type, vacuum_axes).
    vacuum_axes lists axes with significant vacuum on both sides.
    """
    pos = atoms.get_positions()  # Cartesian positions within the cell
    cell = atoms.get_cell()
    axis_lengths = np.array([np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])])
    min_abs = pos.min(axis=0)
    max_abs = pos.max(axis=0)
    margin_low_abs = min_abs
    margin_high_abs = axis_lengths - max_abs
    axes = []
    for i in range(3):
        if margin_low_abs[i] > vacuum_threshold_angstrom and margin_high_abs[i] > vacuum_threshold_angstrom:
            axes.append(i)
    n = len(axes)
    if n == 3:
        geom = 'cluster'
    elif n == 2:
        geom = 'wire'
    elif n == 1:
        geom = 'slab'
    else:
        geom = 'bulk'
    return geom, axes

def get_mobile_atoms(atoms: Atoms, mobile_region) -> np.ndarray:
    """
    Compute boolean mask for mobile atoms.
    True = mobile, False = immobile
    """
    n_atoms = len(atoms)
    if mobile_region is None:
        return np.ones(n_atoms, dtype=bool)

    positions = atoms.get_positions()
    mask = np.zeros(n_atoms, dtype=bool)
    if mobile_region['type'] == 'sphere':
        center = np.array(mobile_region['center'])
        radius = mobile_region['radius']
        distances = np.linalg.norm(positions - center, axis=1)
        mask = distances <= radius
    elif mobile_region['type'] == 'slab':
        normal = np.array(mobile_region['normal'])
        normal = normal / (np.linalg.norm(normal) or 1.0)
        origin = np.array(mobile_region['origin'])
        min_dist = mobile_region['min_dist']
        max_dist = mobile_region['max_dist']
        vectors = positions - origin
        distances = np.dot(vectors, normal)
        mask = (distances >= min_dist) & (distances <= max_dist)
    elif mobile_region['type'] in ('lower', 'upper'):
        axis = mobile_region['axis'].lower()
        threshold = mobile_region['threshold']
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")
        axis_index = axis_map[axis]
        coords = positions[:, axis_index]
        if mobile_region['type'] == 'lower':
            mask = coords <= threshold
        elif mobile_region['type'] == 'upper':
            mask = coords >= threshold
    return np.where(mask)[0].tolist()

def periodic_distance(
    atoms1: Atoms,
    atoms2: Atoms,
    N: Optional[np.ndarray] = None
) -> Tuple[float, Optional[float]]:
    from typing import Optional, Tuple
    """
    Compute the Euclidean distance between two periodic atomic structures,
    taking into account the minimum image convention (MIC).

    Optionally, if a reference direction vector N is provided (shape: (3, n_atoms)),
    also compute the angle (in degrees) between the actual displacement and N.

    Requirements:
    - atoms1 and atoms2 must have the same number of atoms;
    - they must share identical unit cells and PBC settings;
    - atoms must be in the same order.

    Parameters:
    atoms1, atoms2 : ASE Atoms objects
    N : optional array of shape (3, n_atoms)
        Reference displacement direction for each atom (in Cartesian coordinates).
        If provided, the angle between the actual displacement and N is returned.

    Returns:
    distance : float
        L2 norm of the MIC-corrected displacement vector (sqrt(sum |dr|^2)).
    angle : float or None
        Angle in degrees between the displacement vector and N (flattened to 3N-D),
        or None if N is not provided.
    """
    # Validate inputs
    if len(atoms1) != len(atoms2):
        raise ValueError("atoms1 and atoms2 must have the same number of atoms.")
    
    if not np.allclose(atoms1.cell, atoms2.cell, atol=1e-6):
        raise ValueError("The unit cells of atoms1 and atoms2 must be identical.")
    
    if not np.array_equal(atoms1.pbc, atoms2.pbc):
        raise ValueError("PBC settings must be identical.")

    # Get fractional coordinates and compute MIC-corrected displacement
    frac1 = atoms1.get_scaled_positions()
    frac2 = atoms2.get_scaled_positions()
    df = frac2 - frac1
    pbc = atoms1.pbc
    if np.any(pbc):
        df[:, pbc] -= np.round(df[:, pbc])

    # Convert to Cartesian displacement
    cell = atoms1.cell
    dr_cart = df @ cell  # Shape: (n_atoms, 3)

    # Total Euclidean distance
    distance = np.linalg.norm(dr_cart)

    # Handle optional direction vector N
    if N is None:
        return distance, None

    # Flatten both vectors into 3N-dimensional vectors for angle computation
    # Note: dr_cart is (n_atoms, 3) → reshape to (3*n_atoms,)
    dr = dr_cart.flatten()

    # Normalize vectors (avoid division by zero)
    norm_dr = np.linalg.norm(dr)
    norm_N = np.linalg.norm(N)

    if norm_dr == 0 or norm_N == 0:
        raise ValueError("Both displacement vector dr and direction vector N must be non-zero.")
    else:
        # Clamp dot product to [-1, 1] to avoid numerical errors in arccos
        cos_angle = np.dot(dr, N) / (norm_dr * norm_N)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.rad2deg(np.arccos(cos_angle))

    return distance, angle    


class DeepMDCalculatorWithAtomicEnergy(Calculator):
    """Wrapper for DeepMD calculator to enable per-atom energy calculation.
    
    DeepMD's official Python calculator does not expose per-atom energies by default.
    This wrapper extends the standard DP calculator to retrieve atomic energies.
    
    Reference: https://zhuanlan.zhihu.com/p/457374515
    
    Usage:
        calc = DeepMDCalculatorWithAtomicEnergy(model_path='model.pb')
        atoms.calc = calc
        atomic_energies = atoms.get_potential_energies()
    """
    
    name = "DP_AtomicEnergy"
    implemented_properties = ['energy', 'free_energy', 'forces', 'virial', 'stress', 'energies']
    
    def __init__(self, model: str, label: str = "DP_AtomicEnergy", type_dict: Optional[dict] = None, **kwargs) -> None:
        """Initialize DeepMD calculator with atomic energy support.
        
        Args:
            model: Path to the DeepMD model file (.pb)
            label: Calculator label
            type_dict: Mapping of element types and their numbers (optional)
        """
        from deepmd.infer import DeepPot
        from pathlib import Path
        
        Calculator.__init__(self, label=label, **kwargs)
        self.dp = DeepPot(str(Path(model).resolve()))
        
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
            )
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Calculate energy, forces, virial, stress, and per-atom energies."""
        from ase.calculators.calculator import PropertyNotImplementedError
        
        if atoms is not None:
            self.atoms = atoms.copy()
        
        # Prepare input
        coord = self.atoms.get_positions().reshape([1, -1])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        
        # Get fparam and aparam from atoms.info if available
        fparam = self.atoms.info.get('fparam', None)
        aparam = self.atoms.info.get('aparam', None)
        
        # Call DeepPot model to get all properties including atomic energies
        # DeepPot.eval returns: (energy, forces, virial, atomic_energy, atomic_virial)
        e, f, v, atomic_e, _ = self.dp.eval(
            coords=coord, 
            cells=cell, 
            atom_types=atype, 
            fparam=fparam, 
            aparam=aparam,
            atomic=True
        )
        
        # Store standard properties
        self.results['energy'] = e[0][0]
        self.results['free_energy'] = e[0][0]
        self.results['forces'] = f[0]
        self.results['virial'] = v[0].reshape(3, 3)
        
        # Store per-atom energies
        self.results['energies'] = atomic_e[0]
        
        # Convert virial into stress for lattice relaxation
        if cell is not None:
            # Stress = -virial / volume (tensile stress is positive)
            stress = -0.5 * (v[0].copy() + v[0].copy().T) / self.atoms.get_volume()
            # Voigt notation
            self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        elif 'stress' in properties:
            raise PropertyNotImplementedError
    
    def get_potential_energies(self, atoms=None):
        """Return per-atom energies."""
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get('energies', np.zeros(len(self.atoms)))

def print_xyz(atoms, filename, energy, bias_energy, extra_info=None, *args, **kwargs):
    """
    Output xyz structure of atoms, allowing arbitrary number of atom-related parameters

    Parameters:
        atoms: ASE Atoms object
        filename: Optional, output file path. If None, print to console
        energy: Total energy of the structure
        bias_energy: Bias potential energy (0 if on real surface)
        extra_info: Optional dict of additional info fields to write into atoms.info
            e.g., {"scheme": 2, "modes": "atomic", "gaussian_n": 5}
        *args: Optional, list of atom-related parameters, each element is a tuple (list, title)
            where list is the atom parameter array, title is the parameter name
        **kwargs: Optional, keyword argument form of atom-related parameters
            e.g., energy=[...], forces=[...]
    """
    from ase.io import write
    import os
    if not os.path.exists("xyz"):
        os.makedirs("xyz")
    if filename is None:
        raise ValueError("filename must be provided when writing to file")

    n_atoms = len(atoms)
    # args
    params = []
    for arg in args:
        if isinstance(arg, tuple) and len(arg) == 2:
            param_list, param_title = arg
            if len(param_list) != n_atoms:
                raise ValueError(f"{param_title} must have n_atoms elements")
            params.append((param_list, param_title))
    # kwargs
    for key, value in kwargs.items():
        if len(value) != n_atoms:
            print("error:", key, value,len(value), n_atoms)
            raise ValueError(f"{key} must have n_atoms elements")
        params.append((value, key))
    
    atoms_copy = atoms.copy()
    atoms_copy.info['energy'] = energy
    atoms_copy.info['bias_energy'] = bias_energy
    # Write extra info fields (scheme, modes, gaussian number, etc.)
    if extra_info is not None:
        for key, value in extra_info.items():
            atoms_copy.info[key] = value
    
    if params:
        for param_list, param_title in params:
            atoms_copy.info['name'] = param_title
            atoms_copy.arrays['forces'] = param_list
            write("xyz/" + filename, atoms_copy, append=True)
    else:
        write("xyz/" + filename, atoms_copy, append=True)

def get_displace(N_in, mobile_mask, n_mobile, average_dr, max_dr):
    # get displacement vector for mobile atoms
    dist = N_in.reshape(-1, 3)
    #return dist

    dr = np.linalg.norm(dist, axis=1)
    mobile_dr = dr.copy()
    for i in range(len(mobile_dr)):
        if not mobile_mask[i]:
            mobile_dr[i] = 0
    
    actual_average_dr = np.sum(mobile_dr) / n_mobile
    
    scale_factor = average_dr / actual_average_dr
    dist *= scale_factor
    
    # Check maximum displacement and scale if needed
    actual_max_dr = np.max(mobile_dr*scale_factor)
    if actual_max_dr > max_dr:
        scale_factor = max_dr / actual_max_dr
        dist *= scale_factor

    #print("dist:  :",dist,np.linalg.norm(dist))
    return dist

def calc_average_max_displace(dist, mobile_mask, n_mobile):
    # get average maximum displacement vector for mobile atoms

    dr = np.linalg.norm(dist, axis=1)
    mobile_dr = dr.copy()
    for i in range(len(mobile_dr)):
        if not mobile_mask[i]:
            mobile_dr[i] = 0
    actual_average_dr = np.sum(mobile_dr) / n_mobile
    actual_max_dr = np.max(mobile_dr)
    return actual_average_dr, actual_max_dr
