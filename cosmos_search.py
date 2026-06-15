# grasmos_search.py  —  graph-aware Monte Carlo structure search
# Based on Shang & Liu 2013 (SSW) with graph-theoretic collective moves.
# Reference 1: Shang, R., & Liu, J. (2013). J. Chem. Phys. 139(24), 244104.
# Reference 2: J. Chem. Theory Comput. 2012, 8, 2215
import os,sys
import numpy as np
from ase.io import write as ase_write
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from bias_calculator import BiasCalculator
from cosmos_utils import is_duplicate_by_desc_and_energy, periodic_distance, print_xyz, get_displace, calc_average_max_displace
class GraSMoSSearch:
    def __init__(
        self,
        task,                # Task type (e.g., 'global_search','structure_sampling')
        structure_info,      # Dict with 'atoms', 'geometry_type', 'vacuum_axes'
        calculator,          # ASE calculator object
        atomic_calculator,   # ASE calculator object for atomic energy calculation
        monte_carlo,         # Dict with 'steps', 'temperature'
        random_direction,    # Dict with 'mode', 'element_weights', 'atomic_calculator'
        gaussian,            # Dict with 'gaussian_height', 'gaussian_width', 'max_gaussians'
        displace,            # Dict with 'average_dr', 'max_dr'
        optimizer,           # Dict with 'max_steps', 'fmax'
        mobile_control,    # Dict with 'mobile_atoms', 'mobile_region', 'wall_strength', 'wall_offset'
        output,          # Output directory
        **kwargs             # Additional parameters
    ):
        self.task = task     # Task type (e.g., 'global_search','structure_sampling')
        # Extract structure info
        self.atoms = structure_info['atoms'].copy()
        self.geometry_type = structure_info['geometry_type']
        self.vacuum_axes = structure_info['vacuum_axes']
        self.n_atoms = len(self.atoms)

        # Monte Carlo parameters
        self.temperature = monte_carlo['temperature']
        self.kB = 8.617333262e-5  # Boltzmann constant (eV/K)
        
        # Random direction parameters
        self.rd_mode = random_direction['mode']
        self.n_rd_mode = len(self.rd_mode)
        self.rd_ratio= random_direction['ratio']
        self.n_rd_scheme = len(self.rd_ratio)
        self.rd_ratio_scheme = [self.rd_ratio[i][1] for i in range(self.n_rd_scheme)]
        self.rd_ratio_mode  =[self.rd_ratio[i][0] for i in range(self.n_rd_scheme)]
        self.element_weights = random_direction['element_weights']
        self.element_scales = np.repeat([self.element_weights.get(symbol, 1.0) for symbol in self.atoms.symbols], 3)  #[1,2,3] -> [1,1,1,2,2,2,3,3,3]
        self.quadra_a = random_direction['quadra_param']
        self.AE_factor = random_direction['AE_factor']
        self.direction_weights = np.tile(random_direction['direction_weights'], self.n_atoms)  # [1,1,0,1,1,0,1,1,0] repeat n_atom times

        # ── Adaptive mode weighting ──
        self.adaptive = random_direction.get('adaptive', False)
        if self.adaptive:
            self.adaptive_interval = random_direction.get('adaptive_interval', 10)
            self.adaptive_alpha = random_direction.get('adaptive_alpha', 2.0)
            self.adaptive_floor = random_direction.get('adaptive_floor', 0.05)
            self.adaptive_smoothing = random_direction.get('adaptive_smoothing', 0.7)

            # Per-mode tracking: one dict per mode
            #  calls     — how many times this mode contributed to an MC step
            #  accepted  — how many times the step was Metropolis-accepted
            #  energy_drop  — sum of -ΔE (positive = improvement)
            #  new_minima — steps that found a new unique minimum
            self._mode_stats = {}
            self._lock_direction = False  # bond_switch skip-dimer flag
            for mode_name in self.rd_mode:
                self._mode_stats[mode_name] = {
                    'calls': 0, 'accepted': 0, 'energy_drop': 0.0,
                    'new_minima': 0
                }
            # Per-scheme tracking (for adapting scheme-level probabilities)
            self._scheme_stats = {}
            for s in range(self.n_rd_scheme):
                self._scheme_stats[s] = {
                    'calls': 0, 'accepted': 0, 'energy_drop': 0.0,
                    'new_minima': 0
                }
            self._initial_rd_ratio_scheme = list(self.rd_ratio_scheme)
            # Store initial ratios so we never drift too far from user intent
            self._initial_rd_ratio_mode = [list(r) for r in self.rd_ratio_mode]
            self._step_counter = 0
        # Per-scheme parameter overrides (always stored, empty dicts = use defaults)
        self._scheme_params = random_direction.get('scheme_params', [])
        # print(self.direction_weights)
        # Optimizer parameters in climbing with bias potential
        self.cl_opt_fmax=random_direction['climb_optimizer']['fmax']
        self.cl_opt_max_steps=random_direction['climb_optimizer']['max_steps']
        self.cl_relaxed_fmax=random_direction['climb_optimizer']['relaxed_fmax']
        self.cl_adaptive_relaxation=random_direction['climb_optimizer']['adaptive_relaxation']

        # Climbing parameters
        self.gaussian_height = gaussian['gaussian_height']   # Gaussian potential height
        self.max_gaussians = gaussian['max_gaussians']     # Max number of Gaussians
        self.average_dr = displace['average_dr']    # displace average step size parameter
        self.max_dr = displace['max_dr']     # displace max step size parameter
        
        # Optimizer parameters
        self.opt_fmax = optimizer['fmax']
        self.opt_max_steps = optimizer['max_steps']
        
        # Mobile control
        self.mobile_atoms = mobile_control['mobile_atoms']
        self.mobile_region = mobile_control['mobile_region']
        self.wall_strength = mobile_control['wall_strength']
        self.wall_offset = mobile_control['wall_offset']
        
        # Output and debug
        self.output_dir = output['directory']
        self.output_xyz = output['rd_xyz']
        self.debug = output['debug']
        
        # Handle additional parameters
        self.additional_params = kwargs
        
        # Initialize minimum pool
        os.makedirs(self.output_dir, exist_ok=True)
        self.pool = []  # Stores all found minimum energy structures
        self.real_energies = []  # Stores corresponding structure energies
        
        # Compute mobile mask once at initialization
        # mobile_atoms is now always provided as a list from cosmos_run (never None)
        # mobile_atoms is always provided by cosmos_run; if missing, this is a configuration error
        if self.mobile_atoms is None:
            raise ValueError("mobile_control['mobile_atoms'] must be provided and cannot be None.\n"
                             "Please check mobile_control configuration in input.json.")

        self.mobile_mask = np.zeros(self.n_atoms, dtype=bool)
        self.mobile_mask[self.mobile_atoms] = True
        # Cache number of mobile atoms
        self.n_mobile = len(self.mobile_atoms)
        # Apply FixAtoms constraint to immobile atoms
        # This ensures local minimization respects mobile constraints
        fixed_indices = [i for i in range(self.n_atoms) if not self.mobile_mask[i]]
        if len(fixed_indices) > 0:
            constraint = FixAtoms(indices=fixed_indices)
            self.atoms.set_constraint(constraint)
        
        # Set up calculators
        self.base_calc = calculator
        self.atomic_calc = atomic_calculator
        self.bias_calc = BiasCalculator(
            base_calculator=self.base_calc,
            mobile_mask=self.mobile_mask,
            mobile_region=self.mobile_region,
            wall_strength=self.wall_strength,
            wall_offset=self.wall_offset
        )

        # Initial structure optimization (real potential)
        if len(self.vacuum_axes) > 0:
            self.atoms.set_positions(self._position_to_center(self.atoms))
        self.atoms.calc = self.base_calc
        self._local_minimize(self.atoms,self.opt_fmax,self.opt_max_steps)        
        self._add_to_pool(self.atoms)

    def _local_minimize(self, atoms, fmax, max_steps):
        """
        Perform local structure optimization using LBFGS algorithm
        Complies with GraSMoS algorithm documentation steps 4 and 6, using limited-memory BFGS optimizer for efficiency
        
        Parameters:
            atoms: Atomic structure to optimize
            calc: Calculator for optimization, defaults to base_calc
            fmax: Convergence force threshold, defaults to class instance's fmax
            max_steps: Maximum number of steps, defaults to class instance's max_steps
        """
        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator assigned.")
        # Use LBFGS optimizer with custom logging if debug mode
        if self.debug and isinstance(atoms.calc, BiasCalculator):    # mute
            # Debug mode: log energy components at each step
           
            class DebugLBFGS(LBFGS):
                def __init__(self, atoms, parent_search, **kwargs):
                    super().__init__(atoms, **kwargs)
                    self.parent_search = parent_search
                    self.step_count = 0
                
                def step(self, f=None):
                    result = super().step(f)
                    self.step_count += 1
                    
                    # Log energy components
                    if hasattr(self.atoms.calc, 'results') and 'energy_components' in self.atoms.calc.results:
                        E_total = self.atoms.get_potential_energy()
                        forces = self.atoms.get_forces()
                        fmax_current = (forces**2).sum(axis=1).max()**0.5
                        E_base, E_bias, E_wall = self.atoms.calc.results['energy_components']
                                                
                        print(f"  Step {self.step_count}: E_total = {E_total:.6f} eV, "
                                f"E_base = {E_base:.6f} eV, E_bias = {E_bias:.6f} eV, E_wall = {E_wall:.6f} eV, "
                                f"fmax = {fmax_current:.6f} eV/Å")
                    
                    return result
            print("Local minimization using LBFGS optimizer in debug mode")
            opt = DebugLBFGS(atoms, self, logfile=None)
        else:
            opt = LBFGS(atoms, logfile=None)

        if self.debug and isinstance(atoms.calc, BiasCalculator):
            atoms.get_potential_energy()
            print(f"Before opt  No.: {len(atoms.calc.gaussian_params)}  "
              f"E_base: {atoms.calc.results['E_base']:.3f} "
              f"F_base_max: {(atoms.calc.results['F_base']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_gauss: {atoms.calc.results['E_gaussian']:.3f}  "
              f"F_gauss_max: {(atoms.calc.results['F_gaussian']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_wall: {atoms.calc.results['E_wall']:.3f}  "
              f"F_wall_max: {(atoms.calc.results['F_wall']**2).sum(axis=1).max()**0.5:.3f}  "
              f"F_total_max: {(atoms.get_forces()**2).sum(axis=1).max()**0.5:.3f}  ")

        opt.run(fmax=fmax,steps=max_steps)
       
        # Output final energy components if available
        if self.debug and isinstance(atoms.calc, BiasCalculator):
            print(f"After opt   No.: {len(atoms.calc.gaussian_params)}  "
              f"E_base: {atoms.calc.results['E_base']:.3f} "
              f"F_base_max: {(atoms.calc.results['F_base']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_gauss: {atoms.calc.results['E_gaussian']:.3f}  "
              f"F_gauss_max: {(atoms.calc.results['F_gaussian']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_wall: {atoms.calc.results['E_wall']:.3f}  "
              f"F_wall_max: {(atoms.calc.results['F_wall']**2).sum(axis=1).max()**0.5:.3f}  "
              f"F_total_max: {(atoms.calc.results['forces']**2).sum(axis=1).max()**0.5:.3f}  "
              f"F_total_max: {(atoms.get_forces()**2).sum(axis=1).max()**0.5:.3f}  ")                      

    def _add_to_pool(self, atoms):
        """
        Add optimized structure to minima pool and save to file
        """
        # Ensure calculator is attached before getting energy
        if atoms.calc is None:
            atoms.calc = self.base_calc
        real_e = atoms.get_potential_energy()
        self.pool.append(atoms.copy())
        self.real_energies.append(real_e)
        
        # Append to the combined trajectory file instead of individual files
        atoms_copy = atoms.copy()
        atoms_copy.calc=self.atomic_calc
        try:
            atoms_copy.get_potential_energies()
        except:
            pass
        atoms_copy.info['E_real'] = real_e
        atoms_copy.info['minima_index'] = len(self.pool) - 1
        
        # Append mode: add structure to existing file
        trajectory_file = os.path.join(self.output_dir, 'all_minima.xyz')
        ase_write(trajectory_file, atoms_copy, append=True)
        
        print(f"Found new minimum #{len(self.pool) - 1}: E = {real_e:.6f} eV")

    def _get_atomic_energies(self, atoms):
        """
        Returns:
            np.ndarray: Per-atom energies
        """
        # User explicitly specified calculator for atomic energy (already loaded)
        atoms_temp = atoms.copy()
        atoms_temp.calc = self.atomic_calc
        
        try:
            atomic_energies = atoms_temp.get_potential_energies()
            return atomic_energies
        except (AttributeError, NotImplementedError, RuntimeError) as e:
            raise RuntimeError(
                f"User-specified atomic energy calculator failed to compute per-atom energies: {e}\n"
                f"Please check 'random_direction.atomic_calc' configuration in input.json.")
    
    def _get_energy_based_scales(self, atoms):
        """
        Calculate energy-based scales for each atom to guide random direction generation.
        Atoms with higher energy (less stable) get larger scales for Ns components.
        
        Returns:
            np.ndarray: Per-atom scales based on exp(normalized_energy)
        """
        atomic_energies = self._get_atomic_energies(atoms)
        
        # Get chemical symbols and find minimum energy for each element as reference
        # Only consider mobile atoms for energy comparison
        symbols = atoms.get_chemical_symbols()
        unique_elements = list(set(symbols))
        
        # Use pre-computed mobile_atoms from initialization
        mobile_indices = self.mobile_atoms
        
        # Calculate reference energies (minimum energy for each element among mobile atoms)
        reference_energies = {}
        for element in unique_elements:
            # Find mobile atoms of this element
            element_mobile_indices = [i for i in mobile_indices if symbols[i] == element]
            if len(element_mobile_indices) > 0:
                element_energies = atomic_energies[element_mobile_indices]
                reference_energies[element] = np.max(element_energies)
            else:
                # No mobile atoms of this element, use 0 as reference
                reference_energies[element] = 10000.0
        
        # Normalize energies relative to element-specific references
        # This ensures that high-energy atoms get larger scales
        normalized_energies = np.zeros(self.n_atoms)
        for i, symbol in enumerate(symbols):
            temp_atomic_energy=atomic_energies[i]
            if isinstance(temp_atomic_energy, list) or isinstance(temp_atomic_energy, np.ndarray):
                temp_atomic_energy=temp_atomic_energy[0]
            normalized_energies[i] = temp_atomic_energy - reference_energies[symbol]  # negative
        
        # Calculate scales using exp(E_atom)
        # Add small offset to avoid issues with very small energies
        scales = np.exp(normalized_energies * self.AE_factor)   # 
        # Set scales of masked (immobile) atoms to 0
        if self.mobile_mask is not None:
            scales[~self.mobile_mask] = 0.0
            # Normalize scales to have reasonable magnitude (mean = 1)
            # Only use mobile atoms for normalization
            mean_scale = np.mean(scales[mobile_indices])
            if mean_scale > 0:
                scales = scales / mean_scale

        #self._print_mobile(scales=scales,energies=atomic_energies,normalized_energies=normalized_energies)
        return scales

    def _get_energy_factors(self, atoms):
        """
        Compute per-atom energy factors in [0.5, 1.0] for graph-based methods.

        Higher factors correspond to less-stable (higher-energy) atoms.
        Edges incident on high-energy atoms get stronger weights, biasing
        community detection and Laplacian analysis toward "hot" regions.

        Returns:
            np.ndarray of shape (n_atoms,) with values in [0.5, 1.0], or
            None if no atomic-energy calculator is available.
        """
        if self.atomic_calc is None:
            return None
        try:
            energies = self._get_atomic_energies(atoms)
        except RuntimeError:
            return None
        if energies is None or len(energies) == 0:
            return None
        e_min = np.min(energies)
        e_max = np.max(energies)
        if e_max - e_min < 1e-10:
            return np.ones(self.n_atoms)
        normalized = (energies - e_min) / (e_max - e_min)
        return 0.5 + 0.5 * normalized   # [0.5, 1.0]

    def _generate_random_direction(self, atoms):
        """
        Generate random search direction, combining global soft movement and local rigid movement.
        Uses energy-based sampling: atoms with higher energy get larger random components.
        Complies with GraSMoS algorithm step 1: Generate initial random direction N⁰
        
        Returns:
            N: Normalized random direction vector
        """
        scheme=np.random.choice(np.arange(self.n_rd_scheme),p=self.rd_ratio_scheme)
        print(f"Random direction scheme {scheme} with modes {self.rd_mode} and weights {self.rd_ratio_mode[scheme]}")
        modes=self.rd_mode
        ratio_mode=self.rd_ratio_mode[scheme]

        # ── Record which scheme was selected (for adaptive tracking) ──
        if self.adaptive:
            self._last_scheme = scheme
            # Only lock dimer for bond_switch-dominated schemes (weight ≥ 0.5)
            bs_idx = modes.index('bond_switch') if 'bond_switch' in modes else -1
            if bs_idx >= 0 and ratio_mode[bs_idx] >= 0.5:
                self._lock_direction = True

        N=np.zeros(3*self.n_atoms)
        for i,mode in enumerate(modes):
            if mode=='thermo':
                N_temp=self._generate_rd_thermo(atoms)
            elif mode=='atomic':
                N_temp=self._generate_rd_atomic(atoms)
            elif mode=='nl':
                N_temp=self._generate_rd_nl(atoms)
            elif mode=='atnl':
                N_temp=self._generate_rd_atnl(atoms)
            elif mode=='python':
                N_temp=self._generate_rd_python(atoms)
            elif mode=='bond_rotation':
                N_temp=self._generate_rd_bond_rotation(atoms)
            elif mode=='bond_switch':
                N_temp=self._generate_rd_bond_switch(atoms)
            elif mode=='shell':
                N_temp=self._generate_rd_shell(atoms)
            elif mode=='community':
                N_temp=self._generate_rd_community(atoms)
            elif mode=='laplacian':
                N_temp=self._generate_rd_laplacian(atoms)
            else:
                raise ValueError(f"Unknown random direction mode: {mode}")
            N+=self._normalize(N_temp)*ratio_mode[i]
            if self.debug:
                print(f"Mode: {mode}, Ratio: {ratio_mode[i]:.3f}, |N_temp| : {np.linalg.norm(N_temp):.3f}")

        N= self.element_scales * self.direction_weights * N

        if self.geometry_type == 'cluster':
            return self._remove_rotation_and_translation(atoms.positions,N.reshape(-1,3)) # already normalized
        else:
            return self._remove_translation(atoms,N) # already normalized in _remove_translation
        
    def _generate_rd_thermo(self, atoms):
        """
        Generate random direction using thermodynamic method.
        """
        N = np.zeros(3 * self.n_atoms)        
        for i in range(self.n_atoms):
            if self.mobile_mask[i]:
                mass = atoms[i].mass 
                sigma = np.sqrt(self.kB * self.temperature / mass)
                N[3*i:3*i+3] = np.random.normal(0, sigma, 3)
        return N

    def _generate_rd_atomic(self, atoms):
        """
        Generate random direction using atomic energy method.
        """
        energy_scales = np.repeat(self._get_energy_based_scales(atoms), 3)  #[1,2,3] -> [1,1,1,2,2,2,3,3,3]
        N = energy_scales * self._generate_rd_thermo(atoms)
        return N

    def _generate_rd_atnl(self, atoms):  # atomic-energy based nl
        """
        Generate random direction using atomic energy-based method.
        Finds the two atoms with highest energy-based scales and sets their N vectors pointing to each other.
        Other atoms have zero N vectors.
        """
        # Get energy-based scales for each atom
        energy_scales = self._get_energy_based_scales(atoms)
        
        # Find the two atoms with highest energy scales
        # argsort returns indices in ascending order, so we take the last two
        sorted_indices = np.argsort(energy_scales)
        i = sorted_indices[-1]  # Index of atom with highest energy scale
        j = sorted_indices[-2]  # Index of atom with second highest energy scale
        
        # Initialize N vector with zeros
        N = np.zeros(3 * self.n_atoms)
        
        # Set direction vectors: i -> j and j -> i (similar to _generate_rd_nl)
        qi = atoms.positions[i].flatten()
        qj = atoms.positions[j].flatten()

        if np.random.random() < 0.7:
            # Attraction mode (70% probability): atoms point to each other
            N[3*i:3*i+3] = qj - qi  # Atom i points to atom j
            N[3*j:3*j+3] = qi - qj  # Atom j points to atom i
        else:
            # Repulsion mode (30% probability): atoms point away from each other
            N[3*i:3*i+3] = qi - qj  # Atom i points away from atom j
            N[3*j:3*j+3] = qj - qi  # Atom j points away from atom i
        
        return N

    def _generate_rd_nl(self, atoms):
        """
        Generate random direction using non-local bonding pattern method.
        """
        if self.n_mobile < 2:
            raise ValueError("Insufficient mobile atoms (need at least 2) for calculation.")
           
        max_attempts = 100
        attempts = 0
        N = np.zeros(3 * self.n_atoms)
        while attempts < max_attempts:
            # Select two non-neighboring atoms from mobile region
            i, j = np.random.choice(self.mobile_atoms, 2, replace=False)
            distance = atoms.get_distance(i, j, mic=True)
            if distance > 3.0:  # Only when atomic distance > 3Å
                # Generate local rigid movement direction according to equation (2) in paper
                # Nl = [qB - qA at position A, qA - qB at position B, 0, ...]
                qi = atoms.positions[i].flatten()
                qj = atoms.positions[j].flatten()  
                N[3*i:3*i+3] = qj - qi
                N[3*j:3*j+3] = qi - qj
                break
            attempts += 1
        
        if attempts == max_attempts:
            raise ValueError(f"Failed to find atom pair with distance > 3Å after {max_attempts} attempts, cannot generate local rigid movement direction.")

        return N

    def _get_bond_pairs(self, atoms):
        """
        Identify bonded atom pairs using covalent radii.
        Returns list of (i, j) tuples where i < j and both are mobile.
        """
        from ase.data import covalent_radii
        positions = atoms.positions
        numbers = atoms.get_atomic_numbers()
        n_atoms = len(atoms)
        bonds = []
        # Bond tolerance factor: 1.25 * (r_cov_i + r_cov_j)
        for i in range(n_atoms):
            if not self.mobile_mask[i]:
                continue
            for j in range(i + 1, n_atoms):
                if not self.mobile_mask[j]:
                    continue
                ri = covalent_radii[numbers[i]]
                rj = covalent_radii[numbers[j]]
                cutoff = 1.25 * (ri + rj)
                dist = atoms.get_distance(i, j, mic=True)
                if dist < cutoff:
                    bonds.append((i, j))
        return bonds

    def _build_weighted_adjacency(self, atoms, energy_factors=None):
        """
        Build a weighted adjacency matrix from the atomic structure using
        covalent-radii-based exponential weights.

        Edge weight = exp(-distance / (0.5 * r_cov_sum)), which is ~1.0 at
        zero separation, ~0.37 at the covalent sum, and ~0.0 at 2x covalent.
        Weights below 0.01 are dropped (sparsification).

        If *energy_factors* is provided (shape (n_atoms,) with values in
        [0.5, 1.0]), each edge weight is further multiplied by
        min(factor_i, factor_j).  This biases the graph toward high-energy
        (less stable) regions, focusing community detection and Laplacian
        analysis where structural changes are most needed.

        Returns:
            np.ndarray: (n_atoms, n_atoms) symmetric adjacency matrix.
            Only mobile atoms have non-zero entries.
        """
        from ase.data import covalent_radii

        n = self.n_atoms
        positions = atoms.positions
        numbers = atoms.get_atomic_numbers()
        A = np.zeros((n, n), dtype=float)
        use_energy = energy_factors is not None

        for i in range(n):
            if not self.mobile_mask[i]:
                continue
            for j in range(i + 1, n):
                if not self.mobile_mask[j]:
                    continue
                dist = atoms.get_distance(i, j, mic=True)
                r_cov = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
                if r_cov <= 0:
                    continue
                weight = np.exp(-dist / (0.5 * r_cov))
                if weight > 0.01:
                    if use_energy:
                        weight *= min(energy_factors[i], energy_factors[j])
                    A[i, j] = A[j, i] = weight

        return A

    def _get_local_coordination(self, atoms, energy_factors=None):
        """
        Compute per-atom coordination numbers from the bond graph.

        Coordination is defined as the number of neighbours with adjacency
        weight above 0.01.  Immobile atoms are assigned coordination 0.

        Returns:
            np.ndarray of shape (n_atoms,) with integer coordination counts.
        """
        A = self._build_weighted_adjacency(atoms, energy_factors=energy_factors)
        coord = np.zeros(self.n_atoms, dtype=int)
        for i in range(self.n_atoms):
            if self.mobile_mask[i]:
                coord[i] = int(np.sum(A[i] > 0.01))
        return coord

    def _filter_switchable_bonds(self, atoms, min_coordination=2):
        """
        Identify bonds eligible for the bond_switch collective move.

        A bond (i, j) is switchable if:
        - Both i and j are mobile.
        - Each has coordination ≥ *min_coordination* (default 2), ensuring
          there is at least one other neighbour to re-bond with after switching.
        - Each has at least one neighbour other than each other (prevents
          picking isolated dimers).

        When atomic energies are available, the bond list is energy-biased:
        bonds incident on high-energy atoms appear more frequently, focusing
        topological rearrangement where it is most needed.

        Parameters:
            atoms: ASE Atoms object.
            min_coordination: Minimum coordination of both endpoints.

        Returns:
            list[tuple[int, int]]: Eligible bond pairs (i, j) with i < j.
        """
        energy_factors = self._get_energy_factors(atoms)
        coord = self._get_local_coordination(atoms, energy_factors=energy_factors)
        A = self._build_weighted_adjacency(atoms, energy_factors=energy_factors)

        # Helper: does atom k have a neighbour other than exclude_idx?
        def has_other_neighbour(k, exclude_idx):
            for m in range(self.n_atoms):
                if m != exclude_idx and self.mobile_mask[m] and A[k, m] > 0.01:
                    return True
            return False

        candidates = []
        candidate_weights = []

        for i in range(self.n_atoms):
            if not self.mobile_mask[i] or coord[i] < min_coordination:
                continue
            for j in range(i + 1, self.n_atoms):
                if not self.mobile_mask[j] or coord[j] < min_coordination:
                    continue
                if A[i, j] <= 0.01:
                    continue
                if not has_other_neighbour(i, j):
                    continue
                if not has_other_neighbour(j, i):
                    continue
                candidates.append((i, j))
                # Energy bias: weight ∝ max(factor_i, factor_j)
                if energy_factors is not None:
                    candidate_weights.append(
                        max(energy_factors[i], energy_factors[j]))
                else:
                    candidate_weights.append(1.0)

        # If energy is available, use weighted sampling later; just return
        # the list — the caller handles selection.
        # Store weights as an attribute so _generate_rd_bond_switch can use them.
        self._switchable_bond_weights = (
            np.array(candidate_weights, dtype=float)
            if candidate_weights else np.array([])
        )
        return candidates

    def _generate_rd_bond_rotation(self, atoms):
        """
        Generate random direction via coordinated bond rotation (generalized GSW-like move).

        Picks a bonded pair and rotates them about an axis perpendicular to the bond,
        creating a twisting motion that preserves bond topology better than single-atom moves.
        Also includes the first-neighbor shell of each bonded atom in the rotation.

        This is the generalized equivalent of the Stone-Wales bond rotation in sp2 carbon,
        but works for any bonding environment.
        """
        from ase.data import covalent_radii
        bonds = self._get_bond_pairs(atoms)
        N = np.zeros(3 * self.n_atoms)

        if len(bonds) == 0:
            # Fall back to thermo mode if no bonds found
            return self._generate_rd_thermo(atoms)

        # Randomly pick a bond
        i, j = bonds[np.random.choice(len(bonds))]
        qi = atoms.positions[i]
        qj = atoms.positions[j]

        # Bond vector and midpoint
        bond_vec = qj - qi
        bond_len = np.linalg.norm(bond_vec)
        if bond_len < 1e-6:
            return self._generate_rd_thermo(atoms)
        bond_dir = bond_vec / bond_len

        # Generate a random rotation axis perpendicular to the bond
        # Pick a random vector and orthogonalize to bond_dir
        rand_vec = np.random.randn(3)
        rand_vec -= np.dot(rand_vec, bond_dir) * bond_dir
        rot_norm = np.linalg.norm(rand_vec)
        if rot_norm < 1e-6:
            return self._generate_rd_thermo(atoms)
        rot_axis = rand_vec / rot_norm

        # Tangential direction = rot_axis × bond_dir (direction of rotation)
        tang_dir = np.cross(rot_axis, bond_dir)

        # Rotation angle (random, ~30 degrees scaled to temperature)
        angle = np.random.uniform(0.3, 1.0)  # radians

        # Define group: atoms i, j and their bonded neighbors
        group = {i, j}
        numbers = atoms.get_atomic_numbers()
        for k in range(self.n_atoms):
            if not self.mobile_mask[k]:
                continue
            dik = atoms.get_distance(i, k, mic=True)
            djk = atoms.get_distance(j, k, mic=True)
            # Include if bonded to either i or j
            ri = covalent_radii[numbers[i]]
            rk = covalent_radii[numbers[k]]
            rj = covalent_radii[numbers[j]]
            if dik < 1.25 * (ri + rk) or djk < 1.25 * (rj + rk):
                group.add(k)

        # Apply rotation: each atom in group gets tangential displacement
        # proportional to its distance from the bond center
        midpoint = (qi + qj) / 2.0
        for idx in group:
            if not self.mobile_mask[idx]:
                continue
            pos = atoms.positions[idx]
            # Distance from rotation axis (the line through midpoint along rot_axis)
            to_mid = pos - midpoint
            # Project onto plane perpendicular to rot_axis
            radial_dist = np.linalg.norm(to_mid - np.dot(to_mid, rot_axis) * rot_axis)
            scale = max(radial_dist, 0.5) * angle  # At least some minimal displacement
            N[3*idx:3*idx+3] = tang_dir * scale

        return N

    def _generate_rd_bond_switch(self, atoms):
        """
        Generalised bond-switching collective Monte Carlo move.

        Selects a bond in the bond graph whose two endpoints each have at
        least two neighbours, then applies a large-angle (~30°-90°) relative
        rotation of the two local fragments about an axis perpendicular to
        the bond.  The first-neighbour shell of each endpoint follows its
        parent atom, producing a coordinated fragment twist.

        When per-atom energies are available, candidate bonds are sampled
        with a bias toward high-energy (less-stable) atoms, so topological
        rearrangement targets the regions that need it most.

        This is the generalisation of the Stone-Wales transformation for
        arbitrary bonding environments:
        - sp² carbon: the dimer rotation naturally finds the 90° reaction
          coordinate that interconverts 6-6-6-6 ↔ 5-7-7-5.
        - Tetrahedral semiconductors: ~70° switches between staggered and
          eclipsed local motifs.
        - Oxides / chalcogenides: adapts to the local coordination polyhedron.
        - Metallic glasses: identifies shear-transformation zones.

        The subsequent dimer rotation + Gaussian climbing + Metropolis steps
        automatically fine-tune the angle and axis to the material's actual
        potential-energy surface, so the initial 30°-90° range only needs to
        be "close enough" for the dimer to lock onto the correct saddle.
        """
        candidates = self._filter_switchable_bonds(atoms, min_coordination=2)

        if not candidates:
            # Fall back to ordinary bond_rotation if no switchable bonds
            return self._generate_rd_bond_rotation(atoms)

        # ── Pick a candidate bond, optionally energy-biased ──
        w = getattr(self, '_switchable_bond_weights', np.array([]))
        if len(w) > 0 and w.sum() > 1e-12:
            probs = w / w.sum()
            idx_sel = np.random.choice(len(candidates), p=probs)
        else:
            idx_sel = np.random.choice(len(candidates))
        i, j = candidates[idx_sel]

        qi = atoms.positions[i]
        qj = atoms.positions[j]
        bond_vec = qj - qi
        bond_len = np.linalg.norm(bond_vec)
        if bond_len < 1e-6:
            return self._generate_rd_thermo(atoms)
        bond_dir = bond_vec / bond_len
        midpoint = (qi + qj) / 2.0

        # ── Random rotation axis perpendicular to the bond ──
        rand_vec = np.random.randn(3)
        rand_vec -= np.dot(rand_vec, bond_dir) * bond_dir
        rn = np.linalg.norm(rand_vec)
        if rn < 1e-6:
            return self._generate_rd_thermo(atoms)
        rot_axis = rand_vec / rn

        # ── Large rotation angle: uniform in [π/6, π/2] (~30°-90°) ──
        # Both fragments rotate in the same direction → rigid bond rotation,
        # preserving |i−j| at all angles (Rodrigues on the same axis).
        theta = np.random.uniform(np.pi / 6.0, np.pi / 2.0)
        # Random sign: +θ for both groups, or −θ for both
        sign = 1.0 if np.random.random() < 0.5 else -1.0

        # ── Build the two neighbour groups from the adjacency ──
        A = self._build_weighted_adjacency(
            atoms, energy_factors=self._get_energy_factors(atoms))
        group_i = {i}
        group_j = {j}
        for k in range(self.n_atoms):
            if not self.mobile_mask[k]:
                continue
            if A[i, k] > 0.01 and k != j:
                group_i.add(k)
            if A[j, k] > 0.01 and k != i:
                group_j.add(k)

        # ── Apply rotation via Rodrigues' formula ──
        # displacement = R(axis, ±θ)·v - v,  where v = r_k - midpoint
        # Angle is bounded at π/3 so bond atoms stay ≥ 0.7 Å apart.
        N = np.zeros(3 * self.n_atoms)

        def rotate_group(group_indices, angle):
            c = np.cos(angle)
            s = np.sin(angle)
            for idx in group_indices:
                v = atoms.positions[idx] - midpoint
                cross = np.cross(rot_axis, v)
                dot_ax = np.dot(rot_axis, v)
                v_rot = v * c + cross * s + rot_axis * (dot_ax * (1.0 - c))
                N[3 * idx:3 * idx + 3] = v_rot - v

        rotate_group(group_i, sign * theta)
        rotate_group(group_j, sign * theta)  # same direction — rigid bond rotation

        return N

    def _generate_rd_shell(self, atoms):
        """
        Generate random direction via coordination shell collective motion.

        Picks a random mobile atom as center, identifies its coordination shell
        (all atoms within a cutoff), and generates a coordinated displacement
        combining radial breathing and tangential shear components.
        """
        N = np.zeros(3 * self.n_atoms)
        n_mobile = len([i for i in range(self.n_atoms) if self.mobile_mask[i]])
        if n_mobile == 0:
            return N

        # Randomly pick a central atom from mobile atoms
        center = np.random.choice([i for i in range(self.n_atoms) if self.mobile_mask[i]])
        pos_center = atoms.positions[center]

        # Identify shell atoms: all mobile atoms within cutoff distance
        # Use a generous cutoff (3.5 Å) to capture coordination environment
        shell_cutoff = 3.5
        shell_indices = []
        shell_vectors = []  # position relative to center

        for i in range(self.n_atoms):
            if i == center or not self.mobile_mask[i]:
                continue
            dist = atoms.get_distance(center, i, mic=True)
            if dist < shell_cutoff:
                shell_indices.append(i)
                vec = atoms.positions[i] - pos_center
                shell_vectors.append(vec)

        if len(shell_indices) == 0:
            # No shell atoms found, fall back to thermo
            return self._generate_rd_thermo(atoms)

        # Randomly choose mode: breathing (40%), shear (40%), or mixed (20%)
        mode_choice = np.random.random()

        if mode_choice < 0.4:
            # Radial breathing: all shell atoms move toward or away from center
            sign = 1.0 if np.random.random() < 0.5 else -1.0
            for idx, vec in zip(shell_indices, shell_vectors):
                dist = np.linalg.norm(vec)
                if dist > 1e-6:
                    N[3*idx:3*idx+3] = sign * vec / dist
            # Center atom gets opposite displacement to preserve center of mass
            shell_disp = N.reshape(-1, 3)[shell_indices].sum(axis=0)
            N[3*center:3*center+3] = -shell_disp / max(len(shell_indices), 1)

        elif mode_choice < 0.8:
            # Tangential shear: all shell atoms move tangentially
            # Generate random rotation axis
            rand_axis = np.random.randn(3)
            rand_axis /= np.linalg.norm(rand_axis)
            for idx, vec in zip(shell_indices, shell_vectors):
                # Tangential direction = random_axis × radial_vector
                tang = np.cross(rand_axis, vec)
                tang_norm = np.linalg.norm(tang)
                if tang_norm > 1e-6:
                    N[3*idx:3*idx+3] = tang / tang_norm

        else:
            # Mixed: combine radial and tangential
            sign = 1.0 if np.random.random() < 0.5 else -1.0
            rand_axis = np.random.randn(3)
            rand_axis /= np.linalg.norm(rand_axis)
            for idx, vec in zip(shell_indices, shell_vectors):
                dist = np.linalg.norm(vec)
                if dist > 1e-6:
                    radial = sign * vec / dist
                    tang = np.cross(rand_axis, vec)
                    tang_norm_val = np.linalg.norm(tang)
                    if tang_norm_val > 1e-6:
                        tang = tang / tang_norm_val
                    else:
                        tang = np.zeros(3)
                    N[3*idx:3*idx+3] = 0.7 * radial + 0.7 * tang

        return N

    # ── Graph-theoretic collective moves ──────────────────────────────

    def _detect_communities(self, atoms, resolution=1.0):
        """
        Partition the bond graph into communities via the Louvain algorithm.

        Builds a weighted adjacency matrix (see _build_weighted_adjacency),
        converts it to a networkx graph, and runs Louvain community detection.
        Isolated nodes (no edges) are placed in singleton communities.

        Parameters:
            atoms: ASE Atoms object.
            resolution: Louvain resolution parameter (>1 = smaller communities).

        Returns:
            list[list[int]]: Each inner list is a community of atom indices.
        """
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        energy_factors = self._get_energy_factors(atoms)
        A = self._build_weighted_adjacency(atoms, energy_factors=energy_factors)
        # Map global atom indices to contiguous graph node ids, skipping
        # immobile and fully isolated atoms.
        mobile_indices = [i for i in range(self.n_atoms) if self.mobile_mask[i]]
        # Build graph only over mobile atoms that have at least one edge
        n_mobile = len(mobile_indices)
        idx_to_node = {}   # global atom index → graph node id
        node_to_idx = {}   # graph node id → global atom index
        G = nx.Graph()
        node_id = 0
        for idx in mobile_indices:
            idx_to_node[idx] = node_id
            node_to_idx[node_id] = idx
            G.add_node(node_id)
            node_id += 1

        for i in mobile_indices:
            u = idx_to_node[i]
            for j in mobile_indices:
                if j <= i:
                    continue
                w = A[i, j]
                if w > 0:
                    v = idx_to_node[j]
                    G.add_edge(u, v, weight=w)

        if G.number_of_edges() == 0:
            # No edges: every mobile atom is its own community
            return [[idx] for idx in mobile_indices]

        # Louvain community detection
        communities_raw = louvain_communities(G, weight='weight',
                                              resolution=resolution, seed=42)
        # Map back to global atom indices
        communities = []
        for comm in communities_raw:
            communities.append([node_to_idx[n] for n in comm])

        # Add any mobile atoms that were isolated (no edges) as singletons
        accounted = set()
        for c in communities:
            accounted.update(c)
        for idx in mobile_indices:
            if idx not in accounted:
                communities.append([idx])

        return communities

    def _generate_rd_community(self, atoms):
        """
        Community-based collective Monte Carlo move.

        Uses Louvain community detection on the bond graph to identify
        natural clusters (functional groups, coordination polyhedra, rings),
        then applies a coordinated displacement to one randomly chosen
        community. The move type is chosen randomly:

        - 40%: rigid translation of the cluster
        - 30%: rigid rotation about the cluster centre of mass
        - 30%: breathing (radial expansion or contraction)

        This preserves local bonding topology, dramatically reducing
        rejection from bond-breaking events.
        """
        communities = self._detect_communities(atoms)
        if len(communities) < 2:
            return self._generate_rd_thermo(atoms)

        # Pick a random community (preferring multi-atom ones for efficiency)
        weights = np.array([len(c) for c in communities], dtype=float)
        weights = np.maximum(weights - 1.0, 0.1)  # favour larger communities
        probs = weights / weights.sum()
        comm_idx = np.random.choice(len(communities), p=probs)
        comm = communities[comm_idx]

        if len(comm) < 2:
            return self._generate_rd_thermo(atoms)

        N = np.zeros(3 * self.n_atoms)
        mode_choice = np.random.random()

        if mode_choice < 0.4:
            # Rigid translation of the cluster
            direction = np.random.randn(3)
            direction /= (np.linalg.norm(direction) + 1e-12)
            for idx in comm:
                N[3 * idx:3 * idx + 3] = direction

        elif mode_choice < 0.7:
            # Rigid rotation about the cluster centre of mass
            com = atoms.positions[comm].mean(axis=0)
            axis = np.random.randn(3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            for idx in comm:
                r = atoms.positions[idx] - com
                N[3 * idx:3 * idx + 3] = np.cross(axis, r)

        else:
            # Radial breathing (expansion or contraction)
            com = atoms.positions[comm].mean(axis=0)
            sign = 1.0 if np.random.random() < 0.5 else -1.0
            for idx in comm:
                r = atoms.positions[idx] - com
                dist = np.linalg.norm(r)
                if dist > 1e-6:
                    N[3 * idx:3 * idx + 3] = sign * r / dist

        return N

    def _compute_laplacian_modes(self, atoms, n_modes=5):
        """
        Compute the lowest-frequency eigenmodes of the normalised graph
        Laplacian.  These encode collective deformation patterns that cost
        minimal energy in a harmonic approximation — the graph-theoretic
        analogue of soft phonon modes.

        Only mobile atoms contribute to the Laplacian; immobile atoms are
        excluded.

        Parameters:
            atoms: ASE Atoms object.
            n_modes: Number of low-frequency modes to return (default 5).

        Returns:
            list[np.ndarray]: Each element is a flattened (3 * n_atoms)
            direction vector for one Laplacian mode.  Returns empty list
            if the graph is trivial.
        """
        A = self._build_weighted_adjacency(atoms,
                                            energy_factors=self._get_energy_factors(atoms))
        n = A.shape[0]

        # Restrict to mobile atoms for the eigen-decomposition
        mobile_indices = np.where(self.mobile_mask)[0]
        n_mob = len(mobile_indices)
        if n_mob < 2:
            return []

        A_mob = A[np.ix_(mobile_indices, mobile_indices)]
        degrees = A_mob.sum(axis=1)
        if np.all(degrees < 1e-12):
            return []

        # Normalised Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
        L_norm = np.eye(n_mob) - d_inv_sqrt @ A_mob @ d_inv_sqrt
        L_norm = np.nan_to_num(L_norm, nan=0.0, posinf=0.0, neginf=0.0)

        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        modes = []
        max_k = min(n_modes + 1, n_mob)
        for k in range(1, max_k):  # skip eigenvalue 0 (trivial mode)
            vec_mob = eigenvectors[:, k]  # shape (n_mob,)
            # Expand to full (3 * n_atoms) space: each atom gets its scalar
            # multiplied by a random 3D direction to break isotropy.
            rand_dirs = np.random.randn(n, 3)
            rand_dirs /= (np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12)
            mode_full = np.zeros(3 * n)
            for local_i, global_i in enumerate(mobile_indices):
                mode_full[3 * global_i:3 * global_i + 3] = (
                    vec_mob[local_i] * rand_dirs[global_i]
                )
            modes.append(mode_full)

        return modes

    def _generate_rd_laplacian(self, atoms):
        """
        Generate a random direction from a graph-Laplacian soft mode.

        Randomly picks one of the lowest-frequency eigenmodes and returns
        it as the displacement direction.  Unlike random thermo/atomic
        directions that scatter energy across uncorrelated atomic motions
        (most of which are uphill), Laplacian modes concentrate the
        displacement into collective motions the bonding network can
        accommodate.

        When per-atom energies are available, mode selection is biased
        toward modes whose displacement magnitude overlaps strongly with
        high-energy (less stable) atoms, so structural changes target the
        regions that need them most.
        """
        modes = self._compute_laplacian_modes(atoms, n_modes=5)
        if not modes:
            return self._generate_rd_thermo(atoms)

        # ── Energy-biased mode selection ──
        energy_factors = self._get_energy_factors(atoms)
        if energy_factors is not None:
            # Score each mode by overlap with high-energy atoms:
            # score_k = Σ_i |vec_ki| · factor_i
            mode_scores = np.zeros(len(modes))
            for k, mode in enumerate(modes):
                mode_3d = mode.reshape(-1, 3)
                per_atom_mag = np.linalg.norm(mode_3d, axis=1)  # (n_atoms,)
                mode_scores[k] = np.dot(per_atom_mag, energy_factors)
            if mode_scores.sum() > 1e-12:
                probs = mode_scores / mode_scores.sum()
                N = modes[np.random.choice(len(modes), p=probs)]
            else:
                N = modes[np.random.choice(len(modes))]
        else:
            N = modes[np.random.choice(len(modes))]

        # Zero out immobile atoms (should already be zero, but be safe)
        for i in range(self.n_atoms):
            if not self.mobile_mask[i]:
                N[3 * i:3 * i + 3] = 0.0

        return N

    def _generate_rd_python(self, atoms):
        """
        Generate random direction using user-defined Python function.
        Loads generate_random_direction.py from working directory.
        
        Returns:
            N: Random direction vector from user function
        """
        
        # Get current working directory
        cwd = os.getcwd()
        script_path = os.path.join(cwd, 'generate_random_direction.py')
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Custom random direction script not found: {script_path}\n"
                f"When random_direction_mode='python', you must provide generate_random_direction.py "
                f"in the working directory with a function generate_random_direction(atoms) that returns N."
            )
        
        # Import user module
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_random_direction", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load {script_path}")
        
        user_module = importlib.util.module_from_spec(spec)
        sys.modules['user_random_direction'] = user_module
        spec.loader.exec_module(user_module)
        
        # Check if function exists
        if not hasattr(user_module, 'generate_random_direction'):
            raise AttributeError(
                f"Module {script_path} must define a function 'generate_random_direction(atoms)' that returns N."
            )
        
        # Call user function
        user_func = user_module.generate_random_direction
        N = user_func(atoms)
        
        # Validate output
        n_atoms = self.n_atoms
        expected_size = 3 * n_atoms
        if not isinstance(N, np.ndarray):
            N = np.array(N)
        
        if N.shape != (expected_size,):
            raise ValueError(
                f"User function generate_random_direction must return a 1D numpy array of size {expected_size} (3*n_atoms), "
                f"but got shape {N.shape}"
            )
        
        print(f"\n=== Random Direction Generation (Python Mode) ===")
        print(f"Loaded custom function from: {script_path}")
        print(f"Direction vector |N| = {np.linalg.norm(N):.6f}")
        print("="*60)
        
        return N

    # ── Adaptive mode weighting ─────────────────────────────────────────

    def _update_mode_stats(self, scheme, delta_E, metrop_accepted, is_new):
        """
        Update per-mode AND per-scheme performance counters after one MC step.

        All modes that contributed to the step (ratio > 0) receive equal
        fractional credit.  The selected scheme receives full credit.
        """
        if not self.adaptive:
            return
        ratio_mode = self.rd_ratio_mode[scheme]
        active = [i for i, w in enumerate(ratio_mode) if w > 1e-10]
        if not active:
            return
        share = 1.0 / len(active)
        improvement = max(0.0, -delta_E)  # positive = energy dropped
        for i in active:
            m = self.rd_mode[i]
            s = self._mode_stats[m]
            s['calls'] += 1
            if metrop_accepted:
                s['accepted'] += 1
            s['energy_drop'] += improvement * share
            if is_new:
                s['new_minima'] += share

        # Scheme-level credit (full credit to the selected scheme)
        if scheme in self._scheme_stats:
            ss = self._scheme_stats[scheme]
            ss['calls'] += 1
            if metrop_accepted:
                ss['accepted'] += 1
            ss['energy_drop'] += improvement
            if is_new:
                ss['new_minima'] += 1

    def _recompute_adaptive_weights(self):
        """
        Recompute scheme-level probabilities based on historical performance.

        Each scheme is a fixed strategy (mode weights are set by the user and
        never changed).  Adaptive weighting only shifts *which strategy* is
        sampled more often, leaving each strategy's internal mix intact.

        Scoring (same UCB-inspired formula as mode-level):
            score = accept_rate^α  ×  (avg_energy_drop + ε)

        Scheme probabilities are EMA-smoothed and protected by an
        exploration floor to prevent any strategy from being starved.
        """
        epsilon = 0.01
        alpha = self.adaptive_alpha
        n_schemes = self.n_rd_scheme

        scheme_raw = np.zeros(n_schemes)
        for s in range(n_schemes):
            ss = self._scheme_stats.get(s, {'calls': 0, 'accepted': 0, 'energy_drop': 0.0})
            calls = max(ss['calls'], 1)
            acc_rate = ss['accepted'] / calls
            avg_drop = ss['energy_drop'] / calls + epsilon
            scheme_raw[s] = (acc_rate ** alpha) * avg_drop

        prev = np.array(self.rd_ratio_scheme, dtype=float)
        if scheme_raw.sum() > 1e-12:
            new_weights = scheme_raw / scheme_raw.sum()
            blended = (self.adaptive_smoothing * new_weights
                       + (1.0 - self.adaptive_smoothing) * prev)
            floor_val = self.adaptive_floor * blended.max() if blended.max() > 0 else 0.01
            blended = np.maximum(blended, floor_val)
            blended = blended / blended.sum()
            self.rd_ratio_scheme = blended.tolist()

        if self.debug:
            print("\n--- Adaptive scheme weights updated ---")
            for s in range(n_schemes):
                ss = self._scheme_stats.get(s, {})
                c = max(ss.get('calls', 0), 1)
                print(f"  Scheme {s}: calls={c:4d}  "
                      f"acc_rate={ss.get('accepted',0)/c:.2f}  "
                      f"avg_drop={ss.get('energy_drop',0)/c:.3f}  "
                      f"prob={self.rd_ratio_scheme[s]:.3f}  "
                      f"modes={self.rd_ratio_mode[s]}")
            print("")

    def _get_real_energy(self, atoms):
        """
        Get energy of structure on real potential energy surface
        """
        atoms_temp = atoms.copy()
        atoms_temp.calc = self.base_calc
        return atoms_temp.get_potential_energy()
    
    def _get_bias_energy(self, atoms):
        """
        Get energy of structure on bias potential energy surface
        """
        atoms_temp = atoms.copy()
        atoms_temp.calc = self.bias_calc
        E=atoms_temp.get_potential_energy()
        if self.debug:
            E_base=atoms_temp.calc.results['E_base']
            E_bias=atoms_temp.calc.results['E_bias']
            print("bias flag: ",self.bias_calc.flag,"   quadratic:",self.bias_calc.quadra_params)
            print("E_base: ",E_base,"   E_bias: ",E_bias,"   E: ",E)
        return E

    def run(self, steps=100):
        """
        Run GraSMoS global search algorithm, strictly following steps in the paper
        """
        # Initial structure energy log (stdout is already tee'd to grasmos_log.txt by cosmos_run)
        #print(f"Initial structure: Energy = {self._get_real_energy(self.atoms):.6f} eV")
        
        # Initialize combined trajectory file (remove old one if exists)
        trajectory_file = os.path.join(self.output_dir, 'all_minima.xyz')
        if os.path.exists(trajectory_file):
            os.remove(trajectory_file)
        
        climb_info_file=open("climb.info","w")
        climb_info_file.write("#index before0/after1 d_climb_origin  angle  d_clime_base angle\n")
        # Initialize current structure as initial minimum structure
        init_atoms = self.atoms.copy()    # user provided initial structure (relaxed in self.__init__())
        basin_atoms= init_atoms.copy()     # current basin structure (relaxed)
        
        # Clean xyz directory if it exists
        if os.path.exists("xyz"):
            for file in os.listdir("xyz"):
                file_path = os.path.join("xyz", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        elif self.output_xyz:
            os.makedirs("xyz")

        for step in range(steps):
            print(f"\n------------- GraSMoS Step {step + 1}/{steps} -------------")
            basin_energy = self._get_real_energy(basin_atoms)  # already relaxed
            N0 = self._generate_random_direction(basin_atoms)
            if self.output_xyz:
                displace0=get_displace(N0.copy(),self.mobile_mask,self.n_mobile,self.average_dr,self.max_dr) # 很奇怪这里如果不copy会影响rotation！！！
                print_xyz(basin_atoms,filename=f"climb_{step}.xyz",energy=basin_energy,bias_energy=0, displace = displace0)
                
            climb_atoms = basin_atoms.copy() # climbing structure
            gaussian_params = []
            prev_climb_energy = basin_energy  # Track energy for adaptive width
            adaptive_gw = None                # Adaptive Gaussian width (None = auto-initialize)

            # ── Per-scheme parameter overrides (restored after step) ──
            _saved = {}
            scheme = getattr(self, '_last_scheme', 0)
            if scheme < len(self._scheme_params):
                overrides = self._scheme_params[scheme]
                for key in ['average_dr', 'max_dr', 'gaussian_height', 'max_gaussians']:
                    if key in overrides:
                        _saved[key] = getattr(self, key)
                        setattr(self, key, overrides[key])
                if 'rotation_param' in overrides:
                    rp = overrides['rotation_param']
                    if rp is None:
                        self._lock_direction = True
                    else:
                        _saved['quadra_a'] = self.quadra_a
                        self.quadra_a = rp

            for n in range(1, self.max_gaussians + 1):
                # ── Dimer rotation (skipped for topology-changing moves whose
                #     reaction coordinate has high curvature) ──
                locked = getattr(self, '_lock_direction', False)
                if locked:
                    N = N0
                    # Rodrigues rotation produces physically meaningful,
                    # non-uniform displacements (atoms far from rotation
                    # axis move more).  Use a large max_dr to avoid capping
                    # the periphery and shrinking the entire move.
                    dr_avg = self.average_dr * 4.0
                    dr_max = float('inf')
                else:
                    N = self._bias_dimer_rotation_ase(climb_atoms, N0)
                    dr_avg = self.average_dr
                    dr_max = self.max_dr
                displace = get_displace(N, self.mobile_mask, self.n_mobile, dr_avg, dr_max)
                Norm_dist = np.linalg.norm(displace)

                # --- Adaptive Gaussian width ---
                # Base width from displacement, then adjust based on energy landscape steepness
                base_gw = Norm_dist * 2.0
                if adaptive_gw is None:
                    adaptive_gw = base_gw
                gaussian_params.append((N.copy(),climb_atoms.positions.flatten(),self.gaussian_height, adaptive_gw))  #(d, R1, gh,gw)

                climb_atoms.calc = self.bias_calc
                climb_atoms.calc.reset_gaussians(gaussian_params)

                climb_atoms.set_positions(climb_atoms.get_positions() + displace)

                tBE=climb_atoms.get_potential_energy()   # potential energy on bias potential energy surface
                climb_energy = self._get_real_energy(climb_atoms) # real energy on real potential energy surface
                if self.output_xyz:  # before local minimize
                    print_xyz(climb_atoms,filename=f"climb_{step}.xyz",energy=climb_energy,bias_energy=tBE,displace=displace)

                # --- Adaptive climbing fmax ---
                # Use relaxed precision during climbing to save cost; only final relaxation needs tight fmax
                if self.cl_adaptive_relaxation:
                    # Gradually relax: first Gaussian uses exact fmax, later ones relax toward cl_relaxed_fmax
                    climb_fmax = min(self.cl_opt_fmax + n * 0.03, self.cl_relaxed_fmax)
                else:
                    climb_fmax = self.cl_relaxed_fmax
                self._local_minimize(climb_atoms, climb_fmax, self.cl_opt_max_steps)

                # Write to climbing.info file (file handle is kept open)
                if self.debug:
                    print(f"Added Gaussian #{n}: gaussian_height={self.gaussian_height:.4f}, sigma={Norm_dist:.4f} Å, |d|={np.linalg.norm(N):.4f}")
                    distance_0_org, angle_0_org = periodic_distance(init_atoms, climb_atoms, N)
                    distance_0_bas, angle_0_bas = periodic_distance(basin_atoms, climb_atoms, N)
                    climb_info_file.write(f"{step} 0 {distance_0_org:.4f} {angle_0_org:.4f} {distance_0_bas:.4f} {angle_0_bas:.4f}\n")
                    print(f"{step} 0 {distance_0_org:.4f} {angle_0_org:.4f} {distance_0_bas:.4f} {angle_0_bas:.4f}\n")

                tBE=climb_atoms.get_potential_energy()   # potential energy on bias potential energy surface
                climb_energy = self._get_real_energy(climb_atoms) # real energy on real potential energy surface
                if self.output_xyz: # after local minimize
                    print_xyz(climb_atoms,filename=f"climb_{step}.xyz",energy=climb_energy,bias_energy=tBE,displace=displace)

                # --- Adaptive Gaussian width update ---
                # After optimization, check energy change to adapt width for next Gaussian.
                # Flat region (small |ΔE|) → widen to stride faster.
                # Steep region (large |ΔE|) → narrow for finer control.
                delta_E_climb = abs(climb_energy - prev_climb_energy)
                prev_climb_energy = climb_energy
                if delta_E_climb < 0.1:
                    adaptive_gw = min(adaptive_gw * 1.3, base_gw * 4.0)
                elif delta_E_climb > 0.5:
                    adaptive_gw = max(adaptive_gw * 0.7, base_gw * 0.5)
                if self.debug:
                    print(f"  Gaussian #{n}: gw={adaptive_gw:.4f}, ΔE_climb={delta_E_climb:.4f}, fmax={climb_fmax:.4f}")

                if self.debug:
                    # Write to climbing.info file (file handle is kept open)
                    distance_1_org, angle_1_org = periodic_distance(init_atoms, climb_atoms, N)
                    distance_1_bas, angle_1_bas = periodic_distance(basin_atoms, climb_atoms, N)
                    climb_info_file.write(f"{step} 1 {distance_1_org:.4f} {angle_1_org:.4f} {distance_1_bas:.4f} {angle_1_bas:.4f}\n")
                    print(f"{step} 1 {distance_1_org:.4f} {angle_1_org:.4f} {distance_1_bas:.4f} {angle_1_bas:.4f}\n")
                    climb_info_file.flush()
                
                # Algorithm Step 5: Check stopping condition
                # Stop if: (i) reached max Gaussians H, or (ii) structure relaxed back below starting energy
                if n >= self.max_gaussians:
                    print(f"\n--- Climb end ---\n n_gaussian={n}, reached maximum Gaussians")
                    break
                elif climb_energy <= basin_energy-0.05:
                    print(f"\n--- Climb end ---\n n_gaussian={n}, energy {climb_energy:.6f} eV <= basin {basin_energy:.6f} eV")
                    break
            
            temp_average_dr, temp_max_dr = calc_average_max_displace(displace,self.mobile_mask,self.n_mobile)
            print(f"n_mobile,Norm_dist, average_dr,max_dr: {self.n_mobile}, {Norm_dist:.3f}, {temp_average_dr:.3f}, {temp_max_dr:.3f}")
            # Algorithm Step 6: Remove all bias potentials and optimize on real potential energy surface
            new_basin_atoms=climb_atoms.copy()
            new_basin_atoms.calc = self.base_calc
            self._local_minimize(new_basin_atoms,self.opt_fmax,self.opt_max_steps)
            new_basin_energy = self._get_real_energy(new_basin_atoms)

            if self.output_xyz:
                print_xyz(new_basin_atoms,filename=f"climb_{step}.xyz",energy=new_basin_energy,bias_energy=0,displace=displace0)
            
            # Algorithm Step 7: Use Metropolis criterion to accept or reject
            delta_E = new_basin_energy - basin_energy
            if delta_E > 0:
                accept_prob = np.exp(-delta_E / (self.kB * self.temperature))
            else:
                accept_prob = 1.0

            metrop_accepted = False
            is_new_minimum = False

            if np.random.rand() < accept_prob:
                metrop_accepted = True
                print(f"Accept new basin structure: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
                basin_atoms=new_basin_atoms
                if len(self.vacuum_axes) > 0:
                    basin_atoms.set_positions(self._position_to_center(basin_atoms))
                # Check if new structure is a duplicate
                if not is_duplicate_by_desc_and_energy(
                    new_atoms=basin_atoms,
                    pool=self.pool,
                    #species=self.soap_species if self.soap_species is not None else list(set(climb_atoms.get_chemical_symbols())),
                    energy=new_basin_energy,
                    pool_energies=self.real_energies,
                    energy_tol=0.5,
                    mobile_atoms=self.mobile_atoms,
                ):
                    # New unique structure found
                    is_new_minimum = True
                    self._add_to_pool(basin_atoms)
                else:
                    # Duplicate structure, but still update init_atoms to explore from different point
                    # This prevents getting stuck in the same location
                    print("Structure is duplicate, not added to pool")
                    self._add_to_pool(basin_atoms) # still add to pool for the debug stage
                    # No need to update pool
            else:
                print(f"Reject new basin structure: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
                # do not change basin_atoms, keep original basin_atoms as the current structure in Mento Carlo
                self._add_to_pool(new_basin_atoms)

            # ── Adaptive mode tracking ──
            if self.adaptive:
                self._update_mode_stats(self._last_scheme, delta_E,
                                        metrop_accepted, is_new_minimum)
                self._step_counter += 1
                if self._step_counter % self.adaptive_interval == 0:
                    self._recompute_adaptive_weights()

            # Reset per-step flags and restore any scheme-param overrides
            self._lock_direction = False
            for key, val in _saved.items():
                setattr(self, key, val)

            # Output current step information
            print(f"Step {step+1}: Energy = {new_basin_energy:.6f} eV")

        print("\nGraSMoS search completed!")
        print(f"All {len(self.pool)} minima structures saved to: {os.path.join(self.output_dir, 'all_minima.xyz')}")
        
        # Save the lowest energy structure
        if self.pool and self.real_energies:
            min_idx = self.real_energies.index(min(self.real_energies))
            best_atoms = self.pool[min_idx].copy()
            best_energy = self.real_energies[min_idx]
            best_atoms.info['energy'] = best_energy
            best_atoms.info['minima_index'] = min_idx
            best_file = os.path.join(self.output_dir, 'best_str.xyz')
            ase_write(best_file, best_atoms)
            print(f"Lowest energy structure (E = {best_energy:.6f} eV) saved to: {best_file}")

        climb_info_file.close()
        return self.pool, self.real_energies

    def _position_to_center(self, atoms):
        if len(self.vacuum_axes) > 0:    # Pre-center structure along vacuum axes
            cell = atoms.get_cell()
            box_center = (cell[0] + cell[1] + cell[2]) / 2.0
            pos0 = atoms.get_positions()
            curr_center = pos0.mean(axis=0)
            shift = box_center - curr_center
            for i in range(3):
                if i not in self.vacuum_axes:
                    shift[i] = 0.0
            return pos0 + shift
        else:
            return atoms.get_positions()

    def _bias_dimer_rotation(self, atoms, N0):
        """
        Implement bias dimer rotation method according to SSW paper (Eq. 3-6)
        Uses proper dimer method to find the lowest curvature direction with bias potential
        
        Reference: Shang & Liu, J. Chem. Phys. 139, 244104 (2013)
        
        Parameters:
            atoms: Current atomic structure (at minimum R_m)
            initial_direction: Initial search direction N^0
        
        Returns:
            Optimized direction N^1 (normalized)
        """
        # Normalize initial direction
        norm_ini = np.linalg.norm(N0)
        N = N0.copy() / norm_ini
              
        # Dimer parameters from SSW paper
        delta_R = 0.02  # Dimer separation (typical: 0.02 Å)
        theta_trial = 0.5 * np.pi / 180.0  # Trial rotation angle (radians), ~1 degrees
        max_rotations = 100  # Maximum number of rotations
        f_rot_tol = 0.01  # Rotational force tolerance (eV/Å)

        # Current position (flattened)
        R0_flat = atoms.positions.flatten()
        
        print(f"\n--- Dimer Rotation ---")
        print(f"Parameters: ΔR={delta_R:.5f} Å, θ_trial={np.degrees(theta_trial):.3f}°, max_iter={max_rotations}")
        
        self.bias_calc.reset_quadra([self.quadra_a,R0_flat,N0])  #a=10
        # Iteratively rotate dimer to find optimal direction
        for rotation_iter in range(max_rotations):
            # Calculate dimer images: R_1 = R_0 + N * ΔR (Eq. 3)
            R1_flat = R0_flat + N * delta_R
            
            # Compute forces at R_1 (on real PES)
            atoms_temp1 = atoms.copy()
            atoms_temp1.set_positions(R1_flat.reshape(-1, 3))
            atoms_temp1.calc = self.bias_calc

            F1 = atoms_temp1.get_forces().flatten()
            print("quadra:",self.bias_calc.results['E_quadra'],self.bias_calc.results['F_quadra'].max(),self.bias_calc.results['F_quadra'].min())
            
            # Compute curvature C = (F_0 - F_1) · N / ΔR (Eq. 4)
            # Note: F_0 can be extrapolated from F_1 for efficiency
            # F_0 ≈ -F_1 (symmetric approximation for small ΔR)
            F0_approx = -F1
            C = np.dot((F0_approx - F1), N) / delta_R
            
            # Calculate rotational force (perpendicular component)
            # F_rot = F_1 - (F_1 · N) * N  (force perpendicular to N)
            F1_parallel = np.dot(F1, N) * N
            F_rot = F1 - F1_parallel
            
            # Check convergence: if rotational force is small, stop rotation
            F_rot_mag = np.linalg.norm(F_rot)
            
            print(f"  Iter {rotation_iter+1}: C={C:10.4f} eV/Å², |F_rot|={F_rot_mag:8.4f} eV/Å", end="")
            
            if F_rot_mag < f_rot_tol:
                print(" <- CONVERGED")
                break
            
            # Compute rotation angle using finite difference
            # Trial rotation: N' = N * cos(θ) + F_rot_normalized * sin(θ)
            if F_rot_mag > 1e-10:
                F_rot_normalized = F_rot / F_rot_mag
            else:
                print(" <- No rotation needed")
                break  # No significant rotation needed
            
            # Trial rotation with small angle
            N_trial = N * np.cos(theta_trial) + F_rot_normalized * np.sin(theta_trial)
            N_trial = N_trial / np.linalg.norm(N_trial)
            
            # Evaluate curvature at trial position
            R1_trial_flat = R0_flat + N_trial * delta_R
            atoms_temp_trial = atoms.copy()
            atoms_temp_trial.set_positions(R1_trial_flat.reshape(-1, 3))
            atoms_temp_trial.calc = self.bias_calc
            F1_trial = atoms_temp_trial.get_forces().flatten()
            F0_trial_approx = -F1_trial
            C_trial = np.dot((F0_trial_approx - F1_trial), N_trial) / delta_R
            
            # Compute second derivative estimate for parabolic fit
            # d²C/dθ² ≈ 2 * C / θ²  (approximate)
            if abs(theta_trial) > 1e-10:
                # Optimal angle from parabolic approximation: θ_opt = -dC/dθ / (d²C/dθ²)
                # Use simple formula: θ_opt = θ_trial * C / (C - C_trial)
                if abs(C - C_trial) > 1e-10:
                    theta_opt = theta_trial * C / (C - C_trial)
                    # Limit rotation angle to avoid overshooting
                    theta_opt = np.clip(theta_opt, -np.pi/4, np.pi/4)
                else:
                    theta_opt = 0.0
            else:
                theta_opt = 0.0
            
            print(f", θ_opt={np.degrees(theta_opt):7.3f}°")
            
            # Apply optimal rotation
            N = N * np.cos(theta_opt) + F_rot_normalized * np.sin(theta_opt)
            N = N / np.linalg.norm(N)
            
            # Check convergence after updating N
            if abs(theta_opt) < 1e-3:
                #print("  -> Rotation angle too small, converged")
                break  # Converged

        N = N * norm_ini
        if self.debug:
            print(f"Dimer rotation completed after {rotation_iter+1} iterations\n")
            print(f'N vector after rotation:\n|N|={np.linalg.norm(N):.6f}')
            for i in range(self.n_atoms):
                if i<10:
                    n_vec = N[3*i:3*i+3]

                    if(self.mobile_mask[i]):
                        print(f"Atom {i:3d}: N_i=[{n_vec[0]:8.4f}, {n_vec[1]:8.4f}, {n_vec[2]:8.4f}], |N_i|={ns_mag:.4f}")
        
        print()
        return self._remove_translation(atoms,N)   # Return optimized direction N^1 with original magnitude

    def _bias_dimer_rotation_ase(self,atoms,N0):
        """
        Utilize the built-in mask feature of dimer to optimize the eigenmode direction 
        to the lowest curvature direction considering only mobile atoms.
        
        Parameters:
        - atoms: ASE Atoms object
        - initial_direction: Initial direction vector (shape: 3*N,)
        - mobile_mask: Boolean array, True indicates mobile atoms
        - max_iterations: Maximum number of iterations
        - tol: Convergence tolerance
        Returns:
        - optimized_direction: Optimized eigenmode direction
        - curvature: Corresponding curvature value
        """
        from dimer.dimer import MinModeAtoms, DimerControl, DimerEigenmodeSearch

        atoms_temp=atoms.copy()
        atoms_temp.calc=self.bias_calc
        atoms_temp.calc.reset_quadra([self.quadra_a,atoms_temp.positions.flatten(),N0])  #a=10
        if self.debug:
            print("self.quadra_a=",self.quadra_a)

        dimer_mask = self.mobile_mask.tolist()

        initial_direction = N0.reshape(-1, 3) / np.linalg.norm(N0)
        masked_initial_direction = initial_direction * self.mobile_mask[:, None]
       
        # Set control parameters, including mask
        # Create DimerControl with logfile=None to disable output
        control = DimerControl(logfile=None, eigenmode_logfile=None)
        #control.set_parameter('dimer_separation', 0.02)
        control.set_parameter('mask', dimer_mask)  # Set mask parameter
        control.set_parameter('order', 1)  # We only need the first eigenmode
        
        # Create MinModeAtoms object
        min_mode_atoms = MinModeAtoms(atoms_temp, control=control, logfile=None)

        # Process initial direction, considering only mobile atoms
       
        # Set initial eigenmode
        min_mode_atoms.initialize_eigenmodes(eigenmodes=[masked_initial_direction])
        
        # Create eigenmode search object
        eigenmode_search = DimerEigenmodeSearch(min_mode_atoms)
        
        # Ensure eigenmode_search has no logfile
        eigenmode_search.logfile = None
        
        # Set control parameters for eigenmode search
        # Adjust these parameters to control convergence
        eigenmode_search.control.set_parameter('f_rot_min', 0.01)  # Convergence threshold
        eigenmode_search.control.set_parameter('f_rot_max', 0.1)  # Upper limit for stopping
        eigenmode_search.control.set_parameter('max_num_rot', 100)  # Maximum rotations
        
        # Use dimer's built-in method to converge to eigenmode

        if self.debug:
            atoms_temp2=atoms_temp.copy()
            atoms_temp2.set_positions(atoms_temp.positions+0.01*N0.reshape(-1,3))
            atoms_temp2.calc=self.bias_calc
            print("before rotation:  E_total=",atoms_temp2.get_potential_energy(), "E_base=",self.bias_calc.results['E_base']," E_quadra=",self.bias_calc.results['E_quadra'])

        eigenmode_search.converge_to_eigenmode()

        # Get the final converged eigenmode
        final_mode = eigenmode_search.eigenmode

        if self.debug:
            atoms_temp2=atoms_temp.copy()
            atoms_temp2.set_positions(atoms_temp.positions+0.01*final_mode)
            atoms_temp2.calc=self.bias_calc
            print("after rotation:  E_total=",atoms_temp2.get_potential_energy(), "E_base=",self.bias_calc.results['E_base']," E_quadra=",self.bias_calc.results['E_quadra'])

        # Update the eigenmode in MinModeAtoms
        min_mode_atoms.set_eigenmode(final_mode, order=1)

        # Get final curvature
        final_curvature = eigenmode_search.get_curvature()
        if self.debug:
            print(f"Final curvature: {final_curvature:.6f}")

        # Verify the final rotational force
        eigenmode_search.update_virtual_forces()
        final_rot_force = eigenmode_search.get_rotational_force()
        final_rot_force_norm = np.linalg.norm(final_rot_force)
        if self.debug:
            print(f"Final rotational force norm: {final_rot_force_norm:.6f}")

        # Return optimized direction and curvature
        optimized_direction = min_mode_atoms.get_eigenmode(order=1)

        N=optimized_direction.flatten()
        if(np.dot(N,N0)<0):
            N=-N
        return N

    def _remove_translation(self,atoms,N):
        """
        Remove translation from N vector
        """
        # do not remove translation and rotation for bulk structure
        N_temp = N.reshape(-1,3)-N.reshape(-1,3).mean(axis=0)
        return self._normalize(N_temp.flatten())

    def _remove_rotation_and_translation(self,pos, vec):
        """
        Remove global translation and rotation from displacement vector using Kabsch algorithm
        :param pos: Original atomic positions (N, 3)
        :param vec: Original displacement vector (N, 3)
        :return: Corrected displacement vector (N, 3)
        """
        P = pos
        Q = pos + vec
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        H = np.dot(P_centered.T, Q_centered)
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(np.dot(Vt.T, U.T))
        step = np.eye(3)
        if d < 0:
            step[2, 2] = -1
        R = np.dot(Vt.T, np.dot(step, U.T))
        Q_aligned = np.dot(Q_centered, R)
        corrected_displacements = Q_aligned - P_centered
        return self._normalize(corrected_displacements.flatten())

    def _print_mobile(self, **kwargs):
        """
        Print values for mobile atoms from multiple lists with custom titles.
        """
        for title, data_list in kwargs.items():
            if len(data_list) != self.n_atoms:
                raise ValueError(f"List '{title}' must have n_atoms elements")
        for i in np.arange(self.n_atoms):
            if self.mobile_mask[i]:
                info = [f"AtomID: {i:3d}"]
                for title, data_list in kwargs.items():
                    info.append(f"{title} = {data_list[i]}")
                print(" | ".join(info))

    def _normalize(self,N):
        """
        Normalize N vector
        """
        N_mask=np.zeros_like(N)
        for i in range(self.n_atoms):
            if self.mobile_mask[i]:
                N_mask[3*i:3*i+3]=N[3*i:3*i+3]
        try:
            return N_mask / np.linalg.norm(N_mask)
        except:
            raise ValueError("Failed to normalize N vector")
