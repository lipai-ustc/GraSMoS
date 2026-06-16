#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraSMoS Global Optimization Execution Script

Function: Reads structure file and potential model configuration from input.json, performs GraSMoS global search
Usage: grasmos [or] python cosmos_run.py
"""

import os
import sys
import atexit
import json
import numpy as np
from ase.io import read
from cosmos_search import GraSMoSSearch
from cosmos_utils import load_potential, get_version_info, \
                         get_mobile_atoms, infer_geometry_type

class TeeLogger:
    """Redirect print output to both console and log file"""
    def __init__(self, log_file, mode='w'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def main() -> None:
    # Record start time
    import time
    start_time = time.time()
    
    # Get current working directory where input files should be
    cwd = os.getcwd()
    
    # 0. Load configuration file
    config_path = os.path.join(cwd, 'input.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

    # 1. Get task type (required)
    sys_config = config.get('system')
    if not sys_config:
        raise ValueError("Configuration missing 'system' section")
    name = sys_config.get('name')
    task = sys_config.get('task').lower()
    if task not in ['global_search', 'structure_sampling']:
        raise ValueError("Invalid task type. Must be 'global_search' or 'structure_sampling'.")

    # 2. Read structure file (optional)
    structure_path = config.get('system', {}).get('structure', 'init.xyz')
    if not os.path.isabs(structure_path):
        structure_path = os.path.join(cwd, structure_path)
    try:
        atoms = read(structure_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    except Exception as e:
        raise ValueError(f"Error reading structure file: {e}")
    
    geometry_type, vacuum_axes = infer_geometry_type(atoms) # Detect geometry type and prepare structure for search
    structure_info = {'atoms': atoms, 'geometry_type': geometry_type, 'vacuum_axes': vacuum_axes}
    
    # 3. Load potential calculator (required)
    potential_config = config.get('potential')
    potential_type = potential_config.get('type')
    if not potential_config or not potential_type:
        raise ValueError("Configuration missing 'potential' section")
    calculator = load_potential(potential_config,custom_atomic=False)

    # 4. Get Monte Carlo configuration (required)
    mc_config = config.get('monte_carlo')
    mc_steps = mc_config.get('steps')
    temperature = mc_config.get('temperature')
    if not mc_config or not mc_steps or not temperature:
        raise ValueError("Configuration missing 'monte_carlo' section")
    monte_carlo={'steps': mc_steps, 'temperature': temperature}
    
    # 5. Get random direction mode and parameters  (optional)
    climb_config = config.get('climbing',{})
    rd_config = climb_config.get('random_direction', {})
    rd_mode = rd_config.get('mode', ['thermo','atomic']) # Default method according to the original SSW algorithm
    rd_ratio = rd_config.get('ratio', [[[0.5,0.5],1]])
    # Parse per-scheme parameter overrides (optional third element in each ratio entry)
    scheme_params = []
    for entry in rd_ratio:
        if len(entry) >= 3:
            scheme_params.append(entry[2])
        else:
            scheme_params.append({})
    #
    element_weights = rd_config.get('element_weights', {}) # additional weight based on element type
    direction_weights = rd_config.get('direction_weights', [1,1,1]) # additional weight based on direction
    quadra_param=rd_config.get('rotation_param', 10)
    AE_factor = rd_config.get('AE_factor', 4.0)
    adaptive = rd_config.get('adaptive', False)
    adaptive_interval = rd_config.get('adaptive_interval', 10)
    adaptive_alpha = rd_config.get('adaptive_alpha', 2.0)
    adaptive_floor = rd_config.get('adaptive_floor', 0.05)
    adaptive_smoothing = rd_config.get('adaptive_smoothing', 0.7)
    
    climb_optimizer_config = rd_config.get('climbing_optimizer', {})
    cl_max_steps = climb_optimizer_config.get('max_steps', 100)
    cl_fmax = climb_optimizer_config.get('fmax', 0.05)
    cl_relaxed_fmax = climb_optimizer_config.get('relaxed_fmax', 0.2)
    cl_adaptive_relaxation = climb_optimizer_config.get('adaptive_relaxation', True)
    climb_optimizer={'max_steps': cl_max_steps, 'fmax': cl_fmax,
                     'relaxed_fmax': cl_relaxed_fmax,
                     'adaptive_relaxation': cl_adaptive_relaxation}

    #active_area_config = rd_config.get('active_area', {})
    #active_area_model = active_area_config.get('model', 'all')
    #active_area_region=None   # #### lipai working here 

    valid_modes = {'thermo',     # Temperature-based default scale (Boltzmann distribution)
                   'atomic',     # 'thermo' * atomic_energy_scale
                   'nl',         # Nl: non-local pair attraction
                   'atnl',       # atomic-energy based nl
                   'python',     # User-defined Python function
                   'bond_rotation',  # Coordinated bond rotation (generalized GSW-like)
                   'bond_switch',    # Generalized bond-switching (Stone-Wales-like for all materials)
                   'shell',          # Coordination shell collective motion
                   'community',      # Graph community detection (Louvain) collective move
                   'laplacian',      # Graph Laplacian soft-mode collective move
                   }
    for i_mode in rd_mode:
        if i_mode not in valid_modes:
            raise ValueError(f"Invalid random direction mode: '{i_mode}'. Must be one of {valid_modes}.")

    if 'atomic' in rd_mode:
        atomic_energy_calculator_config = rd_config.get('atomic_energy_calculator', None)
        # Load atomic energy calculator if specified
        if atomic_energy_calculator_config:
            atomic_energy_calculator = load_potential(atomic_energy_calculator_config)
            # Verify it supports per-atom energy calculation
            if not hasattr(atomic_energy_calculator, 'get_potential_energies'):
                raise ValueError(
                    f"Atomic energy calculator (type: {atomic_energy_calculator_config.get('type', 'unknown')}) does not support per-atom energies.\n"
                    f"The calculator must have 'get_potential_energies' method.\n"
                    f"Please specify a compatible calculator in 'random_direction.atomic_energy_calculator'."
                )
            rd_info=f"Loaded user-specified atomic energy calculator: {atomic_energy_calculator_config.get('type', 'unknown')}"
        else:
            atomic_energy_calculator = load_potential(potential_config, custom_atomic=True)
            if not hasattr(atomic_energy_calculator, 'get_potential_energies'):
                raise ValueError(
                    f"Primary potential calculator does not support per-atom energies.\n"
                    f"The calculator must have 'get_potential_energies' method.\n"
                    f"Please specify a compatible calculator in 'random_direction.atomic_energy_calculator'."
                )
            rd_info=f"No atomic_energy_calculator specified. Using primary potential calculator for per-atom energies."
    else:
        atomic_energy_calculator = None
    
    random_direction={
        'mode': rd_mode,
        'ratio': rd_ratio,
        'scheme_params': scheme_params,
        'element_weights': element_weights,
        'direction_weights': direction_weights,
        'quadra_param': quadra_param,
        'AE_factor': AE_factor,
        'climb_optimizer': climb_optimizer,
        'adaptive': adaptive,
        'adaptive_interval': adaptive_interval,
        'adaptive_alpha': adaptive_alpha,
        'adaptive_floor': adaptive_floor,
        'adaptive_smoothing': adaptive_smoothing,
    } 

    # 6. Get Climbing configuration (optional)

    gaussian_config=climb_config.get('gaussian',{})
    gaussian_height = gaussian_config.get('height', 0.2)    # w parameter
    gaussian_width  = gaussian_config.get('width', 0.2)     # ds parameter (initial width; adaptively tuned during climb)
    max_gaussians   = gaussian_config.get('Nmax', 20)       # H parameter

    gaussian={'gaussian_height': gaussian_height, 'gaussian_width': gaussian_width, 'max_gaussians': max_gaussians}

    displace_config=climb_config.get('displace',{})
    average_dr = displace_config.get('average_dr', 0.1)    # displace average step size parameter
    max_dr = displace_config.get('max_dr', 0.2)     # displace max step size parameter
    
    displace={'average_dr': average_dr, 'max_dr': max_dr}

    # 7. Get Optimizer configuration (optional)
    optimizer_config = config.get('optimizer',{})
    max_steps = optimizer_config.get('max_steps', 500)
    fmax = optimizer_config.get('fmax', 0.03)
    optimizer={'max_steps': max_steps, 'fmax': fmax}

    # 8. Get Mobile Control configuration and normalize to internal format (optional)
    raw_mc = config.get('mobile_control', {})
    mobile_mode   = raw_mc.get('mode', 'all')
    mobile_region = None
    wall_strength   = raw_mc.get('wall_strength', 10.0)
    wall_offset     = raw_mc.get('wall_offset', 2.0)

    if mobile_mode == 'all':
        mobile_atoms = np.arange(len(atoms), dtype=int).tolist()

    elif mobile_mode == 'indices_free':
        mobile_atoms = np.array(raw_mc.get('indices_free', []), dtype=int).tolist()

    elif mobile_mode == 'indices_fix':
        fixed = np.array(raw_mc.get('indices_fix', []), dtype=int)
        all_idx = np.arange(len(atoms), dtype=int)
        mobile_atoms = np.setdiff1d(all_idx, fixed, assume_unique=False).tolist()

    elif mobile_mode == 'region':
        region_type = raw_mc.get('region_type')
        if not region_type:
            raise ValueError("mobile_control mode 'region' requires 'region_type' to be specified")
        if region_type == 'sphere':
            center = raw_mc.get('center')
            radius = raw_mc.get('radius')
            if center is None or radius is None:
                raise ValueError("region_type 'sphere' requires 'center' and 'radius' to be specified")
            if center == "center".lower():
                center = atoms.get_center_of_mass()
            mobile_region = {
                'type': 'sphere',
                'center': np.array(center).tolist(),
                'radius': radius,
            }
        elif region_type == 'slab':
            origin = raw_mc.get('origin')
            normal = raw_mc.get('normal')
            min_dist = raw_mc.get('min_dist')
            max_dist = raw_mc.get('max_dist')
            if origin is None or normal is None or min_dist is None or max_dist is None:
                raise ValueError("region_type 'slab' requires 'origin', 'normal', 'min_dist', and 'max_dist' to be specified")
            mobile_region = {
                'type': 'slab',
                'origin': np.array(origin).tolist(),
                'normal': np.array(normal).tolist(),
                'min_dist': min_dist,
                'max_dist': max_dist,
            }
        elif region_type in ('lower', 'upper'):
            axis = raw_mc.get('axis')
            threshold = raw_mc.get('threshold')
            if axis is None or threshold is None:
                raise ValueError(f"region_type '{region_type}' requires 'axis' and 'threshold' to be specified")
            mobile_region = {
                'type': region_type,
                'axis': axis,
                'threshold': threshold,
            }
        else:
            raise ValueError(f"Unknown region_type: {region_type}.\n Valid options are 'sphere', 'slab', 'lower', and 'upper'.")
        
        mobile_atoms = get_mobile_atoms(atoms, mobile_region)  # Calculate mobile_atoms from mobile_region

    else:
        raise ValueError(f"Unknown mobile_control mode: {mobile_mode}")
        
    mobile_control = {      # Construct unified mobile_control_param
        'mobile_atoms': mobile_atoms,
        'mobile_region': mobile_region,
        'wall_strength': wall_strength,
        'wall_offset': wall_offset,    }
    
    # 9. Get output configuration with defaults (optional)
    output_config = config.get('output', {})
    output_dir = output_config.get('directory', 'grasmos_output')
    output_xyz=output_config.get('rd_xyz', False)
    debug_mode = output_config.get('debug', False)    
    output={'directory': output_dir, 'rd_xyz': output_xyz, 'debug': debug_mode}

    # Prepare log file and print all configuration parameters at the top
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'grasmos_log.txt')
    # Redirect stdout to TeeLogger (opens in write mode to clear existing content)
    tee_logger = TeeLogger(log_path, mode='w')
    sys.stdout = tee_logger
    # Ensure stdout is restored and log closed even on crash
    atexit.register(lambda: (setattr(sys, 'stdout', tee_logger.terminal),
                              tee_logger.close()))
    print(get_version_info())    #header
    print('\n\n===================================    GraSMoS Input Configuration    ================================\n')
    print(f'\nSystem information:')
    print(f'  System name      : {name}')   
    print(f'  Task type        : {task}')
    print(f'  Structure file   : {structure_path}')
    print(f'  Geometry type    : {geometry_type}')
    print(f'\nPotential information:')
    print(f'  Potential type   : {potential_type}')
    if potential_config.get('model',None) is not None:
        if not os.path.isabs(potential_config['model']):
            potential_config['model'] = os.path.join(cwd, potential_config['model'])
        print(f'  Potential model  : {potential_config["model"]}')
    print(f'\nMonte Carlo information:')
    print(f'  Monte Carlo steps: {mc_steps}')
    print(f'  Temperature (K)  : {temperature}')
    print(f'\nRandom Direction information:')
    print(f'  RD mode          : {rd_mode}')
    for i,ratios in enumerate(rd_ratio):
        print(f'  Scheme {i} : mode ratio={ratios[0]} with Posibility {ratios[-1]}')
    if element_weights:  # Empty dict evaluates to False
        print(f'  Element weights  : {element_weights}')
    if direction_weights:  # Empty dict evaluates to False
        print(f'  Direction weights  : {direction_weights}')
    if climb_optimizer:  # Empty dict evaluates to False
        print(f'  Climbing optimizer : {climb_optimizer}')
    print(f'  AE factor        : {AE_factor}')
    print(f'  Dimer rotation a : {quadra_param}')
    if adaptive:
        print(f'  Adaptive weighting: enabled (interval={adaptive_interval}, alpha={adaptive_alpha}, floor={adaptive_floor})')
    print(f'\nGaussian information:')
    print(f'  Gaussian height w: {gaussian_height}')
    print(f'  Max Gaussians H  : {max_gaussians}')
    print(f'\nDisplace information:')
    print(f'  Displace average dr: {average_dr}')
    print(f'  Displace max dr: : {max_dr}')
    print(f'\nOptimizer information:')
    print(f'  Optimizer steps  : {max_steps}')
    print(f'  Optimizer fmax   : {fmax}')
    if mobile_mode != 'all':
        print(f'\nConstraint information:')
        print(f'  Mobile mode    : {mobile_mode}')
        print(f'  Number of Mobile/All atoms     : {len(mobile_atoms)}/{len(atoms)}')       
        # Create a display copy of mobile_region with rounded coordinates
        mobile_region_display = mobile_region.copy()
        # Round coordinate values for display (center, origin, normal)
        coord_keys = ['center', 'origin', 'normal']
        for key in coord_keys:
            if key in mobile_region_display:
                mobile_region_display[key] = [round(coord, 4) for coord in mobile_region_display[key]]
        print(f'  Mobile region  : {mobile_region_display}')
        print(f'  Wall strength    : {wall_strength}')
        print(f'  Wall offset      : {wall_offset}')
    else:
        print('  Mobile control : None')
    print(f'\nOutput information:')
    print(f'  Output dir       : {output_dir}')
    if output_xyz is not None:
        print(f'  RD info          : {output_xyz}')
    print(f'  Debug mode       : {debug_mode}')
    
    if 'atomic' in rd_mode:
        print(f'\n{rd_info}')

    print('\n\n=====================================    Start of GraSMoS Search    ==================================\n') 

    grasmos = GraSMoSSearch(
        task=task,
        structure_info=structure_info,
        calculator=calculator,
        atomic_calculator=atomic_energy_calculator,
        monte_carlo=monte_carlo,
        random_direction=random_direction,
        gaussian=gaussian,
        displace=displace,
        optimizer=optimizer,
        mobile_control=mobile_control,
        output=output
    )
    
    # Run GraSMoS global optimization
    minima_pool, energies = grasmos.run(steps=mc_steps)

    print("\nGraSMoS search completed!")
    print(f"Found {len(minima_pool)} energy minimum structures")
    if energies:
        print(f"Lowest energy: {min(energies):.6f} eV")
    print(f"Results saved to: {output_dir}")
    print(f"All minima structures in: {os.path.join(output_dir, 'all_minima.xyz')}")
    print(f"Best structure saved to: {os.path.join(output_dir, 'best_str.xyz')}")

    print('\n\n======================================    End of GraSMoS Search    ===================================\n')

    # Calculate and print total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")

if __name__ == '__main__':
    main()
