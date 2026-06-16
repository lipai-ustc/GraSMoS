# GraSMoS: Graph-Aware Structure Search with Monte Carlo Simulation

*lipai@mail.sim.ac.cn*

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Algorithm](#algorithm)
- [References](#references)
- [License](#license)

## Overview

GraSMoS is a tool for finding stable atomic structures through global optimization. It extends the stochastic surface walking (SSW) algorithm (Shang & Liu, 2013) with a graph-theoretic layer: a weighted bond graph is built from the covalent network and used to identify mechanically coupled atom groups. Graph-guided collective moves — community detection, Laplacian soft modes, and generalized bond switching — drive intelligent multi-atom displacements that preserve bonding topology and target high-energy regions. A multi-armed bandit system adaptively tunes the probability of each move type based on historical performance. The framework supports a wide range of calculators (NequIP, DeepMD, CHGNet, FAIRChem, EAM, VASP, LAMMPS, and custom Python) via the ASE interface.

## Features

**10 random-direction modes**, composable via weighted schemes:

| Mode | Description | Atomic energy usage |
|------|-------------|-----|
| `thermo` | Boltzmann-distributed random vectors | None (baseline) |
| `atomic` | Energy-weighted thermo | Scales displacement by exp(normalized_energy × AE_factor) |
| `nl` | Non-local pair attraction | Energy-biased pair selection; high-energy atoms chosen more often |
| `atnl` | Atomic-energy guided pair + neighbour shell | Selects the two highest-energy atoms as pair, extends displacement to their bonded neighbours |
| `bond_rotation` | Coordinated bond rotation (GSW-like) | Energy-biased bond selection; strained bonds rotated first |
| `bond_switch` | Generalized bond-switching (Stone-Wales-like) | Energy-biased candidate screening; fragments rotate in opposite directions |
| `shell` | Coordination shell collective motion | Energy-biased center selection; high-energy atoms chosen as center |
| `community` | Louvain community detection on bond graph | Energy-biased community selection; high-energy communities favour breathing, low-energy ones favour translation |
| `laplacian` | Graph Laplacian soft-mode deformation | Energy-weighted adjacency + energy-biased mode selection; single global direction preserves collective character |
| `python` | User-defined function | None |

All modes go through the same dimer rotation + Gaussian climbing + Metropolis pipeline. When per-atom energies are available (via MLFF calculators), most modes use them to target structurally unfavourable (high-energy) local environments — the core philosophy of GraSMoS. Only `thermo` and `python` remain energy-independent as safe baselines.

**Graph-theoretic collective moves.** A weighted adjacency matrix is built from the structure (w = exp(−d / 0.5 r_cov)), per-atom energies modulate edge weights when available, and the resulting bond graph drives four move types beyond the baseline:

- `community` — Louvain detection identifies natural bonded clusters (rings, coordination polyhedra). Community selection is biased toward high-energy clusters. The move type adapts to the cluster's average energy: high-energy clusters favour breathing (stress relief), low-energy ones favour translation (safe exploration).
- `laplacian` — Low-frequency eigenvectors of the normalized Laplacian L = I − D^{−1/2} A D^{−1/2} encode soft deformation patterns. A single random global direction is applied to all atoms, weighted by the eigenvector magnitudes, preserving the collective character of the mode. Mode selection is energy-biased toward modes overlapping with high-energy atoms.
- `bond_switch` — A generalized Stone-Wales transformation. The two local fragments rotate in opposite directions (30°–90°), producing topology change. The dimer rotation is skipped (locked) for this mode, and displacement is controlled by `lock_dr_factor` (default 2×) with a capped max_dr to prevent atom overlap. Candidate bonds are energy-biased toward strained regions.
- `bond_rotation` — Small-angle coordinated rotation for local relaxation without topology change. Bond selection is energy-biased: strained bonds are rotated first.

**Adaptive mode weighting.** A UCB-inspired multi-armed bandit tracks acceptance rate, energy drop, and new-minima count per scheme, recomputing weights every N steps with EMA smoothing. Atom-overlap steps (where two atoms are too close) are counted as rejections so that schemes producing overlaps are penalised. Underperforming modes are never eliminated (exploration floor ≥ 5%).

**Manual mode selection tips:**
- Covalent networks: `bond_switch` 0.2–0.4 + `bond_rotation` 0.1–0.2
- Clusters / nanoparticles: `shell` 0.1–0.2
- General boost: `community` 0.1–0.3
- Amorphous / disordered: `laplacian` 0.1–0.2
- Baseline: `thermo` + `atomic`
- Adaptive: `thermo:atomic:community:bond_switch:laplacian = 0.2:0.2:0.2:0.3:0.1` with `adaptive: true` is a robust starting point.

**Additional capabilities:**
- Flexible mobile control (index, sphere, slab, lower/upper region, with wall potentials)
- Automatic geometry classification (0D/1D/2D/3D) with translation/rotation removal
- Permutation-invariant duplicate detection via sorted SOAP descriptors with energy gating

## Installation

### Prerequisites
- Python 3.8+
- ASE ≥ 3.26.0, dscribe ≥ 2.1.0
- Optional calculators: [NequIP](https://github.com/mir-group/nequip), [DeepMD-kit](https://github.com/deepmodeling/deepmd-kit), [FAIRChem](https://github.com/facebookresearch/fairchem), [CHGNet](https://github.com/CederGroupHub/chgnet)

### Install from source
```bash
git clone https://github.com/lipai-ustc/GraSMoS.git
cd grasmos
conda create -n grasmos python=3.10 -y && conda activate grasmos  # optional
pip install .
```

## Usage

```bash
# Set NequIP model (default calculator)
export NEQUIP_MODEL=/path/to/deployed_model.pth

# Run with input.json + init.xyz in current directory
grasmos
```

## Configuration

### input.json

#### System and Potential
- `system.name`: (optional)
- `system.task`: `"global_search"` or `"structure_sampling"`
- `system.structure`: path to initial structure (default: `"init.xyz"`)
- `potential.type`: `nequip` / `eam` / `chgnet` / `deepmd` / `fairchem` / `vasp` / `lammps` / `python`
- `potential.model`: model file path or name
- `potential.device`: `"cpu"` (default) or `"cuda"`

**Supported calculators:**

*NequIP (default):* `"potential": {"type": "nequip", "model": "deployed_model.pth"}`

*EAM:* `"potential": {"type": "eam", "model": "AlCu.eam.alloy"}`

*CHGNet:* `"potential": {"type": "chgnet", "model": "pretrained"}`

*DeepMD:* `"potential": {"type": "deepmd", "model": "dp_model.pb"}`

*FAIRChem:* `"potential": {"type": "fairchem", "model": "/path/to/checkpoint.pt", "device": "cuda"}`

*VASP:* `"potential": {"type": "vasp", "model": "INCAR"}`
Reads parameters from the INCAR file; falls back to default PBE if not found.

*LAMMPS:* `"potential": {"type": "lammps", "commands": [...]}`

*Custom Python:* `"potential": {"type": "python"}`
Requires `calculator.py` in the working directory defining a `calculator` variable.

#### Monte Carlo Layer
- `monte_carlo.steps`: total MC steps (required)
- `monte_carlo.temperature`: Metropolis temperature in K (required)

#### Climbing Layer
- `climbing.gaussian.height`: Gaussian bias height in eV (default: 0.2)
- `climbing.gaussian.width`: Gaussian width in Å (default: 0.2)
- `climbing.gaussian.Nmax`: max Gaussians per climb (default: 20)

- `climbing.random_direction.mode`: list of modes (default: `["thermo","atomic"]`)
- `climbing.random_direction.ratio`: `[[weights, scheme_probability, scheme_params], ...]` (default: `[[[0.5,0.5],1]]`). The optional third element per-scheme overrides parameters like `max_dr`, `average_dr`, `gaussian_height`, `max_gaussians`, and `rotation_param` (set to `null` to lock the dimer direction).
- `climbing.random_direction.rotation_param`: dimer rotation strength (default: 10)
- `climbing.random_direction.AE_factor`: atomic energy exponent (default: 4.0). Higher values amplify the displacement difference between high- and low-energy atoms.
- `climbing.random_direction.lock_dr_factor`: displacement multiplier when dimer is locked for bond_switch (default: 2.0). Controls how far atoms move during topology-changing steps; also caps max_dr.
- `climbing.random_direction.element_weights`: e.g. `{"Cu": 1.5, "Al": 1.0}`
- `climbing.random_direction.direction_weights`: per-axis scaling e.g. `[1,1,0]` to freeze z (default: `[1,1,1]`)
- `climbing.random_direction.atomic_energy_calculator`: per-atom energy source for energy-biased modes

- `climbing.random_direction.climbing_optimizer.max_steps`: LBFGS steps per Gaussian (default: 100)
- `climbing.random_direction.climbing_optimizer.fmax`: force convergence eV/Å (default: 0.05)
- `climbing.random_direction.climbing_optimizer.relaxed_fmax`: relaxed convergence (default: 0.2)
- `climbing.random_direction.climbing_optimizer.adaptive_relaxation`: gradually relax fmax (default: true)

For adaptive weighting, add these to `random_direction`:
- `adaptive`: `true` to enable (default: false)
- `adaptive_interval`: steps between recomputation (default: 10)
- `adaptive_alpha`: acceptance-rate exponent, higher = more aggressive (default: 2.0)
- `adaptive_floor`: min weight fraction (default: 0.05)
- `adaptive_smoothing`: EMA factor, 0 = instant (default: 0.7)

#### Optimizer Layer
- `optimizer.max_steps`: max optimization steps (default: 500)
- `optimizer.fmax`: force convergence eV/Å (default: 0.05)

#### Mobile Control Layer (optional)
Default: all atoms mobile. Four modes available:

- `"mode": "all"` — all atoms mobile (default)
- `"mode": "indices_free"` + `"indices_free": [i, j, ...]` — only listed atoms move
- `"mode": "indices_fix"` + `"indices_fix": [i, j, ...]` — listed atoms are fixed
- `"mode": "region"` + `"region_type": "sphere"|"slab"|"lower"|"upper"` — spatial constraint

Wall potential: `V_wall = 0.5 × wall_strength × overshoot²` with `wall_offset` tolerance.

#### Output
- `output.directory`: output path (default: `"grasmos_output"`)
- `output.rd_xyz`: save random direction info (default: false)
- `output.debug`: verbose logging (default: false)

#### Custom Random Direction (Python mode)
When `random_direction.mode` includes `"python"`, create `generate_random_direction.py`:
```python
import numpy as np
from ase import Atoms

def generate_random_direction(atoms: Atoms) -> np.ndarray:
    n_atoms = len(atoms)
    N = np.random.randn(3 * n_atoms)
    return N
```

#### Low-Dimensional Structure Preparation
For clusters/wires/slabs, place the structure near the box center before running. The geometry classifier detects vacuum layers automatically; misplaced structures may be misclassified.

#### Main Parameters Quick Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ds` | Step size (Å) | 0.2 |
| `H` | Gaussian count | 20 |
| `w` | Gaussian height (eV) | 0.2 |
| `temperature` | MC temperature (K) | required |
| `relaxed_fmax` | Climbing convergence (eV/Å) | 0.2 |
| `adaptive_relaxation` | Gradually relax fmax | true |

#### Complete Example

```json
{
  "system": {"name": "C60", "task": "global_search"},
  "potential": {"type": "nep", "model": "C.txt"},
  "monte_carlo": {"steps": 200, "temperature": 500},
  "climbing": {
    "random_direction": {
      "mode": ["thermo","atomic","atnl","community","bond_switch","laplacian"],
      "ratio": [
        [[0.10,0.10,0.30,0.20,0.15,0.15], 0.35],
        [[0.10,0.10,0.30,0.20,0.15,0.15], 0.30, {"max_dr": 0.4, "average_dr": 0.12}],
        [[0.10,0.10,0.30,0.20,0.15,0.15], 0.25, {"max_dr": 0.15, "average_dr": 0.05}],
        [[0.10,0.10,0.30,0.20,0.15,0.15], 0.10, {"max_dr": 0.8, "average_dr": 0.25}]
      ],
      "rotation_param": 15,
      "AE_factor": 2.0,
      "lock_dr_factor": 2.0,
      "adaptive": true,
      "adaptive_interval": 15,
      "adaptive_alpha": 2.0,
      "adaptive_floor": 0.05,
      "adaptive_smoothing": 0.6,
      "atomic_energy_calculator": {"type": "nep", "model": "C.txt"},
      "climbing_optimizer": {"max_steps": 100, "fmax": 0.05, "relaxed_fmax": 0.15, "adaptive_relaxation": true}
    },
    "gaussian": {"height": 0.5, "width": 0.15, "Nmax": 20},
    "displace": {"average_dr": 0.12, "max_dr": 0.3}
  },
  "optimizer": {"max_steps": 150, "fmax": 0.03},
  "mobile_control": {"mode": "all"},
  "output": {"directory": "grasmos_output", "rd_xyz": true, "debug": false}
}
```

## Output

Results are saved in `grasmos_output` (or custom directory):
- `all_minima.xyz` — unique discovered minima only (duplicates and overlap-rejected structures are excluded from the pool)
- `best_str.xyz` — lowest-energy structure
- `grasmos_log.txt` — step-by-step energy log

## Algorithm

GraSMoS extends the stochastic surface walking (SSW) framework with a graph-theoretic layer for intelligent collective moves. Each Monte Carlo step follows the same core pipeline — generate a random direction, climb via Gaussian bias potentials, locally optimize, then accept or reject via Metropolis — but the direction generation is now informed by a bond graph that identifies mechanically coupled atom groups.

### Bond graph construction

A weighted adjacency matrix is built from the atomic structure. For each mobile atom pair, the edge weight is

&emsp; w = exp(−d / 0.5 r<sub>cov</sub>)

where d is the interatomic distance and r<sub>cov</sub> is the sum of the covalent radii. Weights below 0.01 are dropped. Per-atom energies, when available, normalize to [0.5, 1.0] and multiply each edge by min(factor<sub>i</sub>, factor<sub>j</sub>), biasing the graph toward high-energy regions. The resulting sparse adjacency matrix A captures bonding topology and strength while ignoring non-bonded contacts.

### Community detection (`community` mode)

The Louvain algorithm partitions A into communities — groups more densely connected internally than to the rest of the structure. These correspond to functional groups, coordination polyhedra, and ring systems. A randomly selected community undergoes rigid translation, rotation, or radial breathing. Because internal bonding topology is preserved, these moves have much higher acceptance rates than uncorrelated single-atom displacements.

### Laplacian soft modes (`laplacian` mode)

The normalized graph Laplacian L = I − D<sup>−1/2</sup> A D<sup>−1/2</sup> (where D is the degree matrix) is the graph-theoretic analogue of the dynamical matrix in lattice dynamics. Its low-frequency eigenvectors encode collective deformation patterns that cost minimal energy in a harmonic approximation. A single random global direction is applied to all atoms, weighted by the eigenvector magnitudes, preserving the collective character of the soft mode (using per-atom random directions would destroy this collective nature and produce noise). Mode selection is energy-biased: modes whose displacement magnitude overlaps strongly with high-energy atoms are preferred.

### Bond switching (`bond_switch` mode)

Candidate bonds are screened via the graph: both endpoints require coordination ≥ 2 and each must have at least one neighbour other than each other. The two local fragments rotate in **opposite directions** by a large angle (30°–90°), producing a topology-changing twist — the generalised Stone-Wales reaction coordinate. The dimer rotation is skipped (locked) for this mode, and displacement is scaled by `lock_dr_factor` (default 2×) with max_dr also scaled by the same factor to prevent atom overlap while preserving enough displacement for topology change. Candidate bonds are energy-biased when per-atom energies are available.

### Dimer rotation

All modes share a common refinement step: the initial direction N<sup>0</sup> is fed into a dimer eigenmode search that rotates it toward the lowest-curvature direction on the locally biased PES. A quadratic bias potential V = −(a/2)(d·N<sup>0</sup>)<sup>2</sup> is applied along N<sup>0</sup> to prevent the dimer from collapsing back to the basin. The converged eigenmode N<sup>1</sup> is then used as the displacement direction for the Gaussian climbing phase.

### Adaptive mode weighting

Mode selection is treated as a multi-armed bandit at the scheme level. Each scheme accumulates statistics: call count, Metropolis acceptance count, cumulative energy drop max(0, −ΔE), and new-minima count. Atom-overlap steps (where min_pair_distance < 0.4 Å) are counted as rejected so that schemes producing overlaps are penalised. Weights are recomputed every N steps:

&emsp; score = (accept_rate)<sup>α</sup> × (avg_energy_drop + ε)<br>
&emsp; new_weight = smoothing × (score / Σscore) + (1 − smoothing) × old_weight<br>
&emsp; weight = max(weight, floor × max_weight) &emsp; [renormalized]

The floor guarantees every mode retains a minimum sampling probability, and EMA smoothing prevents abrupt drift.

## References

1. Shang, C., & Liu, Z.-P. (2013). Stochastic surface walking method for structure prediction and pathway searching. *J. Chem. Theory Comput.*, 9(5), 1838–1845.
2. Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. *J. Chem. Phys.*, 139(24), 244104.
3. Zhang, X.-J., Shang, C., & Liu, Z.-P. (2012). Double-ended surface walking method for pathway building and transition state location. *J. Chem. Theory Comput.*, 9(12), 5745–5753.
4. Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.*, P10008.
5. Chung, F. R. K. (1997). *Spectral Graph Theory*. AMS.
6. Newman, M. E. J. (2004). Detecting community structure in networks. *Eur. Phys. J. B*, 38(2), 321–330.
7. Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data representation. *Neural Comp.*, 15(6), 1373–1396.
8. Jacobs, D. J., & Thorpe, M. F. (1998). Generic rigidity percolation: The pebble game. *Phys. Rev. Lett.*, 75(22), 4051–4054.
9. Stone, A. J., & Wales, D. J. (1986). Theoretical studies of icosahedral C60 and some related species. *Chem. Phys. Lett.*, 128(5–6), 501–503.
10. Behler, J., & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Phys. Rev. Lett.*, 98(14), 146401.
11. Batzner, S., et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Comm.*, 13, 2453.
12. Zhang, L., et al. (2018). Deep potential molecular dynamics. *Phys. Rev. Lett.*, 120(14), 143001.
13. Schütt, K. T., et al. (2018). SchNet – A deep learning architecture for molecules and materials. *J. Chem. Phys.*, 148(24), 241722.
14. Gasteiger, J., et al. (2021). GemNet: Universal directional graph neural networks for molecules. *NeurIPS*, 34, 6790–6802.
15. Merchant, A., et al. (2023). Scaling deep learning for materials discovery. *Nature*, 624(7990), 80–85.

## License

GNU General Public License v3.0. See https://www.gnu.org/licenses/gpl-3.0.html.
