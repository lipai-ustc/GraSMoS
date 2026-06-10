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

| Mode | Description | Best for |
|------|-------------|----------|
| `thermo` | Boltzmann-distributed random vectors | Baseline, always safe |
| `atomic` | Energy-weighted thermo | Heterogeneous systems |
| `nl` | Non-local pair attraction | Bond formation |
| `atnl` | Atomic-energy guided pair | Targeted bond formation |
| `bond_rotation` | Coordinated bond rotation (GSW-like) | Covalent networks |
| `bond_switch` | Generalized bond-switching (Stone-Wales-like) | Topological rearrangement, all materials |
| `shell` | Coordination shell collective motion | Clusters, local rearrangements |
| `community` | Louvain community detection on bond graph | All systems, increased acceptance |
| `laplacian` | Graph Laplacian soft-mode deformation | Amorphous, disordered networks |
| `python` | User-defined function | Custom applications |

All modes go through the same dimer rotation + Gaussian climbing + Metropolis pipeline.

**Graph-theoretic collective moves.** A weighted adjacency matrix is built from the structure (w = exp(−d / 0.5 r_cov)), per-atom energies modulate edge weights when available, and the resulting bond graph drives four move types beyond the baseline:

- `community` — Louvain detection identifies natural bonded clusters (rings, coordination polyhedra). A selected cluster undergoes rigid translation, rotation, or radial breathing.
- `laplacian` — Low-frequency eigenvectors of the normalized Laplacian L = I − D^{−1/2} A D^{−1/2} encode soft deformation patterns. These concentrate displacement into structurally favourable directions, especially effective in disordered systems.
- `bond_switch` — A generalized Stone-Wales transformation. Candidate bonds are screened (coordination ≥ 2, each endpoint has another neighbour), then a large-angle (30°–90°) relative rotation drives topology change. The dimer rotation fine-tunes the reaction coordinate to the specific material.
- `bond_rotation` — Small-angle coordinated rotation for local relaxation without topology change.

**Adaptive mode weighting.** A UCB-inspired multi-armed bandit tracks acceptance rate, energy drop, and new-minima count per mode, recomputing weights every N steps with EMA smoothing. Underperforming modes are never eliminated (exploration floor ≥ 5%).

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
- `climbing.random_direction.ratio`: `[[weights, scheme_probability], ...]` (default: `[[[0.5,0.5],1]]`)
- `climbing.random_direction.rotation_param`: dimer rotation strength (default: 10)
- `climbing.random_direction.element_weights`: e.g. `{"Cu": 1.5, "Al": 1.0}`
- `climbing.random_direction.atomic_energy_calculator`: per-atom energy source for `atomic` mode

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
  "system": {"name": "Si-I4", "task": "global_search"},
  "potential": {"type": "nep", "model": "Si.txt"},
  "monte_carlo": {"steps": 200, "temperature": 500},
  "climbing": {
    "random_direction": {
      "mode": ["thermo","atomic","community","bond_switch"],
      "ratio": [[[0.2,0.2,0.3,0.3], 1.0]],
      "rotation_param": 30,
      "adaptive": true,
      "climbing_optimizer": {"max_steps": 100, "fmax": 0.05, "relaxed_fmax": 0.2, "adaptive_relaxation": true}
    },
    "gaussian": {"height": 0.1, "width": 0.1, "Nmax": 20}
  },
  "optimizer": {"max_steps": 100, "fmax": 0.05},
  "mobile_control": {"mode": "region", "region_type": "sphere", "center": "center", "radius": 6, "wall_strength": 10.0, "wall_offset": 2.0},
  "output": {"directory": "grasmos_output", "rd_xyz": true, "debug": false}
}
```

## Output

Results are saved in `grasmos_output` (or custom directory):
- `all_minima.xyz` — all discovered minima (trajectory format)
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

The normalized graph Laplacian L = I − D<sup>−1/2</sup> A D<sup>−1/2</sup> (where D is the degree matrix) is the graph-theoretic analogue of the dynamical matrix in lattice dynamics. Its low-frequency eigenvectors encode collective deformation patterns that cost minimal energy in a harmonic approximation. The Fiedler vector (second-smallest eigenvector) captures the most significant natural bipartition. Displacing atoms along a low-frequency Laplacian eigenmode concentrates motion into structurally soft directions, avoiding high-energy bond-stretching. Mode selection is energy-biased: modes whose displacement magnitude overlaps strongly with high-energy atoms are preferred.

### Bond switching (`bond_switch` mode)

Candidate bonds are screened via the graph: both endpoints require coordination ≥ 2 and each must have at least one neighbour other than each other. A large-angle (30°–90°) relative rotation is applied to the two local fragments about a random axis perpendicular to the bond, using Rodrigues' rotation formula to preserve distances. The large initial angle deliberately overshoots so that, after optimization, the two halves can rebond with each other's former neighbours. The subsequent dimer rotation fine-tunes angle and axis to the material's actual PES — for sp² carbon this discovers the 90° Stone-Wales coordinate; for tetrahedral semiconductors ~70°; for arbitrary environments it adapts automatically. Candidate bonds are energy-biased when per-atom energies are available.

### Dimer rotation

All modes share a common refinement step: the initial direction N<sup>0</sup> is fed into a dimer eigenmode search that rotates it toward the lowest-curvature direction on the locally biased PES. A quadratic bias potential V = −(a/2)(d·N<sup>0</sup>)<sup>2</sup> is applied along N<sup>0</sup> to prevent the dimer from collapsing back to the basin. The converged eigenmode N<sup>1</sup> is then used as the displacement direction for the Gaussian climbing phase.

### Adaptive mode weighting

Mode selection is treated as a multi-armed bandit. Each mode accumulates statistics: call count, Metropolis acceptance count, cumulative energy drop max(0, −ΔE), and new-minima count. Weights are recomputed every N steps:

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
