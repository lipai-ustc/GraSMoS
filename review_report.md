# GraSMoS Code Review Report

## 项目概述

GraSMoS (Graph-Aware Structure Search with Monte Carlo Simulation) 是一个基于随机表面行走(SSW)算法的原子结构全局优化工具，核心创新在于引入图论方法（社区检测、Laplacian软模、广义键切换）驱动智能多原子协作移动。

**核心文件结构：**
- `cosmos_search.py` (1851行) — 主算法类 GraSMoSSearch
- `cosmos_run.py` (395行) — 入口脚本，配置解析
- `cosmos_utils.py` (596行) — 工具函数、势函数加载、SOAP去重
- `bias_calculator.py` (173行) — 偏置势能计算器（高斯+二次+壁势）

---

## 🔴 严重问题（会导致崩溃或结果错误）

### 1. `_normalize` 方法体被截断 — 程序无法运行

`cosmos_search.py` 第1850-1852行：
```python
def _normalize(self,N):
    """Normalize N vector"""
```
方法体完全缺失。这个方法在多处被调用（第398、405-407、1812、1836行），每次调用都会抛出 `SyntaxError` 或返回 None 导致后续计算崩溃。

**修复建议：** 补充实现：
```python
def _normalize(self, N):
    """Normalize N vector"""
    norm = np.linalg.norm(N)
    if norm < 1e-12:
        return N
    return N / norm
```

这是整个项目最紧急的bug。Git历史显示 commit `25c112f` 曾修复过截断问题，但当前版本仍然缺失。

### 2. Dimer旋转中引用未定义变量 `ns_mag`

`cosmos_search.py` 第1704行：
```python
print(f"Atom {i:3d}: N_i=[...], |N_i|={ns_mag:.4f}")
```
`ns_mag` 从未定义。应该改为 `np.linalg.norm(n_vec)`：
```python
print(f"Atom {i:3d}: N_i=[...], |N_i|={np.linalg.norm(n_vec):.4f}")
```

### 3. `_bias_dimer_rotation` 中 F0 ≈ -F1 的近似不合理

第1632-1633行：
```python
F0_approx = -F1
C = np.dot((F0_approx - F1), N) / delta_R
```

这意味着 `C = np.dot(-2*F1, N) / delta_R`，这在物理上是错误的。在真正的dimer方法中：
- F0 是原始位置R0的力（可以通过一次力计算得到）
- F1 是偏移位置R1的力
- 曲率 `C = (F1 - F0) · N / delta_R`（或某些变体用 `(F0 - F1)`）

用 `-F1` 代替 F0 的假设只在完全平坦的PES上成立（力完全对称），而这正是dimer方法试图逃离的情况。这会导致：
- 曲率估计系统性偏差
- dimer方向收敛到错误方向
- 在陡峭区域（最关键的区域）误差最大

**修复建议：** 直接在R0处做一次力计算：
```python
atoms_temp0 = atoms.copy()
atoms_temp0.calc = self.bias_calc
F0 = atoms_temp0.get_forces().flatten()
C = np.dot((F0 - F1), N) / delta_R
```
或者使用ASE dimer方法（`_bias_dimer_rotation_ase`），该方法是正确的。

### 4. 重复结构仍然被添加到pool — 违反pool设计意图

第1529行：
```python
print("Structure is duplicate, not added to pool")
self._add_to_pool(basin_atoms) # still add to pool for the debug stage
```

注释说"not added"但紧接着就添加了。第1534行也类似——被拒绝的结构也被添加到pool。这意味着 `pool` 和 `real_energies` 包含所有结构（包括重复和被拒绝的），使得pool不再是"唯一极小值集合"，违反了README中的承诺。

`_add_to_pool` 还会写 `all_minima.xyz`，所以输出文件也会包含重复结构。

**修复建议：** 区分"trace pool"（记录所有尝试）和"unique minima pool"（只记录唯一极小值）。或者简单修复：
```python
# 被接受但重复：不加入unique pool
if not is_duplicate:
    is_new_minimum = True
    self._add_to_pool(basin_atoms)
else:
    print("Duplicate, not added to unique pool")

# 被拒绝：不加入任何pool
# basin_atoms 保持不变
```

---

## 🟠 重要问题（影响效率或结果质量）

### 5. 邻接矩阵构建用 O(n²) 距离计算 — 性能瓶颈

`_build_weighted_adjacency` 第543-558行和 `_get_bond_pairs` 第491-514行都使用双循环 `for i ... for j ...` 计算所有原子间距离。对于大体系（>500原子），这会非常慢。

**修复建议：** 使用 ASE 的 `NeighborList` 或 scipy.spatial.cKDTree 预计算邻居列表：
```python
from ase.neighborlist import NeighborList
nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
nl.update(atoms)
# 只遍历实际邻居，而不是所有原子对
```

### 6. `_filter_switchable_bonds` 每次调用重建整个邻接矩阵

`_generate_rd_bond_switch` 调用 `_filter_switchable_bonds`，后者调用 `_build_weighted_adjacency` 和 `_get_local_coordination`（又调用 `_build_weighted_adjacency`），然后 `_generate_rd_bond_switch` 自己又再调用 `_build_weighted_adjacency`。同一个结构上重复构建3次邻接矩阵。

**修复建议：** 缓存邻接矩阵，只在结构改变时重建：
```python
# 在 run() 循环中，每个MC步开始时构建一次
self._cached_A = self._build_weighted_adjacency(basin_atoms, ...)
self._cached_coord = self._get_local_coordination_from_A(self._cached_A)
```

### 7. Laplacian模式中随机3D方向破坏了模式的集体性

第1078-1084行：
```python
rand_dirs = np.random.randn(n, 3)
rand_dirs /= (np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12)
mode_full[3 * global_i:3 * global_i + 3] = vec_mob[local_i] * rand_dirs[global_i]
```

每个原子给一个随机3D方向乘以Laplacian标量。这破坏了Laplacian模式的物理含义——低频模式应该是所有原子沿着**一致方向**的集体移动，而不是每个原子朝随机方向振动。

随机方向使得位移变成了噪声而非集体运动，极大削弱了Laplacian模式的核心优势。

**修复建议：** 使用全局一致方向或模式本身的物理含义：
```python
# 方案A: 单一全局方向（最简单，保留集体性）
global_dir = np.random.randn(3)
global_dir /= np.linalg.norm(global_dir) + 1e-12
for local_i, global_i in enumerate(mobile_indices):
    mode_full[3*global_i:3*global_i+3] = vec_mob[local_i] * global_dir

# 方案B: 交替符号的呼吸模式
sign = np.random.choice([-1, 1])
for local_i, global_i in enumerate(mobile_indices):
    mode_full[3*global_i:3*global_i+3] = sign * vec_mob[local_i] * global_dir
```

### 8. Bond rotation 中 group 包含了非mobile的原子但不应用位移

第691-703行，group 用 `self.mobile_mask[k]` 过滤了加入条件，但 `dik < 1.25 * (ri + rk)` 的判断没有考虑 MIC (minimum image convention)。对于周期性体系，距离计算应使用 `atoms.get_distance(i, k, mic=True)`。

同样的问题存在于 `_get_bond_pairs`（第511行用了mic=True）和 `_build_weighted_adjacency`（第549行用了mic=True），但 `bond_rotation` 的group构建没有用mic。

**修复建议：**
```python
dik = atoms.get_distance(i, k, mic=True)
djk = atoms.get_distance(j, k, mic=True)
```

### 9. `_min_pair_distance` 实现效率低且可能有bug

第1285-1307行：只计算了 `mobile_idx[0]` 到所有其他mobile原子距离的最小值，然后逐行计算。但这只考虑了mobile原子间的距离，忽略了mobile和fixed原子间的重叠（对于被拒绝的情况，fixed原子约束确实会阻止严重重叠，但0.4Å的阈值可能不够保守）。

更严重的是：算法在循环中只从 `mobile_idx[i]` 到 `mobile_idx[i:]`（只计算"后面"的原子），遗漏了 `mobile_idx[i]` 到 `mobile_idx[:i]` 的距离。虽然这对于无序体系大致可以（因为前一个原子到当前原子的距离已经在之前计算过），但如果原子顺序在climbing中改变，可能会有遗漏。

**修复建议：** 使用 ASE 的 `get_all_distances()` 方法：
```python
mobile_idx = np.where(self.mobile_mask)[0]
if len(mobile_idx) < 2:
    return float('inf')
D = atoms.get_distances(mobile_idx, mobile_idx, mic=True)
D = np.where(D > 1e-6, D, np.inf)
return D.min()
```

### 10. Gaussian宽度参数在配置中被注释掉但代码中仍在使用

`cosmos_run.py` 第189行注释掉了 `gaussian_width`，但 `bias_calculator.py` 第53-61行中 Gaussian 势能公式 `E_bias += gh * np.exp(-(proj**2) / (2 * gw**2))` 使用 `gw` 作为宽度参数。在 `run()` 方法中，第1412行使用 `adaptive_gw` 作为宽度。

这意味着 Gaussian 宽度完全由自适应逻辑控制，用户无法通过配置文件设置初始值。README中提到的 `width` 参数实际上被忽略了。

**修复建议：** 恢复 `gaussian_width` 作为初始值：
```python
gaussian_width = gaussian_config.get('width', 0.2)
gaussian={'gaussian_height': gaussian_height, 'gaussian_width': gaussian_width, ...}
# 在 run() 中：adaptive_gw = self.gaussian_width  # 用用户设置初始化
```

### 11. `_compute_laplacian_modes` 中 `n` 变量含义混乱

第1052行 `n = A.shape[0]` 是总原子数，但第1078行 `rand_dirs = np.random.randn(n, 3)` 也用了 `n`。然而邻接矩阵是对所有原子构建的，而Laplacian只在mobile子集上计算。`rand_dirs` 应该只需要为mobile原子生成，但用全局 `n` 生成了所有原子。虽然后续只使用 `mobile_indices` 对应的行，浪费不大，但如果 `n` 的含义不一致，未来维护容易出错。

### 12. `infer_geometry_type` 使用绝对坐标判断vacuum — 周期性体系不适用

`cosmos_utils.py` 第277-303行：通过检查原子位置与cell边界的绝对距离来判断vacuum。对于周期性体系（pbc=True），原子可以在0附近（fractional坐标），margin计算会误判。

**修复建议：** 使用 fractional 坐标 + PBC 检查：
```python
if not any(atoms.pbc):
    # 非周期性：检查绝对margin
    ...
else:
    # 周期性：沿pbc=False的轴检查vacuum
    ...
```

---

## 🟡 一般改进建议

### 13. 两个dimer方法共存 — `_bias_dimer_rotation` 有bug且未使用

代码中有两个dimer旋转方法：
- `_bias_dimer_rotation` (第1585-1707行) — 手动实现，有F0≈-F1 bug，且只在debug时使用
- `_bias_dimer_rotation_ase` (第1709-1804行) — 使用外部dimer库，是实际使用的方法

`run()` 方法第1401行只调用 `_bias_dimer_rotation_ase`。手动实现的方法既不正确也不使用，应删除或标记为deprecated。

### 14. `print_xyz` 硬编码 "xyz/" 子目录

`cosmos_utils.py` 第530行 `os.makedirs("xyz")` 硬编码了相对路径。在多任务并行或不同工作目录下会出问题。应该使用 `self.output_dir` 参数。

### 15. `climb_info_file` 硬编码路径且只在debug模式写入header

第1343行 `open("climb.info","w")` 硬编码相对路径，且第1344行只写了一行header但后续内容只在debug模式下写入（第1442、1467行），导致非debug模式下文件只有一行header。

### 16. Adaptive权重只调整scheme概率，不调整mode权重

README说"Adaptive mode weighting"暗示会调整各mode的权重，但实际实现 `_recompute_adaptive_weights` 只调整scheme间的概率。Mode-level的统计被收集但从未用于权重计算。

这是一个设计选择而非bug，但与文档不一致。如果确实只调整scheme，README应该更明确。

### 17. `_generate_rd_shell` 使用固定3.5Å cutoff

shell模式使用硬编码的3.5Å cutoff而不是从邻接矩阵获取邻居。对于大原子（如Au, covalent_radius≈1.34Å，1.25×2×1.34=3.35Å）这大致合理，但对小原子（C, 1.25×2×0.76=1.9Å）这会包含远超第一邻居的原子。

**修复建议：** 使用邻接矩阵的邻居定义：
```python
A = self._build_weighted_adjacency(atoms)
shell_indices = [k for k in range(self.n_atoms) if k != center and self.mobile_mask[k] and A[center, k] > 0.01]
```

### 18. `TeeLogger` 不处理异常情况

`cosmos_run.py` 第19-35行：TeeLogger 在 `__init__` 中打开文件但没有在异常时关闭。`close()` 方法存在但从未被调用（main函数没有调用 `sys.stdout.close()`）。如果程序崩溃，日志文件可能不完整。

**修复建议：** 使用 `try/finally` 确保关闭：
```python
logger = TeeLogger(log_path, mode='w')
sys.stdout = logger
try:
    ...
finally:
    sys.stdout = logger.terminal
    logger.close()
```

### 19. `_get_energy_based_scales` 中reference_energies用了max而非min

第297行：
```python
reference_energies[element] = np.max(element_energies)
```

用同种元素中最高原子能作为参考，使得所有其他原子的 `normalized_energies = E_atom - E_ref` 为负值。然后 `scales = np.exp(negative * AE_factor)` 给小于1的值。

这个设计的意图是让低能原子位移小、高能原子位移大，但用max作为参考使得"最高能原子"的scale=exp(0)=1，而不是最大。如果用min作为参考，高能原子会有大于1的scale，低能原子scale≈1，更符合"高能原子大位移"的直觉。

当前设计虽然不是bug（归一化后效果类似），但语义不直观。

### 20. DeepMD calculator的stress计算可能有误差

`cosmos_utils.py` 第504行：
```python
stress = -0.5 * (v[0].copy() + v[0].copy().T) / self.atoms.get_volume()
```

`v[0].copy() + v[0].copy().T` 对一个3×3矩阵做对称化。但virial本身不一定需要对称化——stress的正确公式是 `stress = -virial / volume`，virial在某些定义下已经是对称的。`.copy() + .copy().T` 会给非对称部分乘以0、对称部分乘以2，然后乘0.5得到原始对称部分。这等价于只取virial的对称部分，可能丢失了antisymmetric贡献。

### 21. Bond switch中两个group同方向旋转 — 物理效果需要验证

第815-816行：
```python
rotate_group(group_i, sign * theta)
rotate_group(group_j, sign * theta)  # same direction — rigid bond rotation
```

注释说"same direction — rigid bond rotation"。但两个邻域同方向旋转的效果是：bond i-j 不旋转（两端相对位移为零），而是整个局部结构做刚性旋转。这与 Stone-Wales 变换（两个半面相对旋转90°）的物理机制不同。

对于 Stone-Wales：需要两组**反方向**旋转才能产生拓扑变化。同方向旋转只产生整体刚性旋转，不会改变键拓扑。

**修复建议：** 验证物理效果。如果目标是拓扑变化，应该：
```python
rotate_group(group_i, sign * theta)
rotate_group(group_j, -sign * theta)  # opposite direction — bond switch
```
但如果dimer rotation后续会修正方向，同方向可能只是初始化策略。需要明确设计意图。

---

## 📊 性能优化建议

| 优化项 | 当前 | 建议 | 预期提升 |
|--------|------|------|----------|
| 邻接矩阵构建 | O(n²) 每次重建 | NeighborList + 缓存 | 10-100x for large systems |
| Laplacian eigendecomposition | np.linalg.eigh 全矩阵 | scipy.sparse.linalg.eigsh | 5-50x for >200 atoms |
| SOAP去重 | 每次重算所有pool结构的descriptor | 缓存pool descriptors | 2-10x as pool grows |
| Dimer rotation ASE | 每次新建MinModeAtoms | 复用control对象 | ~2x |

---

## ✅ 设计亮点

1. **图论驱动集体移动的设计思路**非常出色。community/laplacian/bond_switch 三种模式覆盖了不同的物理机制，互补性强。
2. **自适应bandit权重**保证了探索多样性，floor机制防止模式饥饿。
3. **Per-scheme参数覆盖**允许不同策略有不同的步长/Gaussian参数，灵活性好。
4. **偏置势分解**（E_base, E_bias, E_wall）便于debug和物理分析。
5. **移动原子约束**（FixAtoms + wall potential）双保险，确保约束不被违反。

---

## 优先级排序

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P0 | #1 `_normalize`截断 | 程序无法运行 |
| P0 | #2 `ns_mag`未定义 | debug模式崩溃 |
| P1 | #3 F0≈-F1近似 | dimer方向错误（仅影响手动实现） |
| P1 | #4 重复结构加入pool | pool语义错误，输出混乱 |
| P1 | #7 Laplacian随机方向 | 核心算法物理意义被破坏 |
| P1 | #21 bond_switch同方向旋转 | 拓扑变化机制可能有误 |
| P2 | #5/#6 邻接矩阵性能 | 大体系运行慢 |
| P2 | #8/#9 MIC和距离计算 | 周期性体系结果错误 |
| P2 | #10 Gaussian宽度参数 | 配置与代码不一致 |
| P3 | #11-#20 | 可维护性、文档一致性、小优化 |
