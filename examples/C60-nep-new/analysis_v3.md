# GraSMoS C60 运行分析与参数优化

## 两次运行对比

| 指标 | Run 1 (3 scheme, mixed) | Run 2 (15 scheme, focused) |
|------|------------------------|----------------------------|
| 步数 | 1214 (未跑完5000) | 1047 (未跑完5000) |
| 接受率 | 78.5% | 92.3% |
| 重叠拒绝率 | 7.8% (95次) | 2.7% (28次) |
| 最佳能量 | -460.5 eV | -470.8 eV |
| 最佳能量步数 | Step 830 | Step 115 |
| 停滞步数 | ~370 (step 850-1213) | ~850 (step 250-1047) |
| 停滞时当前能量 | -442.1 eV | -430.6 eV |

## 核心问题诊断

### 1. 严重停滞 — 最大问题

两次运行都陷入严重停滞：
- Run 1：找到 -460.5 eV 后，当前结构停在 -442.1 eV（比最佳高 18 eV），370步没有新突破
- Run 2：找到 -470.8 eV 后，停在 -430.6 eV（比最佳高 40 eV），850步没有新突破

**原因：** 温度 T=1000K 太低。C60 的构型转变势垒约 5-10 eV，而 kBT = 0.086 eV。Metropolis 几乎不可能接受任何 uphill move。一旦陷入一个局部极小，就永远被困住。

**修复：** 提高温度到 1500K（kBT = 0.13 eV），或使用退火策略（前 1000 步 2000K → 后 4000 步 800K）。

### 2. bond_switch 位移失控

Run 2 的 step 250 出现 Norm_dist=37.96 Å、max_dr=14.33 Å 的灾难性位移。
原因：`bond_switch` 锁定 dimer 时，代码将 `max_dr` 设为 infinity、`average_dr` 放大 4倍（代码第1398-1399行）。对 60 原子的 C60 cluster，这个放大远远超出合理范围。

**修复：** bond_switch 的 scheme 不应该用很大的 `max_dr`，因为锁定模式本身已经放大了位移。建议 bond_switch scheme 用 `max_dr ≤ 1.5`、`average_dr ≤ 0.5`。

### 3. 大步长的 overlap 问题

Run 1 的 95 次 overlap 全来自 scheme 0（max_dr=2）。Run 2 的 bond_switch 大步长 scheme（3-5）也全是 0% 接受率 + 高 overlap。

自适应系统**跳过 overlap 步**不统计（代码 `if self.adaptive and not overlap`），导致大步长 scheme 不会被自适应惩罚。这很危险：一个 scheme 10次里9次 overlap、1次成功且能量降 20 eV，它的 score 反而很高。

**修复：** 应将 overlap 也纳入统计（overlap = rejected），或限制大步长 scheme 的初始概率。

### 4. ADAPTIVE 输出缺失

两次运行都没有 ADAPTIVE 输出行。虽然代码中有打印语句（第1322行），但在日志中找不到。可能原因：
- `_step_counter % adaptive_interval == 0` 条件在 overlap 步时不增加 counter，导致 counter 增长太慢
- Run 1: 1214步中 95 次 overlap → 有效计数只有 ~1119 → 1119/20 ≈ 56 次自适应更新
- Run 2: 1047步中 28 次 overlap → ~1019 有效 → 1019/30 ≈ 34 次

这些数字说明 adaptive 应该打印过。问题可能是 adaptive_smoothing=0.5（Run 2）使得权重快速漂移，但输出确实应该出现。需要进一步调试。

### 5. 小步长 scheme 能量下降微弱

Run 2 中 scheme 1（nl, medium dr, 95.9% accept）只贡献了 6.2 eV 能量下降；scheme 7（laplacian, small dr, 98.2%）只有 2.8 eV。而 scheme 12（thermo+atomic, 97%）贡献了 12 eV。这意味着小步长虽然接受率高，但探索能力弱。

## 优化后的 input_v3.json 设计思路

### 温度：1000K → 1500K
更高的温度允许 Metropolis 接受更大的 uphill move，减少停滞。对 C60，1500K 的 kBT=0.13 eV 仍然很低，但比 0.086 eV 提升了 50%。

### Scheme 设计：5 个 scheme（不是 15 个）

**3 个稳健 scheme（总概率 80%）：**
- Scheme 0: mixed modes + 大步长 (max_dr=0.8) — 探索
- Scheme 1: mixed modes + 中步长 (max_dr=0.4) — 平衡
- Scheme 2: mixed modes + 小步长 (max_dr=0.15) — 精细

**2 个激进 scheme（总概率 20%）：**
- Scheme 3: graph-heavy (community 30%, bond_switch 25%, laplacian 15%) + 大步长 (max_dr=1.5)
- Scheme 4: graph-heavy + 中步长 (max_dr=0.6)

**关键设计原则：**
1. 没有 pure bond_switch scheme — bond_switch 应与其他 modes 混合使用，避免锁定 dimer 时位移失控
2. 每个稳健 scheme 都包含 thermo/atomic/nl 作为"安全网"，保证总有可接受的基线位移
3. nl 权重最高 (30%) — 从 Run 1/2 的数据看，nl 是 C60 最有效的模式
4. 大步长的 max_dr 上限控制在 1.5（不是 3），避免 overlap

### AE_factor: 1.0 → 2.0
Run 1 的 scheme 0（大步长）能量下降巨大但 50% 接受率。提高 AE_factor 让高能原子得到更大的位移，可能提高成功步的能量下降质量。

### Dimer rotation_param: 8 → 15
两次运行都用了 a=8，这是较小的值（偏置势较弱）。增加到 15 可以让 dimer 更有效地找到低曲率方向，减少在势能面上的无效旋转。

### Gaussian height: 0.8 → 0.5
0.8 eV 的高斯偏置太强，尤其在 60 原子体系上可能导致 climbing 过度偏移。降低到 0.5 eV 可以让 climbing 更可控。

### adaptive_alpha: 3.0 → 2.0
alpha=3.0 使得接受率的权重过于激进（接受率差一点就被大幅惩罚）。降低到 2.0 使权重变化更平滑。

### adaptive_interval: 30 → 15
更频繁的权重更新让系统能更快响应不同 scheme 的实际表现。

### Optimizer fmax: 0.05 → 0.03
最终优化需要更精确的收敛，确保找到真正的极小值而不是鞍点附近。
