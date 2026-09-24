# lll_reduce — 有理 Gram–Schmidt 的 LLL 格基约化与盒约束最近格向量

可复用的离线数论计算模块：

1. 对满行秩整数行基执行 **LLL 约化（δ = 3/4）**，Gram–Schmidt 系数全部用
   `fractions.Fraction` **精确**计算（无浮点、无容差），返回约化基、整数幺模
   变换矩阵与逐步变换日志，并附带**独立验证器**核验证书；
2. **盒约束最近格向量（CVP）**：给定有理目标点与调用者提供的有限**原基整数
   系数盒**，在 n ≤ 3 时精确枚举盒内全部最近格向量，返回原基整数系数、格点
   与精确平方距离，并列点按原系数字典序排列。**只保证盒内最优**。

## 环境与依赖

- Python **3.14.7**（仅使用 3.14 标准库：`fractions`、`dataclasses`、
  `math.isqrt` 等）。
- **无第三方依赖**，无需安装任何包；全部计算离线完成，不访问网络、文件、
  环境变量或任何外部服务。
- Linux 原生运行。

## 运行命令

```bash
# 全部测试（63 个），在仓库根目录执行：
python -m unittest discover -s tests -v

# 固定输入演示（约 1 秒，真实计算，无 sleep、无预录结果）：
python demo.py
```

## 公开接口

```python
from lll_reduce import (
    reduce_basis,        # LLL 主算法；lll_reduce 为同义别名
    verify_reduction,    # LLL 独立验证器
    round_half_even,     # 最近整数、并列向偶数（banker's rounding）
    nearest_vector,      # 盒约束最近格向量（CVP，n <= 3）
    LLLResult, LogEntry, VerificationReport, NearestVectorResult,
    DELTA, MAX_DIM, CVP_MAX_DIM, DEFAULT_BUDGET,
    LLLError, InvalidParameterError, InvalidBasisError,
    RankDeficientError, BudgetExhaustedError,
)
```

### `reduce_basis(A, budget=1_000_000) -> LLLResult`

- `A`：n×m 整数矩阵，**行向量为基**；`1 <= n <= 8`，`m >= n`，满行秩。
  接受 `list`/`tuple` 嵌套；元素必须是精确 `int`。
- `budget`：正整数操作预算。

返回 `LLLResult`（dataclass，字段均为只读）：

| 字段 | 含义 |
|---|---|
| `basis: list[list[int]]` | 约化基 B，满足 B = U·A |
| `transform: list[list[int]]` | 整数幺模矩阵 U（det U = ±1） |
| `log: list[LogEntry]` | 每次原子变换日志，`step` 从 1 连续编号 |
| `reduced: bool` | **仅当 True 时为合格约化证书** |
| `budget / ops_used / swaps / index` | 预算、已用计费操作、交换次数、停滞下标 |

`LogEntry`：`step, kind, index_k, index_j, coefficient, detail`。
- `kind="size_reduce"`：`b_k ← b_k − q·b_j`，`coefficient=q`（整数）；
- `kind="swap"`：相邻行 `b_{k-1} ↔ b_k` 交换，`coefficient=None`。

成功时 `reduced=True`，B 满足（下标 0 起，μ 为 GS 系数，`|b*_i|²` 为正交投影平方范数）：

- **size reduction**：对所有 `j < i`，`|μ_{i,j}| ≤ 1/2`；
- **Lovász 条件**：`|b*_i|² + μ_{i,i-1}² |b*_{i-1}|² ≥ (3/4) |b*_{i-1}|²`。

### `verify_reduction(A, B, U) -> VerificationReport`

**独立实现**（不复用核心代码），用自带的有理 Gram–Schmidt 与 Bareiss
精确整数行列式逐项检查：

1. 形状合法且均为整数矩阵；
2. `B == U·A`（精确整数乘法，逐元素报告首个不符位置）；
3. `U` 幺模：`det U ∈ {+1, −1}`；
4. B 满行秩（GS 投影非零）；
5. 所有 `|μ_{i,j}| ≤ 1/2`；
6. 每对相邻行满足 Lovász 条件。

`report.ok` 为布尔，`report.failures` 给出可定位的失败原因列表。
比较全部是 `Fraction` **精确**比较，没有任何 epsilon/容差。

### `nearest_vector(A, target, bounds, *, budget=DEFAULT_BUDGET) -> NearestVectorResult`

在调用者给定的**有限原基整数系数盒**内，求离有理目标点最近的全部格向量
（精确 CVP 枚举，**n ≤ 3**）。

- `A`：n×m 整数行基，`1 <= n <= 3`，m ≥ n，满行秩（校验规则同
  `reduce_basis`，n > 3 抛 `InvalidBasisError`）。
- `target`：长度 m 的序列，元素为精确 `int` 或 `fractions.Fraction`；
  `float`（含 NaN/±Inf）与 `bool` 一律拒绝。目标点不必在格的行空间内
  （长方形基时，行空间正交补分量是所有格点共有的常数，已计入平方距离）。
- `bounds`：有限系数盒，两种形式：
  - 非负整数 `B`：每个原基系数 `c_i ∈ [-B, B]`；
  - 长度 n 的 `(lo_i, hi_i)` 序列：均为精确 int 且 `lo_i <= hi_i`
    （逐分量非对称，允许单点盒）。
- `budget`：透传给内部 LLL 的正整数操作预算。

返回 `NearestVectorResult`（只读 dataclass）：

| 字段 | 含义 |
|---|---|
| `coefficients: list[tuple[int,...]]` | 全部并列最优点的**原基**整数系数 c，格点 = c·A；按系数元组**字典序升序** |
| `vectors: list[list[int]]` | 与 coefficients 一一对应的格点（整数向量） |
| `squared_distance: Fraction` | 并列点共有的精确平方距离 ‖t−c·A‖²（整数距离时分母为 1） |
| `optimal_in_box: bool` | 恒为 `True`：**只保证盒内最优**，盒外是否存在更近点不予判断 |
| `bounds` | 归一化后的逐分量 `(lo, hi)` 元组 |
| `target / basis` | 归一化后的 Fraction 目标与原基副本 |
| `nodes_visited / leaves_evaluated / box_points` | 搜索节点数、叶子数、盒内格点总数 `∏(hi−lo+1)`（诊断用） |

算法：先复用 `reduce_basis` 得 B = U·A，在约化基坐标上做深度优先的
Gram–Schmidt 椭球枚举（逐层精确有理增量，整数半径用 `isqrt`，不开平方根、
不用浮点）；盒约束在原系数空间，经精确的 **Fourier–Motzkin 消元**（按 U
符号预计算，节点上只做仿射赋值）投影成每层坐标的可行区间，与椭球区间取交。
初始上界由 Babai 最近平面点夹入盒内给出。最底层直接取离 GS 中线最近的盒内
整数（恰半整数时两个并列），因此全部并列点精确无遗漏。

## 错误语义（非法输入不静默纠正）

| 异常 | 触发条件 |
|---|---|
| `InvalidBasisError` | 非矩阵、空行、行不等长、元素非 `int`、列数 < 行数；`reduce_basis` 行数 > 8、`nearest_vector` 行数 > 3；`.location` 给出如 `A[1][2]` 的定位 |
| `InvalidParameterError` | `budget` 非正整数、`target` 长度/类型不对、`bounds` 非法等；`.parameter/.reason` 可定位 |
| `RankDeficientError` | **数学上证明**行线性相关；`.expected/.rank` 给出期望秩与检测到的实际行秩 |
| `BudgetExhaustedError` | LLL 预算耗尽（资源不足，**不是**无解）；`.partial` 为 `reduced=False` 的中间态快照 |

- **NaN/Infinity 拒绝**：矩阵元素只接受 `int`，`target` 只接受 `int`/`Fraction`；
  任何 `float`（包括 `float('nan')`、`±inf`）一律抛 `InvalidBasisError`/
  `InvalidParameterError`，不做静默数值转换。
- **bool 拒绝**：`bool` 虽是 `int` 子类，矩阵元素、`target`、`bounds` 与
  `budget` 中的 `True/False` 都显式拒绝。
- **预算耗尽 ≠ 证明无解**：`BudgetExhaustedError` 与 `RankDeficientError`
  是不同异常类型。耗尽时 `partial.reduced` 必为 `False`，绝不假称约化；
  中间态仍满足 `B == U·A`（基与变换矩阵在同一次原子操作内同步更新，
  失败不留未说明的部分变更）。`nearest_vector` 是无状态查询：输入非法时
  在任何计算前抛出，不修改任何外部状态；连续失败不影响后续合法调用
  （测试 `test_failure_leaves_no_state` 锁定）。
- **空盒拒绝**：`bounds` 中任何 `lo_i > hi_i` 即 `InvalidParameterError`，
  不返回“空结果”，因为调用者更可能是传参错误。单点盒 `lo_i == hi_i` 合法。

### 预算计费口径

- 初始 Gram–Schmidt：计 `n`；
- 一次 size reduction：计 `1`；
- 一次相邻交换（含交换后全量精确 GS 重算）：计 `n`。

计费是确定的、可从日志逐步复算（测试中有用例锁定精确计数）。
`DEFAULT_BUDGET = 1_000_000`，对 n ≤ 8 的输入极为充裕；用户也可传入小预算
观察确定性的中途终止。

## 设计取舍

1. **全程精确有理数**：μ、`|b*|²`、Lovász 比较、CVP 的距离增量与
   Fourier–Motzkin 边界都是 `Fraction`，**不存在浮点舍入误差，也不设容差**；
   10⁸⁰ 量级输入仍精确。代价是大整数运算速度慢于浮点实现，但 n ≤ 8（CVP
   n ≤ 3）的契约下完全可接受（整套 63 测约 3–4 秒）。
2. **交换后全量重算 GS**（n ≤ 8），而非手工维护 μ/D 的增量公式：
   换取实现直观、杜绝增量更新的下标/符号错误；正确性由独立验证器兜底。
3. **并列舍入向偶数**：`round_half_even` 用纯整数比较实现（比较 `2r` 与
   分母，不引入浮点），`±3/2 → ±2`、`±1/2 → 0`。
4. **行基、整数域**：严格按本题契约，不支持列基约定、实数/浮点输入或
   δ 参数化；不宣称兼容 fplll/FPLLL 等行业实现的全部选项与输出排序。
5. **CVP 只做盒内精确枚举**：盒由调用者显式给定且有限，结果带
   `optimal_in_box=True` 标记，**不声明盒外最优性**，也不做无限域截断证明。
   盒在原系数空间定义，经 FM 消元映射到约化坐标参与剪枝，因此“目标极远、
   盒很小”和“格极歪斜、盒很大”两类情形都不会退化成全盒遍历（后者在深层
   仍沿椭球面探索，节点数与盒半径近似无关）。
6. **测试期望值独立性**：测试目录内的 `tests/_reference.py` 是独立编写的
   教科书 LLL（刻意采用不同的并列舍入规则）、二维短向量穷举与盒 CVP 全盒
   穷举，不调用被测核心；CVP 另有手算定值（并列、歪斜格精确点、行空间外
   残差、单点盒）。对约化基再次约化的“往返一致”只作补充断言。

## 已知限制（如实声明）

- LLL 维数上限 n = 8；CVP 维数上限 n = 3；超过即 `InvalidBasisError`，
  不自动放宽（本次迭代不实现 n ≥ 4 的 Schnorr–Euchner/Kannan 枚举）。
- LLL 给的是 δ-约化基，**不保证给出格的最短向量**；二维有
  `‖b₀‖² ≤ 2·λ₁²` 的界，测试与 demo 用独立穷举对照但不要求首向量最短。
- CVP **只保证给定有限盒内的最优性**：盒外可能有更近的格点；函数不判断
  盒是否“足够大”，也不返回盒外任何信息。不提供无限域 CVP、最近 k 点或
  暂停/恢复接口（本迭代按题面不做）。
- CVP 枚举代价随实例形状变化：LLL 约化 + 椭球/FM 剪枝后，常规小盒与
  歪斜幺模格大盒（如 4×10¹² 个候选）均为毫秒级；但“极大盒且无约束最近点
  在盒外”的极端长条情形需要沿整个椭球截面探索（实测 B=10⁶ 量级约十秒、
  数万节点）。题面未给大规模性能指标，未为此引入增量枚举等复杂结构。
- 输出向量的具体次序/符号与并列舍入策略有关；本库保证的是约化条件与
  `B = U·A` 证书，不保证与其他 LLL 实现逐向量相同。CVP 的并列点次序则是
  契约的一部分：严格按原基系数元组字典序。
- 约化基的长度受格协体积约束：若输入基的行列式本身为 N 量级，约化基中
  保留长向量是数学必然，LLL 不能把它变短。
- 不提供 LLL 续算/断点恢复 API（预算耗尽后需重新调用）；日志本身已足够
  完整，外部可据此重建任意中间态。
- 未做也不需要并行化、GPU、数据库或网络组件。

## 目录结构

```
lll_reduce/
    __init__.py    # 公开接口导出
    core.py        # LLL 主算法、舍入、预算、输入校验
    cvp.py         # 盒约束最近格向量：GS 椭球枚举 + FM 盒剪枝（n <= 3）
    verifier.py    # LLL 独立验证器（自带 GS + Bareiss 行列式）
    errors.py      # 错误类型
tests/
    test_lll.py    # LLL 必要测试（unittest）
    test_cvp.py    # CVP 必要测试（unittest）
    _reference.py  # 独立教科书 LLL、二维短向量穷举、盒 CVP 全盒穷举
demo.py            # 固定输入演示（LLL + CVP）
README.md
```
