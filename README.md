# lll_reduce — 有理 Gram–Schmidt 的 LLL 格基约化

可复用的离线数论计算模块：对满行秩整数行基执行 **LLL 约化（δ = 3/4）**，
Gram–Schmidt 系数全部用 `fractions.Fraction` **精确**计算（无浮点、无容差），
返回约化基、整数幺模变换矩阵与逐步变换日志，并附带**独立验证器**核验证书。

## 环境与依赖

- Python **3.14.7**（仅使用 3.14 标准库：`fractions`、`dataclasses` 等）。
- **无第三方依赖**，无需安装任何包；全部计算离线完成，不访问网络、文件、
  环境变量或任何外部服务。
- Linux 原生运行。

## 运行命令

```bash
# 必要测试（38 个），在仓库根目录执行：
python -m unittest discover -s tests -v

# 固定输入演示（约 1 秒，真实计算，无 sleep、无预录结果）：
python demo.py
```

## 公开接口

```python
from lll_reduce import (
    reduce_basis,        # 主算法；lll_reduce 为同义别名
    verify_reduction,    # 独立验证器
    round_half_even,     # 最近整数、并列向偶数（banker's rounding）
    LLLResult, LogEntry, VerificationReport,
    DELTA, MAX_DIM, DEFAULT_BUDGET,
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

## 错误语义（非法输入不静默纠正）

| 异常 | 触发条件 |
|---|---|
| `InvalidBasisError` | 非矩阵、空行、行不等长、元素非 `int`、列数 < 行数、行数 > 8；`.location` 给出如 `A[1][2]` 的定位 |
| `InvalidParameterError` | `budget` 非正整数等；`.parameter/.reason` 可定位 |
| `RankDeficientError` | **数学上证明**行线性相关；`.expected/.rank` 给出期望秩与检测到的实际行秩 |
| `BudgetExhaustedError` | 预算耗尽（资源不足，**不是**无解）；`.partial` 为 `reduced=False` 的中间态快照 |

- **NaN/Infinity 拒绝**：矩阵元素只接受 `int`；任何 `float`（包括
  `float('nan')`、`±inf`）一律抛 `InvalidBasisError`，不做静默数值转换。
- **bool 拒绝**：`bool` 虽是 `int` 子类，矩阵元素与 `budget` 中的
  `True/False` 都显式拒绝。
- **预算耗尽 ≠ 证明无解**：`BudgetExhaustedError` 与 `RankDeficientError`
  是不同异常类型。耗尽时 `partial.reduced` 必为 `False`，绝不假称约化；
  中间态仍满足 `B == U·A`（基与变换矩阵在同一次原子操作内同步更新，
  失败不留未说明的部分变更）。

### 预算计费口径

- 初始 Gram–Schmidt：计 `n`；
- 一次 size reduction：计 `1`；
- 一次相邻交换（含交换后全量精确 GS 重算）：计 `n`。

计费是确定的、可从日志逐步复算（测试中有用例锁定精确计数）。
`DEFAULT_BUDGET = 1_000_000`，对 n ≤ 8 的输入极为充裕；用户也可传入小预算
观察确定性的中途终止。

## 设计取舍

1. **全程精确有理数**：μ、`|b*|²`、Lovász 比较都是 `Fraction`，因此
   **不存在浮点舍入误差，也不设容差**；10⁸⁰ 量级输入仍精确。代价是大整数
   运算速度慢于浮点实现，但 n ≤ 8 的契约下完全可接受（测试套件含 10⁸⁰
   用例，整套 38 测约 3–6 秒）。
2. **交换后全量重算 GS**（n ≤ 8），而非手工维护 μ/D 的增量公式：
   换取实现直观、杜绝增量更新的下标/符号错误；正确性由独立验证器兜底。
3. **并列舍入向偶数**：`round_half_even` 用纯整数比较实现（比较 `2r` 与
   分母，不引入浮点），`±3/2 → ±2`、`±1/2 → 0`。
4. **行基、整数域**：严格按本题契约，不支持列基约定、实数/浮点输入或
   δ 参数化；不宣称兼容 fplll/FPLLL 等行业实现的全部选项与输出排序。
5. **测试期望值独立性**：测试目录内的 `tests/_reference.py` 是独立编写的
   教科书 LLL（刻意采用不同的并列舍入规则）与二维短向量穷举，不调用被测
   核心；另有手算定值、日志重放与独立验证器。对约化基再次约化的
   “往返一致”只作补充断言。

## 已知限制（如实声明）

- 维数上限 n = 8；超过即 `InvalidBasisError`，不自动放宽。
- LLL 给的是 δ-约化基，**不保证给出格的最短向量**；二维有
  `‖b₀‖² ≤ 2·λ₁²` 的界，测试与 demo 用独立穷举对照但不要求首向量最短。
- 输出向量的具体次序/符号与并列舍入策略有关；本库保证的是约化条件与
  `B = U·A` 证书，不保证与其他 LLL 实现逐向量相同。
- 约化基的长度受格协体积约束：若输入基的行列式本身为 N 量级，约化基中
  保留长向量是数学必然，LLL 不能把它变短。
- 不提供续算/断点恢复 API（预算耗尽后需重新调用）；日志本身已足够完整，
  外部可据此重建任意中间态。
- 未做也不需要并行化、GPU、数据库或网络组件。

## 目录结构

```
lll_reduce/
    __init__.py    # 公开接口导出
    core.py        # LLL 主算法、舍入、预算、输入校验
    verifier.py    # 独立验证器（自带 GS + Bareiss 行列式）
    errors.py      # 错误类型
tests/
    test_lll.py    # 必要测试（unittest）
    _reference.py  # 独立教科书 LLL + 二维短向量穷举（测试专用）
demo.py            # 固定输入演示
README.md
```
