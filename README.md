# 有理 Gram–Schmidt 的 LLL 格基约化（Python 标准库）

在精确有理数（`fractions.Fraction`）上实现的可复用 LLL 格基约化模块：输入满行秩
整数 **行基**，固定参数 `delta = 3/4`，返回 LLL-约化基 `B`、整数幺模变换 `U`
（`B = U*A`）以及每次交换/约化的完整日志；附带与核心算法 **相互独立实现** 的
验证器。全程无浮点、无第三方依赖、完全离线。

## Python 版本与依赖

- Python **3.14.7**（Linux 原生验证通过；仅使用 3.10+ 通用语法，3.10+ 应可运行）。
- **无需安装任何第三方依赖**：只使用标准库（`fractions`、`dataclasses`、`math`、`unittest` 等）。
- 不访问网络、账号、密钥、数据库、中间件、云 API 或 Docker；示例数据由仓库内
  确定性代码（固定种子的 `random.Random` 与硬编码矩阵）生成。

## 运行命令

```bash
# 必要测试（28 个用例，约 1 秒）
python -m unittest discover -s tests -v

# 固定输入演示（真实计算，约 0.1 秒，远小于 8 秒上限；无 sleep、无预录结果）
python demo.py
```

终端命令仅用于启动测试和演示；本项目不提供 Web/CLI 产品。

## 公开接口与数据结构

```python
from lll_reduce import lll_reduce, verify_result, round_half_even
from lll_reduce import (
    DELTA, MAX_DIM, DEFAULT_BUDGET,
    ReductionEvent, LLLResult,
    LLLReduceError, InvalidBasisError, InvalidParameterError,
    LatticeDependentError,
)
from lll_reduce.verify import verify_reduction, VerificationReport
```

### `lll_reduce(A, budget=1_000_000) -> LLLResult`

| 参数 | 契约 |
|---|---|
| `A` | `n` 行 `m` 列整数矩阵（`list`/`tuple` 嵌套），按 **行** 为基向量；`1 <= n <= m`，**满行秩**，`n <= MAX_DIM = 8`。元素必须是真 `int`：拒绝 `bool`、`float`（含 `NaN`/`±Infinity`）、字符串等。 |
| `budget` | 正整数（真 `int`，拒绝 `bool`/浮点）。默认 `1_000_000`。 |

返回不可变 `LLLResult`：

- `basis: list[list[int]]` — 当前基 `B`；
- `transform: list[list[int]]` — 整数幺模矩阵 `U`，恒有 `B = U*A`，`det U = ±1`；
- `events: tuple[ReductionEvent, ...]` — 有序日志。`ReductionEvent` 字段：
  `step`（从 1 起，与预算计数一致）、`kind`（`"size_reduce"` / `"swap"`）、
  `i`、`j`、`coefficient`（size reduction 的整数 `m`，使 `b_i -= m*b_j`；swap 为 `None`）、
  `k`（触发时的算法下标）、`mu_before`（相关的 `Fraction` 投影系数）、`message`；
- `reduced: bool` — 是否完整达到 LLL 约化；
- `exhausted: bool` — 是否因预算耗尽停止（与 `reduced` 互斥）；
- `operations`、`budget`、`rank`。

`as_dict()` 给出可 JSON 化表示（`Fraction` 序列化为 `"p/q"`）。

### 错误语义

- `InvalidBasisError`：类型/形状/有限性违约，信息带可定位坐标（如 `A[1][0] is non-finite (NaN/Infinity)`）。`m < n` 等形状上不可能满秩的情况按输入契约错误处理。
- `InvalidParameterError`：`budget` 非正整数或为 `bool`/浮点。
- `LatticeDependentError(index=k)`：精确算术判定线性相关（第 `k` 个 GS 向量为零，0 起）。这是 **被证明的无解**，立即抛出，与预算无关。
- 预算耗尽 **不抛异常**：返回 `reduced=False, exhausted=True` 的自洽中间态。

### 验证器（独立代码路径）

`verify_reduction(A, B, U, events=None)` / `verify_result(result, A, require_reduced=True)`
返回 `VerificationReport`，全部为精确整数/有理数判定，**无浮点、无容差**：

1. 形状与整数性；2. `B = U*A` 逐项精确相等；3. `det U = ±1`（独立分数 Gauss 消元）；
4. `A` 满行秩；5. `B` 满行秩；6. 所有 `|mu[i][j]| <= 1/2`；
7. Lovász 条件 `D[i] >= (delta - mu[i][i-1]^2) D[i-1]`；
8. 若提供 `events`，在单位阵上重放日志，结果必须等于 `U`。

验证器的 Gram–Schmidt 走 **Gram 矩阵递推** 公式，与核心模块的逐向量正交化是
两套独立实现，不存在“被测代码自证”。测试中的期望值（二维最短向量枚举、行列式）
也在测试文件内独立实现；`B=U*A` 往返一致仅作补充检查。

## 算法与设计取舍

- **GS 表示**：显式保存正交向量 `b*_i` 与系数 `mu`，每次初等行变换后整体重算
  Gram–Schmidt。维数 `n <= 8`，重算代价可忽略，换取实现直白、不变量始终一致，
  避免增量更新 `mu` 的下标错误。
- **Lovász 判定**（经典形式）：`D_k >= (3/4 - mu_{k,k-1}^2) D_{k-1}`，
  `D_i = ||b*_i||^2`。失败则交换 `b_{k-1}, b_k` 并令 `k = max(k-1, 1)`。
- **Size reduction**：惰性执行——先只约化 `mu_{k,k-1}`，Lovász 通过后再补做
  `ell = k-2, ..., 0`。仅当严格 `|mu| > 1/2` 才变换；恰为半整数不动。
- **并列舍入（round half to even）**：`round_half_even(p/q)` 用 `divmod` 取非负
  余数比较 `2r` 与 `q`，半整数并列时选偶数。故 `+3/2 → +2`、`-3/2 → -2`、
  `+5/2 → +2`、`-5/2 → -2`（负并列向偶数而非远离零）。
- **精确性**：任何路径都不出现浮点；`10^60` 量级公共分量与小分量的消元是精确的。
  本库因此没有“浮点误差口径/容差”需要声明——所有判定都是有理数精确比较。
  （若调用方自行把结果转成 `float`，转换误差是 Python `Fraction → float` 的
  相对误差约 2^-53，与本库判定无关。）
- **预算口径**：每次 Lovász 判定、每次真正执行的初等行变换（size reduction 或
  相邻交换）各计 1 次操作；初始 GS 不计入。耗尽时在改动 `B/U` **之前** 判定并
  停止，因此返回态没有未说明的部分变更：`B = U*A`、`U` 幺模仍然成立，只是
  不满足 LLL 条件，`reduced=False` 明确标记。
- **无状态**：`lll_reduce` 不修改输入，重复调用结果一致。
- 不实现 LLL 的可调 `delta`、BKZ、浮点近似 LLL、非整数输入等；本项目只实现
  本题契约的数学子集，不宣称行业标准（如某篇论文的具体交换变体）完整兼容。

## 已知限制

- 维数硬上限 8 行；列数不限但实际随整数位宽增长（精确有理数的分子分母会变大；
  本库规模与 8 秒演示规模下耗时为毫秒级）。
- LLL 只保证 `||b_1|| <= 2^{(n-1)/2} · λ_1`，**不保证输出最短向量**；
  测试仅在二维小例上用独立枚举逐向量对照，并断言理论界。
- 舍入并列与交换次序按本文实现固定；不同教材的 LLL 变体可能产生不同的同样合法
  的约化基（行序/符号差异），验证器接受任何满足证书的基。
- 预算默认 100 万次操作；人为把预算设得极小会如实返回 `exhausted=True`。

## 仓库结构

```
lll_reduce/
  __init__.py    # 公开 API
  errors.py      # 异常类型
  core.py        # 有理 GS + LLL 核心（被验证对象）
  verify.py      # 独立验证器（Gram 矩阵路径，不调用 core 的算法）
tests/test_lll.py# 28 个独立测试
demo.py          # 固定输入演示
requirements.txt # 无第三方依赖（仅注释说明）
```
