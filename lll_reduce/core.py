"""LLL 格基约化核心（精确有理 Gram-Schmidt，delta = 3/4）。

约定
----
* 输入 A 为 n×m 整数矩阵，行向量为基，n <= m <= ...，n <= 8，满行秩。
* 全部 Gram-Schmidt 系数与平方范数用 :class:`fractions.Fraction` 精确表示，
  算法不使用浮点数，因此没有舍入误差。
* 约化基满足（行序 0 起）：
    - size reduction: 对所有 j < i，|mu_{i,j}| <= 1/2；
    - Lovasz 条件:   |b*_i|^2 + mu_{i,i-1}^2 |b*_{i-1}|^2
                       >= (3/4) |b*_{i-1}|^2。
* B = U A，U 为整数幺模矩阵（det U = ±1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Sequence

from .errors import (
    BudgetExhaustedError,
    InvalidBasisError,
    InvalidParameterError,
    RankDeficientError,
)

#: 固定 Lovasz 参数（本题契约不做参数化）。
DELTA: Fraction = Fraction(3, 4)

#: 支持的最大基维数（行数）。
MAX_DIM: int = 8

#: 默认操作预算（计费单元见模块文档与 README）。
DEFAULT_BUDGET: int = 1_000_000


@dataclass(frozen=True)
class LogEntry:
    """一次原子变换的结构化日志。

    kind 为 ``"size_reduce"`` 时记录 b_k <- b_k - q*b_j（含 index_k/index_j/coefficient）；
    kind 为 ``"swap"`` 时记录相邻行 index_k 与 index_j 交换，coefficient 为 None。
    """

    step: int
    kind: str
    index_k: int
    index_j: int
    coefficient: Optional[int]
    detail: str


@dataclass(frozen=True)
class LLLResult:
    """LLL 运行结果（同时作为预算耗尽时的中间态快照载体）。

    ``reduced`` 为 True 才是合格的约化证书；False 表示未完成，
    此时 basis/transform 仅是中间态，禁止当作约化基使用。
    """

    basis: list[list[int]]
    transform: list[list[int]]
    log: list[LogEntry]
    reduced: bool
    budget: int
    ops_used: int
    swaps: int
    index: Optional[int]
    delta: Fraction = field(default=DELTA)


def round_half_even(x: Fraction) -> int:
    """最近整数舍入，半整数并列时向偶数取值（banker's rounding）。

    纯有理数实现：比较 2*余数 与分母，无浮点参与。对负数同样正确，
    例如 -3/2 -> -2，-5/2 -> -2，-1/2 -> 0。
    """
    if not isinstance(x, Fraction):
        raise InvalidParameterError("x", "round_half_even expects Fraction")
    q, r = divmod(x.numerator, x.denominator)  # q = floor(x)，0 <= r < d
    twice = 2 * r
    if twice < x.denominator:
        return q
    if twice > x.denominator:
        return q + 1
    # 恰好半数：取 q 与 q+1 中的偶数
    return q if q % 2 == 0 else q + 1


def _validate_matrix(A: object) -> list[list[int]]:
    if not isinstance(A, (list, tuple)) or len(A) == 0:
        raise InvalidBasisError("A", "must be a non-empty sequence of rows")
    n = len(A)
    if n > MAX_DIM:
        raise InvalidBasisError("A", f"row count {n} exceeds MAX_DIM={MAX_DIM}")
    rows: list[list[int]] = []
    width: Optional[int] = None
    for i, row in enumerate(A):
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise InvalidBasisError(f"A[{i}]", "row must be a non-empty sequence")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise InvalidBasisError(
                f"A[{i}]", f"row length {len(row)} != earlier width {width}"
            )
        out_row: list[int] = []
        for j, x in enumerate(row):
            # bool 是 int 的子类，按契约显式拒绝；float（含 NaN/Infinity）一律拒绝，
            # 本库只接受精确整数，不做任何静默浮点转换。
            if isinstance(x, bool) or not isinstance(x, int):
                raise InvalidBasisError(
                    f"A[{i}][{j}]", f"element must be exact int, got {type(x).__name__}"
                )
            out_row.append(x)
        rows.append(out_row)
    assert width is not None
    if width < n:
        raise InvalidBasisError(
            "A", f"full row rank requires columns >= rows: {width} < {n}"
        )
    return rows


def _dot(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(u, v)), Fraction(0))


def _gram_schmidt(
    b: list[list[Fraction]], n: int
) -> tuple[list[list[Fraction]], list[list[Fraction]], list[Fraction]]:
    """有理 Gram-Schmidt（不取平方根，平方范数直接用内积）。

    返回 (bstar, mu, norm2)。第 i 个投影为零向量即证明前若干行线性相关，
    此时抛 :class:`RankDeficientError`（rank 为检测到的行秩）。
    """
    bstar: list[list[Fraction]] = []
    mu = [[Fraction(0)] * n for _ in range(n)]
    norm2: list[Fraction] = []
    for i in range(n):
        v = list(b[i])
        for j in range(i):
            if norm2[j] == 0:
                raise RankDeficientError(n, i)
            mu[i][j] = _dot(b[i], bstar[j]) / norm2[j]
            v = [v[t] - mu[i][j] * bstar[j][t] for t in range(len(v))]
        n2 = _dot(v, v)
        if n2 == 0:
            # 有序 GS 中第 i 个投影为零 <=> 行秩恰为 i。
            raise RankDeficientError(n, i)
        bstar.append(v)
        norm2.append(n2)
        mu[i][i] = Fraction(1)
    return bstar, mu, norm2


def reduce_basis(A: object, budget: int = DEFAULT_BUDGET) -> LLLResult:
    """对满行秩整数行基 A 执行 LLL(delta=3/4) 约化。

    参数
    ----
    A : n×m 整数矩阵（嵌套序列），1 <= n <= 8，m >= n，满行秩。
    budget : 正整数操作预算（bool 不接受）。一次 size reduction 计 1；
        一次相邻交换（含交换后精确 GS 全量重算）计 n；初始 GS 计 n。

    返回
    ----
    LLLResult，``reduced=True`` 时 basis 为约化基 B、transform 满足 B=U A。

    异常
    ----
    InvalidParameterError / InvalidBasisError : 输入不合法（可定位错误）。
    RankDeficientError : 数学上证明行线性相关（与预算耗尽严格区分）。
    BudgetExhaustedError : 预算耗尽，``.partial`` 为 reduced=False 的中间态快照，
        绝不假称已约化。
    """
    rows = _validate_matrix(A)
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise InvalidParameterError("budget", "must be an int (bool rejected)")
    if budget < 1:
        raise InvalidParameterError("budget", f"must be >= 1, got {budget}")

    n = len(rows)
    m = len(rows[0])
    # 工作基（Fraction），变换矩阵 U（int），二者同步变换。
    b: list[list[Fraction]] = [[Fraction(x) for x in row] for row in rows]
    u_mat: list[list[int]] = [
        [1 if i == j else 0 for j in range(n)] for i in range(n)
    ]
    log: list[LogEntry] = []

    ops = 0

    def charge(amount: int) -> None:
        nonlocal ops
        ops += amount
        if ops > budget:
            # 仅在主循环内计费，故 k 已绑定；快照记录停滞的循环位置。
            raise BudgetExhaustedError(budget, ops, _snapshot(k))

    def _snapshot(index: Optional[int]) -> LLLResult:
        # 预算耗尽时的显式中间态（与异常文档一一对应），绝非静默部分变更；
        # 基坐标此时仍全为整数（变换只做整数线性组合/交换）。
        return LLLResult(
            basis=_to_int_matrix(b),
            transform=[row[:] for row in u_mat],
            log=list(log),
            reduced=False,
            budget=budget,
            ops_used=ops,
            swaps=sum(1 for e in log if e.kind == "swap"),
            index=index,
        )

    # 初始有理 GS；同时完成满行秩检验。
    bstar, mu, norm2 = _gram_schmidt(b, n)
    ops += n  # 初始 GS 计 n 个内积级单元
    if ops > budget:
        raise BudgetExhaustedError(budget, ops, _snapshot(None))

    step = 0
    swaps = 0
    k = 1
    while k < n:
        # ---- size reduction：j 从 k-1 递减到 0 ----
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                if ops > budget:
                    raise BudgetExhaustedError(budget, ops, _snapshot(k))
                q = round_half_even(mu[k][j])
                step += 1
                b[k] = [b[k][t] - q * b[j][t] for t in range(m)]
                u_mat[k] = [u_mat[k][t] - q * u_mat[j][t] for t in range(n)]
                # b*_0..b*_k 与全部平方范数不变；只更新 mu 第 k 行前 j+1 项。
                for ell in range(j + 1):
                    mu[k][ell] -= q * mu[j][ell]
                log.append(
                    LogEntry(
                        step=step,
                        kind="size_reduce",
                        index_k=k,
                        index_j=j,
                        coefficient=q,
                        detail=f"b{k} <- b{k} - ({q})*b{j}",
                    )
                )
                charge(1)

        # ---- Lovasz 检验（精确比较）----
        if norm2[k] >= (DELTA - mu[k][k - 1] ** 2) * norm2[k - 1]:
            k += 1
        else:
            step += 1
            b[k - 1], b[k] = b[k], b[k - 1]
            u_mat[k - 1], u_mat[k] = u_mat[k], u_mat[k - 1]
            log.append(
                LogEntry(
                    step=step,
                    kind="swap",
                    index_k=k - 1,
                    index_j=k,
                    coefficient=None,
                    detail=f"swap b{k - 1} <-> b{k}",
                )
            )
            swaps += 1
            # n <= 8：交换后直接全量重算 GS，最直观且杜绝增量更新的符号错误；
            # 行张成的子空间不变，满秩性在此处必然保持。
            bstar, mu, norm2 = _gram_schmidt(b, n)
            charge(n)
            k = max(k - 1, 1)

    # 防御性自检：所有行必须仍为精确整数（q 为整数、交换不改变元素）。
    int_basis = _to_int_matrix(b)
    return LLLResult(
        basis=int_basis,
        transform=u_mat,
        log=log,
        reduced=True,
        budget=budget,
        ops_used=ops,
        swaps=swaps,
        index=None,
    )


def _to_int_matrix(b: list[list[Fraction]]) -> list[list[int]]:
    out: list[list[int]] = []
    for row in b:
        int_row: list[int] = []
        for x in row:
            if x.denominator != 1:
                raise AssertionError(
                    f"internal invariant violated: non-integral coordinate {x}"
                )
            int_row.append(x.numerator)
        out.append(int_row)
    return out


#: 语义别名。
lll_reduce = reduce_basis
