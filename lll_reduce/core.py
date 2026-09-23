"""核心算法：精确有理 Gram--Schmidt 上的 LLL 格基约化。

约定
----
* 基按 **行** 给出：``A`` 是 ``n`` 行 ``m`` 列整数矩阵，``n <= m``，满行秩，``n <= 8``。
* ``delta = 3/4`` 固定（见 :data:`DELTA`），不作为可调参数。
* 全部内积、投影系数 ``mu``、Gram--Schmidt 范数平方 ``D`` 都用
  :class:`fractions.Fraction` 精确计算，任何路径都不出现浮点。
* Size reduction 的最近整数在并列（半整数）时向偶数舍入（round half to even）。
* Lovász 判定使用经典形式 ``D_k >= (delta - mu_{k,k-1}^2) * D_{k-1}``，
  失败时交换相邻的第 ``k-1``、``k`` 行。
* 输出 ``B = U * A``，其中 ``U`` 为整数幺模矩阵（行列式 ±1）。

预算语义（与“证明无解”严格分开）
--------------------------------
每次 Lovász 判定、每次真正执行的初等行变换（size reduction 或相邻交换）各计 1 次
操作。预算耗尽时：

* **不** 抛异常；
* 立即停止，返回的 ``B``、``U`` 仍然满足 ``B = U*A``（无未说明的部分变更）；
* ``reduced=False``、``exhausted=True`` 明确标记“尚未完成”，绝不谎称已约化。

线性相关不属于预算问题：初始 Gram--Schmidt 一旦出现零向量，立即抛
:class:`~lll_reduce.errors.LatticeDependentError`，这是被精确证明的无解。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from .errors import InvalidBasisError, InvalidParameterError, LatticeDependentError

#: 固定的 LLL 参数 delta。
DELTA: Fraction = Fraction(3, 4)

#: 行数（格维数）上限。
MAX_DIM: int = 8

#: 默认操作预算（每次 Lovász 判定或初等行变换计 1）。
DEFAULT_BUDGET: int = 1_000_000

_HALF = Fraction(1, 2)


@dataclass(frozen=True)
class ReductionEvent:
    """一条约化日志，按实际执行顺序产生。"""

    #: 1 起始的操作序号（与预算计数一致）。
    step: int
    #: ``"size_reduce"``（b_i -= coefficient * b_j）或 ``"swap"``（交换 b_i,b_{i+1}）。
    kind: str
    #: 主行下标（0 起）；size_reduce 的被改行，swap 的前一行。
    i: int
    #: 辅行下标；size_reduce 中被减去的行，swap 时等于 ``i + 1``。
    j: int
    #: size_reduce 的整数系数 m（b_i -= m * b_j）；swap 时为 ``None``。
    coefficient: Optional[int]
    #: 触发该操作时算法所在的 k（0 起，k>=1）。
    k: int
    #: 操作前相关的 mu_{k,.} 值；swap 记录 mu_{k,k-1}，无法适用时为 ``None``。
    mu_before: Optional[Fraction]
    #: 人类可读描述（数学记号，语言中性）。
    message: str


@dataclass(frozen=True)
class LLLResult:
    """LLL 结果。无论是否耗尽预算，``basis`` 与 ``transform`` 始终自洽。"""

    #: 当前基 B（整数行矩阵）。预算耗尽时它是合法的中间态，不是约化基。
    basis: list[list[int]]
    #: 整数幺模变换 U，恒有 B = U * A。
    transform: list[list[int]]
    #: 按序产生的交换/约化日志。
    events: tuple[ReductionEvent, ...]
    #: 是否完整通过 LLL 约化（所有 |mu|<=1/2 且 Lovász 条件成立）。
    reduced: bool
    #: 是否因操作预算耗尽而停止。与 ``reduced`` 独立，永不在成功时为真。
    exhausted: bool
    #: 已消耗操作数。
    operations: int
    #: 本次调用的预算上限。
    budget: int
    #: 格的秩（行数）。
    rank: int

    def as_dict(self) -> dict:
        """转成可 JSON 化的字典（Fraction 以 ``"p/q"`` 字符串表示）。"""

        return {
            "basis": self.basis,
            "transform": self.transform,
            "events": [
                {
                    "step": e.step,
                    "kind": e.kind,
                    "i": e.i,
                    "j": e.j,
                    "coefficient": e.coefficient,
                    "k": e.k,
                    "mu_before": None if e.mu_before is None else str(e.mu_before),
                    "message": e.message,
                }
                for e in self.events
            ],
            "reduced": self.reduced,
            "exhausted": self.exhausted,
            "operations": self.operations,
            "budget": self.budget,
            "rank": self.rank,
        }


def round_half_even(x: Fraction) -> int:
    """最近整数；恰为半整数时向偶数舍入。

    例：``3/2 -> 2``、``-3/2 -> -2``、``5/2 -> 2``、``-5/2 -> -2``、
    ``1/2 -> 0``、``-1/2 -> 0``。
    """

    q, r = divmod(x.numerator, x.denominator)  # 0 <= r < denominator，负数也成立
    twice = 2 * r
    if twice < x.denominator:
        return q
    if twice > x.denominator:
        return q + 1
    return q if q % 2 == 0 else q + 1  # 并列：取偶数


def _is_real_int(x: object) -> bool:
    # bool 是 int 的子类型，按契约必须显式拒绝。
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_matrix(A: Sequence[Sequence[object]]) -> list[list[int]]:
    if not isinstance(A, (list, tuple)) or len(A) == 0:
        raise InvalidBasisError(
            "A must be a non-empty list/tuple of rows (got "
            f"{type(A).__name__})"
        )
    n = len(A)
    rows: list[list[int]] = []
    width: Optional[int] = None
    for i, row in enumerate(A):
        if not isinstance(row, (list, tuple)):
            raise InvalidBasisError(
                f"A[{i}] must be a list/tuple row (got {type(row).__name__})"
            )
        if width is None:
            width = len(row)
            if width == 0:
                raise InvalidBasisError(f"A[{i}] is an empty row")
        elif len(row) != width:
            raise InvalidBasisError(
                f"A[{i}] has length {len(row)}, expected {width} (rows must be "
                "rectangular)"
            )
        out_row: list[int] = []
        for j, x in enumerate(row):
            if isinstance(x, bool):
                raise InvalidBasisError(
                    f"A[{i}][{j}] is bool, which is not accepted as an integer"
                )
            if isinstance(x, float):
                if not math.isfinite(x):
                    raise InvalidBasisError(
                        f"A[{i}][{j}] is non-finite (NaN/Infinity)"
                    )
                raise InvalidBasisError(
                    f"A[{i}][{j}]={x!r} is a float; exact integers are required"
                )
            if not _is_real_int(x):
                raise InvalidBasisError(
                    f"A[{i}][{j}] must be an exact integer (got "
                    f"{type(x).__name__})"
                )
            out_row.append(int(x))
        rows.append(out_row)

    m = width
    if n > MAX_DIM:
        raise InvalidBasisError(f"row count {n} exceeds MAX_DIM={MAX_DIM}")
    if m < n:
        # 形状已使满行秩不可能，这是输入契约错误，不当作“运行时相关”处理。
        raise InvalidBasisError(
            f"A is {n}x{m}: fewer columns than rows, full row rank impossible"
        )
    return rows


def _validate_budget(budget: object) -> int:
    if isinstance(budget, bool) or not _is_real_int(budget):
        raise InvalidParameterError(
            f"budget must be a positive int (got {type(budget).__name__})"
        )
    if budget < 1:
        raise InvalidParameterError(f"budget must be >= 1 (got {budget})")
    return int(budget)


def gram_schmidt(
    B: Sequence[Sequence[int]],
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """经典 Gram--Schmidt（精确有理）。

    返回 ``(mu, D)``：``mu[i][j]`` 为投影系数（``j < i``），
    ``D[i] = <b*_i, b*_i>`` 为正交向量范数平方。
    若第 ``i`` 个正交向量为零向量，抛 :class:`LatticeDependentError`（``index=i``）。
    """

    n = len(B)
    stars: list[list[Fraction]] = []
    mu: list[list[Fraction]] = [
        [Fraction(0) for _ in range(n)] for _ in range(n)
    ]
    norms: list[Fraction] = []
    for i in range(n):
        v = [Fraction(t) for t in B[i]]
        for j in range(i):
            num = sum(Fraction(B[i][t]) * stars[j][t] for t in range(len(v)))
            c = num / norms[j]
            mu[i][j] = c
            if c:
                for t in range(len(v)):
                    v[t] -= c * stars[j][t]
        norm = sum(t * t for t in v)
        if norm == 0:
            raise LatticeDependentError(i)
        stars.append(v)
        norms.append(norm)
    return mu, norms


def _row_sub(M: list[list[int]], i: int, m_coef: int, j: int) -> None:
    """M[i] -= m_coef * M[j]（整数初等行变换）。"""

    if m_coef == 0:
        return
    row_i, row_j = M[i], M[j]
    for t in range(len(row_i)):
        row_i[t] -= m_coef * row_j[t]


def _swap_rows(M: list[list[int]], i: int, j: int) -> None:
    M[i], M[j] = M[j], M[i]


def lll_reduce(
    A: Sequence[Sequence[int]],
    budget: int = DEFAULT_BUDGET,
) -> LLLResult:
    """对满行秩整数行基 ``A`` 执行 LLL（delta=3/4，精确有理 GS）。

    :param A: ``n x m`` 整数矩阵（list/tuple 嵌套），``1 <= n <= m``、``n <= 8``、
        满行秩；元素必须是真正的 ``int``（拒绝 bool、float、NaN/Infinity）。
    :param budget: 操作预算正整数（每次 Lovász 判定或初等行变换计 1）。
    :return: :class:`LLLResult`。成功时 ``reduced=True``；预算耗尽时
        ``reduced=False, exhausted=True``，但仍有 ``B = U*A``。
    :raises InvalidBasisError: 形状/类型/有限性不满足契约。
    :raises InvalidParameterError: budget 非法（含 bool）。
    :raises LatticeDependentError: 基线性相关（精确判定）。
    """

    B = _validate_matrix(A)
    budget = _validate_budget(budget)
    n = len(B)

    # 初始相关性检查：满行秩是硬契约。
    mu, norms = gram_schmidt(B)

    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    events: list[ReductionEvent] = []
    ops = 0

    def spend() -> None:
        # 在执行操作前判定：耗尽则不改动 B/U，状态保持自洽。
        if ops >= budget:
            raise _BudgetExhausted

    class _BudgetExhausted(Exception):
        pass

    def log(kind: str, i: int, j: int, k: int, coef: Optional[int],
            mu_before: Optional[Fraction]) -> None:
        if kind == "size_reduce":
            msg = f"b{i} -= ({coef}) * b{j}"
        else:
            msg = f"swap b{i}, b{j}"
        events.append(
            ReductionEvent(
                step=ops, kind=kind, i=i, j=j, k=k, coefficient=coef,
                mu_before=mu_before, message=msg,
            )
        )

    reduced = False
    try:
        k = 1
        while k < n:
            # 先针对 k-1 做惰性 size reduction。
            coef_mu = mu[k][k - 1]
            if abs(coef_mu) > _HALF:
                m = round_half_even(coef_mu)
                spend()
                ops += 1
                _row_sub(B, k, m, k - 1)
                _row_sub(U, k, m, k - 1)
                log("size_reduce", k, k - 1, k, m, coef_mu)
                mu, norms = gram_schmidt(B)

            # Lovász 判定本身计 1 次操作。
            spend()
            ops += 1
            if norms[k] >= (DELTA - mu[k][k - 1] ** 2) * norms[k - 1]:
                # 条件成立：补做对更前行的 size reduction，然后右移 k。
                for ell in range(k - 2, -1, -1):
                    coef_mu = mu[k][ell]
                    if abs(coef_mu) > _HALF:
                        m = round_half_even(coef_mu)
                        spend()
                        ops += 1
                        _row_sub(B, k, m, ell)
                        _row_sub(U, k, m, ell)
                        log("size_reduce", k, ell, k, m, coef_mu)
                        mu, norms = gram_schmidt(B)
                k += 1
            else:
                # Lovász 失败：交换相邻基并回退 k。
                spend()
                ops += 1
                _swap_rows(B, k - 1, k)
                _swap_rows(U, k - 1, k)
                log("swap", k - 1, k, k, None, mu[k][k - 1])
                mu, norms = gram_schmidt(B)
                k = max(k - 1, 1)
        reduced = True
    except _BudgetExhausted:
        reduced = False

    return LLLResult(
        basis=B,
        transform=U,
        events=tuple(events),
        reduced=reduced,
        exhausted=not reduced,
        operations=ops,
        budget=budget,
        rank=n,
    )
