"""独立验证器：独立代码路径复核 LLL 结果，不调用 :mod:`lll_reduce.core`。

验证项（全部精确整数/有理数判定，无浮点、无容差）：

1. 形状与整数性；
2. ``B = U * A``（逐项精确相等）；
3. ``U`` 为整数幺模矩阵：``det(U) == ±1``；
4. 原始输入 ``A`` 满行秩（独立的 Gram 矩阵正交化判定）；
5. 所有投影系数 ``|mu[i][j]| <= 1/2``（``j < i``）；
6. Lovász 条件 ``D[i] >= (delta - mu[i][i-1]^2) * D[i-1]``（``i >= 1``）；
7. （可选）按事件日志在单位阵上重放初等行变换，结果必须等于 ``U``。

这里的 Gram--Schmidt 系数走 **Gram 矩阵递推** 公式，与核心模块的逐向量
正交化是不同实现，避免“被测代码自证”。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from .core import DELTA, ReductionEvent

_HALF = Fraction(1, 2)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    checks: tuple[Check, ...]
    errors: tuple[str, ...]
    mu: tuple[tuple[Fraction, ...], ...]
    norms: tuple[Fraction, ...]

    def summary(self) -> str:
        lines = ["verification " + ("PASSED" if self.ok else "FAILED")]
        for c in self.checks:
            mark = "+" if c.passed else "-"
            extra = f"  ({c.detail})" if c.detail else ""
            lines.append(f"  [{mark}] {c.name}{extra}")
        for e in self.errors:
            lines.append(f"  [!] {e}")
        return "\n".join(lines)


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _dot_int(x: Sequence[int], y: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(x, y))


def _mat_mul_int(U: Sequence[Sequence[int]],
                 A: Sequence[Sequence[int]]) -> list[list[int]]:
    n, r = len(U), len(A)
    m = len(A[0])
    return [
        [sum(U[i][t] * A[t][j] for t in range(r)) for j in range(m)]
        for i in range(n)
    ]


def _det_int(M: Sequence[Sequence[int]]) -> Fraction:
    """整数/有理数方阵行列式，独立的分数 Gauss 消元（带行交换）。"""

    n = len(M)
    X = [[Fraction(v) for v in row] for row in M]
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if X[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            X[col], X[pivot] = X[pivot], X[col]
            det = -det
        piv = X[col][col]
        det *= piv
        for r in range(col + 1, n):
            if X[r][col] != 0:
                factor = X[r][col] / piv
                for c in range(col, n):
                    X[r][c] -= factor * X[col][c]
    return det


def gram_data(
    B: Sequence[Sequence[int]],
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """由 Gram 矩阵独立递推 ``(mu, D)``。

    ``D[i] = G[i][i] - sum_{j<i} mu[i][j]^2 D[j]``；
    ``mu[i][j] = (G[i][j] - sum_{t<j} mu[i][t] mu[j][t] D[t]) / D[j]``。
    若 ``D[i] == 0`` 抛 :class:`ValueError`（调用方据此判定相关）。
    """

    n = len(B)
    G = [[_dot_int(B[i], B[j]) for j in range(n)] for i in range(n)]
    mu: list[list[Fraction]] = [
        [Fraction(0) for _ in range(n)] for _ in range(n)
    ]
    D: list[Fraction] = []
    for i in range(n):
        d = Fraction(G[i][i])
        for j in range(i):
            s = Fraction(G[i][j])
            for t in range(j):
                s -= mu[i][t] * mu[j][t] * D[t]
            mu[i][j] = s / D[j]
            d -= mu[i][j] ** 2 * D[j]
        if d == 0:
            raise ValueError(f"D[{i}] == 0: linearly dependent rows")
        D.append(d)
    return mu, D


def _replay_events(events: Sequence[ReductionEvent], n: int) -> list[list[int]]:
    """在单位阵上按日志重放行变换，独立重建 U。"""

    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for e in events:
        if e.kind == "size_reduce":
            if e.coefficient is None:
                raise ValueError(f"event step {e.step} missing coefficient")
            for t in range(n):
                U[e.i][t] -= e.coefficient * U[e.j][t]
        elif e.kind == "swap":
            U[e.i], U[e.j] = U[e.j], U[e.i]
        else:
            raise ValueError(f"event step {e.step} has unknown kind {e.kind!r}")
    return U


def verify_reduction(
    A: Sequence[Sequence[int]],
    B: Sequence[Sequence[int]],
    U: Sequence[Sequence[int]],
    events: Optional[Sequence[ReductionEvent]] = None,
    *,
    delta: Fraction = DELTA,
) -> VerificationReport:
    """对 ``(A, B, U)`` 做完整独立验证。任何一项失败则 ``ok=False``。

    本函数不抛“验证失败”异常，而是把失败收集进报告；仅当输入结构连检查
    都无法进行（例如不是矩阵）时才抛 :class:`TypeError`/:class:`ValueError`。
    """

    checks: list[Check] = []
    errors: list[str] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append(Check(name, passed, detail))
        if not passed:
            errors.append(f"{name}: {detail}")

    n = len(A)
    m = len(A[0]) if n else 0

    # 1. 形状与整数性
    shape_ok = (
        len(B) == n
        and all(len(r) == m for r in B)
        and len(U) == n
        and all(len(r) == n for r in U)
        and all(_is_int(v) for row in A for v in row)
        and all(_is_int(v) for row in B for v in row)
        and all(_is_int(v) for row in U for v in row)
    )
    add("shapes and integral types", shape_ok,
        "" if shape_ok else f"expected B {n}x{m}, U {n}x{n}, all integer")
    if not shape_ok:
        return VerificationReport(False, tuple(checks), tuple(errors), (), ())

    # 2. B = U*A
    UA = _mat_mul_int(U, A)
    product_ok = UA == [list(r) for r in B]
    add("B == U * A (exact)", product_ok,
        "" if product_ok else "row/column mismatch under exact integer multiply")

    # 3. U 幺模
    d = _det_int(U)
    unimod = d == 1 or d == -1
    add("U unimodular (det U = +/-1)", unimod, f"det U = {d}")

    # 4. A 满行秩
    try:
        gram_data(A)
        add("A full row rank (independent GS)", True)
        rank_ok = True
    except ValueError as exc:
        add("A full row rank (independent GS)", False, str(exc))
        rank_ok = False

    mu: list[list[Fraction]] = []
    D: list[Fraction] = []
    if rank_ok:
        try:
            mu, D = gram_data(B)
        except ValueError as exc:
            add("B full row rank (independent GS)", False, str(exc))
            return VerificationReport(False, tuple(checks),
                                      tuple(errors) + (str(exc),), (), ())
        add("B full row rank (independent GS)", True)

        # 5. size reduction 条件
        bad = [
            (i, j, mu[i][j])
            for i in range(n) for j in range(i)
            if abs(mu[i][j]) > _HALF
        ]
        add(
            "all |mu[i][j]| <= 1/2",
            not bad,
            "violators: "
            + ", ".join(f"mu[{i}][{j}]={v}" for i, j, v in bad)
            if bad else f"{n * (n - 1) // 2} coefficients checked",
        )

        # 6. Lovász 条件
        lov_bad = []
        for i in range(1, n):
            if D[i] + (mu[i][i - 1] ** 2 - delta) * D[i - 1] < 0:
                lov_bad.append((i, D[i] / D[i - 1]))
        add(
            "Lovasz D[i] >= (delta - mu[i,i-1]^2) D[i-1]",
            not lov_bad,
            "violators: "
            + ", ".join(f"i={i} (D_i/D_i-1={r})" for i, r in lov_bad)
            if lov_bad else f"{max(n - 1, 0)} adjacent pairs checked",
        )

    # 7. 事件日志重放
    if events is not None:
        try:
            U_re = _replay_events(events, n)
            replay_ok = U_re == [list(r) for r in U]
            add("event log replays to U", replay_ok,
                "" if replay_ok else "replayed transform differs from U")
        except ValueError as exc:
            add("event log replays to U", False, str(exc))

    ok = all(c.passed for c in checks)
    return VerificationReport(
        ok,
        tuple(checks),
        tuple(errors),
        tuple(tuple(r) for r in mu),
        tuple(D),
    )


def verify_result(result: object, A: Sequence[Sequence[int]],
                  *, require_reduced: bool = True) -> VerificationReport:
    """便捷封装：直接验证 :class:`~lll_reduce.core.LLLResult`。

    ``require_reduced=True`` 时，``result.reduced`` 必须为真，否则报告失败
    （防止把预算耗尽的中间态当作约化基使用）。
    """

    report = verify_reduction(
        A, result.basis, result.transform, result.events
    )
    if require_reduced and not result.reduced:
        chk = Check("result claims reduced", False,
                    f"reduced={result.reduced}, exhausted={result.exhausted}, "
                    f"operations={result.operations}/{result.budget}")
        report = VerificationReport(
            False,
            report.checks + (chk,),
            report.errors + (chk.detail,),
            report.mu,
            report.norms,
        )
    return report
