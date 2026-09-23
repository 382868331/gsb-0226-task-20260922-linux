"""独立验证器：校验 LLL 约化证书。

本模块刻意不调用 :mod:`lll_reduce.core` 的任何实现，独立完成：
  1. B == U A（精确整数矩阵乘法）；
  2. U 为整数幺模矩阵（det U = ±1，且 B 满行秩）；
  3. size reduction：所有 j < i 有 |mu_{i,j}| <= 1/2；
  4. Lovasz(delta=3/4)：
       |b*_i|^2 + mu_{i,i-1}^2 |b*_{i-1}|^2 >= 3/4 * |b*_{i-1}|^2。
全部使用 Fraction 精确比较，无浮点、无容差。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

DELTA: Fraction = Fraction(3, 4)


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    failures: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("verification failed: " + "; ".join(self.failures))


def _is_int_matrix(M: object) -> bool:
    return (
        isinstance(M, (list, tuple))
        and len(M) > 0
        and all(
            isinstance(r, (list, tuple))
            and len(r) > 0
            and all(
                isinstance(x, int) and not isinstance(x, bool) for x in r
            )
            for r in M
        )
    )


def _mat_mul_int(
    U: Sequence[Sequence[int]], A: Sequence[Sequence[int]]
) -> list[list[int]]:
    n, r, m = len(U), len(A), len(A[0])
    return [
        [sum(U[i][t] * A[t][j] for t in range(r)) for j in range(m)]
        for i in range(n)
    ]


def _det_int(M: Sequence[Sequence[int]]) -> int:
    """Bareiss 精确整数行列式（独立实现）。"""
    n = len(M)
    X = [list(map(int, row)) for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        # 选主元（行交换），保持精确除法。
        if X[k][k] == 0:
            for t in range(k + 1, n):
                if X[t][k] != 0:
                    X[k], X[t] = X[t], X[k]
                    sign = -sign
                    break
            else:
                return 0
        pivot = X[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                X[i][j] = (X[i][j] * pivot - X[i][k] * X[k][j]) // prev
        prev = pivot
        for i in range(k + 1, n):
            X[i][k] = 0
    return sign * X[n - 1][n - 1]


def _independent_gs(
    B: Sequence[Sequence[int]],
) -> tuple[list[list[Fraction]], list[list[Fraction]], list[Fraction]]:
    """验证器自带的 Gram-Schmidt，与核心实现独立。"""
    n = len(B)
    dim = len(B[0])
    bstar: list[list[Fraction]] = []
    mu = [[Fraction(0)] * n for _ in range(n)]
    norm2: list[Fraction] = []
    for i in range(n):
        v = [Fraction(B[i][t]) for t in range(dim)]
        for j in range(i):
            d = sum(v[t] * bstar[j][t] for t in range(dim))
            c = d / norm2[j]
            mu[i][j] = c
            for t in range(dim):
                v[t] -= c * bstar[j][t]
        n2 = sum(x * x for x in v)
        bstar.append(v)
        norm2.append(n2)
    return bstar, mu, norm2


def verify_reduction(
    A: object,
    B: object,
    U: object,
    *,
    delta: Fraction = DELTA,
) -> VerificationReport:
    """独立验证 B = U A、U 幺模、size reduction 与 Lovasz 条件。

    返回 :class:`VerificationReport`；任何形状/类型问题同样记录在 failures
    中（ok=False），而不是静默通过。
    """
    failures: list[str] = []

    for name, M in (("A", A), ("B", B), ("U", U)):
        if not _is_int_matrix(M):
            failures.append(f"{name} is not a non-empty integer matrix")
    if failures:
        return VerificationReport(False, failures)

    A = [list(r) for r in A]  # type: ignore[union-attr]
    B = [list(r) for r in B]  # type: ignore[union-attr]
    U = [list(r) for r in U]  # type: ignore[union-attr]
    n, m = len(A), len(A[0])

    if len(B) != n or any(len(r) != m for r in B):
        failures.append(f"B has wrong shape: expect {n}x{m}")
    if len(U) != n or any(len(r) != n for r in U):
        failures.append(f"U has wrong shape: expect {n}x{n}")
    if any(len(r) != m for r in A):
        failures.append("A is not rectangular")
    if failures:
        return VerificationReport(False, failures)

    # 1) B = U A
    UA = _mat_mul_int(U, A)
    if UA != B:
        for i in range(n):
            for j in range(m):
                if UA[i][j] != B[i][j]:
                    failures.append(
                        f"B != U*A at [{i}][{j}]: {B[i][j]} != {UA[i][j]}"
                    )
                    break
            if failures and failures[-1].startswith("B != U*A"):
                break

    # 2) U 幺模（整数且 det = ±1）
    det = _det_int(U)
    if det not in (1, -1):
        failures.append(f"U is not unimodular: det(U) = {det}")

    # 3)+4) 独立 GS
    _, mu, norm2 = _independent_gs(B)
    for i in range(n):
        if norm2[i] == 0:
            failures.append(f"B is rank deficient: b*_{i} = 0")
    if any(x == 0 for x in norm2):
        return VerificationReport(False, failures)

    for i in range(n):
        for j in range(i):
            if abs(mu[i][j]) > Fraction(1, 2):
                failures.append(
                    f"size reduction violated: |mu[{i}][{j}]| = "
                    f"{mu[i][j]} > 1/2"
                )

    for i in range(1, n):
        lhs = norm2[i] + mu[i][i - 1] ** 2 * norm2[i - 1]
        rhs = delta * norm2[i - 1]
        if lhs < rhs:
            failures.append(
                f"Lovasz condition violated at i={i}: "
                f"{float(lhs):.6g} < {float(rhs):.6g} (exact rational comparison)"
            )

    return VerificationReport(not failures, failures)
