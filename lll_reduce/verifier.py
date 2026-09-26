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


def _sum_lattice_det(B1, B2):
    """和格 L(B1)+L(B2) 的协体积：[B1|B2] 的列 Hermite 标准形行列式。

    验证器独立实现（不调用 intersection 模块），用于完备性恒等式。
    """
    n = len(B1)
    H = [list(map(int, B1[i])) + list(map(int, B2[i])) for i in range(n)]
    m = 2 * n
    det = 1
    for i in range(n):
        # 在列 i..m-1 中对第 i 行整数辗转相除，使只剩 H[i][i] 非零。
        while True:
            piv = -1
            for j in range(i, m):
                if H[i][j] != 0 and (piv < 0 or abs(H[i][j]) < abs(H[i][piv])):
                    piv = j
            if piv < 0:
                return 0
            if piv != i:
                for r in range(n):
                    H[r][i], H[r][piv] = H[r][piv], H[r][i]
            for j in range(i + 1, m):
                if H[i][j] != 0:
                    q = H[i][j] // H[i][i]
                    for r in range(n):
                        H[r][j] -= q * H[r][i]
            if all(H[i][j] == 0 for j in range(i + 1, m)):
                break
        det *= H[i][i]
    return abs(det)


def verify_intersection(B1, B2, G, U, V):
    """独立验证交格证书：成员关系 + 行列式/整数标准形完备性证据。

    逐项检查（全部精确整数运算，无浮点、无容差）：
      1. 五个矩阵均为同维 n x n 整数矩阵（1 <= n <= 4）；
      2. 成员关系：B1 @ U == G 且 B2 @ V == G（G 的列同时在两格中）；
      3. 非奇异：det G、det U、det V 均非零；
      4. 指数一致：|det G| == |det B1| * |det U| == |det B2| * |det V|；
      5. 完备性（无遗漏共同格点）：记 s 为和格 L(B1)+L(B2) 的协体积
         （由 [B1|B2] 的列 Hermite 标准形独立计算），第二同构定理要求
         [L1 : L1∩L2] = [L1+L2 : L2]，即
             |det U| * s == |det B2| 且 |det V| * s == |det B1|。
    """
    failures = []

    mats = {}
    for name, M in (("B1", B1), ("B2", B2), ("G", G), ("U", U), ("V", V)):
        if not _is_int_matrix(M):
            failures.append(name + " is not a non-empty integer matrix")
        else:
            mats[name] = [list(r) for r in M]
    if failures:
        return VerificationReport(False, failures)

    n = len(mats["B1"])
    if n > 4:
        failures.append("dimension %d exceeds maximum 4" % n)
    for name, M in mats.items():
        if len(M) != n or any(len(r) != n for r in M):
            failures.append("%s is not %dx%d" % (name, n, n))
    if failures:
        return VerificationReport(False, failures)

    B1m, B2m = mats["B1"], mats["B2"]
    Gm, Um, Vm = mats["G"], mats["U"], mats["V"]

    # 2) 成员关系
    for name, P in (("B1@U", _mat_mul_int(B1m, Um)), ("B2@V", _mat_mul_int(B2m, Vm))):
        if P != Gm:
            for i in range(n):
                for j in range(n):
                    if P[i][j] != Gm[i][j]:
                        failures.append(
                            "%s != G at [%d][%d]: %d != %d"
                            % (name, i, j, P[i][j], Gm[i][j])
                        )
                        break
                else:
                    continue
                break

    # 3) 非奇异
    det_g, det_u, det_v = _det_int(Gm), _det_int(Um), _det_int(Vm)
    det_b1, det_b2 = _det_int(B1m), _det_int(B2m)
    for name, d in (("G", det_g), ("U", det_u), ("V", det_v)):
        if d == 0:
            failures.append("det(%s) == 0: singular" % name)
    if det_b1 == 0 or det_b2 == 0:
        failures.append("B1 or B2 is singular")
    if failures:
        return VerificationReport(False, failures)

    # 4) 指数一致性
    if abs(det_g) != abs(det_b1) * abs(det_u):
        failures.append(
            "|det G| = %d != |det B1|*|det U| = %d"
            % (abs(det_g), abs(det_b1) * abs(det_u))
        )
    if abs(det_g) != abs(det_b2) * abs(det_v):
        failures.append(
            "|det G| = %d != |det B2|*|det V| = %d"
            % (abs(det_g), abs(det_b2) * abs(det_v))
        )

    # 5) 完备性：[Li : L1∩L2] = [L1+L2 : Lj]（第二同构定理，HNF 证据）
    s = _sum_lattice_det(B1m, B2m)
    if s == 0:
        failures.append("sum lattice L(B1)+L(B2) is degenerate")
    else:
        if abs(det_u) * s != abs(det_b2):
            failures.append(
                "completeness failed: [L1:L∩]*det(L1+L2) = %d != |det B2| = %d"
                % (abs(det_u) * s, abs(det_b2))
            )
        if abs(det_v) * s != abs(det_b1):
            failures.append(
                "completeness failed: [L2:L∩]*det(L1+L2) = %d != |det B1| = %d"
                % (abs(det_v) * s, abs(det_b1))
            )

    return VerificationReport(not failures, failures)
