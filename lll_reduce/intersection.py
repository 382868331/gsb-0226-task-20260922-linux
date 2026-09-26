"""两个同维满秩整数列格的交格（精确整数算法）。

约定
----
* B1、B2 为 n×n 整数方阵（1 <= n <= 4），列向量张成格
  L(Bi) = { Bi x : x in Z^n }。
* 共同点 (u, v) 满足 B1 u = B2 v，即满行秩整数矩阵
  M = [B1 | -B2] 的整数核 M [u; v] = 0。
* 整数核用**列 Hermite 消元**（扩展欧几里得 + 幺正列变换）精确求出，
  绝不使用浮点子空间交：浮点只能给出 Q-张成空间，无法识别整数同余。
* 核基前 n 行给出 u、后 n 行给出 v：B1 u = B2 v = C，C 即交格的一个基；
  再用现有（行基）LLL 对 C 的转置约化，幺正变换同步作用到 U、V，
  得到缩短后的交格列基。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .core import LLLResult, reduce_basis
from .errors import InvalidBasisError, RankDeficientError

#: 交格接口支持的最大维数。
MAX_INTERSECTION_DIM: int = 4


@dataclass(frozen=True)
class IntersectionResult:
    """交格计算结果（全部为精确整数证据）。"""

    basis: list[list[int]]
    U: list[list[int]]
    V: list[list[int]]
    index_in_lattice1: int
    index_in_lattice2: int
    determinant: int
    kernel_basis: list[list[int]]
    lll_result: LLLResult = field(repr=False)


def _validate_square(B: object, name: str) -> list[list[int]]:
    if not isinstance(B, (list, tuple)) or len(B) == 0:
        raise InvalidBasisError(name, "must be a non-empty square matrix")
    n = len(B)
    if n > MAX_INTERSECTION_DIM:
        raise InvalidBasisError(
            name,
            "dimension " + str(n) + " exceeds MAX_INTERSECTION_DIM=" + str(MAX_INTERSECTION_DIM),
        )
    rows: list[list[int]] = []
    for i, row in enumerate(B):
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise InvalidBasisError(
                name + "[" + str(i) + "]",
                "must have " + str(n) + " columns for a square matrix",
            )
        out_row: list[int] = []
        for j, x in enumerate(row):
            if isinstance(x, bool) or not isinstance(x, int):
                raise InvalidBasisError(
                    name + "[" + str(i) + "][" + str(j) + "]",
                    "element must be exact int, got " + type(x).__name__,
                )
            out_row.append(x)
        rows.append(out_row)
    return rows


def bareiss_det(M: Sequence[Sequence[int]]) -> int:
    """Bareiss 精确整数行列式（无浮点）。"""
    n = len(M)
    X = [list(row) for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
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


def mat_mul_int(A, B):
    n, r, m = len(A), len(B), len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(r)) for j in range(m)]
        for i in range(n)
    ]


def _transpose(A: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*A)]


def _egcd(a: int, b: int):
    """扩展欧几里得：返回 (g, s, t)，g = s*a + t*b（允许负输入）。"""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def integer_kernel_full_row_rank(M: Sequence[Sequence[int]]) -> list[list[int]]:
    """求满行秩整数矩阵 M（n 行 m 列，m >= n）的整数核基（按列返回 m×(m-n)）。

    列 Hermite 消元：只用整数幺正列变换（列交换与扩展欧几里得 2×2 变换）
    把 M 化为 [H | 0]，同步作用于单位阵 Q；末尾 m-n 列即整数核基。
    整数幺正变换保证取到的是**整数核**，而不是有理/浮点子空间。
    """
    n = len(M)
    m = len(M[0])
    if m < n:
        raise InvalidBasisError("M", "kernel requires columns >= rows")
    A = [list(row) for row in M]
    Q = [[1 if i == j else 0 for j in range(m)] for i in range(m)]

    def col_replace(p, j, s, t, cp, cq):
        for X in (A, Q):
            for i in range(len(X)):
                vp, vj = X[i][p], X[i][j]
                X[i][p] = s * vp + t * vj
                X[i][j] = cp * vp + cq * vj

    pivot = 0
    for i in range(n):
        nz = pivot
        while nz < m and A[i][nz] == 0:
            nz += 1
        if nz == m:
            raise InvalidBasisError("M", "no pivot at row " + str(i) + " during HNF")
        if nz != pivot:
            for X in (A, Q):
                for r in range(len(X)):
                    X[r][pivot], X[r][nz] = X[r][nz], X[r][pivot]
        for j in range(pivot + 1, m):
            b = A[i][j]
            if b == 0:
                continue
            a = A[i][pivot]
            g, s, t = _egcd(a, b)
            # [s t; b/g -a/g] 行列式为 -1，整数幺正。
            col_replace(pivot, j, s, t, b // g, -a // g)
        pivot += 1

    for j in range(n, m):
        if any(A[i][j] != 0 for i in range(n)):
            raise AssertionError("internal error: HNF zero block is not zero")
    return [[Q[i][j] for j in range(n, m)] for i in range(m)]


def _rank_square(M: Sequence[Sequence[int]]) -> int:
    """对小整数方阵用有理消元求秩（仅用于奇异输入的错误信息）。"""
    from fractions import Fraction

    n = len(M)
    A = [[Fraction(x) for x in row] for row in M]
    rank = 0
    for col in range(n):
        pivot = next((r for r in range(rank, n) if A[r][col] != 0), None)
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        pv = A[rank][col]
        for j in range(col, n):
            A[rank][j] /= pv
        for r in range(n):
            if r != rank and A[r][col] != 0:
                f = A[r][col]
                for j in range(col, n):
                    A[r][j] -= f * A[rank][j]
        rank += 1
    return rank


def intersection_basis(B1: object, B2: object) -> IntersectionResult:
    """求两个同维满秩整数列格的交格基。

    B1, B2 为 n×n 整数方阵，1 <= n <= 4，满秩（det != 0）；列向量为基，
    负行列式与任意列序都只是同一个格的等价基。
    返回 IntersectionResult：LLL 缩短后的交格列基 C、整数坐标 U/V
    （B1 U = B2 V = C）、两边指数、det C 与整数核基证书。
    """
    M1 = _validate_square(B1, "B1")
    M2 = _validate_square(B2, "B2")
    if len(M2) != len(M1):
        raise InvalidBasisError(
            "B2", "dimension " + str(len(M2)) + " != B1 dimension " + str(len(M1))
        )
    n = len(M1)

    d1 = bareiss_det(M1)
    if d1 == 0:
        raise RankDeficientError(n, _rank_square(M1))
    d2 = bareiss_det(M2)
    if d2 == 0:
        raise RankDeficientError(n, _rank_square(M2))

    # M = [B1 | -B2]，核向量 (u, v) 满足 B1 u = B2 v。
    M = [M1[i] + [-M2[i][j] for j in range(n)] for i in range(n)]
    K = integer_kernel_full_row_rank(M)  # 2n × n，按列
    U = [K[i][:] for i in range(n)]
    V = [K[n + i][:] for i in range(n)]

    if bareiss_det(V) == 0 or bareiss_det(U) == 0:
        raise AssertionError("internal error: intersection coordinate map singular")

    C = mat_mul_int(M2, V)

    # 复用现有行基 LLL：对 C^T 约化，R = W C^T ⇒ C_out = C W^T。
    lll = reduce_basis(_transpose(C))
    Wt = _transpose(lll.transform)
    C_out = _transpose(lll.basis)
    U_out = mat_mul_int(U, Wt)
    V_out = mat_mul_int(V, Wt)
    K_out = U_out + V_out

    det_c = bareiss_det(C_out)
    if det_c == 0:
        raise AssertionError("internal error: intersection basis singular")
    i1, r1 = divmod(abs(det_c), abs(d1))
    i2, r2 = divmod(abs(det_c), abs(d2))
    if r1 != 0 or r2 != 0:
        raise AssertionError("internal error: intersection index is not integral")

    return IntersectionResult(
        basis=C_out,
        U=U_out,
        V=V_out,
        index_in_lattice1=i1,
        index_in_lattice2=i2,
        determinant=det_c,
        kernel_basis=K_out,
        lll_result=lll,
    )
