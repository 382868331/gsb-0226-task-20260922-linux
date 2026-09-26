"""两套整数列格的交格：精确整数构造 + LLL 缩短。

约定
----
* 输入 B1、B2 为同维满秩整数方阵（n x n，1 <= n <= 4），**列向量为基**，
  即格 L(Bi) = { Bi z : z in Z^n }。
* 共同向量通过精确整数方程求得：整数核
      ker_Z [B1 | -B2] = { (x, y) in Z^{2n} : B1 x = B2 y }
  由列 Hermite 标准形（幺模列变换）给出 Z-基；其 x 分量经 B1 映回即得
  交格 L(B1) ∩ L(B2) 的一组基。全程整数运算，**不使用浮点子空间交**。
* 得到的交格基再交给 reduce_basis（行基约定，故转置传入）缩短，
  输出仍为列基。

证书
----
返回的 U、V 为整数坐标矩阵，满足 B1 @ U == B2 @ V == G（输出基）；
index_in_left = |det U| = [L(B1) : L(G)]，index_in_right = |det V|；
snf_left / snf_right 为 U、V 的 Smith 标准形对角元（不变因子），其乘积
等于对应指数；sum_det 为和格 L(B1)+L(B2) 的协体积（由 [B1|B2] 的列
Hermite 标准形行列式给出）。完备性恒等式（第二同构定理）：

    [L1 : L1∩L2] = [L1+L2 : L2]  <=>  index_in_left * sum_det == |det B2|

由 lll_reduce.verifier.verify_intersection 独立复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .core import DEFAULT_BUDGET, LogEntry, reduce_basis
from .errors import InvalidBasisError, InvalidParameterError, RankDeficientError

#: 交格计算的最大维数（本题契约）。
MAX_INTERSECTION_DIM: int = 4


@dataclass(frozen=True)
class IntersectionResult:
    """交格计算结果。

    basis : n x n 整数矩阵，**列**为交格 L(B1)∩L(B2) 的 LLL 约化基。
    left_coords / right_coords : 整数矩阵 U、V，满足 B1@U == B2@V == basis。
    index_in_left / index_in_right : 交格在两边格中的指数 [Li : L1∩L2]。
    snf_left / snf_right : U、V 的 Smith 标准形对角元（正的不变因子，
        逐个整除，乘积等于对应指数）。
    sum_det : 和格 L(B1)+L(B2) 的协体积（行列式绝对值）。
    lll_log : 最后一步 LLL 缩短的变换日志（可追溯）。
    """

    basis: list[list[int]]
    left_coords: list[list[int]]
    right_coords: list[list[int]]
    index_in_left: int
    index_in_right: int
    snf_left: list[int]
    snf_right: list[int]
    sum_det: int
    lll_log: list[LogEntry] = field(default_factory=list)


def _validate_square(B: object, name: str) -> list[list[int]]:
    if not isinstance(B, (list, tuple)) or len(B) == 0:
        raise InvalidBasisError(name, "must be a non-empty square matrix")
    n = len(B)
    if n > MAX_INTERSECTION_DIM:
        raise InvalidBasisError(
            name,
            f"dimension {n} exceeds MAX_INTERSECTION_DIM={MAX_INTERSECTION_DIM}",
        )
    rows: list[list[int]] = []
    for i, row in enumerate(B):
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise InvalidBasisError(
                f"{name}[{i}]", f"row must be a sequence of length {n}"
            )
        out: list[int] = []
        for j, x in enumerate(row):
            if isinstance(x, bool) or not isinstance(x, int):
                raise InvalidBasisError(
                    f"{name}[{i}][{j}]",
                    f"element must be exact int, got {type(x).__name__}",
                )
            out.append(x)
        rows.append(out)
    return rows


def _det_int(M: Sequence[Sequence[int]]) -> int:
    """Bareiss 精确整数行列式。"""
    n = len(M)
    X = [list(map(int, row)) for row in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if X[k][k] == 0:
            for t in range(k + 1, n):
                if X[t][k]:
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
    return sign * X[-1][-1]


def _rank_int(M: Sequence[Sequence[int]]) -> int:
    """Bareiss 风格消元求整数矩阵的秩（用于秩亏定位）。"""
    X = [list(row) for row in M]
    n, m = len(X), len(X[0])
    rank, col = 0, 0
    while rank < n and col < m:
        piv = -1
        for i in range(rank, n):
            if X[i][col] != 0:
                piv = i
                break
        if piv < 0:
            col += 1
            continue
        X[rank], X[piv] = X[piv], X[rank]
        for i in range(rank + 1, n):
            if X[i][col] != 0:
                g = X[i][col]
                for j in range(col, m):
                    X[i][j] = X[i][j] * X[rank][col] - g * X[rank][j]
        rank += 1
        col += 1
    return rank


def _mat_mul(
    A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]
) -> list[list[int]]:
    n, r, m = len(A), len(B), len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(r)) for j in range(m)]
        for i in range(n)
    ]


def _transpose(A: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*A)]


def _col_hnf(
    A: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """列 Hermite 标准形（带幺模变换矩阵）。

    输入 n x m 整数矩阵（行秩 n，m >= n）。返回 (H, U)，U 为 m x m 幺模
    矩阵，A @ U == H，其中 H 的前 n 列为下三角列 HNF（对角元为正、
    左侧元素已约化到 [0, 对角元)），后 m-n 列为零。U 的后 m-n 列即
    A 的整数核的一组 Z-基。
    """
    n, m = len(A), len(A[0])
    H = [list(map(int, row)) for row in A]
    U = [[1 if i == j else 0 for j in range(m)] for i in range(m)]

    def swap_cols(p: int, q: int) -> None:
        if p == q:
            return
        for r in range(n):
            H[r][p], H[r][q] = H[r][q], H[r][p]
        for r in range(m):
            U[r][p], U[r][q] = U[r][q], U[r][p]

    def add_col(p: int, q: int, mul: int) -> None:  # col p += mul * col q
        if mul == 0:
            return
        for r in range(n):
            H[r][p] += mul * H[r][q]
        for r in range(m):
            U[r][p] += mul * U[r][q]

    def neg_col(p: int) -> None:
        for r in range(n):
            H[r][p] = -H[r][p]
        for r in range(m):
            U[r][p] = -U[r][p]

    for i in range(n):
        while True:
            piv = -1
            for j in range(i, m):
                if H[i][j] != 0 and (piv < 0 or abs(H[i][j]) < abs(H[i][piv])):
                    piv = j
            if piv < 0:
                raise RankDeficientError(n, _rank_int(A))
            swap_cols(i, piv)
            for j in range(i + 1, m):
                if H[i][j] != 0:
                    add_col(j, i, -(H[i][j] // H[i][i]))
            if all(H[i][j] == 0 for j in range(i + 1, m)):
                break
        if H[i][i] < 0:
            neg_col(i)
        for k in range(i - 1, -1, -1):  # 约化对角元左侧元素到 [0, 对角元)
            add_col(k, i, -(H[k][i] // H[i][i]))
    return H, U


def _snf_diag(M: Sequence[Sequence[int]]) -> list[int]:
    """Smith 标准形对角元（正不变因子，d1 | d2 | ... | dn）。"""
    A = [list(map(int, row)) for row in M]
    n = len(A)
    diag: list[int] = []
    for k in range(n):
        while True:
            # 每轮把子矩阵中绝对值最小的非零元换到 (k,k)，保证主元严格递减。
            pos: Optional[tuple[int, int]] = None
            for i in range(k, n):
                for j in range(k, n):
                    if A[i][j] != 0 and (
                        pos is None or abs(A[i][j]) < abs(A[pos[0]][pos[1]])
                    ):
                        pos = (i, j)
            if pos is None:
                break
            i0, j0 = pos
            if i0 != k:
                A[k], A[i0] = A[i0], A[k]
            if j0 != k:
                for r in A:
                    r[k], r[j0] = r[j0], r[k]
            for j in range(k + 1, n):
                if A[k][j] != 0:
                    q = A[k][j] // A[k][k]
                    for r in range(n):
                        A[r][j] -= q * A[r][k]
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    q = A[i][k] // A[k][k]
                    for c in range(n):
                        A[i][c] -= q * A[k][c]
            # 行/列 k 未清零：余数严格小于主元，下一轮自动换更小主元。
            if any(A[k][j] != 0 for j in range(k + 1, n)) or any(
                A[i][k] != 0 for i in range(k + 1, n)
            ):
                continue
            bad: Optional[int] = None
            for i in range(k + 1, n):
                if any(A[i][j] % A[k][k] != 0 for j in range(k + 1, n)):
                    bad = i
                    break
            if bad is None:
                break
            for c in range(n):  # 重新引入不可整除项，继续辗转相除
                A[k][c] += A[bad][c]
        diag.append(abs(A[k][k]))
    return diag


def intersection_basis(
    B1: object, B2: object, budget: int = DEFAULT_BUDGET
) -> IntersectionResult:
    """计算两个同维满秩整数列格的交格基（LLL 缩短）与双边坐标证书。

    参数
    ----
    B1, B2 : n x n 整数矩阵（嵌套序列），列向量为格基，1 <= n <= 4，
        均须满秩（det != 0）。
    budget : 传给内部 LLL 缩短的操作预算（正整数，bool 拒绝）。

    返回
    ----
    IntersectionResult；basis 的列生成 L(B1) ∩ L(B2)，且
    B1 @ left_coords == B2 @ right_coords == basis。

    异常
    ----
    InvalidBasisError : 形状/元素非法（可定位）。
    InvalidParameterError : budget 非法。
    RankDeficientError : 任一输入奇异（数学上证明非满秩）。
    """
    A1 = _validate_square(B1, "B1")
    A2 = _validate_square(B2, "B2")
    if len(A1) != len(A2):
        raise InvalidBasisError(
            "B2", f"dimension {len(A2)} != dimension {len(A1)} of B1"
        )
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise InvalidParameterError(
            "budget", "must be a positive int (bool rejected)"
        )
    n = len(A1)
    for A in (A1, A2):
        if _det_int(A) == 0:
            raise RankDeficientError(n, _rank_int(A))

    # 1) 精确整数核：ker_Z [B1 | -B2]，列 HNF 变换矩阵的末 n 列即核的 Z-基。
    joint = [A1[i] + [-x for x in A2[i]] for i in range(n)]
    H, Ucol = _col_hnf(joint)
    kernel = [row[n:] for row in Ucol]  # 2n x n，[B1|-B2] @ kernel == 0
    kx = kernel[:n]  # B1 侧坐标
    ky = kernel[n:]  # B2 侧坐标（B1 kx == B2 ky）

    # 2) 交格基（列）：C = B1 @ kx == B2 @ ky。
    C = _mat_mul(A1, kx)

    # 3) 现有 LLL（行基约定）缩短：对 C 的转置约化，再转回列基。
    res = reduce_basis(_transpose(C), budget=budget)
    T = _transpose(res.transform)  # 列约定下的幺模右乘变换
    G = _transpose(res.basis)
    U = _mat_mul(kx, T)
    V = _mat_mul(ky, T)

    index_left = abs(_det_int(U))
    index_right = abs(_det_int(V))
    sum_det = 1
    for i in range(n):
        sum_det *= abs(H[i][i])
    return IntersectionResult(
        basis=G,
        left_coords=U,
        right_coords=V,
        index_in_left=index_left,
        index_in_right=index_right,
        snf_left=_snf_diag(U),
        snf_right=_snf_diag(V),
        sum_det=sum_det,
        lll_log=list(res.log),
    )
