"""盒内最近格向量（限定维数 <= 3，调用者给定有限系数盒）。

给定满行秩整数行基 ``A``（n <= 3，行向量为基）与有理目标点 ``t``，
在调用者给定的有限整数系数盒
``box = [(lo_0, hi_0), ..., (lo_{n-1}, hi_{n-1})]`` 内**穷举**全部整系数
向量 ``c``，返回使平方距离 ``||c·A - t||^2`` 最小的格向量。

契约要点
--------
* **只保证盒内最优**：盒外可能存在更近的格向量；本模块不做无限域
  截断证明，也不声称给出全格最近向量。
* **并列规则**：多个系数向量达到同一最小距离时，返回**原基系数
  字典序最小**者（按 c = (c_0, ..., c_{n-1}) 的元组序）。
* **精确性**：目标允许 ``int``/``Fraction``，平方距离以
  :class:`fractions.Fraction` 精确返回，无浮点、无容差。
* 返回的格向量 ``vector = coefficients · A``（对**原基** A，不是约化基）。

复杂度与盒的体积 ``prod(hi_i - lo_i + 1)`` 成正比（n <= 3 的纯穷举），
盒的大小由调用者负责；本模块不另设搜索预算。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from .core import _gram_schmidt, _validate_matrix
from .errors import InvalidBasisError, InvalidParameterError

#: 最近向量查询支持的最大基维数（行数）。与 LLL 的 MAX_DIM=8 不同，
#: 盒内穷举的契约上限为 3。
CVP_MAX_DIM: int = 3


@dataclass(frozen=True)
class CVPResult:
    """盒内最近格向量结果。

    ``coefficients`` 为原基整数系数 c（``vector == c · A``）；
    ``distance2`` 为精确平方距离 ``||vector - target||^2``（Fraction）；
    ``box`` 回显实际使用的系数盒；``nodes_visited`` 为穷举的系数向量数，
    等于盒体积，确定可复算。
    """

    coefficients: list[int]
    vector: list[int]
    distance2: Fraction
    box: list[tuple[int, int]]
    nodes_visited: int


def _validate_target(target: object, m: int) -> list[Fraction]:
    if not isinstance(target, (list, tuple)) or len(target) == 0:
        raise InvalidParameterError(
            "target", "must be a non-empty sequence of int/Fraction"
        )
    if len(target) != m:
        raise InvalidParameterError(
            "target",
            f"length {len(target)} != basis column count {m}",
        )
    out: list[Fraction] = []
    for j, x in enumerate(target):
        # 与全库口径一致：拒绝 bool 与一切 float（含 NaN/Infinity），
        # 只接受精确 int / Fraction，不做静默数值转换。
        if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
            raise InvalidParameterError(
                f"target[{j}]",
                f"must be exact int or Fraction, got {type(x).__name__}",
            )
        out.append(Fraction(x))
    return out


def _validate_box(box: object, n: int) -> list[tuple[int, int]]:
    if not isinstance(box, (list, tuple)) or len(box) != n:
        raise InvalidParameterError(
            "box", f"must provide exactly n={n} (lo, hi) pairs"
        )
    out = []
    for i, pair in enumerate(box):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise InvalidParameterError(
                f"box[{i}]", "must be a (lo, hi) pair of ints"
            )
        lo, hi = pair
        for name, v in (("lo", lo), ("hi", hi)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise InvalidParameterError(
                    f"box[{i}].{name}",
                    f"bound must be exact int (bool rejected), "
                    f"got {type(v).__name__}",
                )
        if lo > hi:
            raise InvalidParameterError(
                f"box[{i}]", f"empty interval: lo={lo} > hi={hi}"
            )
        out.append((lo, hi))
    return out


def closest_vector(A: object, target: object, box: object) -> CVPResult:
    """在有限系数盒内找离有理目标最近的格向量。

    参数
    ----
    A : n×m 整数矩阵（嵌套序列），行向量为基；``1 <= n <= 3``，``m >= n``，
        满行秩。元素必须是精确 ``int``。
    target : 长度 m 的目标点，元素为 ``int`` 或 ``Fraction``。
    box : n 个 ``(lo, hi)`` 整数对，``lo <= hi``，给出每个原基系数的
        闭区间搜索范围。

    返回
    ----
    CVPResult：``coefficients``（原基整数系数）、``vector``（格向量）、
    ``distance2``（精确平方距离，Fraction）。多个系数达到同一最小距离时，
    返回系数字典序最小者。**只保证盒内最优。**

    异常
    ----
    InvalidBasisError : 基矩阵形状/元素非法，或行数超过 CVP_MAX_DIM=3。
    InvalidParameterError : target/box 非法（可定位到具体分量）。
    RankDeficientError : 数学上证明基行线性相关。
    """
    rows = _validate_matrix(A)  # 形状/元素/列数校验（n <= MAX_DIM=8）
    n = len(rows)
    if n > CVP_MAX_DIM:
        raise InvalidBasisError(
            "A", f"row count {n} exceeds CVP_MAX_DIM={CVP_MAX_DIM}"
        )
    m = len(rows[0])
    t = _validate_target(target, m)
    bounds = _validate_box(box, n)
    # 满行秩检验：复用核心的精确 Gram-Schmidt，秩亏抛 RankDeficientError。
    _gram_schmidt([[Fraction(x) for x in row] for row in rows], n)

    # product 按 (c_0, ..., c_{n-1}) 字典序升序产出；配合严格小于更新，
    # 并列时自然保留字典序最小的系数向量。
    best_c: list[int] | None = None
    best_v: list[int] | None = None
    best_d: Fraction | None = None
    nodes = 0
    ranges = [range(lo, hi + 1) for lo, hi in bounds]
    for c in product(*ranges):
        nodes += 1
        v = [sum(c[i] * rows[i][j] for i in range(n)) for j in range(m)]
        d = Fraction(0)
        for j in range(m):
            diff = v[j] - t[j]
            d += diff * diff
        if best_d is None or d < best_d:
            best_d = d
            best_c = list(c)
            best_v = v

    assert best_c is not None and best_v is not None and best_d is not None
    return CVPResult(
        coefficients=best_c,
        vector=best_v,
        distance2=best_d,
        box=bounds,
        nodes_visited=nodes,
    )
