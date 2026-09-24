"""盒约束最近格向量（CVP）精确枚举，维数 n <= 3。

问题
----
给定满行秩整数行基 A（n×m，n <= 3）、有理目标点 t（长度 m）与有限的
**原基整数系数盒**::

    lo_i <= c_i <= hi_i,  c_i ∈ Z,  格点 v = c·A,

求使平方距离 ‖t - v‖² 最小的全部系数 c（并列全部返回，按 c 的字典序排列）。

只保证**盒内最优**：盒外可能存在更近的格点，本函数不作任何全局声明
（``NearestVectorResult.optimal_in_box`` 恒为 True，是这份有限保证的显式标记）。

算法
----
1. 复用 :func:`lll_reduce.core.reduce_basis` 得 B = U·A（U 整数幺模）。
2. 在约化基坐标 z（格点 = z·B = z·U·A）上做深度优先椭球枚举：
   Gram–Schmidt 分解把平方距离拆成逐层精确有理增量，用整数运算求每个
   z_i 的可行整数区间（不开平方根、不用浮点），超出当前最优即剪枝。
3. 盒约束在原基系数空间（c = z·U）。把固定的深层坐标代入盒不等式后，
   对剩余自由变量做精确有理的 Fourier–Motzkin 消元，得到当前坐标的
   （实松弛）可行区间，与椭球区间取交。FM 的消元结构只依赖 U，故每个
   搜索层在枚举前**符号预计算一次**（z_i 的上下界为深层坐标的仿射式），
   节点上只做线性赋值。盒边界直接参与剪枝，而不是只在叶子上过滤。
4. 初始可行解用 Babai 最近平面点的原基系数逐分量夹到盒内得到
   （夹逼后仍是盒内格点），保证一开始就有紧的上界。最底层固定深层坐标
   后直接取离 GS 中线最近的盒内整数（恰半整数时两个并列），全部并列点
   精确无遗漏。

全部比较使用 :class:`fractions.Fraction`，无浮点、无容差。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Optional

from .core import DEFAULT_BUDGET, _gram_schmidt, _validate_matrix, reduce_basis
from .errors import InvalidBasisError, InvalidParameterError

#: 本函数支持的最大格维数（基的行数）。
CVP_MAX_DIM: int = 3


@dataclass(frozen=True)
class NearestVectorResult:
    """盒约束最近格向量结果（只读）。

    ``coefficients[k]`` 与 ``vectors[k]`` 一一对应：
    ``vectors[k] == coefficients[k] · A``；所有并列点的平方距离都等于
    ``squared_distance``（精确 :class:`fractions.Fraction`，整数距离时分母为 1）。
    系数列表按原基系数元组的字典序升序排列。

    保证范围仅限盒内：``optimal_in_box`` 恒为 True；盒外是否更近不予判断。
    """

    target: list[Fraction]
    basis: list[list[int]]
    bounds: tuple[tuple[int, int], ...]
    coefficients: list[tuple[int, ...]]
    vectors: list[list[int]]
    squared_distance: Fraction
    optimal_in_box: bool
    nodes_visited: int
    leaves_evaluated: int
    box_points: int


def nearest_vector(
    A: object,
    target: object,
    bounds: object,
    *,
    budget: int = DEFAULT_BUDGET,
) -> NearestVectorResult:
    """在给定有限系数盒内求离有理目标点最近的格点（n <= 3，精确）。

    参数
    ----
    A : n×m 整数矩阵（嵌套序列），行向量为基，1 <= n <= 3，m >= n，满行秩。
        元素只接受精确 ``int``（``bool``、``float`` 一律拒绝）。
    target : 长度 m 的序列，元素为 ``int`` 或 ``fractions.Fraction``
        （``bool``、``float`` 包括 NaN/Inf 一律拒绝，不做静默浮点转换）。
    bounds : 有限系数盒，两种形式：

        * 非负整数 B：每个系数 c_i ∈ [-B, B]；
        * 长度 n 的序列，每项为 (lo_i, hi_i)，均为 int 且 lo_i <= hi_i。

    budget : 透传给内部 LLL 的正整数操作预算；LLL 预算耗尽时抛
        :class:`~lll_reduce.errors.BudgetExhaustedError`（无任何用户可见状态
        被更新——本函数是无状态查询）。

    返回
    ----
    :class:`NearestVectorResult`。

    异常
    ----
    InvalidBasisError : A 形状/元素非法、维数 > 3，或秩亏（秩亏由内部
        LLL 以 :class:`RankDeficientError` 抛出）。
    InvalidParameterError : target/bounds/budget 不合法。
    """
    # ---- 输入校验：全部在任何计算之前完成，失败即无副作用 ----
    rows = _validate_matrix(A)
    n = len(rows)
    if n > CVP_MAX_DIM:
        raise InvalidBasisError(
            "A",
            f"nearest_vector supports at most {CVP_MAX_DIM} rows, got {n}",
        )
    m = len(rows[0])

    t = _validate_target(target, m)
    normed_bounds = _validate_bounds(bounds, n)
    # budget 由 reduce_basis 统一校验（类型/正整数/bool），错误语义不变。

    # ---- LLL 约化（复用既有公开入口与其全部错误语义）----
    red = reduce_basis(rows, budget)
    B = [[Fraction(x) for x in row] for row in red.basis]
    U = red.transform
    V = _inverse_unimodular(U)  # U^{-1}，精确整数

    # ---- 约化基的有理 GS 与目标点的 GS 坐标 ----
    bstar, mu, norm2 = _gram_schmidt(B, n)
    y = [
        sum(t[k] * bstar[i][k] for k in range(m)) / norm2[i]
        for i in range(n)
    ]

    los = [lo for lo, _ in normed_bounds]
    his = [hi for _, hi in normed_bounds]
    box_points = 1
    for lo, hi in normed_bounds:
        box_points *= hi - lo + 1

    # 盒不等式（对全部 n 个 z 变量）：sum_j U[j][i]*z[j] >= lo_i
    # 且 -sum_j U[j][i]*z[j] >= -hi_i。系数为整数，rhs 为整数。
    base_rows: list[tuple[list[int], int]] = []
    for i in range(n):
        col = [U[j][i] for j in range(n)]
        base_rows.append((col, los[i]))
        base_rows.append(([-x for x in col], -his[i]))

    # FM 消元的结构只依赖 U（整数系数），与节点上固定的深层坐标无关：
    # 每个搜索层只预计算一次，得到 z_i 上下界关于 z_{i+1..n-1} 的仿射式，
    # 节点上只求值，不重复消元。层 0 不做任何消元，行系数全为整数，
    # 节点上走纯整数路径（叶子层是访问量最大的层）。
    level_bounds = [_fm_bounds_for_level(base_rows, n, i) for i in range(n)]
    integral_levels = [
        all(
            ci.denominator == 1
            and rhs.denominator == 1
            and all(d.denominator == 1 for d in deep)
            for ci, deep, rhs in level_bounds[i]
        )
        for i in range(n)
    ]

    # 目标点在格行空间正交补上的平方分量：对所有格点为同一常数，必须计入
    # 距离（长方形基或不在行空间内的有理目标时非零），但不影响最优性比较。
    residual = sum(x * x for x in t) - sum(
        y[i] ** 2 * norm2[i] for i in range(n)
    )
    # t = Σ y_i b*_i + res，res ⟂ 行空间；精确恒等式下残差平方必非负。
    if residual < 0:
        raise AssertionError("internal error: negative orthogonal residual")

    # ---- 初始可行解：Babai 最近平面 -> 原基系数 -> 逐分量夹入盒内 ----
    z_seed_r = [0] * n
    for i in range(n - 1, -1, -1):
        center = y[i] - sum(mu[j][i] * z_seed_r[j] for j in range(i + 1, n))
        z_seed_r[i] = _round_half_even_local(center)
    c_seed_r = [sum(z_seed_r[j] * U[j][i] for j in range(n)) for i in range(n)]
    c_seed = [min(his[i], max(los[i], c_seed_r[i])) for i in range(n)]
    z_seed = [sum(c_seed[i] * V[i][j] for i in range(n)) for j in range(n)]
    # 幺模恒等式自检：夹逼只改 c，z = c·U^{-1} 必精确还原。
    if [sum(z_seed[j] * U[j][i] for j in range(n)) for i in range(n)] != c_seed:
        raise AssertionError("internal error: seed coefficient conversion")
    best = _squared_distance(z_seed, B, t, m)

    # ---- 精确椭球枚举（约化坐标 z），盒约束用 FM 消元逐层剪枝 ----
    z_fixed = [0] * n
    nodes = 0
    leaves = 0
    winners: set[tuple[int, ...]] = {tuple(c_seed)}

    def box_bounds(i: int) -> Optional[tuple[Optional[int], Optional[int]]]:
        """固定 z[i+1..n-1] 后，盒内可行 z_i 的整数边界（已取整）。

        层 0 的 FM 行全为整数系数，走纯整数路径（叶子层访问量最大）；
        其余层为 Fraction 仿射赋值后 ceil/floor。任一端点 None 表示该侧
        无约束，整体不可行返回 None。实松弛取整后只会多探索、绝不错剪
        可行整数点。
        """
        lo: Optional[Fraction] = None
        hi: Optional[Fraction] = None
        rows = level_bounds[i]
        if integral_levels[i]:
            lo_i: Optional[int] = None
            hi_i: Optional[int] = None
            for ci, deep, rhs0 in rows:
                shifted = rhs0.numerator - sum(
                    d.numerator * z_fixed[i + 1 + k]
                    for k, d in enumerate(deep)
                )
                civ = ci.numerator
                if civ == 0:
                    if shifted > 0:
                        return None
                elif civ > 0:  # z_i >= ceil(shifted/ci)，整数时整除向上
                    bnd = -((-shifted) // civ)
                    lo_i = bnd if lo_i is None else max(lo_i, bnd)
                else:  # z_i <= floor(shifted/ci)
                    bnd = shifted // civ
                    hi_i = bnd if hi_i is None else min(hi_i, bnd)
            return lo_i, hi_i

        for ci, deep, rhs0 in rows:
            shifted = rhs0 - sum(
                deep[k] * z_fixed[i + 1 + k] for k in range(n - i - 1)
            )
            if ci == 0:
                if shifted > 0:
                    return None
            elif ci > 0:
                bnd = shifted / ci
                lo = bnd if lo is None else max(lo, bnd)
            else:
                bnd = shifted / ci
                hi = bnd if hi is None else min(hi, bnd)
        if lo is not None and hi is not None and lo > hi:
            return None
        return (
            None if lo is None else _ceil_frac(lo),
            None if hi is None else _floor_frac(hi),
        )

    def consider_leaf(z0: int, spent_above: Fraction) -> None:
        """叶子：z0 给定，总距离 = spent_above + 本层增量 + 正交补常数。"""
        nonlocal leaves, best
        leaves += 1
        center0 = y[0] - sum(mu[j][0] * z_fixed[j] for j in range(1, n))
        total = spent_above + (center0 - z0) ** 2 * norm2[0] + residual
        if total <= best:
            coeff = tuple(
                sum(z_fixed[j] * U[j][r] for j in range(n)) for r in range(n)
            )
            if total < best:
                best = total
                winners.clear()
            winners.add(coeff)

    def dfs(i: int, spent: Fraction) -> None:
        nonlocal nodes, best
        nodes += 1
        center = y[i] - sum(mu[j][i] * z_fixed[j] for j in range(i + 1, n))

        # 最底层：固定 z1..z_{n-1} 后，z0 只出现在自己的二次项里，盒内
        # 最优值是离 center 最近的整数（恰好半整数时并列两个），不必枚举
        # 整个椭球区间——这是精确的，不是近似。
        if i == 0:
            boxed0 = box_bounds(0)
            if boxed0 is None:
                return
            blo, bhi = boxed0
            if blo is None or bhi is None:
                # 盒在每个原系数分量上有限，层 0 的 FM 行必给双侧整数界。
                raise AssertionError("internal error: finite box gave unbounded z0")
            # center 的最近整数：floor/ceil 比较距离；恰半整数时并列两个。
            q, r = divmod(center.numerator, center.denominator)
            twice = 2 * r
            if twice < center.denominator:
                nearest = [q]
            elif twice > center.denominator:
                nearest = [q + 1]
            else:
                nearest = [q, q + 1]
            cand = {max(blo, min(bhi, z)) for z in nearest}
            for z0 in sorted(cand):
                z_fixed[0] = z0
                consider_leaf(z0, spent)
            return

        remaining = best - residual - spent
        ell = _ellipsoid_integer_bounds(center, remaining, norm2[i])
        if ell is None:
            return
        boxed = box_bounds(i)
        if boxed is None:
            return
        blo, bhi = boxed
        lo_z = ell[0] if blo is None else max(ell[0], blo)
        hi_z = ell[1] if bhi is None else min(ell[1], bhi)
        if lo_z > hi_z:
            return

        # 按离 center 的距离从近到远枚举；先夹到区间内，再两侧交替，
        # 保证区间内每个整数恰好出现一次（含中线落在区间外的情形）。
        mid = max(lo_z, min(hi_z, _round_half_even_local(center)))

        def visit(zc: int) -> None:
            delta = (center - zc) ** 2 * norm2[i]
            spent2 = spent + delta
            # 枚举区间本身保证 delta <= remaining；保留精确防御性比较。
            if spent2 + residual > best:
                return
            z_fixed[i] = zc
            dfs(i - 1, spent2)

        visit(mid)
        down, up = mid - 1, mid + 1
        while down >= lo_z or up <= hi_z:
            if down >= lo_z:
                visit(down)
                down -= 1
            if up <= hi_z:
                visit(up)
                up += 1

    dfs(n - 1, Fraction(0))

    coeffs_sorted = sorted(winners)
    vectors = [
        [sum(c[i] * rows[i][k] for i in range(n)) for k in range(m)]
        for c in coeffs_sorted
    ]

    return NearestVectorResult(
        target=t[:],
        basis=[row[:] for row in rows],
        bounds=normed_bounds,
        coefficients=coeffs_sorted,
        vectors=vectors,
        squared_distance=best,
        optimal_in_box=True,
        nodes_visited=nodes,
        leaves_evaluated=leaves,
        box_points=box_points,
    )


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------

def _validate_target(target: object, m: int) -> list[Fraction]:
    if not isinstance(target, (list, tuple)):
        raise InvalidParameterError(
            "target", f"must be a sequence of {m} rational numbers"
        )
    if len(target) != m:
        raise InvalidParameterError(
            "target", f"length {len(target)} != basis column count {m}"
        )
    out: list[Fraction] = []
    for j, x in enumerate(target):
        if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
            raise InvalidParameterError(
                f"target[{j}]",
                f"element must be int or Fraction, got {type(x).__name__}",
            )
        out.append(Fraction(x))
    return out


def _validate_bounds(bounds: object, n: int) -> tuple[tuple[int, int], ...]:
    if isinstance(bounds, bool) or not isinstance(bounds, (int, Sequence)):
        raise InvalidParameterError(
            "bounds", "must be a non-negative int or a sequence of (lo, hi) pairs"
        )
    if isinstance(bounds, int):
        B = bounds
        if B < 0:
            raise InvalidParameterError("bounds", f"symmetric bound must be >= 0, got {B}")
        return tuple((-B, B) for _ in range(n))

    if len(bounds) != n:
        raise InvalidParameterError(
            "bounds", f"expected {n} (lo, hi) pairs, got {len(bounds)}"
        )
    out: list[tuple[int, int]] = []
    for i, pair in enumerate(bounds):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise InvalidParameterError(
                f"bounds[{i}]", "must be a (lo, hi) pair of exact ints"
            )
        lo, hi = pair
        if isinstance(lo, bool) or not isinstance(lo, int):
            raise InvalidParameterError(
                f"bounds[{i}].lo", f"must be exact int, got {type(lo).__name__}"
            )
        if isinstance(hi, bool) or not isinstance(hi, int):
            raise InvalidParameterError(
                f"bounds[{i}].hi", f"must be exact int, got {type(hi).__name__}"
            )
        if lo > hi:
            raise InvalidParameterError(
                f"bounds[{i}]", f"lo {lo} > hi {hi}: empty coefficient box"
            )
        out.append((lo, hi))
    return tuple(out)


# --------------------------------------------------------------------------
# 精确有理 / 整数辅助
# --------------------------------------------------------------------------

def _round_half_even_local(x: Fraction) -> int:
    """最近整数、并列向偶数（与 core.round_half_even 同一规则，本地复制以避免
    循环导入时的公开耦合；纯整数比较，无浮点）。"""
    q, r = divmod(x.numerator, x.denominator)
    twice = 2 * r
    if twice < x.denominator:
        return q
    if twice > x.denominator:
        return q + 1
    return q if q % 2 == 0 else q + 1


def _ceil_frac(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _floor_frac(x: Fraction) -> int:
    return x.numerator // x.denominator


def _ellipsoid_integer_bounds(
    center: Fraction, budget: Fraction, weight: Fraction
) -> Optional[tuple[int, int]]:
    """满足 (z - center)^2 * weight <= budget 的全部整数 z 的区间（精确）。

    不开平方根、不构造中间 Fraction：化成分数平方
    ``(qz-p)^2 <= budget·q²/weight`` 后，右端用一次整数分子分母组装与
    :func:`math.isqrt` 求 floor(sqrt(·))。budget < 0（当前累计已超最优）时
    无可行点。``isqrt(floor(T)) == floor(sqrt(T))``（floor(T) 为完全平方而
    T 有非零小数部分时仍成立），故不重不漏；叶子另有精确距离比较兜底。
    """
    if budget < 0:
        return None
    q, p = center.denominator, center.numerator
    num = budget.numerator * q * q * weight.denominator
    den = budget.denominator * weight.numerator
    radius = isqrt(num // den)
    lo = -((-(p - radius)) // q)  # ceil((p - radius) / q)
    hi = (p + radius) // q        # floor((p + radius) / q)
    return lo, hi


def _squared_distance(
    z: Sequence[int], B: Sequence[Sequence[Fraction]], t: Sequence[Fraction], m: int
) -> Fraction:
    return sum(
        (t[k] - sum(z[j] * B[j][k] for j in range(len(z)))) ** 2
        for k in range(m)
    )


def _inverse_unimodular(U: Sequence[Sequence[int]]) -> list[list[int]]:
    """整数幺模矩阵的精确整数逆（Gauss–Jordan 有理消元，结果分母必须全为 1）。"""
    n = len(U)
    M: list[list[Fraction]] = [
        [Fraction(U[i][j]) for j in range(n)]
        + [Fraction(1 if i == j else 0) for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            raise AssertionError("internal error: U from LLL is singular")
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        M[col] = [x / p for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [M[r][j] - f * M[col][j] for j in range(2 * n)]
    inv: list[list[int]] = []
    for i in range(n):
        row: list[int] = []
        for j in range(n, 2 * n):
            x = M[i][j]
            if x.denominator != 1:
                raise AssertionError("internal error: unimodular inverse non-integral")
            row.append(x.numerator)
        inv.append(row)
    return inv


def _fm_bounds_for_level(
    base_rows: Sequence[tuple[Sequence[int], int]], n: int, i: int
) -> list[tuple[Fraction, list[Fraction], Fraction]]:
    """为搜索层 i 符号预计算盒约束的 Fourier–Motzkin 投影。

    输入全部 n 个不等式 ``sum_j a_j z_j >= r``；保留 z_i 与深层参数
    z_{i+1..n-1}，消去 z_0..z_{i-1}。返回行列表
    ``(c_i, [d_{i+1},...,d_{n-1}], r0)``，表示::

        c_i * z_i + sum_k d_k z_k >= r0。

    结构只依赖 U，因此每层只在搜索开始前计算一次；节点上对深层坐标做
    线性赋值即可。消元规则（pos 下界与 neg 上界联立产生新行）：
    r_p - Σp·x ≤ r_n + Σn·x ⇒ Σ(p+n)·x ≥ r_p - r_n。
    """
    cur: list[tuple[list[Fraction], list[Fraction], Fraction]] = []
    for a, r in base_rows:
        free = [Fraction(a[j]) for j in range(i + 1)]
        deep = [Fraction(a[j]) for j in range(i + 1, n)]
        cur.append((free, deep, Fraction(r)))

    free_cols = list(range(i + 1))
    for v in range(i - 1, -1, -1):
        vpos = free_cols.index(v)
        new_cols = [c for c in free_cols if c != v]
        pos: list[tuple[list[Fraction], list[Fraction], Fraction]] = []
        neg: list[tuple[list[Fraction], list[Fraction], Fraction]] = []
        out: list[tuple[list[Fraction], list[Fraction], Fraction]] = []
        for free, deep, r in cur:
            c = free[vpos]
            rest = [free[k] for k in range(len(free)) if k != vpos]
            if c == 0:
                out.append((rest, deep, r))
                continue
            scale_f = [x / abs(c) for x in rest]
            scale_d = [x / abs(c) for x in deep]
            if c > 0:
                # z_v >= r/|c| - Σ scale_f·x - Σ scale_d·w
                pos.append((scale_f, scale_d, r / abs(c)))
            else:
                # z_v <= -r/|c| + Σ scale_f·x + Σ scale_d·w
                neg.append((scale_f, scale_d, -r / abs(c)))
        for pf, pd, rp in pos:
            for nf, nd, rn in neg:
                out.append(
                    (
                        [pf[k] + nf[k] for k in range(len(new_cols))],
                        [pd[k] + nd[k] for k in range(n - i - 1)],
                        rp - rn,
                    )
                )
        cur = out
        free_cols = new_cols

    # 自由变量只剩 z_i（free_cols == [i]，在第 0 位）。
    return [(free[0], deep, r) for free, deep, r in cur]
