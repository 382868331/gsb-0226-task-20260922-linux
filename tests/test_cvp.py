"""盒约束最近格向量（CVP）的必要测试。

期望值来源彼此独立：
  * 手算定值（I2/2I 的并列、歪斜幺模格的精确格点系数、目标远离盒、
    行空间外目标的正交补距离、单点盒、1 维）；
  * tests/_reference.py::enumerate_box_cvp —— 直接遍历原系数盒的独立
    穷举，不调用被测包、不做 LLL、不做剪枝；
  * 返回值自洽性（vectors == coefficients·A、平方距离用 Fraction 复算、
    列字典序、盒内每一点都不优于最优）。
不使用被测实现自身生成期望值。
"""

import unittest
from fractions import Fraction
from itertools import product

from lll_reduce import (
    BudgetExhaustedError,
    CVP_MAX_DIM,
    InvalidBasisError,
    InvalidParameterError,
    RankDeficientError,
    nearest_vector,
)

from _reference import enumerate_box_cvp


def dist2(c, A, t):
    n, m = len(A), len(A[0])
    v = [sum(c[i] * A[i][k] for i in range(n)) for k in range(m)]
    return v, sum((t[k] - v[k]) ** 2 for k in range(m))


def lcg_sequence(seed=20260924, mod=9):
    state = seed

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)
        return (state >> 33) % mod - mod // 2

    return rnd


def det_n(M):
    n = len(M)
    X = [r[:] for r in M]
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


class TestHandComputed(unittest.TestCase):
    def test_half_integer_four_way_tie_sorted(self):
        # I2，目标 (1/2,1/2)，盒 [-1,1]^2：四个点距离平方都是 1/2，
        # 必须全部返回且按原系数字典序。
        r = nearest_vector(
            [[1, 0], [0, 1]],
            [Fraction(1, 2), Fraction(1, 2)],
            1,
        )
        self.assertEqual(
            r.coefficients, [(0, 0), (0, 1), (1, 0), (1, 1)]
        )
        self.assertEqual(r.squared_distance, Fraction(1, 2))
        self.assertEqual(
            r.vectors, [[0, 0], [0, 1], [1, 0], [1, 1]]
        )

    def test_scaled_lattice_four_way_tie(self):
        # A = 2I，目标 (1,1)：c∈{0,1}² 四点距离平方都是 2。
        r = nearest_vector([[2, 0], [0, 2]], [1, 1], 1)
        self.assertEqual(
            r.coefficients, [(0, 0), (0, 1), (1, 0), (1, 1)]
        )
        self.assertEqual(r.squared_distance, 2)

    def test_half_integer_axis_tie(self):
        # 2I，目标 (1,0)：c=(0,0) 与 (1,0) 并列，平方距离 1。
        r = nearest_vector([[2, 0], [0, 2]], [1, 0], 1)
        self.assertEqual(r.coefficients, [(0, 0), (1, 0)])
        self.assertEqual(r.squared_distance, 1)

    def test_skew_unimodular_lattice_exact_lattice_point(self):
        # det=-1 的歪斜格；t=(0,1) 恰是格点，原基系数手算为 (99,-100)：
        # 99*(100,99) - 100*(99,98) = (0,1)。盒 [-100,100]² 有 40401 点，
        # FM 盒剪枝应只探索极少数节点，且精确命中距离 0。
        A = [[100, 99], [99, 98]]
        r = nearest_vector(A, [0, 1], 100)
        self.assertEqual(r.coefficients, [(99, -100)])
        self.assertEqual(r.vectors, [[0, 1]])
        self.assertEqual(r.squared_distance, 0)
        self.assertEqual(r.box_points, 201 * 201)
        self.assertLess(r.nodes_visited, 50)

    def test_skew_lattice_asymmetric_box(self):
        # 同一格点系数 (99,-100)，非对称盒恰好包含它；验证盒按原系数
        # 而不是约化坐标解释。
        A = [[100, 99], [99, 98]]
        bounds = [(50, 120), (-120, -50)]
        r = nearest_vector(A, [0, 1], bounds)
        self.assertEqual(r.coefficients, [(99, -100)])
        self.assertEqual(r.bounds, tuple(tuple(b) for b in bounds))

    def test_target_far_outside_small_box(self):
        # 目标 (10,10)，盒 [0,2]²：无约束最近点 (10,10) 在盒外，
        # 盒内最优只能是角点 (2,2)，距离平方 2*8²=128。
        r = nearest_vector([[1, 0], [0, 1]], [10, 10], [(0, 2), (0, 2)])
        self.assertEqual(r.coefficients, [(2, 2)])
        self.assertEqual(r.vectors, [[2, 2]])
        self.assertEqual(r.squared_distance, 128)

    def test_target_outside_row_space_includes_residual(self):
        # 长方形基，格点 (c0,c1,c0+c1) 都在平面 x3=x1+x2 上；目标 (0,0,1)。
        # 平面最近点是 (1/3,1/3,2/3)（距离平方 1/3）但不是格点；最近格点
        # 为 (0,0,0)、(0,1,1)、(1,0,1)，距离平方都是 1。若漏算目标的正交
        # 补分量、只在格平面内量距离，会错答距离 0。
        A = [[1, 0, 1], [0, 1, 1]]
        r = nearest_vector(A, [0, 0, 1], 2)
        self.assertEqual(r.coefficients, [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(r.vectors, [[0, 0, 0], [0, 1, 1], [1, 0, 1]])
        self.assertEqual(r.squared_distance, 1)

    def test_rectangular_integer_target(self):
        A = [[1, 0, 10], [0, 1, 10]]
        # 目标 (0,0,1)：最近格点为原点，残差平方 = 1（精确手算）。
        r = nearest_vector(A, [0, 0, 1], 2)
        self.assertEqual(r.coefficients, [(0, 0)])
        self.assertEqual(r.vectors, [[0, 0, 0]])
        self.assertEqual(r.squared_distance, 1)

    def test_one_dimensional_fractional(self):
        # A=[2]，目标 1/2：c=0 距离 1/4；目标 1：c=0,1 并列距离 1。
        r = nearest_vector([[2]], [Fraction(1, 2)], 2)
        self.assertEqual(r.coefficients, [(0,)])
        self.assertEqual(r.squared_distance, Fraction(1, 4))
        r = nearest_vector([[2]], [1], 2)
        self.assertEqual(r.coefficients, [(0,), (1,)])
        self.assertEqual(r.squared_distance, 1)

    def test_single_point_box(self):
        # 盒退化为单点：无论目标在哪，唯一候选就是该点。
        r = nearest_vector([[1, 0], [0, 1]], [3, 3], [(0, 0), (0, 0)])
        self.assertEqual(r.coefficients, [(0, 0)])
        self.assertEqual(r.squared_distance, 18)
        r = nearest_vector(
            [[1, 0], [0, 1]], [0, 0], [(5, 5), (-3, -3)]
        )
        self.assertEqual(r.coefficients, [(5, -3)])
        self.assertEqual(r.vectors, [[5, -3]])
        self.assertEqual(r.squared_distance, 34)

    def test_three_dimensional_half_integer_tie(self):
        # I3，目标 (1/2,1/2,1/2)，B=1：2^3=8 个并列点，字典序输出。
        r = nearest_vector(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [Fraction(1, 2)] * 3,
            1,
        )
        want = list(product((0, 1), repeat=3))
        self.assertEqual(r.coefficients, want)
        self.assertEqual(r.squared_distance, Fraction(3, 4))
        self.assertEqual(r.leaves_evaluated, 8)

    def test_three_dimensional_skew_exact_point(self):
        # 3 维幺模歪斜基，格点系数取在盒深处，验证深层 FM 剪枝。
        A = [[1000, 1001, 0], [0, 1000, 1001], [1, 0, 1000]]
        t = [1000, 2001, 1001]  # = 1*b0 + 1*b1 + 0*b2
        r = nearest_vector(A, t, 2000)
        self.assertEqual(r.coefficients, [(1, 1, 0)])
        self.assertEqual(r.squared_distance, 0)
        self.assertLess(r.nodes_visited, 20)

    def test_zero_symmetric_box(self):
        r = nearest_vector([[3, 1], [1, 2]], [Fraction(1, 3), Fraction(2, 3)], 0)
        self.assertEqual(r.coefficients, [(0, 0)])
        self.assertEqual(r.squared_distance, Fraction(5, 9))

    def test_tuple_inputs_accepted(self):
        # A、target、bounds 全部用 tuple 提供，与 list 等价。
        r = nearest_vector(((1, 2), (3, 5)), (Fraction(1, 2), 0), ((-2, 2), (-2, 2)))
        w, d, _ = enumerate_box_cvp(
            [[1, 2], [3, 5]],
            [Fraction(1, 2), 0],
            [(-2, 2), (-2, 2)],
        )
        self.assertEqual(r.coefficients, [tuple(x) for x in w])
        self.assertEqual(r.squared_distance, d)


class TestResultInvariants(unittest.TestCase):
    def test_fields_self_consistent(self):
        A = [[19, 7, 2], [8, 3, 11], [-4, 9, 6]]
        r = nearest_vector(A, [Fraction(1, 2), 1, -2], 3)
        self.assertTrue(r.optimal_in_box)
        self.assertEqual(CVP_MAX_DIM, 3)
        self.assertEqual(r.basis, A)
        self.assertEqual(len(r.target), 3)
        # vectors == coefficients·A；距离可用 Fraction 独立复算。
        for c, v in zip(r.coefficients, r.vectors):
            vv, d = dist2(c, A, r.target)
            self.assertEqual(v, vv)
            self.assertEqual(d, r.squared_distance)
        # 严格字典序、无重复；每个系数都在盒内。
        self.assertEqual(r.coefficients, sorted(set(r.coefficients)))
        for c in r.coefficients:
            for i, (lo, hi) in enumerate(r.bounds):
                self.assertTrue(lo <= c[i] <= hi)
        self.assertEqual(r.box_points, 7 ** 3)

    def test_deterministic_and_stateless(self):
        A = [[31, 12, -7], [-4, 19, 8], [6, -3, 22]]
        r1 = nearest_vector(A, [Fraction(1, 3), Fraction(-2, 5), 4], 4)
        r2 = nearest_vector(A, [Fraction(1, 3), Fraction(-2, 5), 4], 4)
        self.assertEqual(r1.coefficients, r2.coefficients)
        self.assertEqual(r1.squared_distance, r2.squared_distance)
        self.assertEqual(r1.nodes_visited, r2.nodes_visited)

    def test_no_box_point_beats_winner(self):
        # 独立遍历盒内每一个点：无一严格优于返回的最优。
        A = [[7, 3, 1], [-2, 5, 8]]
        t = [Fraction(3, 2), Fraction(-1, 4), 2]
        bounds = [(-3, 4), (-5, 2)]
        r = nearest_vector(A, t, bounds)
        for c in product(range(-3, 5), range(-5, 3)):
            _, d = dist2(c, A, t)
            self.assertGreaterEqual(d, r.squared_distance)


class TestBruteForceCrossCheck(unittest.TestCase):
    def test_randomized_matches_independent_enumeration(self):
        rnd = lcg_sequence()
        cases = 0
        for idx in range(120):
            n = 1 + idx % 3
            m = n + (idx % 2)
            while True:
                A = [[rnd() for _ in range(m)] for _ in range(n)]
                if det_n([[A[i][j] for j in range(n)] for i in range(n)]) != 0:
                    break
            kind = idx % 4
            if kind == 0:
                t = [Fraction(rnd()) for _ in range(m)]
            elif kind == 1:
                t = [Fraction(rnd(), 1 + abs(rnd())) for _ in range(m)]
            elif kind == 2:
                t = [Fraction(1, 2)] * m
            else:
                t = [Fraction(50 + rnd()) for _ in range(m)]  # 远离小盒
            B = 1 + idx % 4 if kind != 3 else 2
            if idx % 5 == 0:
                bounds = tuple(
                    (-B + (i % 2), B - (1 if i % 2 == 0 else 0))
                    for i in range(n)
                )
            else:
                bounds = tuple((-B, B) for _ in range(n))
            r = nearest_vector(A, t, list(bounds))
            w, d, vecs = enumerate_box_cvp(A, t, bounds)
            self.assertEqual(r.coefficients, [tuple(x) for x in w], (A, t, bounds))
            self.assertEqual(r.squared_distance, d)
            self.assertEqual(r.vectors, vecs)
            cases += 1
        self.assertEqual(cases, 120)

    def test_extreme_common_component(self):
        # 与 LLL 测试同型的巨大公共分量基（det=1），目标是分数点：
        # 盒内结果与独立穷举一致，且全程精确。
        N = 10 ** 20
        A = [[N, 1, 0], [N, 0, 1], [N + 1, 1, 0]]
        t = [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]
        bounds = [(-3, 3)] * 3
        r = nearest_vector(A, t, bounds)
        w, d, _ = enumerate_box_cvp(A, t, bounds)
        self.assertEqual(r.coefficients, [tuple(x) for x in w])
        self.assertEqual(r.squared_distance, d)
        # det=1 的格覆盖 Z^3，半整数目标最近距离平方恰为 3/4。
        self.assertEqual(r.squared_distance, Fraction(3, 4))

    def test_three_dimensional_moderate_full_box(self):
        # 31^3 = 29791 个盒点的独立全盒穷举，覆盖非叶子层 Fraction FM
        # 投影在稍大盒上的行为。目标由盒内整数系数 (5,-3,2) 真实构造，
        # 故精确距离应为 0 且唯一最优（满秩格只有该系数表示该点）。
        A = [[7, 3, -2], [-2, 5, 4], [3, -1, 6]]
        c0 = (5, -3, 2)
        t = [Fraction(sum(c0[i] * A[i][k] for i in range(3))) for k in range(3)]
        bounds = [(-15, 15)] * 3
        r = nearest_vector(A, t, bounds)
        w, d, vecs = enumerate_box_cvp(A, t, bounds)
        self.assertEqual(r.coefficients, [tuple(x) for x in w])
        self.assertEqual(r.squared_distance, d)
        self.assertEqual(r.vectors, vecs)
        self.assertEqual(r.box_points, 31 ** 3)
        self.assertEqual(r.coefficients, [c0])
        self.assertEqual(r.squared_distance, 0)


class TestInvalidInput(unittest.TestCase):
    I2 = [[1, 0], [0, 1]]

    def test_dimension_above_three_rejected(self):
        with self.assertRaises(InvalidBasisError):
            nearest_vector(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [0, 0, 0, 0],
                1,
            )

    def test_rank_deficient_rejected(self):
        with self.assertRaises(RankDeficientError):
            nearest_vector([[1, 1], [2, 2]], [0, 0], 1)
        with self.assertRaises(RankDeficientError):
            nearest_vector([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [0, 0, 0], 1)

    def test_bad_basis_still_rejected(self):
        with self.assertRaises(InvalidBasisError):
            nearest_vector([[1, 0.5], [0, 1]], [0, 0], 1)
        with self.assertRaises(InvalidBasisError):
            nearest_vector([], [], 1)
        with self.assertRaises(InvalidBasisError):
            nearest_vector([[1], [2]], [0], 1)  # 列数 1 < 行数 2

    def test_bad_target(self):
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0, 0], 1)       # 长度不符
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, 3, 1)                # 非序列
        for bad in (0.5, True, float("nan"), float("inf")):
            with self.assertRaises(InvalidParameterError):
                nearest_vector(self.I2, [bad, 0], 1)

    def test_bad_bounds(self):
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], -1)          # 负对称界
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], True)        # bool
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], 1.5)         # float
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], "5")         # 字符串
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], [(0, 1)])    # 对数不符
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], [(0, 1), (2, 1)])  # 空盒
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], [(0, 1), (0, 1.0)])
        with self.assertRaises(InvalidParameterError):
            nearest_vector(self.I2, [0, 0], [(0, 1), [0, 2, 3]])

    def test_budget_passthrough(self):
        A = [[19, 7, 3], [8, 3, 11], [-4, 9, 6]]
        with self.assertRaises(BudgetExhaustedError):
            nearest_vector(A, [0, 0, 0], 2, budget=3)
        with self.assertRaises(InvalidParameterError):
            nearest_vector(A, [0, 0, 0], 2, budget=0)
        with self.assertRaises(InvalidParameterError):
            nearest_vector(A, [0, 0, 0], 2, budget=False)

    def test_failure_leaves_no_state(self):
        # 本函数是无状态查询：非法调用前后，合法调用结果不受任何影响。
        before = nearest_vector(self.I2, [0, 0], 1)
        for bad_call in (
            lambda: nearest_vector([[1, 1], [2, 2]], [0, 0], 1),
            lambda: nearest_vector(self.I2, [0], 1),
            lambda: nearest_vector(self.I2, [0, 0], -1),
        ):
            with self.assertRaises(Exception):
                bad_call()
        after = nearest_vector(self.I2, [0, 0], 1)
        self.assertEqual(before.coefficients, after.coefficients)
        self.assertEqual(before.squared_distance, after.squared_distance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
