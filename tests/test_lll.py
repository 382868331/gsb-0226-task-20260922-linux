"""LLL 库的必要测试。

期望值来源彼此独立：
  * 手算定值（舍入表、固定小基、预算计数）；
  * tests/_reference.py 中独立编写的教科书 LLL 与二维短向量穷举；
  * 日志重放（不经过被测代码，直接按日志从 A 重建 B 与 U）；
  * 独立验证器 lll_reduce.verify_reduction（自带 GS 与 Bareiss 行列式）。
“再次调用结果一致”只作为补充断言，不作为期望值来源。
"""

import unittest
from fractions import Fraction

from lll_reduce import (
    BudgetExhaustedError,
    InvalidBasisError,
    InvalidParameterError,
    RankDeficientError,
    reduce_basis,
    round_half_even,
    verify_reduction,
)

from _reference import enumerate_shortest_2d, ref_lll


def mat_mul(U, A):
    n, r, m = len(U), len(A), len(A[0])
    return [
        [sum(U[i][t] * A[t][j] for t in range(r)) for j in range(m)]
        for i in range(n)
    ]


def det(M):
    n = len(M)
    X = [row[:] for row in M]
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


def replay_log(A, log):
    """完全按日志重建 B 与 U（独立于被测实现的变换代码）。"""
    n = len(A)
    b = [row[:] for row in A]
    u = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for e in log:
        if e.kind == "size_reduce":
            k, j, q = e.index_k, e.index_j, e.coefficient
            b[k] = [b[k][t] - q * b[j][t] for t in range(len(b[k]))]
            u[k] = [u[k][t] - q * u[j][t] for t in range(n)]
        elif e.kind == "swap":
            b[e.index_k], b[e.index_j] = b[e.index_j], b[e.index_k]
            u[e.index_k], u[e.index_j] = u[e.index_j], u[e.index_k]
        else:
            raise AssertionError(f"unknown log kind: {e.kind}")
    return b, u


def lcg_matrices(seed=20260923, count=18):
    """确定性生成一批小整数满秩倾向矩阵（2~5 维，含长方形）。"""
    state = seed

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)
        return (state >> 33) % 41 - 20

    out = []
    for idx in range(count):
        n = 2 + idx % 4
        m = n + (idx % 3)
        while True:
            A = [[rnd() for _ in range(m)] for _ in range(n)]
            # 避开秩亏：检查前 n 列行列式，为零则重新生成。
            if det([[A[i][j] for j in range(n)] for i in range(n)]) != 0:
                out.append(A)
                break
    return out


class TestRoundHalfEven(unittest.TestCase):
    def test_hand_computed_table(self):
        cases = {
            Fraction(0): 0,
            Fraction(1, 2): 0,      # 并列 -> 偶数 0
            Fraction(-1, 2): 0,    # 并列 -> 偶数 0
            Fraction(3, 2): 2,     # 并列 -> 偶数 2
            Fraction(-3, 2): -2,   # 并列 -> 偶数 -2
            Fraction(5, 2): 2,
            Fraction(-5, 2): -2,
            Fraction(7, 2): 4,
            Fraction(-7, 2): -4,
            Fraction(9, 2): 4,
            Fraction(-9, 2): -4,
            Fraction(9, 4): 2,     # 2.25
            Fraction(-9, 4): -2,
            Fraction(11, 4): 3,    # 2.75
            Fraction(-11, 4): -3,
            Fraction(100): 100,
            Fraction(-100): -100,
            Fraction(123456789123456789, 2): 61728394561728394,  # 大半数并列->偶
        }
        for x, want in cases.items():
            with self.subTest(x=str(x)):
                self.assertEqual(round_half_even(x), want)

    def test_rejects_non_fraction(self):
        with self.assertRaises(InvalidParameterError):
            round_half_even(2.5)
        with self.assertRaises(InvalidParameterError):
            round_half_even(True)  # bool 不冒充整数


class TestBasicReduction(unittest.TestCase):
    CLASSIC = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]

    def test_classic_example_certificate(self):
        r = reduce_basis(self.CLASSIC)
        self.assertTrue(r.reduced)
        # 教科书常见结果之一（精确值，手算可核对）。
        self.assertEqual(
            sorted(tuple(row) for row in r.basis),
            [(-1, 0, 2), (0, 1, 0), (1, 0, 1)],
        )
        report = verify_reduction(self.CLASSIC, r.basis, r.transform)
        self.assertTrue(report.ok, report.failures)
        self.assertEqual(r.basis, mat_mul(r.transform, self.CLASSIC))
        self.assertEqual(det(r.transform), 1)

    def test_identity_and_already_reduced(self):
        for A in (
            [[1, 0], [0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[2, 0], [1, 1]],  # mu=1/2，恰好不触发 size reduction
        ):
            r = reduce_basis(A)
            self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)
            self.assertEqual(r.basis, mat_mul(r.transform, A))

    def test_tuple_input_accepted(self):
        r = reduce_basis(((1, 2), (3, 5)))
        self.assertTrue(r.reduced)
        self.assertTrue(verify_reduction(((1, 2), (3, 5)), r.basis, r.transform).ok)

    def test_log_replay_reconstructs_basis_and_transform(self):
        for A in lcg_matrices()[:10] + [self.CLASSIC]:
            with self.subTest(A=A):
                r = reduce_basis(A)
                b2, u2 = replay_log(A, r.log)
                self.assertEqual(b2, r.basis)
                self.assertEqual(u2, r.transform)
                # 步骤号严格递增，类型只有两种。
                self.assertEqual([e.step for e in r.log], list(range(1, len(r.log) + 1)))
                self.assertTrue(all(e.kind in ("swap", "size_reduce") for e in r.log))

    def test_swap_count_matches_log(self):
        r = reduce_basis([[1, 1], [0, 1]])
        self.assertEqual(r.swaps, sum(1 for e in r.log if e.kind == "swap"))
        self.assertGreaterEqual(r.swaps, 1)

    def test_dimension_up_to_8(self):
        # 确定性 8 维基：对角大值加下方小扰动。
        n = 8
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            A[i][i] = 9 - i
            for j in range(i):
                A[i][j] = 1
        r = reduce_basis(A)
        self.assertTrue(r.reduced)
        self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)
        self.assertEqual(det(r.transform) in (1, -1), True)

    def test_rectangular_basis(self):
        A = [[10, 1, 19, -3], [7, 13, 5, 2], [-4, 8, 11, 6]]
        r = reduce_basis(A)
        self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)
        self.assertEqual(r.basis, mat_mul(r.transform, A))


class TestCrossCheckReference(unittest.TestCase):
    def test_matches_independent_textbook_lattice(self):
        # 与独立参考不要求逐向量相同（舍入并列策略不同），但必须张成同一格、
        # 各自通过约化条件，且协体积（Gram 行列式）相等。
        for A in lcg_matrices(count=24):
            with self.subTest(A=A):
                r = reduce_basis(A)
                rb = ref_lll(A)
                self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)
                n = len(A)

                def gram_det(M):
                    G = [
                        [sum(M[i][t] * M[j][t] for t in range(len(M[0]))) for j in range(n)]
                        for i in range(n)
                    ]
                    return det(G)

                self.assertEqual(gram_det(r.basis), gram_det(A))
                self.assertEqual(gram_det(rb), gram_det(A))
                # 参考基向量必须能用 U 的整数组合表示（同格的成员抽查）。
                for v in rb:
                    # v 在格中当且仅当它是 A 行的整数组合；用 B(=UA) 反解整数系数。
                    self.assertTrue(
                        _is_integer_combo(v, A),
                        f"reference vector {v} not in lattice",
                    )

    def test_2d_short_vector_enumeration(self):
        # 独立穷举二维格最短向量，对照 LLL 首向量：允许不最短，但必须在
        # LLL(delta=3/4) 的二维近似界 ||b0||^2 <= 2*lambda1^2 之内。
        lattices = [
            [[19, 7], [8, 3]],
            [[15, 4], [11, 3]],
            [[31, 12], [-7, 5]],
            [[5, 0], [3, 1]],
            [[100, 99], [99, 98]],
        ]
        equal_count = 0
        for A in lattices:
            r = reduce_basis(A)
            lam2, short_vecs = enumerate_shortest_2d(A)
            got = sum(x * x for x in r.basis[0])
            self.assertGreaterEqual(got, lam2)
            self.assertLessEqual(got, 2 * lam2 + 0)  # 精确整数界
            if got == lam2:
                equal_count += 1
                self.assertIn(tuple(r.basis[0]), short_vecs)
        # 这些小格大部分应直接命中最短（经验事实，非算法保证）。
        self.assertGreaterEqual(equal_count, 3)


def _is_integer_combo(v, A):
    """v 是否为 A 行的整数组合（A 满行秩 n×m，v 长度 m）。

    v = c·A，c 为长度 n 的整数行向量。任选 n 个线性无关列 cols，
    解 (A 取这些列后的转置) c = v[cols]，再用其余列核验。
    """
    from itertools import combinations

    n, m = len(A), len(A[0])

    def square(cols):
        return [[Fraction(A[i][j]) for i in range(n)] for j in cols]

    chosen = None
    if det([[A[i][j] for i in range(n)] for j in range(n)]) != 0:
        chosen = tuple(range(n))
    else:
        for cols in combinations(range(m), n):
            if det([[A[i][j] for i in range(n)] for j in cols]) != 0:
                chosen = cols
                break
    if chosen is None:
        return False  # A 不满行秩

    # 方程组：行=所选列 j，列=系数 i；X c = rhs。
    X = square(chosen)
    X = [X[r][:] + [Fraction(v[chosen[r]])] for r in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if X[r][col] != 0), None)
        if piv is None:
            return False
        X[col], X[piv] = X[piv], X[col]
        p = X[col][col]
        X[col] = [x / p for x in X[col]]
        for r in range(n):
            if r != col and X[r][col] != 0:
                f = X[r][col]
                X[r] = [X[r][t] - f * X[col][t] for t in range(n + 1)]
    coeffs = [X[i][n] for i in range(n)]
    if any(c.denominator != 1 for c in coeffs):
        return False
    coeffs = [c.numerator for c in coeffs]
    return all(
        sum(coeffs[i] * A[i][j] for i in range(n)) == v[j] for j in range(m)
    )


class TestLargeCommonComponent(unittest.TestCase):
    def test_huge_common_component(self):
        # 三行共享 ~N 的公共分量；取 det=1 的基（前两行叉积 (1,-N,N) 与
        # 第三行内积为 1），约化后数学上必然全为单位长度——大公共分量被
        # 精确消成小向量。注意：若基的协体积本身为 N 量级，则约化基中
        # 保留一个长向量是数学必然，LLL 不承诺把它变短（见 README 已知限制）。
        N = 10 ** 40
        A = [[N, 1, 0], [N, 0, 1], [N + 1, 1, 0]]
        r = reduce_basis(A)
        self.assertTrue(r.reduced)
        report = verify_reduction(A, r.basis, r.transform)
        self.assertTrue(report.ok, report.failures)
        for row in r.basis:
            for x in row:
                self.assertLessEqual(abs(x), 1)
        self.assertEqual(
            sorted(tuple(row) for row in r.basis),
            [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
        )
        self.assertEqual(r.basis, mat_mul(r.transform, A))

    def test_huge_values_exact(self):
        # 直接用超大整数，检验全程精确、无浮点。
        A = [[10 ** 80 + 1, 10 ** 80], [10 ** 80, 10 ** 80 - 1]]
        r = reduce_basis(A)
        self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)


class TestNegativeTieRounding(unittest.TestCase):
    def test_mu_minus_three_halves_rounds_to_minus_two(self):
        # b1=(1,1), |b1|^2=2；b2=(-2,-1) 使 mu = -3/2（恰好并列）。
        A = [[1, 1], [-2, -1]]
        r = reduce_basis(A)
        first = r.log[0]
        self.assertEqual(first.kind, "size_reduce")
        self.assertEqual(first.coefficient, -2)  # -3/2 向偶 -> -2（不是 -1）
        self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)

    def test_mu_plus_three_halves_rounds_to_plus_two(self):
        A = [[1, 1], [2, 1]]
        r = reduce_basis(A)
        self.assertEqual(r.log[0].kind, "size_reduce")
        self.assertEqual(r.log[0].coefficient, 2)
        self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)

    def test_mu_minus_half_is_not_reduced(self):
        # mu = -1/2 时 |mu| <= 1/2，不应产生 size_reduce 日志（k=0,j=0）。
        A = [[1, 1, 0], [-1, 0, 1]]
        r = reduce_basis(A)
        self.assertFalse(
            any(
                e.kind == "size_reduce" and e.index_k == 1 and e.index_j == 0
                for e in r.log
            )
        )
        self.assertTrue(verify_reduction(A, r.basis, r.transform).ok)


class TestRankDeficiency(unittest.TestCase):
    def test_dependent_rows_rejected(self):
        with self.assertRaises(RankDeficientError) as cm:
            reduce_basis([[1, 2, 3], [2, 4, 6]])
        self.assertEqual(cm.exception.expected, 2)
        self.assertEqual(cm.exception.rank, 1)

    def test_zero_row_rejected(self):
        with self.assertRaises(RankDeficientError) as cm:
            reduce_basis([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.assertEqual(cm.exception.rank, 0)

    def test_dependent_three_rows(self):
        with self.assertRaises(RankDeficientError) as cm:
            reduce_basis([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.assertEqual(cm.exception.rank, 2)

    def test_rank_deficiency_distinct_from_budget(self):
        # 即使预算极小，秩亏也必须先作为数学错误抛出，绝不报预算耗尽。
        with self.assertRaises(RankDeficientError):
            reduce_basis([[1, 1], [2, 2]], budget=1)

    def test_columns_smaller_than_rows_is_shape_error(self):
        with self.assertRaises(InvalidBasisError):
            reduce_basis([[1, 0], [0, 1], [1, 1]])


class TestBudget(unittest.TestCase):
    def test_budget_boundary_deterministic(self):
        # [[1,1],[0,1]]：初始 GS 计 2；交换计 2；交换后新 b1 的 mu=1 再做
        # 一次 size reduction 计 1 —— 合计恰为 5，逐步计费可复算。
        A = [[1, 1], [0, 1]]
        r = reduce_basis(A, budget=5)
        self.assertTrue(r.reduced)
        self.assertEqual(r.ops_used, 5)
        with self.assertRaises(BudgetExhaustedError):
            reduce_basis(A, budget=4)
        with self.assertRaises(BudgetExhaustedError):
            reduce_basis(A, budget=1)

    def test_partial_snapshot_is_consistent_but_not_certified(self):
        A = [
            [19, 7, 3],
            [8, 3, 11],
            [-4, 9, 6],
        ]
        try:
            reduce_basis(A, budget=5)
        except BudgetExhaustedError as e:
            p = e.partial
            self.assertFalse(p.reduced)
            # 中间态仍满足 B = U A（有状态失败不留“未说明的部分变更”）。
            self.assertEqual(p.basis, mat_mul(p.transform, A))
        else:
            self.fail("BudgetExhaustedError not raised")

    def test_invalid_budget(self):
        A = [[1, 0], [0, 1]]
        for bad in (0, -1, -10 ** 6):
            with self.assertRaises(InvalidParameterError):
                reduce_basis(A, budget=bad)
        with self.assertRaises(InvalidParameterError):
            reduce_basis(A, budget=True)   # bool 不冒充整数
        with self.assertRaises(InvalidParameterError):
            reduce_basis(A, budget=1.5)
        with self.assertRaises(InvalidParameterError):
            reduce_basis(A, budget="100")

    def test_completion_is_not_claimed_after_exhaustion(self):
        # 大预算一定完成；极小预算一定抛错——成功与资源耗尽不可混淆。
        A = [[31, 12, -7], [-4, 19, 8], [6, -3, 22]]
        r = reduce_basis(A)
        self.assertTrue(r.reduced)
        with self.assertRaises(BudgetExhaustedError) as cm:
            reduce_basis(A, budget=5)
        self.assertFalse(cm.exception.partial.reduced)


class TestInvalidInput(unittest.TestCase):
    def test_nan_infinity_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(InvalidBasisError):
                reduce_basis([[1, bad], [0, 1]])

    def test_float_element_rejected(self):
        with self.assertRaises(InvalidBasisError):
            reduce_basis([[1.0, 0], [0, 1]])

    def test_bool_element_rejected(self):
        with self.assertRaises(InvalidBasisError):
            reduce_basis([[True, 0], [0, 1]])

    def test_bad_shapes_and_types(self):
        with self.assertRaises(InvalidBasisError):
            reduce_basis([])
        with self.assertRaises(InvalidBasisError):
            reduce_basis([[]])
        with self.assertRaises(InvalidBasisError):
            reduce_basis([[1, 2], [3]])           # 不等长
        with self.assertRaises(InvalidBasisError):
            reduce_basis([[1, "x"], [0, 1]])      # 非整数
        with self.assertRaises(InvalidBasisError):
            reduce_basis([[1, 2], [3, 4], [5, 6], [7, 8],
                          [9, 0], [1, 1], [2, 2], [3, 3], [4, 4]])  # 9 行超上限
        with self.assertRaises(InvalidBasisError):
            reduce_basis("not a matrix")

    def test_error_is_locatable(self):
        try:
            reduce_basis([[1, 2], [3, 0.5]])
        except InvalidBasisError as e:
            self.assertIn("A[1][1]", e.location)
        else:
            self.fail()

    def test_n_minus_one_accepted(self):
        # 1 维平凡基可用。
        r = reduce_basis([[3, 4]])
        self.assertEqual(r.basis, [[3, 4]])
        self.assertTrue(verify_reduction([[3, 4]], r.basis, r.transform).ok)


class TestVerifierIndependent(unittest.TestCase):
    def test_detects_b_equals_ua_violation(self):
        A = [[1, 0], [0, 1]]
        rep = verify_reduction(A, [[2, 0], [0, 1]], [[1, 0], [0, 1]])
        self.assertFalse(rep.ok)
        self.assertTrue(any("B != U*A" in f for f in rep.failures))

    def test_detects_non_unimodular(self):
        A = [[2, 0], [0, 2]]
        rep = verify_reduction(A, A, [[2, 0], [0, 1]])
        self.assertFalse(rep.ok)
        self.assertTrue(any("unimodular" in f for f in rep.failures))

    def test_detects_lovasz_violation(self):
        rep = verify_reduction([[3, 0], [0, 1]], [[3, 0], [0, 1]],
                               [[1, 0], [0, 1]])
        self.assertFalse(rep.ok)
        self.assertTrue(any("Lovasz" in f for f in rep.failures))

    def test_detects_size_reduction_violation(self):
        # mu = 3/2 > 1/2，且 Lovasz 也必然失败；至少 size reduction 要点名。
        rep = verify_reduction([[2, 0], [3, 1]], [[2, 0], [3, 1]],
                               [[1, 0], [0, 1]])
        self.assertFalse(rep.ok)
        self.assertTrue(any("size reduction" in f for f in rep.failures))

    def test_detects_rank_deficient_certificate(self):
        rep = verify_reduction([[1, 0], [2, 0]], [[1, 0], [2, 0]],
                               [[1, 0], [0, 1]])
        self.assertFalse(rep.ok)

    def test_bad_inputs_to_verifier(self):
        self.assertFalse(verify_reduction([], [[1]], [[1]]).ok)
        self.assertFalse(
            verify_reduction([[1, 0]], [[1.0, 0]], [[1]]).ok
        )


class TestIdempotenceSupplement(unittest.TestCase):
    """往返一致仅作补充：对约化基再约化，格不变且仍有证书。"""

    def test_reduce_reduced_basis(self):
        A = [[19, 7, 2], [8, 3, 11], [-4, 9, 6]]
        r1 = reduce_basis(A)
        r2 = reduce_basis(r1.basis)
        self.assertTrue(verify_reduction(r1.basis, r2.basis, r2.transform).ok)
        self.assertEqual(
            sorted(tuple(x) for x in r1.basis),
            sorted(tuple(x) for x in r2.basis),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
