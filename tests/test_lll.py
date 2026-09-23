"""必要测试。

独立参考（二维短向量枚举、Gram 行列式）在本文件内自行实现，
不调用 ``lll_reduce.lll_reduce`` 来产生期望值；核心的往返一致性只作补充检查。
"""

import fractions
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lll_reduce import (  # noqa: E402
    DEFAULT_BUDGET,
    DELTA,
    InvalidBasisError,
    InvalidParameterError,
    LatticeDependentError,
    lll_reduce,
    round_half_even,
)
from lll_reduce.verify import gram_data, verify_reduction, verify_result  # noqa: E402

F = fractions.Fraction


# ---------------------------------------------------------------------------
# 独立参考实现（不依赖被测核心）
# ---------------------------------------------------------------------------

def independent_mat_mul(U, A):
    n, r, m = len(U), len(A), len(A[0])
    return [[sum(U[i][t] * A[t][j] for t in range(r)) for j in range(m)]
            for i in range(n)]


def independent_det(M):
    n = len(M)
    X = [[F(v) for v in row] for row in M]
    d = F(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if X[r][col] != 0), None)
        if piv is None:
            return F(0)
        if piv != col:
            X[col], X[piv] = X[piv], X[col]
            d = -d
        p = X[col][col]
        d *= p
        for r in range(col + 1, n):
            f = X[r][col] / p
            for c in range(col, n):
                X[r][c] -= f * X[col][c]
    return d


def enumerate_shortest_2d(A, bound=60):
    """暴力枚举二维整格的非零最短向量平方（独立参考）。

    对随机小基，bound=60 的系数范围远大于覆盖最短向量所需。
    """
    a1, a2 = A[0], A[1]
    best = None
    best_vecs = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            if x == 0 and y == 0:
                continue
            vx = x * a1[0] + y * a2[0]
            vy = x * a1[1] + y * a2[1]
            L = vx * vx + vy * vy
            if best is None or L < best:
                best = L
                best_vecs = [(vx, vy)]
            elif L == best:
                best_vecs.append((vx, vy))
    return best, best_vecs


def norm2(v):
    return sum(t * t for t in v)


# ---------------------------------------------------------------------------
# 舍入规则
# ---------------------------------------------------------------------------

class TestRoundHalfEven(unittest.TestCase):
    def test_exact_integers(self):
        for z in (-7, -1, 0, 1, 7, 1000003):
            self.assertEqual(round_half_even(F(z)), z)

    def test_non_ties(self):
        cases = {
            F(1, 3): 0, F(2, 3): 1, F(-1, 3): 0, F(-2, 3): -1,
            F(7, 3): 2, F(-7, 3): -2, F(8, 3): 3, F(-8, 3): -3,
        }
        for x, want in cases.items():
            self.assertEqual(round_half_even(x), want, (x, want))

    def test_negative_ties_go_to_even(self):
        # 这些取值能区分“向偶数”与“远离零/向上”舍入。
        cases = {
            F(1, 2): 0, F(-1, 2): 0,
            F(3, 2): 2, F(-3, 2): -2,
            F(5, 2): 2, F(-5, 2): -2,
            F(7, 2): 4, F(-7, 2): -4,
            F(9, 2): 4, F(-9, 2): -4,
        }
        for x, want in cases.items():
            self.assertEqual(round_half_even(x), want, (x, want))


# ---------------------------------------------------------------------------
# 手工小例：独立核对最终矩阵
# ---------------------------------------------------------------------------

class TestHandComputed(unittest.TestCase):
    def test_41_21(self):
        # 手算轨迹（GS: ||b*1||^2=17, mu=9/17, D2=4/17 -> 交换 ->
        # size reduction m=2 -> 再交换 -> m=-1）。LLL 不固定向量符号，
        # 故按“向量集合至多差符号”独立核对：应得到长度 1 与 2 的两条正交轴。
        A = [[4, 1], [2, 1]]
        res = lll_reduce(A)
        self.assertTrue(res.reduced)
        up_to_sign = sorted(tuple(abs(t) for t in v) for v in res.basis)
        self.assertEqual(up_to_sign, [(0, 1), (2, 0)])
        self.assertEqual(res.basis, independent_mat_mul(res.transform, A))
        rep = verify_result(res, A)
        self.assertTrue(rep.ok, rep.summary())

    def test_tie_coefficient_positive(self):
        # b1=(2,0), b2=(5,1): mu=5/2，并列 -> m=2（向偶数），不是 3。
        A = [[2, 0], [5, 1]]
        res = lll_reduce(A)
        self.assertTrue(res.reduced)
        first = res.events[0]
        self.assertEqual(first.kind, "size_reduce")
        self.assertEqual(first.coefficient, 2)
        rep = verify_result(res, A)
        self.assertTrue(rep.ok, rep.summary())

    def test_negative_tie_coefficient(self):
        # mu=-5/2 -> m=-2（向偶数），远离零舍入会错误地给 -3。
        A = [[2, 0], [-5, 1]]
        res = lll_reduce(A)
        self.assertTrue(res.reduced)
        first = res.events[0]
        self.assertEqual(first.kind, "size_reduce")
        self.assertEqual(first.coefficient, -2)
        # 负舍入并列且 |mu|=1/2 时不得做 size reduction：
        # (2,0),(-1,1) 的 mu=-1/2，第一条日志必须是交换而非约化。
        res2 = lll_reduce([[2, 0], [-1, 1]])
        self.assertTrue(res2.reduced)
        self.assertEqual(res2.events[0].kind, "swap")
        rep = verify_result(res2, [[2, 0], [-1, 1]])
        self.assertTrue(rep.ok, rep.summary())


# ---------------------------------------------------------------------------
# 二维枚举对照
# ---------------------------------------------------------------------------

class TestEnumeration2D(unittest.TestCase):
    def test_fixed_cases_match_enumeration(self):
        bases = [
            [[4, 1], [2, 1]],
            [[15, 9], [7, 5]],
            [[12, 7], [31, 18]],
            [[-9, 13], [4, -6]],
        ]
        for A in bases:
            with self.subTest(A=A):
                res = lll_reduce(A)
                self.assertTrue(res.reduced)
                shortest, vecs = enumerate_shortest_2d(A)
                # 本题不要求 LLL 总给最短；但这些二维实例上逐向量对照应当一致。
                self.assertEqual(norm2(res.basis[0]), shortest,
                                 (res.basis, vecs))
                # 枚举得到的最短向量必须能由约化基整数表出（同格的独立佐证，
                # 坐标是对基向量 b_i 而非 GS 正交向量取的）。
                (x0, y0), (x1, y1) = res.basis
                detB = x0 * y1 - y0 * x1
                self.assertNotEqual(detB, 0)
                for vx, vy in vecs[:4]:
                    c0 = F(vx * y1 - vy * x1, detB)
                    c1 = F(x0 * vy - y0 * vx, detB)
                    self.assertEqual(c0.denominator, 1, (vx, vy, c0))
                    self.assertEqual(c1.denominator, 1, (vx, vy, c1))

    def test_random_2d_enumeration(self):
        rng = random.Random(20260922)
        matches = 0
        for _ in range(30):
            A = [[rng.randint(-12, 12), rng.randint(-12, 12)],
                 [rng.randint(-12, 12), rng.randint(-12, 12)]]
            if A[0][0] * A[1][1] - A[0][1] * A[1][0] == 0:
                continue
            res = lll_reduce(A)
            self.assertTrue(res.reduced)
            rep = verify_result(res, A)
            self.assertTrue(rep.ok, rep.summary())
            shortest, _ = enumerate_shortest_2d(A)
            if norm2(res.basis[0]) == shortest:
                matches += 1
            # 不要求总相等，但必须满足 LLL 的理论界 ||b1||^2 <= 2^(n-1) * lambda1^2。
            self.assertLessEqual(norm2(res.basis[0]), 2 * shortest)
        self.assertGreaterEqual(matches, 1)


# ---------------------------------------------------------------------------
# 大公共分量
# ---------------------------------------------------------------------------

class TestLargeCommonComponent(unittest.TestCase):
    def test_huge_shared_prefix(self):
        N = 10**40
        A = [[N, 1, 0], [N, 0, 1], [N, 1, 1]]
        # 行列式：N*(0*1-1*1) - 1*(N*1-1*N) = -N，满秩。
        res = lll_reduce(A)
        self.assertTrue(res.reduced)
        rep = verify_result(res, A)
        self.assertTrue(rep.ok, rep.summary())
        # 公共大分量必须被消掉：首向量不应再含 10^40 量级。
        self.assertLessEqual(norm2(res.basis[0]), 3)
        self.assertEqual(res.basis, independent_mat_mul(res.transform, A))
        self.assertEqual(abs(independent_det(res.transform)), 1)

    def test_large_2d(self):
        N = 10**60
        A = [[N, 1], [N, 0]]
        res = lll_reduce(A)
        self.assertTrue(res.reduced)
        self.assertEqual(norm2(res.basis[0]), 1)
        rep = verify_result(res, A)
        self.assertTrue(rep.ok, rep.summary())


# ---------------------------------------------------------------------------
# 秩亏与输入契约
# ---------------------------------------------------------------------------

class TestRankDeficient(unittest.TestCase):
    def test_explicit_multiple(self):
        A = [[1, 2, 3], [2, 4, 6], [1, 0, 1]]
        with self.assertRaises(LatticeDependentError) as cm:
            lll_reduce(A)
        self.assertEqual(cm.exception.index, 1)

    def test_zero_and_duplicate_rows(self):
        with self.assertRaises(LatticeDependentError):
            lll_reduce([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        with self.assertRaises(LatticeDependentError):
            lll_reduce([[3, 4, 5], [3, 4, 5], [1, 1, 1]])

    def test_dependence_distinct_from_budget(self):
        # “证明无解”（相关）必须与预算耗尽是两条不同的错误通道。
        with self.assertRaises(LatticeDependentError):
            lll_reduce([[1, 1], [2, 2]], budget=10**9)


class TestInputContract(unittest.TestCase):
    def test_non_finite_floats_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidBasisError):
                lll_reduce([[1, 0], [bad, 1]])

    def test_bool_rejected_everywhere(self):
        with self.assertRaises(InvalidBasisError):
            lll_reduce([[True, 0], [0, 1]])
        with self.assertRaises(InvalidParameterError):
            lll_reduce([[1, 0], [0, 1]], budget=True)  # type: ignore[arg-type]

    def test_float_and_str_rejected(self):
        with self.assertRaises(InvalidBasisError):
            lll_reduce([[1.0, 0], [0, 1]])
        with self.assertRaises(InvalidBasisError):
            lll_reduce([[1, "0"], [0, 1]])

    def test_shape_errors_are_localizable(self):
        with self.assertRaises(InvalidBasisError):
            lll_reduce([])
        with self.assertRaises(InvalidBasisError):
            lll_reduce([[1, 2], [3]])  # 长短不一
        with self.assertRaises(InvalidBasisError):
            lll_reduce([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                        [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 0, 1], [0, 1, 1]])  # 9 行 > 8

    def test_fewer_columns_than_rows(self):
        with self.assertRaises(InvalidBasisError):
            lll_reduce([[1, 2], [3, 4], [5, 6]])

    def test_budget_values(self):
        with self.assertRaises(InvalidParameterError):
            lll_reduce([[1, 0], [0, 1]], budget=0)
        with self.assertRaises(InvalidParameterError):
            lll_reduce([[1, 0], [0, 1]], budget=-3)
        with self.assertRaises(InvalidParameterError):
            lll_reduce([[1, 0], [0, 1]], budget=1.5)  # type: ignore[arg-type]

    def test_tuple_input_accepted(self):
        res = lll_reduce(((3, 4), (1, 1)))
        self.assertTrue(res.reduced)
        self.assertIsInstance(res.basis, list)


# ---------------------------------------------------------------------------
# 预算：耗尽时状态自洽、不谎称约化
# ---------------------------------------------------------------------------

class TestBudget(unittest.TestCase):
    def test_exhaustion_is_honest(self):
        A = [[4, 1], [2, 1]]
        res = lll_reduce(A, budget=1)
        self.assertFalse(res.reduced)
        self.assertTrue(res.exhausted)
        self.assertLessEqual(res.operations, res.budget + 1)
        # 无未说明的部分变更：B = U*A 仍成立，U 仍幺模。
        self.assertEqual(res.basis, independent_mat_mul(res.transform, A))
        self.assertIn(abs(independent_det(res.transform)), (1,))
        # 验证器不得把中间态当成约化基。
        rep = verify_result(res, A)
        self.assertFalse(rep.ok)
        # 但纯结构关系 B=U*A、U 幺模仍然通过（不要求 reduced 标记）。
        rep_struct = verify_reduction(A, res.basis, res.transform)
        self.assertTrue(
            any(c.name.startswith("B == U") and c.passed
                for c in rep_struct.checks)
        )
        self.assertTrue(
            any(c.name.startswith("U unimodular") and c.passed
                for c in rep_struct.checks)
        )

    def test_budget_one_terminates_cleanly_all_dims(self):
        # 预算=1 在任何输入上都必须给出自洽结果或精确的相关性错误，绝不挂起。
        rng = random.Random(99)
        for n in range(1, 9):
            A = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]
            try:
                res = lll_reduce(A, budget=1)
            except LatticeDependentError:
                continue
            self.assertEqual(res.basis,
                             independent_mat_mul(res.transform, A))
            if n == 1:
                self.assertTrue(res.reduced)  # 一维基无需任何操作
            else:
                self.assertFalse(res.reduced, n)

    def test_default_budget_enough_for_dim8(self):
        rng = random.Random(7)
        A = [[rng.randint(-50, 50) for _ in range(8)] for _ in range(8)]
        if independent_det(A) == 0:
            A[0][0] += 1
        res = lll_reduce(A)
        self.assertTrue(res.reduced, res.operations)
        self.assertFalse(res.exhausted)
        self.assertLess(res.operations, DEFAULT_BUDGET)

    def test_statelessness_across_calls(self):
        A = [[4, 1], [2, 1]]
        r1 = lll_reduce(A, budget=1)
        r2 = lll_reduce(A, budget=1)
        self.assertEqual(r1.basis, r2.basis)
        self.assertEqual(A, [[4, 1], [2, 1]])  # 输入未被改动


# ---------------------------------------------------------------------------
# 证书与日志
# ---------------------------------------------------------------------------

class TestCertificate(unittest.TestCase):
    def test_log_replay_and_unimodularity(self):
        rng = random.Random(31337)
        for _ in range(20):
            n = rng.randint(1, 6)
            m = rng.randint(n, n + 2)
            A = [[rng.randint(-15, 15) for _ in range(m)] for _ in range(n)]
            try:
                res = lll_reduce(A)
            except LatticeDependentError:
                continue
            self.assertTrue(res.reduced)
            # 日志步号严格递增且系数为整数。
            steps = [e.step for e in res.events]
            self.assertEqual(steps, sorted(set(steps)))
            for e in res.events:
                if e.kind == "size_reduce":
                    self.assertIsInstance(e.coefficient, int)
                    self.assertNotEqual(e.coefficient, 0)
                else:
                    self.assertEqual(e.kind, "swap")
                    self.assertEqual(e.j, e.i + 1)
            rep = verify_result(res, A)
            self.assertTrue(rep.ok, rep.summary())

    def test_gram_determinant_preserved(self):
        # 同维方阵：|det B| = |det A|（独立行列式参考）。
        A = [[6, 1, 2], [0, 5, 3], [4, -2, 7]]
        res = lll_reduce(A)
        self.assertTrue(res.reduced)
        self.assertEqual(abs(independent_det(res.basis)),
                         abs(independent_det(A)))

    def test_dim1_trivial(self):
        res = lll_reduce([[6, -8]])
        self.assertTrue(res.reduced)
        self.assertEqual(res.basis, [[6, -8]])
        self.assertEqual(res.transform, [[1]])
        self.assertEqual(res.events, ())


# ---------------------------------------------------------------------------
# 确定性随机全维模糊：独立验证器逐项核对
# ---------------------------------------------------------------------------

class TestFuzz(unittest.TestCase):
    def test_random_bases_up_to_dim8(self):
        rng = random.Random(20260923)
        checked = 0
        for _ in range(60):
            n = rng.randint(1, 8)
            m = rng.randint(n, n + 2)
            span = rng.choice([4, 9, 30])
            A = [[rng.randint(-span, span) for _ in range(m)] for _ in range(n)]
            try:
                res = lll_reduce(A)
            except LatticeDependentError:
                # 相关基：独立确认它确实相关，然后跳过。
                with self.assertRaises(ValueError):
                    gram_data(A)
                continue
            self.assertTrue(res.reduced, (n, m, res.operations))
            self.assertEqual(res.basis, independent_mat_mul(res.transform, A))
            self.assertEqual(abs(independent_det(res.transform)), 1)
            rep = verify_result(res, A)
            self.assertTrue(rep.ok, rep.summary())
            # mu 全部满足 |.|<=1/2，Lovasz 全通过（独立数值复核一次）。
            for i in range(n):
                for j in range(i):
                    self.assertLessEqual(abs(rep.mu[i][j]), F(1, 2))
                if i >= 1:
                    self.assertGreaterEqual(
                        rep.norms[i],
                        (DELTA - rep.mu[i][i - 1] ** 2) * rep.norms[i - 1])
            checked += 1
        self.assertGreaterEqual(checked, 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
