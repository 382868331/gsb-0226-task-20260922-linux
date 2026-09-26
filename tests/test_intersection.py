"""intersection_basis（整数列格交格）的必要测试。

期望值来源彼此独立：
  * 手算定值（相等格、互素对角格、Z^n 情形）；
  * 小坐标盒**枚举共同点**：独立用 Fraction 求 Bi^{-1}，列出盒内属于
    两边格的全部点，再与交格 C 在同盒内的点逐一对照（生成关系检验）；
  * 独立验证器 verify_intersection（Bareiss 行列式 + 核基本原性子式 gcd）。
全程精确整数/Fraction，无浮点、无容差。
"""

import unittest
from fractions import Fraction
from itertools import product

from lll_reduce import (
    InvalidBasisError,
    RankDeficientError,
    bareiss_det,
    integer_kernel_full_row_rank,
    intersection_basis,
    verify_intersection,
    verify_reduction,
)


def mat_mul(A, B):
    n, r, m = len(A), len(B), len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(r)) for j in range(m)]
        for i in range(n)
    ]


def inverse_fraction(B):
    """独立有理求逆（伴随矩阵 / 行列式），仅供测试枚举用。"""
    n = len(B)
    d = bareiss_det(B)
    assert d != 0
    if n == 1:
        return [[Fraction(1, d)]]
    cof = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [
                [B[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]
            sign = -1 if (i + j) % 2 else 1
            cof[j][i] = sign * bareiss_det(minor)  # 转置伴随
    return [[Fraction(cof[i][j], d) for j in range(n)] for i in range(n)]


def box_lattice_points(B, R):
    """盒 [-R,R]^n 中属于列格 L(B) 的点（用 B^{-1} 坐标全为整数判定）。"""
    n = len(B)
    Bi = inverse_fraction(B)
    out = set()
    rng = range(-R, R + 1)
    for p in product(rng, repeat=n):
        coords = [
            sum(Bi[i][t] * p[t] for t in range(n)) for i in range(n)
        ]
        if all(q.denominator == 1 for q in coords):
            out.add(p)
    return out


def assert_exact_intersection(testcase, B1, B2, R=12):
    res = intersection_basis(B1, B2)
    n = len(B1)
    # 证书：B1 U = B2 V = C。
    testcase.assertEqual(mat_mul(B1, res.U), res.basis)
    testcase.assertEqual(mat_mul(B2, res.V), res.basis)
    # 独立验证器（含核基本原性 gcd=1：无漏点）。
    report = verify_intersection(B1, B2, res)
    testcase.assertTrue(report.ok, report.failures)
    # 缩短后的 C 转置确实是 LLL 约化基。
    C = res.basis
    Ct = [list(row) for row in zip(*C)]
    eye = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    red = verify_reduction(Ct, Ct, eye)
    testcase.assertTrue(red.ok, red.failures)
    # 小坐标盒枚举：共同点集合 == 交格点集合。
    common = box_lattice_points(B1, R) & box_lattice_points(B2, R)
    testcase.assertEqual(common, box_lattice_points(C, R))
    # 指数行列式证据。
    testcase.assertEqual(
        abs(bareiss_det(C)) // abs(bareiss_det(B1)),
        res.index_in_lattice1,
    )
    testcase.assertEqual(
        abs(bareiss_det(C)) // abs(bareiss_det(B2)),
        res.index_in_lattice2,
    )
    return res


class TestIntersection(unittest.TestCase):
    def test_equal_lattices(self):
        # 两个等价（斜）基张成同一格：交就是本身，两边指数都为 1。
        B1 = [[3, 1], [2, 4]]
        # B2 = B1 * [[1,1],[0,-1]]（列幺正变换）。
        B2 = mat_mul(B1, [[1, 1], [0, -1]])
        res = assert_exact_intersection(self, B1, B2)
        self.assertEqual(res.index_in_lattice1, 1)
        self.assertEqual(res.index_in_lattice2, 1)
        self.assertEqual(abs(res.determinant), abs(bareiss_det(B1)))

    def test_coprime_diagonal_lattices(self):
        # 对角互素：L(diag(2,3)) ∩ L(diag(3,5)) = diag(lcm)=diag(6,15)。
        res = assert_exact_intersection(self, [[2, 0], [0, 3]], [[3, 0], [0, 5]])
        self.assertEqual(res.index_in_lattice1, 15)
        self.assertEqual(res.index_in_lattice2, 6)
        self.assertEqual(
            sorted(abs(x) for col in zip(*res.basis) for x in col if x),
            [6, 15],
        )

    def test_z2_and_sublattice(self):
        res = assert_exact_intersection(
            self, [[1, 0], [0, 1]], [[2, 0], [0, 3]], R=14
        )
        self.assertEqual(res.index_in_lattice2, 1)
        self.assertEqual(res.index_in_lattice1, 6)

    def test_skewed_bases(self):
        res = assert_exact_intersection(
            self, [[3, 1], [2, 4]], [[5, 1], [1, 2]], R=10
        )
        self.assertEqual(abs(res.determinant), 90)
        self.assertEqual(res.index_in_lattice1, 9)
        self.assertEqual(res.index_in_lattice2, 10)

    def test_negative_determinant_and_column_permutation(self):
        # 负行列式 / 列换序给出的是同一个格。
        B1 = [[0, 2], [3, 0]]  # det = -6
        B2 = [[2, 0], [0, 3]]  # B1 的列换序（同一格，det 符号相反）
        res = assert_exact_intersection(self, B1, B2, R=14)
        self.assertEqual(abs(res.determinant), 6)
        self.assertEqual((res.index_in_lattice1, res.index_in_lattice2), (1, 1))

    def test_dimensions_1_3_4(self):
        res1 = assert_exact_intersection(self, [[6]], [[10]], R=40)
        self.assertEqual(res1.basis, [[30]])
        self.assertEqual((res1.index_in_lattice1, res1.index_in_lattice2), (5, 3))
        B3a = [[2, 1, 0], [0, 3, 1], [1, 0, 4]]
        B3b = [[5, 0, 1], [1, 4, 0], [0, 1, 6]]
        assert_exact_intersection(self, B3a, B3b, R=6)
        B4a = [
            [2, 1, 0, 0],
            [0, 3, 1, 0],
            [1, 0, 4, 1],
            [0, 1, 0, 5],
        ]
        B4b = [
            [6, 1, 0, 2],
            [0, 5, 2, 0],
            [2, 0, 7, 1],
            [1, 1, 0, 9],
        ]
        assert_exact_intersection(self, B4a, B4b, R=4)

    def test_integer_kernel_exactness(self):
        # M = [2 -4]：整数核基 (2,1)（而非有理核 (1, 1/2)）。
        K = integer_kernel_full_row_rank([[2, -4]])
        self.assertEqual(mat_mul([[2, -4]], K), [[0]])
        import math

        self.assertEqual(math.gcd(abs(K[0][0]), abs(K[1][0])), 1)  # 本原核基
        # 3x5 满行秩矩阵：MK=0，且核的全部 2 阶（最大）子式 gcd=1（饱和子格）。
        M = [[1, 2, 3, 0, 0], [0, 1, 4, 5, 0], [2, 0, -1, 0, 7]]
        K = integer_kernel_full_row_rank(M)
        self.assertEqual(mat_mul(M, K), [[0] * 2 for _ in range(3)])
        import itertools

        g = 0
        for combo in itertools.combinations(range(5), 2):
            minor = [K[i] for i in combo]
            g = math.gcd(g, abs(bareiss_det(minor)))
        self.assertEqual(g, 1)

    def test_singular_inputs_rejected(self):
        with self.assertRaises(RankDeficientError) as cm:
            intersection_basis([[1, 2], [2, 4]], [[1, 0], [0, 1]])
        self.assertEqual(cm.exception.expected, 2)
        self.assertEqual(cm.exception.rank, 1)
        with self.assertRaises(RankDeficientError):
            intersection_basis([[1, 0], [0, 1]], [[0, 0], [0, 1]])

    def test_bad_shapes_types_dimension(self):
        with self.assertRaises(InvalidBasisError):
            intersection_basis([[1, 0]], [[1, 0], [0, 1]])
        with self.assertRaises(InvalidBasisError):
            intersection_basis([[1, 0.5], [0, 1]], [[1, 0], [0, 1]])
        with self.assertRaises(InvalidBasisError):
            intersection_basis([[True, 0], [0, 1]], [[1, 0], [0, 1]])
        big = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
        with self.assertRaises(InvalidBasisError):
            intersection_basis(big, big)
        with self.assertRaises(InvalidBasisError):
            intersection_basis([[1, 0], [0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_verifier_detects_missing_points(self):
        # 取真实交格的 2 倍子格：包含成立但漏掉共同点，gcd 子式证书必须报错。
        res = intersection_basis([[2, 0], [0, 2]], [[3, 0], [0, 3]])
        U2 = [[2 * x for x in row] for row in res.U]
        V2 = [[2 * x for x in row] for row in res.V]
        C2 = mat_mul([[2, 0], [0, 2]], U2)
        Fake = type("Fake", (), {
            "basis": C2,
            "U": U2,
            "V": V2,
            "kernel_basis": U2 + V2,
            "index_in_lattice1": abs(bareiss_det(C2)) // 4,
            "index_in_lattice2": abs(bareiss_det(C2)) // 9,
        })()
        report = verify_intersection([[2, 0], [0, 2]], [[3, 0], [0, 3]], Fake)
        self.assertFalse(report.ok)
        self.assertTrue(any("primitive" in f for f in report.failures))

    def test_verifier_detects_inclusion_failure(self):
        res = intersection_basis([[2, 0], [0, 3]], [[4, 0], [0, 5]])
        BadU = [[x + 1 for x in row] for row in res.U]
        Fake = type("Fake", (), {
            "basis": res.basis,
            "U": BadU,
            "V": res.V,
            "kernel_basis": BadU + res.V,
            "index_in_lattice1": res.index_in_lattice1,
            "index_in_lattice2": res.index_in_lattice2,
        })()
        report = verify_intersection([[2, 0], [0, 3]], [[4, 0], [0, 5]], Fake)
        self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
