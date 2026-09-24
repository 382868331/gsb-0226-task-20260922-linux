"""盒内最近格向量（closest_vector）的必要测试。

期望值来源彼此独立，均不调用被测实现生成：
  * 手算定值（一维/二维/三维小例，含并列字典序）；
  * 解析关系（目标恰为格点则距离为 0 且系数可手算；目标平移一个
    格向量则系数同步平移）；
  * tests/_reference.py 中独立编写的递归穷举 ref_closest_vector
    （结构与被测实现不同）做小规模枚举对照。
"""

import unittest
from fractions import Fraction

from lll_reduce import (
    CVP_MAX_DIM,
    InvalidBasisError,
    InvalidParameterError,
    RankDeficientError,
    closest_vector,
)

from _reference import ref_closest_vector


def lcg_cases(seed=20260924, count=12):
    """确定性生成 (A, target, box) 小例：n in 1..3，m in n..n+2。"""
    state = seed

    def rnd(mod):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)
        return (state >> 33) % mod

    def det2(x):
        return x[0][0] * x[1][1] - x[0][1] * x[1][0]

    def det3(x):
        return (
            x[0][0] * det2([[x[1][1], x[1][2]], [x[2][1], x[2][2]]])
            - x[0][1] * det2([[x[1][0], x[1][2]], [x[2][0], x[2][2]]])
            + x[0][2] * det2([[x[1][0], x[1][1]], [x[2][0], x[2][1]]])
        )

    out = []
    for idx in range(count):
        n = 1 + idx % 3
        m = n + (idx % 3)
        while True:
            A = [[rnd(9) - 4 for _ in range(m)] for _ in range(n)]
            if n == 1:
                ok = any(x != 0 for x in A[0])
            elif n == 2:
                ok = any(
                    det2([[A[0][j], A[0][k]], [A[1][j], A[1][k]]]) != 0
                    for j in range(m)
                    for k in range(j + 1, m)
                )
            else:
                # 前 3 列行列式非零是满行秩的充分条件（m>3 时仍安全）。
                ok = det3([row[:3] for row in A]) != 0
            if ok:
                break
        target = [Fraction(rnd(11) - 5, 2) for _ in range(m)]
        box = []
        for _ in range(n):
            lo = rnd(5) - 3
            hi = lo + rnd(4) + 1
            box.append((lo, hi))
        out.append((A, target, box))
    return out


class TestClosestVectorHandComputed(unittest.TestCase):
    def test_1d_basic(self):
        # A=[[4]], t=7/2：c=1 -> v=4, d=(1/2)^2=1/4；c=0 -> d=49/4。
        r = closest_vector([[4]], [Fraction(7, 2)], [(-3, 3)])
        self.assertEqual(r.coefficients, [1])
        self.assertEqual(r.vector, [4])
        self.assertEqual(r.distance2, Fraction(1, 4))

    def test_1d_negative_direction(self):
        # t=-9，A=[[4]]：c=-2 -> v=-8, d=1；c=-3 -> v=-12, d=9。
        r = closest_vector([[4]], [-9], [(-5, 5)])
        self.assertEqual(r.coefficients, [-2])
        self.assertEqual(r.vector, [-8])
        self.assertEqual(r.distance2, 1)

    def test_tie_broken_by_coefficient_lexicographic_order(self):
        # A=[[2]], t=1：c=0 与 c=1 的平方距离都是 1，取系数字典序最小 (0,)。
        r = closest_vector([[2]], [1], [(-2, 2)])
        self.assertEqual(r.distance2, 1)
        self.assertEqual(r.coefficients, [0])
        # t=-1：c=-1 与 c=0 并列，(-1,) < (0,)，取 c=-1。
        r = closest_vector([[2]], [-1], [(-2, 2)])
        self.assertEqual(r.distance2, 1)
        self.assertEqual(r.coefficients, [-1])

    def test_tie_2d_identity_half(self):
        # t=(1/2,1/2)，四个角点距离同为 1/2，取系数字典序最小 (0,0)。
        r = closest_vector(
            [[1, 0], [0, 1]],
            [Fraction(1, 2), Fraction(1, 2)],
            [(-1, 2), (-1, 2)],
        )
        self.assertEqual(r.distance2, Fraction(1, 2))
        self.assertEqual(r.coefficients, [0, 0])
        self.assertEqual(r.vector, [0, 0])

    def test_exact_lattice_point_zero_distance(self):
        # t = 3*A0 - 2*A1 = (4, -6)，手算系数 (3, -2)，距离恰为 0。
        A = [[2, 0], [1, 3]]
        r = closest_vector(A, [4, -6], [(-6, 6), (-6, 6)])
        self.assertEqual(r.coefficients, [3, -2])
        self.assertEqual(r.vector, [4, -6])
        self.assertEqual(r.distance2, 0)

    def test_3d_hand_computed(self):
        # A = [[2,0,0],[0,2,0],[0,0,2]]，t=(3,0,0)：
        # c0=1 -> |2-3|^2=1，c0=2 -> |4-3|^2=1，并列取字典序最小 (1,0,0)。
        A = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        r = closest_vector(A, [3, 0, 0], [(-2, 3)] * 3)
        self.assertEqual(r.coefficients, [1, 0, 0])
        self.assertEqual(r.vector, [2, 0, 0])
        self.assertEqual(r.distance2, 1)

    def test_rectangular_basis(self):
        # A=[[1,0,0],[0,1,0]]（嵌入 3 维的平面格），t=(1/2,1/2,7)：
        # 第三坐标恒为 0，最近点 (0,0,0)，d = 1/4+1/4+49 = 99/2。
        r = closest_vector(
            [[1, 0, 0], [0, 1, 0]],
            [Fraction(1, 2), Fraction(1, 2), 7],
            [(-2, 2), (-2, 2)],
        )
        self.assertEqual(r.coefficients, [0, 0])
        self.assertEqual(r.vector, [0, 0, 0])
        self.assertEqual(r.distance2, Fraction(99, 2))


class TestClosestVectorAnalyticRelations(unittest.TestCase):
    def test_shift_by_lattice_vector_shifts_coefficients(self):
        # 解析关系：t' = t + k·A 时，盒同步平移 k 后最近系数平移 k、距离不变。
        A = [[3, 1], [1, 4]]
        t = [Fraction(5, 2), Fraction(-3, 2)]
        box = [(-4, 4), (-4, 4)]
        k = (2, -3)
        r1 = closest_vector(A, t, box)
        t2 = [t[j] + k[0] * A[0][j] + k[1] * A[1][j] for j in range(2)]
        box2 = [(lo + k[i], hi + k[i]) for i, (lo, hi) in enumerate(box)]
        r2 = closest_vector(A, t2, box2)
        self.assertEqual(
            r2.coefficients, [r1.coefficients[i] + k[i] for i in range(2)]
        )
        self.assertEqual(r2.distance2, r1.distance2)

    def test_box_edge_optimum(self):
        # 盒约束真实起作用：全格最近系数是 0，但盒从 3 开始，只能取 c=3。
        r = closest_vector([[10]], [0], [(3, 5)])
        self.assertEqual(r.coefficients, [3])
        self.assertEqual(r.vector, [30])
        self.assertEqual(r.distance2, 900)

    def test_singleton_box(self):
        # 盒退化为单点：结果就是该点，无搜索余地。
        r = closest_vector([[2, 0], [0, 3]], [100, -100], [(4, 4), (-2, -2)])
        self.assertEqual(r.coefficients, [4, -2])
        self.assertEqual(r.vector, [8, -6])
        self.assertEqual(r.distance2, (8 - 100) ** 2 + (-6 + 100) ** 2)
        self.assertEqual(r.nodes_visited, 1)


class TestClosestVectorCrossCheck(unittest.TestCase):
    def test_matches_independent_recursive_enumeration(self):
        for A, target, box in lcg_cases():
            with self.subTest(A=A, target=target, box=box):
                r = closest_vector(A, target, box)
                c_ref, v_ref, d_ref = ref_closest_vector(A, target, box)
                self.assertEqual(r.coefficients, c_ref)
                self.assertEqual(r.vector, v_ref)
                self.assertEqual(r.distance2, d_ref)
                # 返回向量必须等于 系数·原基（不经过被测实现另行核验）。
                n, m = len(A), len(A[0])
                expect_v = [
                    sum(r.coefficients[i] * A[i][j] for i in range(n))
                    for j in range(m)
                ]
                self.assertEqual(r.vector, expect_v)
                # 平方距离直接按定义复算。
                expect_d = sum(
                    (Fraction(expect_v[j]) - target[j]) ** 2 for j in range(m)
                )
                self.assertEqual(r.distance2, expect_d)
                # 穷举节点数等于盒体积（确定可复算）。
                vol = 1
                for lo, hi in box:
                    vol *= hi - lo + 1
                self.assertEqual(r.nodes_visited, vol)
                # 系数确实落在盒内。
                for i, (lo, hi) in enumerate(box):
                    self.assertTrue(lo <= r.coefficients[i] <= hi)

    def test_result_is_readonly_snapshot(self):
        r = closest_vector([[2]], [3], [(-2, 2)])
        with self.assertRaises(AttributeError):
            r.distance2 = 0  # frozen dataclass 不允许改写


class TestClosestVectorFailures(unittest.TestCase):
    def test_dimension_limit_3(self):
        A = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
        self.assertEqual(CVP_MAX_DIM, 3)
        with self.assertRaises(InvalidBasisError):
            closest_vector(A, [0, 0, 0, 0], [(-1, 1)] * 4)

    def test_rank_deficient_rejected(self):
        with self.assertRaises(RankDeficientError):
            closest_vector([[1, 2], [2, 4]], [0, 0], [(-1, 1)] * 2)
        with self.assertRaises(RankDeficientError):
            closest_vector([[0, 0]], [1, 1], [(-1, 1)])

    def test_bad_basis_shape_and_elements(self):
        with self.assertRaises(InvalidBasisError):
            closest_vector([[1.5, 0], [0, 1]], [0, 0], [(-1, 1)] * 2)
        with self.assertRaises(InvalidBasisError):
            closest_vector([[True, 0], [0, 1]], [0, 0], [(-1, 1)] * 2)
        with self.assertRaises(InvalidBasisError):
            closest_vector([[1, 2], [3]], [0, 0], [(-1, 1)] * 2)
        with self.assertRaises(InvalidBasisError):
            closest_vector([], [], [])
        with self.assertRaises(InvalidBasisError):
            # 列数 < 行数（不可能满行秩）
            closest_vector([[1], [2]], [0], [(-1, 1)] * 2)

    def test_bad_target(self):
        A = [[1, 0], [0, 1]]
        box = [(-1, 1)] * 2
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, [0], box)  # 长度不符
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, [0, 0, 0], box)
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, [0.5, 0], box)  # float 拒绝（含静默转换风险）
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, [float("nan"), 0], box)
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, [float("inf"), 0], box)
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, [True, 0], box)  # bool 拒绝
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, "xy", box)

    def test_bad_box(self):
        A = [[1, 0], [0, 1]]
        t = [0, 0]
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, t, [(-1, 1)])  # 对数不足
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, t, [(-1, 1)] * 3)  # 对数过多
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, t, [(2, 1), (-1, 1)])  # lo > hi
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, t, [(-1.0, 1), (-1, 1)])  # 非整数界
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, t, [(True, 1), (-1, 1)])  # bool 界拒绝
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, t, [(-1, 1, 0), (-1, 1)])  # 非二元对
        with self.assertRaises(InvalidParameterError):
            closest_vector(A, t, None)

    def test_error_is_locatable(self):
        try:
            closest_vector([[1, 0], [0, 1]], [0, "x"], [(-1, 1)] * 2)
        except InvalidParameterError as e:
            self.assertIn("target[1]", e.parameter)
        else:
            self.fail("expected InvalidParameterError")
        try:
            closest_vector([[1, 0], [0, 1]], [0, 0], [(-1, 1), (5, 2)])
        except InvalidParameterError as e:
            self.assertIn("box[1]", e.parameter)
        else:
            self.fail("expected InvalidParameterError")

    def test_tuple_inputs_accepted(self):
        r = closest_vector(((2, 0), (0, 2)), (Fraction(3, 2), 1), ((-2, 2),) * 2)
        # t=(3/2,1)：c=(1,0) -> v=(2,0), d=(1/2)^2+1^2=5/4；
        # c=(1,1) -> v=(2,2), d=1/4+1=5/4 并列，取系数字典序最小 (1,0)。
        self.assertEqual(r.coefficients, [1, 0])
        self.assertEqual(r.distance2, Fraction(5, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
