"""固定输入演示：两个同维满秩整数列格的交格（精确整数 + LLL 缩短）。

运行：python demo_iteration.py（秒级，纯标准库、离线、无 sleep、无预录结果）。

演示内容：
  1. 取两个 3 维整数列格 B1、B2，用列 Hermite 消元精确求 [B1|-B2] 的
     整数核，得到共同点的全部整数坐标 (U, V)，再用现有 LLL 缩短交格基；
  2. 打印 C、U、V、两边指数与 det 证据，并用独立验证器核验“包含 +
     无漏点（核 n 阶子式 gcd=1）”证书；
  3. 用小坐标盒枚举共同点，逐一核对它们与交格点完全一致；
  4. 边界：奇异（秩亏）输入被 RankDeficientError 拒绝。
"""

from itertools import product

from lll_reduce import (
    RankDeficientError,
    intersection_basis,
    verify_intersection,
)


def fmt(M):
    w = max(len(str(x)) for row in M for x in row)
    return "\n".join(
        "  [" + " ".join(str(x).rjust(w) for x in row) + "]" for row in M
    )


def box_points(B, R):
    """盒 [-R,R]^3 中属于列格 L(B) 的点（独立有理坐标判定）。"""
    from fractions import Fraction
    from lll_reduce import bareiss_det

    n = len(B)
    d = bareiss_det(B)
    # 用伴随/行列式直接算 B^{-1}（n=3，测试/演示规模）。
    cof = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [
                [B[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]
            s = -1 if (i + j) % 2 else 1
            cof[j][i] = s * bareiss_det(minor)
    Bi = [[Fraction(cof[i][j], d) for j in range(n)] for i in range(n)]
    pts = set()
    for p in product(range(-R, R + 1), repeat=n):
        coords = [sum(Bi[i][t] * p[t] for t in range(n)) for i in range(n)]
        if all(q.denominator == 1 for q in coords):
            pts.add(p)
    return pts


def main():
    B1 = [
        [2, 1, 0],
        [0, 3, 1],
        [1, 0, 4],
    ]
    B2 = [
        [5, 0, 1],
        [1, 4, 0],
        [0, 1, 6],
    ]
    print("B1 (列格):")
    print(fmt(B1))
    print("B2 (列格):")
    print(fmt(B2))

    res = intersection_basis(B1, B2)
    print("\n交格基 C = B1 U = B2 V（LLL 缩短后）:")
    print(fmt(res.basis))
    print("整数坐标 U:")
    print(fmt(res.U))
    print("整数坐标 V:")
    print(fmt(res.V))
    print(f"det(C) = {res.determinant}")
    print(f"[L(B1):L(C)] = {res.index_in_lattice1},  "
          f"[L(B2):L(C)] = {res.index_in_lattice2}")

    report = verify_intersection(B1, B2, res)
    print("\n独立验证（包含 + 核子式 gcd=1 无漏点 + LLL）:",
          "PASS" if report.ok else "FAIL " + str(report.failures))

    R = 5
    common = box_points(B1, R) & box_points(B2, R)
    cpts = box_points(res.basis, R)
    print(f"盒 [-{R},{R}]^3 枚举：共同点 {len(common)} 个，交格点 {len(cpts)} 个，"
          f"集合一致: {common == cpts}")

    print("\n边界演示：奇异输入必须拒绝。")
    try:
        intersection_basis([[1, 2, 0], [2, 4, 0], [0, 0, 1]], B2)
    except RankDeficientError as exc:
        print(f"  RankDeficientError: {exc}")


if __name__ == "__main__":
    main()
