"""交格计算的固定输入离线演示（真实计算，无 sleep、无预录结果）。

运行：python demo_iteration.py   （预期数秒内完成，远低于 90 秒上限）

演示内容：
  1. 互素对角格（手算可对照）；
  2. 明显斜基二维格；
  3. 幺模斜基与对角格（三维）；
  4. 奇异输入的实际拒绝。
每个用例打印交格基、双边坐标矩阵 U/V、指数与 Smith 标准形证据，
并由独立验证器 verify_intersection 复核完备性恒等式。
"""

from lll_reduce import (
    RankDeficientError,
    intersection_basis,
    verify_intersection,
)


def show(title, B1, B2):
    print("=" * 64)
    print(title)
    print(f"B1 = {B1}")
    print(f"B2 = {B2}")
    r = intersection_basis(B1, B2)
    print(f"交格基 G（列）= {r.basis}")
    print(f"U = {r.left_coords}   （B1 @ U == G）")
    print(f"V = {r.right_coords}   （B2 @ V == G）")
    print(
        f"指数 [L1:L∩] = {r.index_in_left}（SNF {r.snf_left}），"
        f"[L2:L∩] = {r.index_in_right}（SNF {r.snf_right}）"
    )
    print(f"和格协体积 det(L1+L2) = {r.sum_det}")
    rep = verify_intersection(B1, B2, r.basis, r.left_coords, r.right_coords)
    assert rep.ok, rep.failures
    print(
        "验证器：成员关系 B1@U==G==B2@V 成立；完备性 "
        f"{r.index_in_left}*{r.sum_det}==|det B2| 且 "
        f"{r.index_in_right}*{r.sum_det}==|det B1| 成立 -> OK"
    )


def main():
    show(
        "用例 1：互素对角格 diag(2,3) 与 diag(5,7)（交格 diag(10,21)）",
        [[2, 0], [0, 3]],
        [[5, 0], [0, 7]],
    )
    show(
        "用例 2：斜基 [[1,1],[0,2]] 与 [[1,0],[1,3]]（交格协体积 6）",
        [[1, 1], [0, 2]],
        [[1, 0], [1, 3]],
    )
    show(
        "用例 3：三维对角格 diag(2,3,5) 与幺模斜基（L2 = Z^3）",
        [[2, 0, 0], [0, 3, 0], [0, 0, 5]],
        [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
    )
    print("=" * 64)
    print("用例 4：奇异输入的实际拒绝")
    try:
        intersection_basis([[1, 2], [2, 4]], [[1, 0], [0, 1]])
    except RankDeficientError as exc:
        print(f"按预期拒绝：{type(exc).__name__}: {exc}")
    else:
        raise AssertionError("singular input was not rejected")
    print("=" * 64)
    print("全部演示用例完成（真实离线计算）。")


if __name__ == "__main__":
    main()

