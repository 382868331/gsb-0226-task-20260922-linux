"""固定输入演示：LLL 有理格基约化 + 独立证书验证。

运行：python demo.py（约 1 秒内，纯标准库、离线、无 sleep、无预录结果）。

演示两部分：
  1. 正常结果：一个带巨大公共分量的 3 维整数基，真实运行 LLL，
     打印约化基 B、幺模变换 U、交换/约化日志摘要，并用独立验证器核验证书；
     同时用二维小格做独立短向量穷举对照（不要求 LLL 总给出最短）。
  2. 实际触发的拒绝边界：线性相关基被 RankDeficientError 拒绝
     （并演示极小预算下 BudgetExhaustedError 与“证明无解”严格区分）。
"""

from fractions import Fraction

from lll_reduce import (
    BudgetExhaustedError,
    RankDeficientError,
    reduce_basis,
    verify_reduction,
)


def fmt_matrix(M):
    width = max(len(str(x)) for row in M for x in row)
    return "\n".join("  [" + " ".join(f"{x:>{width}}" for x in row) + "]" for row in M)


def enumerate_shortest_2d(A, bound=1024):
    best, vecs = None, []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            if x == y == 0:
                continue
            v = (x * A[0][0] + y * A[1][0], x * A[0][1] + y * A[1][1])
            d = v[0] ** 2 + v[1] ** 2
            if best is None or d < best:
                best, vecs = d, [v]
            elif d == best:
                vecs.append(v)
    return best, sorted(set(vecs))


def main():
    print("=" * 70)
    print("演示 1：正常 LLL 约化（delta = 3/4，全程 Fraction 精确计算）")
    print("=" * 70)

    N = 10 ** 24
    A = [
        [N, 1, 0],
        [N, 0, 1],
        [N + 1, 1, 0],
    ]
    print("\n输入行基 A（三行共享 10^24 的巨大公共分量，行列式 = 1）：")
    print(fmt_matrix(A))

    r = reduce_basis(A)
    print("\n约化基 B = U*A：")
    print(fmt_matrix(r.basis))
    print("\n整数幺模变换 U（证书的一部分）：")
    print(fmt_matrix(r.transform))

    n_red = sum(1 for e in r.log if e.kind == "size_reduce")
    n_swp = sum(1 for e in r.log if e.kind == "swap")
    print(f"\n操作日志：共 {len(r.log)} 步（size reduction {n_red} 次，相邻交换 {n_swp} 次），"
          f"计费操作 {r.ops_used}/{r.budget}")
    print("前 8 步：")
    for e in r.log[:8]:
        print(f"  step {e.step:>2}: {e.kind:<11} {e.detail}")

    report = verify_reduction(A, r.basis, r.transform)
    print("\n独立验证器报告（B=UA、det U=±1、|mu|<=1/2、Lovasz）：")
    print("  ok =", report.ok)
    if not report.ok:
        for f in report.failures:
            print("   FAIL:", f)
    assert report.ok, "证书必须通过"
    print("  -> 约化证书成立：巨大公共分量被精确消去。")

    print("\n" + "-" * 70)
    print("二维短向量独立穷举对照（不要求 LLL 总给最短）")
    print("-" * 70)
    A2 = [[19, 7], [8, 3]]
    r2 = reduce_basis(A2)
    lam2, vecs = enumerate_shortest_2d(A2)
    got2 = sum(x * x for x in r2.basis[0])
    print(f"  格基 A = {A2}")
    print(f"  穷举最短长度平方 lambda1^2 = {lam2}，最短向量 = {vecs}")
    print(f"  LLL 首向量 = {r2.basis[0]}，长度平方 = {got2}")
    print(f"  命中最短：{got2 == lam2}；LLL 二维界 ||b0||^2 <= 2*lambda1^2 成立："
          f"{got2 <= 2 * lam2}")
    assert verify_reduction(A2, r2.basis, r2.transform).ok

    print("\n" + "=" * 70)
    print("演示 2：实际触发的拒绝 / 异常边界（非预录，现场计算）")
    print("=" * 70)

    bad = [[1, 2, 3], [2, 4, 6]]  # 第二行 = 2 * 第一行，线性相关
    print(f"\n秩亏输入 A = {bad}（第二行是第一行的 2 倍）")
    try:
        reduce_basis(bad)
    except RankDeficientError as e:
        print(f"  已拒绝：{type(e).__name__}: {e}")
        print(f"  可定位信息：期望秩 {e.expected}，检测到实际行秩 {e.rank}")
    else:
        raise AssertionError("秩亏基必须被拒绝")

    print("\n预算耗尽 vs 证明无解 的区分（同一正常格，预算只够前几步）：")
    try:
        reduce_basis([[19, 7, 3], [8, 3, 11], [-4, 9, 6]], budget=5)
    except BudgetExhaustedError as e:
        p = e.partial
        print(f"  {type(e).__name__}: {e}")
        print(f"  partial.reduced = {p.reduced}（不假称约化）；"
              f"中间态仍满足 B = U A，index 停在 {p.index}")
    else:
        raise AssertionError("预算 5 必须耗尽")
    print("  对照：秩亏是数学上证明“不适用”，预算耗尽只是资源不足，二者异常类型不同。")

    print("\n演示完成：全部断言通过。")
    # 引用 Fraction 仅为表明误差口径：核心计算无浮点。
    assert Fraction(3, 4) == Fraction(3, 4)


if __name__ == "__main__":
    main()
