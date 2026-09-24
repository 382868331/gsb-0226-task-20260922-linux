"""固定输入演示：LLL 有理格基约化 + 盒约束最近格向量（CVP）+ 独立核对。

运行：python demo.py（约 1 秒内，纯标准库、离线、无 sleep、无预录结果）。

演示内容：
  1. LLL 正常约化（带巨大公共分量的 3 维基），独立验证器核验证书，
     并用二维小格做独立短向量穷举对照；
  2. LLL 实际触发的拒绝边界：秩亏基与极小预算（预算耗尽 ≠ 证明无解）；
  3. CVP 正常结果：歪斜幺模格上，目标是原系数 (99,-100) 对应的格点，
     盒有 40401 个候选，精确命中距离 0；半整数目标的多点并列按字典序；
  4. CVP 边界：目标远离小盒时的盒内最优（只保证盒内）；
  5. CVP 实际错误：维数 > 3、非法系数盒、float 目标均被明确拒绝。
"""

from fractions import Fraction
from itertools import product

from lll_reduce import (
    BudgetExhaustedError,
    InvalidBasisError,
    InvalidParameterError,
    RankDeficientError,
    nearest_vector,
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


def brute_cvp(A, t, bounds):
    """独立盒穷举（不调用 lll_reduce 的 CVP 代码），用于演示现场核对。"""
    n, m = len(A), len(A[0])
    best, winners = None, []
    for c in product(*[range(lo, hi + 1) for lo, hi in bounds]):
        v = [sum(c[i] * A[i][k] for i in range(n)) for k in range(m)]
        d = sum((t[k] - v[k]) ** 2 for k in range(m))
        if best is None or d < best:
            best, winners = d, [c]
        elif d == best:
            winners.append(c)
    return sorted(winners), best


def demo_lll():
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
    assert report.ok, report.failures
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
    print(f"  LLL 首向量 = {r2.basis[0]}，长度平方 = {got2}，命中最短：{got2 == lam2}")
    assert verify_reduction(A2, r2.basis, r2.transform).ok

    print("\n" + "=" * 70)
    print("演示 2：LLL 实际触发的拒绝 / 异常边界（现场计算）")
    print("=" * 70)

    bad = [[1, 2, 3], [2, 4, 6]]
    print(f"\n秩亏输入 A = {bad}（第二行是第一行的 2 倍）")
    try:
        reduce_basis(bad)
    except RankDeficientError as e:
        print(f"  已拒绝：{type(e).__name__}：期望秩 {e.expected}，实际行秩 {e.rank}")
    else:
        raise AssertionError("秩亏基必须被拒绝")

    print("\n预算耗尽 vs 证明无解 的区分（同一正常格，预算只够前几步）：")
    try:
        reduce_basis([[19, 7, 3], [8, 3, 11], [-4, 9, 6]], budget=5)
    except BudgetExhaustedError as e:
        p = e.partial
        print(f"  {type(e).__name__}：partial.reduced = {p.reduced}（不假称约化），"
              f"index 停在 {p.index}")
    else:
        raise AssertionError("预算 5 必须耗尽")


def demo_cvp():
    print("\n" + "=" * 70)
    print("演示 3：CVP 正常结果——歪斜格深处的精确格点（盒内最优）")
    print("=" * 70)

    # det = -1 的二维歪斜格；t=(0,1) 的原基系数手算为 (99,-100)。
    A = [[100, 99], [99, 98]]
    t = [0, 1]
    print(f"\n行基 A = {A}，目标 t = {t}，系数盒 [-100,100]^2（共 201^2=40401 点）")
    r = nearest_vector(A, t, 100)
    print(f"  最近格点系数（原基）：{r.coefficients}")
    print(f"  对应格点 c·A：{r.vectors}")
    print(f"  精确平方距离：{r.squared_distance}")
    print(f"  搜索统计：访问 {r.nodes_visited} 个节点、评估 {r.leaves_evaluated} 个叶子"
          f"（盒剪枝后，未遍历 40401 个点）")
    print("  独立盒穷举核对……")
    w, d = brute_cvp(A, [Fraction(0), Fraction(1)], [(-100, 100), (-100, 100)])
    assert r.coefficients == [tuple(x) for x in w] and r.squared_distance == d
    print(f"  独立穷举最优系数 {w}、平方距离 {d}，一致。")

    print("\n" + "-" * 70)
    print("并列全部返回，按原系数字典序（有理目标）")
    print("-" * 70)
    r = nearest_vector(
        [[1, 0], [0, 1]],
        [Fraction(1, 2), Fraction(1, 2)],
        1,
    )
    print("  A=I2，t=(1/2,1/2)，盒 [-1,1]^2：")
    for c, v in zip(r.coefficients, r.vectors):
        print(f"    系数 {tuple(c)} -> 格点 {tuple(v)}")
    print(f"  {len(r.coefficients)} 个并列点，平方距离均为 {r.squared_distance}")
    assert r.coefficients == [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("\n" + "=" * 70)
    print("演示 4：CVP 边界——目标远离小盒，只保证盒内最优")
    print("=" * 70)
    r = nearest_vector([[1, 0], [0, 1]], [10, 10], [(0, 2), (0, 2)])
    print("\n  A=I2，t=(10,10)，盒 [0,2]^2（无约束最近点 (10,10) 在盒外）：")
    print(f"  盒内最优系数 {r.coefficients}，格点 {r.vectors}，"
          f"平方距离 {r.squared_distance}（= 2·8^2）")
    assert r.coefficients == [(2, 2)] and r.squared_distance == 128
    print("  optimal_in_box =", r.optimal_in_box, "（不声明盒外情况）")

    # 单点盒：唯一候选
    r = nearest_vector([[3, 1], [1, 2]], [5, -5], [(2, 2), (-1, -1)])
    print(f"\n  单点盒 c=(2,-1)：唯一候选格点 {r.vectors}，平方距离 {r.squared_distance}")

    print("\n" + "=" * 70)
    print("演示 5：CVP 实际触发的输入错误（不静默纠正、无状态残留）")
    print("=" * 70)

    cases = [
        ("维数 4 > 上限 3",
         lambda: nearest_vector(
             [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
             [0, 0, 0, 0], 1)),
        ("系数盒 lo > hi（空盒）",
         lambda: nearest_vector([[1, 0], [0, 1]], [0, 0], [(0, 1), (2, 1)])),
        ("目标含 float（拒绝 NaN/浮点静默转换）",
         lambda: nearest_vector([[1, 0], [0, 1]], [0.5, 0], 1)),
        ("秩亏基",
         lambda: nearest_vector([[1, 1], [2, 2]], [0, 0], 1)),
    ]
    for label, fn in cases:
        try:
            fn()
        except (InvalidBasisError, InvalidParameterError, RankDeficientError) as e:
            print(f"  {label}\n    -> {type(e).__name__}: {e}")
        else:
            raise AssertionError(f"必须拒绝：{label}")

    # 错误调用后合法调用照常（无状态查询，不存在半更新）
    r = nearest_vector([[1, 0], [0, 1]], [0, 0], 1)
    assert r.coefficients == [(0, 0)] and r.squared_distance == 0
    print("\n  连续非法调用后，合法调用结果不变：", r.coefficients, r.squared_distance)


def main():
    demo_lll()
    demo_cvp()
    print("\n演示完成：全部断言通过。")
    # 引用 Fraction 仅为表明误差口径：核心计算无浮点。
    assert Fraction(3, 4) == Fraction(3, 4)


if __name__ == "__main__":
    main()
