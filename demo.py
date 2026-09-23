#!/usr/bin/env python3
"""固定输入演示：真实运行 LLL 并用独立验证器检查证书。

运行：``python demo.py``（标准库，离线，秒级完成）。

内容：
1. 正常结果：二维基的 LLL，打印 B、U、日志，独立枚举最短向量对照，
   独立验证器逐项给出通过证书；
2. 大公共分量 + 负舍入并列：展示大整数精确消元与半整数向偶数舍入；
3. 实际触发的拒绝/异常边界：秩亏基被精确拒绝（区别于预算耗尽通道），
   以及预算耗尽时不假称约化。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lll_reduce import (
    LatticeDependentError,
    InvalidBasisError,
    lll_reduce,
)
from lll_reduce.verify import verify_result


def fmt_mat(M):
    return "[" + ", ".join("[" + ", ".join(map(str, row)) + "]" for row in M) + "]"


def enumerate_shortest_2d(A, bound=40):
    """独立暴力枚举二维最短向量平方（仅用于演示对照，不经过被测核心）。"""
    a1, a2 = A
    best, vecs = None, []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            if x == y == 0:
                continue
            vx = x * a1[0] + y * a2[0]
            vy = x * a1[1] + y * a2[1]
            L = vx * vx + vy * vy
            if best is None or L < best:
                best, vecs = L, [(vx, vy)]
            elif L == best:
                vecs.append((vx, vy))
    return best, vecs


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    failures = []

    # ------------------------------------------------------------------
    # 1. 正常结果：二维 LLL + 证书 + 独立枚举对照
    # ------------------------------------------------------------------
    section("1. 正常结果：二维整数基的 LLL（delta=3/4，精确有理 GS）")
    A = [[4, 1], [2, 1]]
    print("输入 A =", fmt_mat(A))
    res = lll_reduce(A)
    print("约化基 B =", fmt_mat(res.basis))
    print("幺模变换 U =\n", "\n".join("  " + str(r) for r in res.transform))
    print(f"det(U) 校验：B=U*A，操作数 {res.operations}，reduced={res.reduced}")
    print("约化日志：")
    for e in res.events:
        mu = "" if e.mu_before is None else f"  (mu_before={e.mu_before})"
        print(f"  step {e.step:>2}: {e.message}{mu}")

    shortest, vecs = enumerate_shortest_2d(A)
    b0_norm = sum(t * t for t in res.basis[0])
    print(f"\n独立枚举最短向量平方 = {shortest}，例如 {sorted(set(vecs))[:4]}")
    print(f"LLL 首向量 ||b0||^2   = {b0_norm}（本例恰好给出最短；"
          "一般二维以上不保证）")

    report = verify_result(res, A)
    print()
    print(report.summary())
    if not report.ok:
        failures.append("normal case verification failed")

    # ------------------------------------------------------------------
    # 2. 大公共分量 + 负舍入并列
    # ------------------------------------------------------------------
    section("2. 大公共分量精确消元；半整数并列向偶数舍入")
    N = 10**40
    A_big = [[N, 1, 0], [N, 0, 1], [N, 1, 1]]
    print("输入 A（三行共享 10^40 公共分量）= ...")
    for row in A_big:
        print("  ", row)
    res_big = lll_reduce(A_big)
    print("约化基 B =")
    for row in res_big.basis:
        print("  ", row)
    rep_big = verify_result(res_big, A_big)
    print(rep_big.summary())
    if not rep_big.ok:
        failures.append("large-common-component verification failed")

    A_tie = [[2, 0], [-5, 1]]
    res_tie = lll_reduce(A_tie)
    first = res_tie.events[0]
    print(f"\n负舍入并列：A={A_tie}，mu=-5/2 时 size-reduction 系数 = "
          f"{first.coefficient}（向偶数 -> -2，而非远离零的 -3）")
    if first.coefficient != -2:
        failures.append("round-half-even coefficient wrong")
    rep_tie = verify_result(res_tie, A_tie)
    if not rep_tie.ok:
        failures.append("tie case verification failed")

    # ------------------------------------------------------------------
    # 3. 实际触发的异常边界：秩亏基被精确拒绝
    # ------------------------------------------------------------------
    section("3. 异常边界：线性相关基必须被拒绝（不是预算问题）")
    A_dep = [[1, 2, 3], [2, 4, 6], [1, 0, 1]]
    print("输入 A =", fmt_mat(A_dep), "（第二行 = 2 * 第一行）")
    try:
        lll_reduce(A_dep)
    except LatticeDependentError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
        print("=> 这是被精确算术证明的无解，与搜索预算耗尽是两条独立通道。")
    else:
        failures.append("rank-deficient basis was NOT rejected")

    # 非有限输入同样被可定位地拒绝。
    try:
        lll_reduce([[1, 0], [float("nan"), 1]])
    except InvalidBasisError as exc:
        print(f"非有限输入实际捕获 {type(exc).__name__}: {exc}")
    else:
        failures.append("NaN input was NOT rejected")

    # ------------------------------------------------------------------
    # 4. 预算耗尽：返回未完成，不抛异常、不谎称约化，但 B=U*A 仍自洽
    # ------------------------------------------------------------------
    section("4. 预算边界：耗尽时返回未完成且状态自洽（不假称约化）")
    res_cut = lll_reduce([[4, 1], [2, 1]], budget=1)
    print(f"budget=1 -> reduced={res_cut.reduced}, exhausted={res_cut.exhausted}, "
          f"operations={res_cut.operations}")
    print("中间基 B =", fmt_mat(res_cut.basis),
          " U =", fmt_mat(res_cut.transform))
    rep_cut = verify_result(res_cut, A, require_reduced=False)
    structural = all(
        c.passed for c in rep_cut.checks
        if c.name in ("B == U * A (exact)", "U unimodular (det U = +/-1)")
    )
    claim_check = verify_result(res_cut, A)
    print(f"结构证书 B=U*A、U 幺模：{'通过' if structural else '失败'}；"
          f"require_reduced=True 时验证器{'正确拒绝' if not claim_check.ok else '误放行'}")
    if not structural or claim_check.ok:
        failures.append("budget-exhaustion semantics wrong")

    section("演示结束")
    if failures:
        print("存在失败项：", failures)
        return 1
    print("全部演示断言通过：正常结果证书有效，拒绝/异常边界均真实触发。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
