"""测试用独立参考实现（刻意不 import 被测核心 lll_reduce）。

包含：
  * ref_lll : 按教科书（H. Cohen, A Course in Computational Algebraic Number
    Theory, Alg. 2.6.3 的常见变体）独立编写的精确有理 LLL，用于交叉检验；
  * enumerate_shortest_2d : 二维格短向量穷举，独立求最短长度 lambda1；
  * frac_gs : 另一份独立 Gram-Schmidt。
这些参考只用于生成/核对期望值，不调用 lll_reduce 包的任何代码。
"""

from fractions import Fraction


def frac_gs(B):
    n = len(B)
    dim = len(B[0])
    bstar = []
    mu = [[Fraction(0)] * n for _ in range(n)]
    norm2 = []
    for i in range(n):
        v = [Fraction(B[i][t]) for t in range(dim)]
        for j in range(i):
            c = sum(B[i][t] * bstar[j][t] for t in range(dim)) / norm2[j]
            mu[i][j] = c
            for t in range(dim):
                v[t] -= c * bstar[j][t]
        bstar.append(v)
        norm2.append(sum(x * x for x in v))
    return bstar, mu, norm2


def _round_half_up_floor(x: Fraction) -> int:
    """参考实现使用「半数远离零」之外的另一种确定性规则：半数向上取整，
    与被测实现的向偶数舍入刻意不同，用于检验二者输出各自可验证，
    而不是互相复制。"""
    q, r = divmod(x.numerator, x.denominator)
    return q + (1 if 2 * r >= x.denominator else 0)


def ref_lll(A, delta=Fraction(3, 4)):
    """独立教科书 LLL（行基，返回约化基，不返回 U/日志）。"""
    b = [list(map(int, row)) for row in A]
    n = len(b)
    m = len(b[0])
    _, mu, norm2 = frac_gs(b)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                q = _round_half_up_floor(mu[k][j])
                b[k] = [b[k][t] - q * b[j][t] for t in range(m)]
                for l in range(j + 1):
                    mu[k][l] -= q * mu[j][l]
        if norm2[k] >= (delta - mu[k][k - 1] ** 2) * norm2[k - 1]:
            k += 1
        else:
            b[k - 1], b[k] = b[k], b[k - 1]
            _, mu, norm2 = frac_gs(b)
            k = max(k - 1, 1)
    return b


def enumerate_shortest_2d(A, bound=1024):
    """独立穷举二维格所有非零向量（系数范围 [-bound, bound]），返回最短长度平方。

    bound 必须覆盖最短向量的系数范围：二维时最短向量系数不超过基向量长度
    之比所限定的范围，本测试基元素 <= 100，取 bound=1024 有充裕余量
    （例如 [[100,99],[99,98]] 的最短向量系数为 (99,-100)）。
    """
    best = None
    best_vecs = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            if x == 0 and y == 0:
                continue
            v0 = x * A[0][0] + y * A[1][0]
            v1 = x * A[0][1] + y * A[1][1]
            d = v0 * v0 + v1 * v1
            if best is None or d < best:
                best = d
                best_vecs = [(v0, v1)]
            elif d == best:
                best_vecs.append((v0, v1))
    return best, sorted(set(best_vecs))
