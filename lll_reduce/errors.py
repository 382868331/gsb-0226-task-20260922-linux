"""LLL 库的公开错误类型。"""

from __future__ import annotations


class LLLError(Exception):
    """本库所有错误的基类。"""


class InvalidParameterError(LLLError, ValueError):
    """非法参数（NaN/Infinity、bool 冒充整数、越界等）。

    ``parameter`` 为可定位的参数名或位置说明，``reason`` 为具体原因。
    """

    def __init__(self, parameter: str, reason: str) -> None:
        self.parameter = parameter
        self.reason = reason
        super().__init__(f"invalid parameter {parameter!r}: {reason}")


class InvalidBasisError(LLLError, ValueError):
    """输入矩阵形状/元素非法。"""

    def __init__(self, location: str, reason: str) -> None:
        self.location = location
        self.reason = reason
        super().__init__(f"invalid basis at {location}: {reason}")


class RankDeficientError(LLLError, ValueError):
    """输入基线性相关（非满行秩）。

    ``rank`` 为算法检测到的实际（行）秩，``expected`` 为应有秩。
    """

    def __init__(self, expected: int, rank: int) -> None:
        self.expected = expected
        self.rank = rank
        super().__init__(
            f"basis is linearly dependent: row rank {rank} < {expected} rows"
        )


class BudgetExhaustedError(LLLError):
    """操作预算耗尽；算法未完成，不能宣称结果已约化。

    与“证明无解”严格区分：这里只是资源耗尽，输入本身仍可能存在 LLL 约化基。
    ``partial`` 携带耗尽时刻的中间态（供调试/续算），但它不是约化证书。
    """

    def __init__(self, budget: int, ops_used: int, partial: object) -> None:
        self.budget = budget
        self.ops_used = ops_used
        self.partial = partial
        super().__init__(
            f"operation budget exhausted: {ops_used}/{budget} elementary "
            f"operations used; returned basis is NOT certified LLL-reduced"
        )
