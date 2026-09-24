"""LLL 格基约化库（精确有理数算法，Python 3.14 标准库）。

公开接口：
    reduce_basis / lll_reduce : LLL 主算法
    LLLResult, LogEntry       : 结果与日志数据结构
    verify_reduction          : 独立验证器
    VerificationReport        : 验证报告
    closest_vector            : 盒内最近格向量（CVP，n <= 3，调用者给系数盒）
    CVPResult, CVP_MAX_DIM    : 最近向量结果与维数上限
    round_half_even           : 并列向偶数的最近整数舍入
    DELTA, MAX_DIM, DEFAULT_BUDGET
"""

from .errors import (
    BudgetExhaustedError,
    InvalidBasisError,
    InvalidParameterError,
    LLLError,
    RankDeficientError,
)
from .core import (
    DEFAULT_BUDGET,
    DELTA,
    MAX_DIM,
    LLLResult,
    LogEntry,
    lll_reduce,
    reduce_basis,
    round_half_even,
)
from .cvp import CVP_MAX_DIM, CVPResult, closest_vector
from .verifier import VerificationReport, verify_reduction

__all__ = [
    "reduce_basis",
    "lll_reduce",
    "LLLResult",
    "LogEntry",
    "verify_reduction",
    "VerificationReport",
    "closest_vector",
    "CVPResult",
    "CVP_MAX_DIM",
    "round_half_even",
    "DELTA",
    "MAX_DIM",
    "DEFAULT_BUDGET",
    "LLLError",
    "InvalidBasisError",
    "InvalidParameterError",
    "RankDeficientError",
    "BudgetExhaustedError",
]
