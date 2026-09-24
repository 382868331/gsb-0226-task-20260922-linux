"""LLL 格基约化库（精确有理数算法，Python 3.14 标准库）。

公开接口：
    reduce_basis / lll_reduce : LLL 主算法
    LLLResult, LogEntry       : 结果与日志数据结构
    verify_reduction          : 独立验证器
    VerificationReport        : 验证报告
    round_half_even           : 并列向偶数的最近整数舍入
    nearest_vector            : 盒约束最近格向量（CVP，n <= 3，精确枚举）
    NearestVectorResult       : CVP 结果数据结构
    DELTA, MAX_DIM, DEFAULT_BUDGET, CVP_MAX_DIM
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
from .verifier import VerificationReport, verify_reduction
from .cvp import CVP_MAX_DIM, NearestVectorResult, nearest_vector

__all__ = [
    "reduce_basis",
    "lll_reduce",
    "LLLResult",
    "LogEntry",
    "verify_reduction",
    "VerificationReport",
    "round_half_even",
    "nearest_vector",
    "NearestVectorResult",
    "CVP_MAX_DIM",
    "DELTA",
    "MAX_DIM",
    "DEFAULT_BUDGET",
    "LLLError",
    "InvalidBasisError",
    "InvalidParameterError",
    "RankDeficientError",
    "BudgetExhaustedError",
]
