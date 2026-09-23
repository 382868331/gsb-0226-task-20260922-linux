"""错误类型。

所有本库抛出的异常都继承 :class:`LLLReduceError`，便于调用方统一捕获。
预算耗尽 **不** 抛异常，而是通过 ``LLLResult.reduced=False`` /
``LLLResult.exhausted=True`` 报告，以便与“被证明无解”（线性相关）明确区分。
"""


class LLLReduceError(Exception):
    """本库所有错误的基类。"""


class InvalidBasisError(LLLReduceError, ValueError):
    """输入基矩阵不满足契约（类型、形状、有限性等）。

    错误信息包含可定位的坐标（例如 ``A[1][2]``）。
    """


class InvalidParameterError(LLLReduceError, ValueError):
    """标量参数不满足契约（例如 budget 不是正整数，或误传 bool）。"""


class LatticeDependentError(LLLReduceError):
    """输入基被精确算术判定为线性相关（满行秩契约被破坏）。

    :ivar index: 第一个使 Gram--Schmidt 正交向量为零向量的行下标（0 起）。
    """

    def __init__(self, index: int, detail: str = ""):
        self.index = index
        text = f"input basis is linearly dependent: b*_{index + 1} == 0"
        if detail:
            text += f" ({detail})"
        super().__init__(text)
