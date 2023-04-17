from math import exp


def derivative_err_func(s: float) -> float:
    """Derivative of the error function: 1 / (1 + exp(-s))."""
    return exp(s) / (1 + exp(s)) ** 2


def logic_AND_activation_func(s: float) -> bool:
    return s >= 1.5


def logic_OR_activation_func(s: float) -> bool:
    return s >= 0.5


def logic_NOT_activation_func(s: float) -> bool:
    return s >= -1.0


def logic_XOR_activation_func(s: float) -> bool:
    return s >= 0.5
