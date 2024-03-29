from math import exp


def derivative_err_func(s: float) -> float:
    """Derivative of the error function: 1 / (1 + exp(-s))."""
    return exp(s) / (1 + exp(s)) ** 2
