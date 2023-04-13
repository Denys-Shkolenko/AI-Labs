from math import exp


def derivative_err_func(s: int | float) -> float:
    """Derivative of the error function: 1 / (1 + exp(-s))."""
    return exp(s) / (1 + exp(s)) ** 2

    # TODO: write functions for AND, OR, XOR, NOT
