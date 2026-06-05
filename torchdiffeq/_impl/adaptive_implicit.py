import torch
from .rk_common import _ButcherTableau
from .rk_common import FIRKAdaptiveStepsizeODESolver, DIRKAdaptiveStepsizeODESolver

_sqrt_2 = torch.sqrt(torch.tensor(2, dtype=torch.float64)).item()
_sqrt_3 = torch.sqrt(torch.tensor(3, dtype=torch.float64)).item()
_sqrt_6 = torch.sqrt(torch.tensor(6, dtype=torch.float64)).item()
_sqrt_15 = torch.sqrt(torch.tensor(15, dtype=torch.float64)).item()

def polyval(coeffs: list, x: float):
    """
    Evaluate a polynomial for a given x.
    Degree is len(coeffs)-1

    Args:
        coeffs (list): list of coefficients from highest power to constant

        x (float): number to evaluate at
    """
    max_degree = len(coeffs) - 1
    return sum(coeffs[i] * (x**(max_degree-i)) for i in range(len(coeffs)))

_GAUSS_LEGENDRE_4_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2 - _sqrt_3 / 6, 1 / 2 + _sqrt_3 / 6], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 4, 1 / 4 - _sqrt_3 / 6], dtype=torch.float64),
        torch.tensor([1 / 4 + _sqrt_3 / 6, 1 / 4], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 2, 1 / 2], dtype=torch.float64),
    c_error=torch.tensor([1 / 2 + _sqrt_3 / 2, 1 / 2 - _sqrt_3 / 2], dtype=torch.float64),
)

_GL4_C_MID = torch.tensor([1 / 4 + _sqrt_3 / 8, 1 / 4 - _sqrt_3 / 8], dtype=torch.float64)

class AdaptiveGaussLegendre4(FIRKAdaptiveStepsizeODESolver):
    order = 4
    tableau = _GAUSS_LEGENDRE_4_TABLEAU
    mid = _GL4_C_MID

_GAUSS_LEGENDRE_6_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2 - _sqrt_15 / 10, 1 / 2, 1 / 2 + _sqrt_15 / 10], dtype=torch.float64),
    beta=[
        torch.tensor([5 / 36                , 2 / 9 - _sqrt_15 / 15, 5 / 36 - _sqrt_15 / 30], dtype=torch.float64),
        torch.tensor([5 / 36 + _sqrt_15 / 24, 2 / 9                , 5 / 36 - _sqrt_15 / 24], dtype=torch.float64),
        torch.tensor([5 / 36 + _sqrt_15 / 30, 2 / 9 + _sqrt_15 / 15, 5 / 36                ], dtype=torch.float64),
    ],
    c_sol=torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=torch.float64),
    c_error=torch.tensor([-5 / 6, 8 / 3, -5 / 6], dtype=torch.float64),
)

_GL6_C_MID = torch.tensor([-5 / 18, 19 / 18, -5 / 18], dtype=torch.float64)

class AdaptiveGaussLegendre6(FIRKAdaptiveStepsizeODESolver):
    order = 6
    tableau = _GAUSS_LEGENDRE_6_TABLEAU
    mid = _GL6_C_MID

# https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_solver/kvaerno3.py
gamma = 0.43586652150
b31 = (-4 * gamma**2 + 6 * gamma - 1) / (4 * gamma)
b32 = (-2 * gamma + 1) / (4 * gamma)
b41 = (6 * gamma - 1) / (12 * gamma)
b42 = -1 / ((24 * gamma - 12) * gamma)
b43 = (-6 * gamma**2 + 6 * gamma - 1) / (6 * gamma - 3)
_KVAERNO_3_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([0, 2 * gamma, 1.0, 1.0], dtype=torch.float64),
    beta = [
        torch.tensor([0], dtype=torch.float64),
        torch.tensor([gamma, gamma], dtype=torch.float64),
        torch.tensor([b31, b32, gamma], dtype=torch.float64),
        torch.tensor([b41, b42, b43, gamma], dtype=torch.float64),
    ],
    c_sol=torch.tensor([b41, b42, b43, gamma], dtype=torch.float64),
    c_error=torch.tensor([b41 - b31, b42 - b32, b43 - gamma, gamma])
)

_KV3_C_MID = torch.tensor([0.35414591, 0.08861962, 0.09340915, -0.03617468], dtype=torch.float64)

class Kvaerno3(DIRKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _KVAERNO_3_TABLEAU
    mid = _KV3_C_MID

# gamma = 0.5728160625
# b31 = polyval([144,-180,81,-15,1,0], gamma) / polyval([12,-6,1], gamma)
# b32 = polyval([-36,39,-15,2,0], gamma) / polyval([12,-6,1], gamma)
# b41 = polyval([-144,396,-330,117,18,1], gamma) / 12 / gamma / gamma / polyval([12,-9,0], gamma)
# b42 = polyval([72,-126,69,-15,1], gamma) / (12*gamma*gamma) / (3*gamma - 1)
# b43 = polyval([-6,6,-1], gamma) * polyval([12,-6,1], gamma) / (12*gamma*gamma) / polyval([12,-9,2], gamma) / (3*gamma -1)
# b51 = polyval([288,-312,120,-18,1], gamma) / (48*gamma*gamma) / polyval([12,-9,2], gamma)
# b52 = polyval([24,-12,1], gamma) / (48*gamma*gamma) / (3*gamma - 1)
# b53 = -(polyval([12,-6,1], gamma)**3) / (48*gamma*gamma) / (3*gamma - 1) / polyval([12,-9,2], gamma) / polyval([6,-6,1], gamma)
# b54 = polyval([-24,36,-12,1], gamma) / polyval([24,-24,4], gamma)
# alpha3 = gamma + b31 + b32
# _KVAERNO_4_TABLEAU = _ButcherTableau(
#     alpha=torch.tensor([0, 2 * gamma, alpha3, 1.0, 1.0], dtype=torch.float64),
#     beta = [
#         torch.tensor([0], dtype=torch.float64),
#         torch.tensor([gamma, gamma], dtype=torch.float64),
#         torch.tensor([b31, b32, gamma], dtype=torch.float64),
#         torch.tensor([b41, b42, b43, gamma], dtype=torch.float64),
#         torch.tensor([b51, b52, b53, b54, gamma], dtype=torch.float64),
#     ],
#     c_sol=torch.tensor([b51, b52, b53, b54, gamma], dtype=torch.float64),
#     c_error=torch.tensor([b51 - b41, b52 - b42, b53 - b43, b54 - gamma, gamma])
# )

# This is not evaluated correctly.
# _KV4_C_MID = torch.tensor([0.35414591, 0.08861962, 0.09340915, -0.03617468], dtype=torch.float64)

# class Kvaerno4(DIRKAdaptiveStepsizeODESolver):
#     order = 4
#     tableau = _KVAERNO_4_TABLEAU
#     mid = _KV4_C_MID

# Kvaerno5
