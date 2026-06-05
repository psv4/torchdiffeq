import torch
from .rk_common import _ButcherTableau
from .rk_common import FIRKAdaptiveStepsizeODESolver, DIRKAdaptiveStepsizeODESolver

_sqrt_2 = torch.sqrt(torch.tensor(2, dtype=torch.float64)).item()
_sqrt_3 = torch.sqrt(torch.tensor(3, dtype=torch.float64)).item()
_sqrt_6 = torch.sqrt(torch.tensor(6, dtype=torch.float64)).item()
_sqrt_15 = torch.sqrt(torch.tensor(15, dtype=torch.float64)).item()

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

# Kvaerno4

# Kvaerno5
