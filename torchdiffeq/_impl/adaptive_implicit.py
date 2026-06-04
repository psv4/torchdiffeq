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

# Kvaerno3

# Kvaerno4

# Kvaerno5
