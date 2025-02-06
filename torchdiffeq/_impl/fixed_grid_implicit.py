from .solvers import FixedGridImplicitODESolver
from .misc import Perturb
import warnings


class ImplicitEuler(FixedGridImplicitODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        dy = dt * f0
        for _ in range(self.max_iters):
            dy_old = dy
            f = func(t1, y0 + dy, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            dy = (dt * f).type_as(y0)
            converged = self._has_converged(dy_old, dy)
            if converged:
                break
        if not converged:
            warnings.warn('Functional iteration did not converge. Solution may be incorrect.')
        return dy, f0
    

class ImplicitMidpoint(FixedGridImplicitODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        f = func(t0 + half_dt, y_mid, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        dy = dt * f
        for _ in range(self.max_iters):
            dy_old = dy
            y_mid = y0 + f * half_dt
            f = func(t0 + half_dt, y_mid, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            dy = (dt * f).type_as(y0)
            converged = self._has_converged(dy_old, dy)
            if converged:
                break
        if not converged:
            warnings.warn('Functional iteration did not converge. Solution may be incorrect.')
        return dy, f0
