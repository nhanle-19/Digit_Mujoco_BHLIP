"""
Biased H-LIP deadbeat stepping controller (sagittal plane).

    x_ddot = alpha^2 * (x - x_eq)

    alpha = sqrt(g_eff_z / z0),   x_eq = z0 * g_eff_x / g_eff_z

    K_db  = [1,  Td + (1/alpha)*coth(alpha*Ts)]

    u = u* + K_db @ ([x_pre, v_pre] - z*),  z* = [x*_pre, v*_pre] in original x (paper Eq. 11)
"""

import numpy as np


class BiasedHLIPController:

    def __init__(self, z0=0.80, Ts=0.5, Td=0.0, vx_des=0.2, mass=47.2281, **_ignored):
        self.z0 = z0
        self.Ts = Ts
        self.Td = Td
        self.vx_des = vx_des
        self.mass = mass
        self.geff_x = 0.0
        self.geff_z = 9.81
        self._recompute()

    def _recompute(self):
        gz = max(self.geff_z, 1.0)
        a  = np.sqrt(gz / self.z0)
        self.alpha = a
        self.alpha_sq = float(a * a)  # for instantaneous dynamics x_ddot = alpha^2 * (x - x_eq)
        self.x_eq  = self.z0 * self.geff_x / gz

        # sigma, K_db, fixed point
        self.sigma = a * np.cosh(a * self.Ts / 2) / np.sinh(a * self.Ts / 2)
        self.K     = np.array([[1.0, self.Td + np.cosh(a * self.Ts) / (np.sinh(a * self.Ts) * a)]])
        self.u_star = self.vx_des * (self.Ts + self.Td)
        xt_star = self.u_star / (2.0 + self.sigma * self.Td)
        vt_star = self.sigma * self.u_star / (2.0 + self.sigma * self.Td)
        # Pre-impact fixed point in original x (paper Eq. (11)): x* = x_eq + tilde_x*, v* unchanged
        self.x_des_pre = self.x_eq + xt_star
        self.v_des_pre = vt_star 
        self.z_star = np.array([self.x_des_pre, self.v_des_pre])

    def sagittal_com_accel_desired(self, x_sc):
        """Instantaneous biased-LIP sagittal acceleration (paper ROM dynamics).

        Uses :math:`\\ddot{x} = \\alpha^2 (x - x_\\mathrm{eq})` with current ``alpha``,
        ``x_eq`` from :meth:`_recompute` / :meth:`update_geff` (same ``geff`` as stepping).
        """
        return float(self.alpha_sq * (float(x_sc) - float(self.x_eq)))

    def update_geff(self, geff_x, geff_z):
        self.geff_x = geff_x
        self.geff_z = geff_z
        gz = max(geff_z, 1.0)
        self.x_eq  = self.z0 * geff_x / gz
        self.alpha = np.sqrt(gz / self.z0)
        # Recompute every update so desired pre-impact x/v track current geff.
        self._recompute()

    def compute_step(self, x_pre, v_pre):
        z_pre = np.array([x_pre, v_pre])
        u = float(self.u_star + self.K @ (z_pre - self.z_star))
        return u, z_pre

    def _propagate_hlip(self, x, v, T): 
        """Propagate H-LIP state forward by time T. x_ddot = alpha^2*(x - x_eq).""" 
        if T <= 0: 
            return float(x), float(v) 
        a = self.alpha 
        x_tilde = x - self.x_eq
        x_t = self.x_eq + x_tilde * np.cosh(a * T) + (v / a) * np.sinh(a * T) 
        v_t = a * x_tilde * np.sinh(a * T) + v * np.cosh(a * T) 
        return float(x_t), float(v_t)

    def compute_step_for_remaining_stance(
        self,
        x_cur,
        v_cur,
        t,
        t_td_last,
        t_td_next,
    ):
        """Deadbeat step u using the same touchdown schedule as the main controller.

        Time-to-nominal-impact is (t_td_next - t). T_remaining is clipped to
        [0, t_td_next - t_td_last].
        """
        den = float(t_td_next) - float(t_td_last)
        T_remaining = float(t_td_next) - float(t)
        T_remaining = float(np.clip(T_remaining, 0.0, den))
        x_pre_est, v_pre_est = self._propagate_hlip(float(x_cur), float(v_cur), T_remaining)
        return self.compute_step(x_pre_est, v_pre_est)

