"""
cummins.py — Time-domain Cummins heave solver (Module 2.3C).

Solves the single-DOF heave equation of motion for a floating point-absorber:

    (M + A_inf) * x_ddot(t)
    + C33 * x(t)
    + integral_0^t K(t-tau) * x_dot(tau) dtau
    = F_exc(t)

where:
    x(t)       heave displacement [m], positive upward
    x_dot(t)   heave velocity [m/s]
    x_ddot(t)  heave acceleration [m/s^2]
    M          buoy mass [kg]
    A_inf      infinite-frequency added mass [kg]
    C33        hydrostatic stiffness [N/m]
    K(t)       radiation impulse-response kernel [N·s/m^2]
    F_exc(t)   wave excitation force [N]

Optional fixed linear PTO (Module 2.4): pass pto=PTOParameters(...) to
solve_cummins.  With pto=None the solver is identical to Module 2.3C.

============================================================
SIGN CONVENTION
============================================================

Positive heave displacement x is UPWARD.

Force balance (Newton's second law):

    M * x_ddot = F_exc + F_rad + F_restore

where:
    F_restore = -C33 * x                          [N]
    F_rad     = -A_inf * x_ddot
                - integral_0^t K(t-tau) * x_dot(tau) dtau   [N]

Rearranging:

    (M + A_inf) * x_ddot
    + C33 * x
    + integral_0^t K(t-tau) * x_dot(tau) dtau
    = F_exc

This is the standard Cummins (1962) form.

A_inf appears ONLY in the effective inertia (M + A_inf).
K(t) represents the memory (radiation damping) component.
They must NOT be double-counted.

============================================================
NUMERICAL METHOD
============================================================

Time integration: explicit Euler predictor with trapezoidal
radiation convolution (Euler-Cromer variant).

At each step n (t = n*dt):

Step 1 — Radiation convolution (causal, left-endpoint quadrature):

    F_mem[n] = -dt * sum_{j=0}^{n-1} K((n-j)*dt) * x_dot[j]

    Quadrature convention: left-endpoint rectangle rule.
    K is evaluated at lag tau = (n-j)*dt for j = 0, ..., n-1.
    The j=n term (zero lag) is excluded because K(0) multiplies
    x_dot[n] which is not yet known at the start of step n.
    This makes the convolution explicit (no implicit solve needed).

    K values at lags beyond the stored kernel duration are set to zero
    (kernel has decayed to negligible amplitude by t_max = 60 s).

Step 2 — Acceleration:

    x_ddot[n] = (F_exc[n] - C33*x[n] + F_mem[n]) / (M + A_inf)

Step 3 — Velocity update (Euler):

    x_dot[n+1] = x_dot[n] + dt * x_ddot[n]

Step 4 — Displacement update (Euler):

    x[n+1] = x[n] + dt * x_dot[n+1]

    (Euler-Cromer: uses updated velocity, which improves energy
    conservation for oscillatory systems compared to standard Euler.)

Kernel interpolation:
    K is stored on a uniform grid with spacing dt_K (from Module 2.3A).
    For the solver timestep dt_s, K is pre-interpolated onto a uniform
    grid with spacing dt_s using linear interpolation.  This avoids
    repeated interpolation inside the time loop.

Timestep requirement:
    dt_s must satisfy the Nyquist criterion for the highest trusted
    frequency: dt_s < pi / omega_max = pi / 1.4 ≈ 2.24 s.
    In practice dt_s <= 0.1 s is recommended for accuracy.

============================================================
REFERENCES
============================================================
- Cummins (1962), The impulse response function and ship motions.
- Ogilvie (1964), Recent progress toward the understanding and prediction
  of ship motions.
- Falnes (2002), Ocean Waves and Oscillating Systems, §5.
- Yu & Falnes (1995), State-space modelling of a vertical cylinder in heave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from module2_wec.pto import PTOParameters, PTOResult, compute_pto_result


# ---------------------------------------------------------------------------
# CumminsParameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CumminsParameters:
    """
    Physical and numerical parameters for the Cummins heave solver.

    Parameters
    ----------
    mass : float
        Buoy mass M [kg].  Must be > 0.
    hydrostatic_stiffness : float
        Hydrostatic restoring stiffness C33 [N/m].  Must be > 0.
    added_mass_infinity : float
        Infinite-frequency added mass A_inf [kg].  Must be >= 0.
        DEVELOPMENT ESTIMATE — configurable, not hard-coded.
    kernel_time : np.ndarray
        Time grid of the radiation kernel [s], shape (n_K,).
        Must start at 0, be strictly increasing, uniform spacing.
    kernel_values : np.ndarray
        Radiation impulse-response K(t) [N·s/m^2], shape (n_K,).
        Produced by Module 2.3A compute_radiation_kernel().
    """

    mass: float
    hydrostatic_stiffness: float
    added_mass_infinity: float
    kernel_time: np.ndarray
    kernel_values: np.ndarray

    @property
    def effective_mass(self) -> float:
        """M + A_inf [kg] — the effective inertia in the Cummins equation."""
        return self.mass + self.added_mass_infinity

    @property
    def omega_n(self) -> float:
        """Undamped natural frequency sqrt(C33 / (M + A_inf)) [rad/s]."""
        return float(np.sqrt(self.hydrostatic_stiffness / self.effective_mass))


# ---------------------------------------------------------------------------
# CumminsResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CumminsResult:
    """
    Time-domain solution of the Cummins heave equation.

    All arrays have shape (n_time,).

    Units
    -----
    time            : s
    displacement    : m      (positive upward)
    velocity        : m/s
    acceleration    : m/s^2
    excitation_force: N
    radiation_force : N      (= -A_inf*x_ddot - convolution term)
    restoring_force : N      (= -C33*x)
    total_force     : N      (= F_exc + F_rad + F_restore = M*x_ddot)
    """

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    excitation_force: np.ndarray
    radiation_force: np.ndarray
    restoring_force: np.ndarray
    total_force: np.ndarray
    params: CumminsParameters
    pto_result: Optional[PTOResult] = None

    @property
    def kinetic_energy(self) -> np.ndarray:
        """0.5 * M * v^2 [J].  Uses buoy mass only (not added mass)."""
        return 0.5 * self.params.mass * self.velocity**2

    @property
    def potential_energy(self) -> np.ndarray:
        """0.5 * C33 * x^2 [J]."""
        return 0.5 * self.params.hydrostatic_stiffness * self.displacement**2

    @property
    def mechanical_energy(self) -> np.ndarray:
        """Kinetic + potential energy [J]."""
        return self.kinetic_energy + self.potential_energy


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_cummins(
    params: CumminsParameters,
    time: np.ndarray,
    excitation_force: np.ndarray,
    x0: float = 0.0,
    v0: float = 0.0,
    pto: Optional[PTOParameters] = None,
) -> CumminsResult:
    """
    Solve the Cummins heave equation using explicit Euler-Cromer integration.

    With pto=None (default) the equation is the frozen Module 2.3C form:

        (M + A_inf) x_ddot + C33 x + convolution = F_exc

    With pto=PTOParameters(damping=B_pto, stiffness=K_pto) the equation is:

        (M + A_inf) x_ddot + (C33 + K_pto) x + convolution + B_pto v = F_exc

    The PTO force on the buoy is F_PTO = -B_pto*v - K_pto*x.
    Backward-compatible: pto=None gives identical results to Module 2.3C.

    Parameters
    ----------
    params : CumminsParameters
        Physical and kernel parameters.
    time : np.ndarray
        Uniform time grid [s], shape (n_time,).  Must be strictly increasing
        with uniform spacing dt.
    excitation_force : np.ndarray
        Wave excitation force F_exc(t) [N], shape (n_time,).
    x0 : float, optional
        Initial displacement [m].  Default 0.0.
    v0 : float, optional
        Initial velocity [m/s].  Default 0.0.
    pto : PTOParameters or None, optional
        Fixed linear PTO parameters.  None (default) = no PTO.

    Returns
    -------
    CumminsResult
        Full time-domain solution.  If pto is not None, result.pto_result
        contains PTO force, power, and energy metrics.

    Raises
    ------
    ValueError
        If time is not uniform, excitation_force shape mismatches, or
        params are non-physical.
    """
    time = np.asarray(time, dtype=float)
    excitation_force = np.asarray(excitation_force, dtype=float)

    if time.ndim != 1 or len(time) < 2:
        raise ValueError("time must be a 1-D array with at least 2 elements")
    if excitation_force.shape != time.shape:
        raise ValueError(
            f"excitation_force shape {excitation_force.shape} != time shape {time.shape}"
        )
    if params.mass <= 0:
        raise ValueError(f"mass must be positive, got {params.mass}")
    if params.hydrostatic_stiffness <= 0:
        raise ValueError(f"hydrostatic_stiffness must be positive, got {params.hydrostatic_stiffness}")
    if params.added_mass_infinity < 0:
        raise ValueError(f"added_mass_infinity must be >= 0, got {params.added_mass_infinity}")

    # Uniform timestep
    dts = np.diff(time)
    dt = float(dts[0])
    if not np.allclose(dts, dt, rtol=1e-6):
        raise ValueError("time grid must be uniformly spaced")

    n = len(time)
    M_eff = params.effective_mass   # M + A_inf
    C33   = params.hydrostatic_stiffness
    B_pto = pto.damping   if pto is not None else 0.0
    K_pto = pto.stiffness if pto is not None else 0.0

    # Pre-interpolate kernel onto solver timestep grid
    # Lags: 0, dt, 2*dt, ..., up to kernel duration
    K_t = params.kernel_time
    K_v = params.kernel_values
    dt_K = float(K_t[1] - K_t[0])
    n_K_max = len(K_t)

    # Maximum number of kernel lags needed (capped at kernel length)
    n_lag_max = min(n, n_K_max)

    # Pre-compute K at lags 1*dt, 2*dt, ..., n_lag_max*dt
    # (lag=0 is excluded from the explicit convolution — see docstring)
    lags = np.arange(1, n_lag_max + 1) * dt
    K_interp = np.interp(lags, K_t, K_v, left=0.0, right=0.0)

    # Output arrays
    x   = np.zeros(n)
    v   = np.zeros(n)
    a   = np.zeros(n)
    F_mem_arr = np.zeros(n)   # memory (convolution) part of radiation force
    F_rad_arr = np.zeros(n)   # total radiation force
    F_res_arr = np.zeros(n)   # restoring force
    F_tot_arr = np.zeros(n)   # total force = M*a

    x[0] = x0
    v[0] = v0

    for i in range(n):
        # Radiation memory convolution: F_mem = -dt * sum_{j=0}^{i-1} K((i-j)*dt) * v[j]
        # Lag index k = i-j ranges from 1 to i (j from i-1 down to 0)
        # K_interp[k-1] = K(k*dt)
        if i > 0:
            n_lag = min(i, n_lag_max)
            # v[i-1], v[i-2], ..., v[i-n_lag] multiplied by K_interp[0], K_interp[1], ...
            v_hist = v[i - n_lag:i][::-1]   # v[i-1], v[i-2], ..., v[i-n_lag]
            F_mem = -dt * np.dot(K_interp[:n_lag], v_hist)
        else:
            F_mem = 0.0

        # Acceleration: (M+A_inf)*a = F_exc - (C33+K_pto)*x - B_pto*v + F_mem
        a[i] = (excitation_force[i]
                - (C33 + K_pto) * x[i]
                - B_pto * v[i]
                + F_mem) / M_eff

        # Radiation force = -A_inf*a - memory term
        F_rad = -params.added_mass_infinity * a[i] + F_mem
        F_res = -C33 * x[i]

        F_mem_arr[i] = F_mem
        F_rad_arr[i] = F_rad
        F_res_arr[i] = F_res
        F_tot_arr[i] = params.mass * a[i]

        # Euler-Cromer update (if not last step)
        if i < n - 1:
            v[i + 1] = v[i] + dt * a[i]
            x[i + 1] = x[i] + dt * v[i + 1]   # uses updated v (Euler-Cromer)

    pto_result = compute_pto_result(time, x, v, pto) if pto is not None else None

    return CumminsResult(
        time=time,
        displacement=x,
        velocity=v,
        acceleration=a,
        excitation_force=excitation_force,
        radiation_force=F_rad_arr,
        restoring_force=F_res_arr,
        total_force=F_tot_arr,
        params=params,
        pto_result=pto_result,
    )


# ---------------------------------------------------------------------------
# Frequency-domain RAO (for validation)
# ---------------------------------------------------------------------------

def frequency_domain_rao(
    omega: float,
    mass: float,
    added_mass: float,
    hydrostatic_stiffness: float,
    radiation_damping: float,
    excitation_complex: complex,
) -> complex:
    """
    Compute the complex displacement RAO from frequency-domain impedance.

    The linear frequency-domain equation of motion is:

        Z(omega) * X(omega) = F_exc(omega)

    ============================================================
    DERIVATION (from the Cummins TD equation)
    ============================================================

    The Cummins time-domain equation is:

        (M + A_inf) x_ddot + C33 x + integral_0^t K(t-tau) v(tau) dtau = F_exc

    PHASOR CONVENTION USED IN THIS PROJECT:

    excitation.py synthesises the force as:
        F(t) = a * |F_exc| * cos(omega*t + angle(F_exc))
             = Re[ a * F_exc * e^{+i*omega*t} ]   -- e^{+i*omega*t} convention

    _measure_amplitude_phase (in tests and demo) uses:
        phasor = mean(x(t) * e^{-i*omega*t}) * 2
    which extracts the e^{+i*omega*t} component:
        x(t) = Re[ X * e^{+i*omega*t} ]

    Therefore the consistent phasor convention throughout this project is
    e^{+i*omega*t}.  Under this convention:
        x_dot  -> +i*omega * X
        x_ddot -> -omega^2 * X

    KERNEL TRANSFORM under e^{+i*omega*t}:

    The one-sided Fourier transform of K(t) is:
        hat_K_+(omega) = integral_0^inf K(t) e^{+i*omega*t} dt

    Since K(t) is real and causal, its cosine/sine transforms give:
        Re[hat_K_+] = integral_0^inf K(t) cos(omega*t) dt = B(omega)   [Ogilvie]
        Im[hat_K_+] = integral_0^inf K(t) sin(omega*t) dt = omega*(A(omega) - A_inf)

    So:  hat_K_+(omega) = B(omega) + i*omega*(A(omega) - A_inf)

    CONVOLUTION TERM under e^{+i*omega*t}:

        integral K(t-tau) v(tau) dtau  ->  hat_K_+(omega) * (i*omega) * X
            = [B + i*omega*(A-A_inf)] * i*omega * X
            = i*omega*B*X - omega^2*(A-A_inf)*X

    SUBSTITUTING into the Cummins equation:
        -omega^2*(M+A_inf)*X + C33*X + i*omega*B*X - omega^2*(A-A_inf)*X = F_exc
        [C33 - omega^2*(M + A(omega)) + i*omega*B(omega)] * X = F_exc

    The CORRECT frequency-domain impedance for this project is:

        Z(omega) = C33 - omega^2*(M + A(omega)) + i*omega*B(omega)

    Key points:
    - A_inf cancels algebraically; A(omega) appears in the final FD equation.
    - A_inf is a computational device in the TD split; it does NOT appear
      in the FD impedance.
    - The sign of the B(omega) term is POSITIVE under e^{+i*omega*t}.
      (Under e^{-i*omega*t} it would be negative — the two conventions
      give conjugate impedances and conjugate X, but identical |X|.)
    - excitation.py uses e^{+i*omega*t}, so Z_+ must be used here for
      phase consistency.

    DOUBLE-COUNTING CHECK:
    - M + A_inf appears exactly once in the inertial term.
    - The convolution integral carries only the memory part of radiation;
      A_inf is NOT re-introduced through K(t) (it cancels in the derivation).
    - A(omega) in the FD impedance is the full BEM added mass, not A_inf.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    mass : float
        Buoy mass M [kg].
    added_mass : float
        Frequency-dependent added mass A(omega) [kg] at this omega.
        Must be the BEM value A(omega), NOT A_inf.
        Using A_inf here gives wrong results near resonance.
    hydrostatic_stiffness : float
        C33 [N/m].
    radiation_damping : float
        B(omega) [N*s/m], physical (positive, as stored in HydrodynamicCoefficients).
    excitation_complex : complex
        Complex excitation force amplitude [N].
        = wave_amplitude * (F_real + i*F_imag) from Capytaine.
        Consistent with excitation.py: F(t) = Re[F_exc * e^{+i*omega*t}].

    Returns
    -------
    complex
        Complex displacement amplitude X(omega) [m] under e^{+i*omega*t} convention.
        |X| is the displacement amplitude; angle(X) is the phase such that
        x(t) = |X| * cos(omega*t + angle(X)).
    """
    Z = (hydrostatic_stiffness
         - omega**2 * (mass + added_mass)
         + 1j * omega * radiation_damping)
    return excitation_complex / Z


def frequency_domain_rao_pto(
    omega: float,
    mass: float,
    added_mass: float,
    hydrostatic_stiffness: float,
    radiation_damping: float,
    excitation_complex: complex,
    pto: Optional[PTOParameters] = None,
) -> complex:
    """
    Frequency-domain RAO with optional fixed linear PTO.

    DERIVATION (e^{+i*omega*t} convention, consistent with Module 2.3C):

    The time-domain equation with PTO is:

        (M + A_inf) x_ddot + (C33 + K_pto) x + convolution + B_pto v = F_exc

    Under e^{+i*omega*t}: v -> i*omega*X, x_ddot -> -omega^2*X.
    The convolution term gives i*omega*B*X - omega^2*(A-A_inf)*X (as in 2.3C).
    The PTO terms give: B_pto*(i*omega)*X + K_pto*X.

    Collecting:
        [(C33 + K_pto) - omega^2*(M + A(omega)) + i*omega*(B(omega) + B_pto)] * X
        = F_exc

    FD impedance with PTO:

        Z_PTO = (C33 + K_pto) - omega^2*(M + A(omega)) + i*omega*(B(omega) + B_pto)

    With pto=None (or B_pto=K_pto=0) this reduces to the no-PTO impedance.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    mass : float
        Buoy mass M [kg].
    added_mass : float
        BEM added mass A(omega) [kg].  NOT A_inf.
    hydrostatic_stiffness : float
        C33 [N/m].
    radiation_damping : float
        B(omega) [N·s/m], physical (positive).
    excitation_complex : complex
        Complex excitation force [N].
    pto : PTOParameters or None, optional
        PTO parameters.  None = no PTO (recovers frequency_domain_rao).

    Returns
    -------
    complex
        Complex displacement amplitude X(omega) [m].
    """
    B_pto = pto.damping   if pto is not None else 0.0
    K_pto = pto.stiffness if pto is not None else 0.0
    Z = ((hydrostatic_stiffness + K_pto)
         - omega**2 * (mass + added_mass)
         + 1j * omega * (radiation_damping + B_pto))
    return excitation_complex / Z
