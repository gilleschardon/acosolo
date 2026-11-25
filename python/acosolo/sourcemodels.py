#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acoustic steering vector models for free-field sources.

Computes the complex propagation from source locations to microphone
positions using the free-field Green's function (1/r * exp(-ikr)).
Supports both monopole and 2D (cylindrical) propagation.

Also supports spherical harmonic source directivities with full Hankel
function radial decay (near-field capable).

Key functions
-------------
freefield : 3D free-field propagation (spherical waves)
freefield2D : 2D free-field propagation (fixed source height Z)
freefield_sh : 3D propagation with spherical harmonic source directivity
freefield_sph_dictionary : Full SH steering vector dictionary basis
pw_rigid_sphere : Plane-wave field on a rigid sphere (rigid-sphere array model)
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1, spherical_jn

try:
    from scipy.special import sph_harm_y as _sph_harm_y
except ImportError:  # scipy < 1.15
    from scipy.special import sph_harm as _sph_harm_old
    _sph_harm_y = None


def _sph_harm_safe(m: int, n: int, phi: np.ndarray, theta: np.ndarray
                   ) -> np.ndarray:
    """Wrapper for spherical harmonics with scipy version compatibility.

    scipy.special.sph_harm(m, n, phi, theta) is deprecated in 1.15 and
    removed in 1.17. The replacement sph_harm_y(n, m, theta, phi) has a
    different arg order.
    """
    if _sph_harm_y is not None:
        return _sph_harm_y(n, m, theta, phi)
    return _sph_harm_old(m, n, phi, theta)


def freefield(PX: np.ndarray, PS: np.ndarray, k: float,
              normalized: bool = False) -> np.ndarray:
    """3D free-field propagation (spherical waves).
    
    Parameters
    ----------
    PX : ndarray (Nm, 3)
        Microphone positions.
    PS : ndarray (3,) or (Ns, 3)
        Source positions. 1-D for a single source, 2-D for multiple sources.
    k : float
        Wavenumber.
    normalized : bool, optional
        If True, normalize output so that sum(|D|^2) = 1 per source.
        
    Returns
    -------
    D : ndarray (Nm,) or (Nm, Ns)
        Complex steering vector(s).
    """
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[:, 1] - PS[1]
        dz = PX[:, 2] - PS[2]
    else:
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T
        dz = PX[:, 2:3] - PS[:, 2:3].T

    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    D = np.exp(-1j * k * d) / d
    
    if normalized:
        D = D / np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        
    return D


def freefield2D(PX: np.ndarray, PS: np.ndarray, Z: float, k: float,
                normalized: bool = False) -> np.ndarray:
    """2D free-field propagation (fixed source height Z).
    
    Parameters
    ----------
    PX : ndarray (Nm, 2)
        Microphone x,y positions.
    PS : ndarray (2,) or (Ns, 2)
        Source x,y positions.
    Z : float
        Source height above mic plane.
    k : float
        Wavenumber.
    normalized : bool, optional
        If True, normalize output.
        
    Returns
    -------
    D : ndarray (Nm,) or (Nm, Ns)
    """
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[:, 1] - PS[1]
    else:
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T

    d = np.sqrt(dx * dx + dy * dy + Z * Z)
    D = np.exp(-1j * k * d) / d
    
    if normalized:
        D = D / np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        
    return D


def freefield_sh(PX: np.ndarray, PS: np.ndarray, k: float,
                 coeffs: np.ndarray, max_order: int | None = None,
                 axis: str = 'z', normalized: bool = False
                 ) -> np.ndarray:
    """3D propagation with spherical harmonic source directivity.
    
    Computes the complex steering vector for a source with arbitrary
    spherical harmonic directivity, using full Hankel function radial
    decay (near-field capable).
    
    Parameters
    ----------
    PX : ndarray (Nm, 3)
        Microphone positions.
    PS : ndarray (3,) or (Ns, 3)
        Source positions. 1-D for a single source, 2-D for multiple sources.
    k : float
        Wavenumber.
    coeffs : ndarray (max_order+1, 2*max_order+1)
        Spherical harmonic emission coefficients.
        coeffs[n, m+max_order] gives the coefficient c_{n,m}.
    max_order : int, optional
        Maximum SH order. Order N has (N+1)^2 coefficients. If None
        (default), it is inferred from ``coeffs.shape[0] - 1``; if given,
        it must match the shape of ``coeffs``.
    axis : {'x', 'y', 'z'}, optional
        Polar axis for spherical angles (default 'z').
    normalized : bool, optional
        If True, normalize the output so that sum(|D|^2) = 1 (default False).
        
    Returns
    -------
    D : ndarray (Nm,) or (Nm, Ns)
        Complex steering vector(s).
        
    Notes
    -----
    The steering vector is computed as:
    
        D_mic = sum_{n=0}^{N} sum_{m=-n}^{n} c_{n,m} * h_n^{(2)}(k*r) * Y_{n,m}(phi, theta)
    
    where h_n^{(2)} is the spherical Hankel function of the second kind,
    and Y_{n,m} uses scipy's convention sph_harm_y(n, m, theta, phi).
    
    Spherical angles are computed from the direction r = PX - PS:
    - theta = arccos(r_axis / |r|)  (polar angle from the axis)
    - phi = arctan2(r_perp2, r_perp1)  (azimuth around the axis)
    
    To match freefield()'s exp(-ikr)/r convention for a monopole,
    set c_00 = -i*k*sqrt(4*pi). With c_00=1, the result is
    h_0^(2)(kr)*Y_00, which differs by a factor of -i*k*sqrt(4*pi).
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.ndim != 2:
        raise ValueError(f"coeffs must be 2-D, got shape {coeffs.shape}")
    n_coef = coeffs.shape[0] - 1
    if coeffs.shape != (n_coef + 1, 2 * n_coef + 1):
        raise ValueError(
            f"coeffs must have shape (N+1, 2N+1), got {coeffs.shape}")
    if max_order is None:
        max_order = n_coef
    elif max_order != n_coef:
        raise ValueError(
            f"max_order={max_order} does not match coeffs of shape "
            f"{coeffs.shape} (order {n_coef})")

    if PS.ndim == 1:
        n_sources = 1
        PS = PS.reshape(1, 3)
    else:
        n_sources = PS.shape[0]
    
    PX = np.asarray(PX)
    n_mics = PX.shape[0]
    
    PS_b = PS[:, np.newaxis, :]       # (n_sources, 1, 3)
    PX_b = PX[np.newaxis, :, :]       # (1, n_mics, 3)
    
    dx = PX_b[:, :, 0] - PS_b[:, :, 0]
    dy = PX_b[:, :, 1] - PS_b[:, :, 1]
    dz = PX_b[:, :, 2] - PS_b[:, :, 2]
    d = np.sqrt(dx * dx + dy * dy + dz * dz)  # (n_sources, n_mics)
    
    k_b = k
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax_idx = axis_map[axis]
    
    if ax_idx == 0:
        r_perp1, r_perp2, r_ax = dy, dz, dx
    elif ax_idx == 1:
        r_perp1, r_perp2, r_ax = dx, dz, dy
    else:
        r_perp1, r_perp2, r_ax = dx, dy, dz
    
    phi = np.arctan2(r_perp2, r_perp1)
    theta = np.arccos(np.clip(r_ax / d, -1, 1))
    
    D_out = np.zeros((n_mics, n_sources), dtype=np.complex128)
    n_max = max_order
    
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            if abs(coeffs[n, m + n_max]) < np.finfo(float).eps:
                continue
            c = coeffs[n, m + n_max]
            Y_nm = _sph_harm_safe(m, n, phi, theta)
            order_half = n + 0.5
            x_r = k_b * d
            h_n = np.sqrt(np.pi / (2 * x_r)) * hankel1(order_half, x_r).conj()
            D_out += c * (Y_nm.T * h_n.T)
    
    if n_sources == 1:
        D_out = D_out.ravel()
    
    if normalized:
        norm = np.sqrt(np.real(np.sum(D_out * np.conj(D_out), axis=0)))
        norm = np.where(norm < np.finfo(float).eps, 1.0, norm)
        D_out = D_out / norm
    
    return D_out


def freefield_sph_dictionary(
    PX: np.ndarray,
    PS: np.ndarray,
    k: float,
    M,
    kind: str = 'complex',
    axis: str = 'z',
    normalized: bool = False,
    m0_only: bool = False,
) -> np.ndarray:
    """Spherical harmonic steering vector dictionary.

    Returns a (Nm, Ns, Z) array containing the complex free-field
    propagation for every spherical harmonic basis mode up to the
    requested order(s), for every source position in *PS* and every
    microphone in *PX*.

    Parameters
    ----------
    PX : ndarray (Nm, 3)
        Microphone positions.
    PS : ndarray (3,) or (Ns, 3)
        Source positions. 1-D for a single source, 2-D for multiple.
    k : float
        Wavenumber.
    M : int or list of int
        Maximum SH order (all modes 0..M included) or list of specific
        orders to include.  Total basis size Z = sum((m+1)**2 for m in orders).
    kind : {'complex', 'real'}, optional
        'complex' uses the standard complex-valued spherical harmonics
        Y_n^m.  'real' uses the orthonormal real SH basis
        (sqrt(2)*Re / sqrt(2)*Im for m != 0, Y_n^0 for m == 0).
    axis : {'x', 'y', 'z'}, optional
        Polar axis for spherical angles (default 'z').
    normalized : bool, optional
        If True, orthogonalize (QR) each (Nm, Z) sub-array so that the
        Z columns are orthonormal for every mic-source pair.
    m0_only : bool, optional
        If True, include only the m=0 (axisymmetric) modes.
        This reduces the basis to Z = M + 1 modes.

    Returns
    -------
    D : ndarray
        If PS was 1-D: (Nm, Z).
        If PS was 2-D: (Nm, Ns, Z).
        Each column (or column-pair for real SH) is the steering vector
        for one spherical harmonic basis mode.

    Notes
    -----
    The basis function for index z is

        D_z(mic, src) = h_{n_z}^{(2)}(k r_{src, mic}) * Y_{n_z, m_z}

    where h^{(2)} is the spherical Hankel function of the second kind
    (outgoing wave) and Y_{n,m} is the spherical harmonic evaluated
    at the direction from source to microphone.

    For kind='complex' the ordering of the Z basis indices is
    lexicographic (n, m) with n=0..M and m=-n..n.

    For kind='real' the ordering is n=0..M, then m=0, then pairs
    (m>0, m<0) mapped to cos/sin linear combinations.

    When ``m0_only=True`` only the m=0 modes are included for each
    order n, giving Z = M+1 columns (or len(M)+1 if M is a list).
    """
    # --- determine orders ---
    if isinstance(M, (list, tuple)):
        orders = sorted(M)
    else:
        orders = list(range(int(M) + 1))

    n_mics = PX.shape[0]
    PX = np.asarray(PX)

    # --- build linearised (n, m) index list ---
    index_list = []  # list of (n, m, kind_tag)
    if kind == 'complex':
        for n in orders:
            if m0_only:
                m_values = [0]
            else:
                m_values = range(-n, n + 1)
            for m in m_values:
                index_list.append((n, m, 'c'))
    elif kind == 'real':
        for n in orders:
            if m0_only:
                # m=0 mode only (already real)
                index_list.append((n, 0, 'r'))
            else:
                index_list.append((n, 0, 'r'))
                for m in range(1, n + 1):
                    index_list.append((n, +m, 'c'))  # cos mode
                    index_list.append((n, -m, 's'))  # sin mode
    else:
        raise ValueError(f"kind must be 'complex' or 'real', got '{kind}'")

    Z = len(index_list)

    # --- reshape PS to 2-D ---
    single_source = PS.ndim == 1
    if single_source:
        PS = np.asarray(PS).reshape(1, 3)
    else:
        PS = np.asarray(PS)
    n_sources = PS.shape[0]

    # --- geometry: distances and angles ---
    PS_b = PS[:, np.newaxis, :]   # (n_sources, 1, 3)
    PX_b = PX[np.newaxis, :, :]   # (1, n_mics, 3)

    dx = PX_b[:, :, 0] - PS_b[:, :, 0]  # (ns, nm)
    dy = PX_b[:, :, 1] - PS_b[:, :, 1]
    dz = PX_b[:, :, 2] - PS_b[:, :, 2]
    d = np.sqrt(dx * dx + dy * dy + dz * dz)  # (ns, nm)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax_idx = axis_map[axis]

    if ax_idx == 0:
        r_perp1, r_perp2, r_ax = dy, dz, dx
    elif ax_idx == 1:
        r_perp1, r_perp2, r_ax = dx, dz, dy
    else:
        r_perp1, r_perp2, r_ax = dx, dy, dz

    phi = np.arctan2(r_perp2, r_perp1)  # (ns, nm)
    theta = np.arccos(np.clip(r_ax / d, -1, 1))  # (ns, nm)

    # --- precompute Hankel factors per order ---
    # hankel_factors[n] has shape (ns, nm)
    max_n = max(orders)
    hankel_factors = {}
    x_r = k * d  # (ns, nm)
    for n in range(max_n + 1):
        order_half = n + 0.5
        hankel_factors[n] = (np.sqrt(np.pi / (2 * x_r)) *
                             hankel1(order_half, x_r).conj())

    # --- evaluate basis ---
    D = np.zeros((n_mics, n_sources, Z), dtype=np.complex128)

    for z, (n, m, tag) in enumerate(index_list):
        # spherical harmonic
        Y = _sph_harm_safe(m, n, phi, theta)  # (ns, nm)

        if tag == 'c' and kind == 'real':
            # real SH: sqrt(2)*Re(Y)
            val = np.sqrt(2.0) * np.real(Y)
        elif tag == 's' and kind == 'real':
            # real SH: sqrt(2)*Im(Y)
            val = np.sqrt(2.0) * np.imag(Y)
        else:
            # complex or m=0 in real mode
            val = Y

        # radial part
        h_n = hankel_factors[n]  # (ns, nm)

        # D[:, :, z] = (Y * h_n).T → shape (nm, ns)
        D[:, :, z] = (val * h_n).T

    # --- reshape for single source ---
    if single_source:
        D = D[:, 0, :]  # (nm, Z)

    # --- orthogonalise if requested ---
    if normalized:
        if n_mics < Z:
            raise ValueError(
                f"normalized=True needs at least as many microphones as "
                f"basis functions to orthonormalize them (Nm={n_mics}, "
                f"Z={Z})")
        D_new = np.empty_like(D)
        if D.ndim == 3:
            for s in range(D.shape[1]):
                Q, _ = np.linalg.qr(D[:, s, :])  # Q: (nm, Z)
                D_new[:, s, :] = Q
        else:
            # D is (nm, Z)
            Q, _ = np.linalg.qr(D)
            D_new = Q
        D = D_new

    return D


def pw_rigid_sphere(diameter: float,
                    mic_elev: np.ndarray,
                    mic_az: np.ndarray,
                    pw_elev: np.ndarray,
                    pw_az: np.ndarray,
                    k: float,
                    N: int | None = None,
                    normalized: bool = False) -> np.ndarray:
    """Plane-wave field on a rigid sphere (rigid-sphere microphone array model).

    Computes the total complex pressure (incident plane wave + field scattered
    by a rigid sphere) evaluated at microphone positions on the sphere surface,
    for one or more incident plane-wave directions.

    A rigid sphere enforces the Neumann boundary condition
    (d p / dr = 0 at r = a).  The total field is expanded in spherical
    harmonics and truncated at order N.

    Parameters
    ----------
    diameter : float
        Sphere diameter.  The radius is a = diameter / 2.
    mic_elev : ndarray (Nm,)
        Polar angle (from the +z axis) of each microphone on the sphere,
        in radians.
    mic_az : ndarray (Nm,)
        Azimuth (around the +z axis, phi = arctan2(y, x)) of each microphone,
        in radians.
    pw_elev : float or ndarray (Np,)
        Polar angle(s) of the plane-wave source direction(s), in radians.
        Scalar for a single plane wave, array for multiple.  The wave is
        assumed to come FROM this direction (travel toward the sphere center).
    pw_az : float or ndarray (Np,)
        Azimuth(s) of the plane-wave source direction(s), in radians.
    k : float
        Wavenumber.
    N : int, optional
        Maximum spherical-harmonic order for the series truncation.
        If None, N = ceil(k*a + 5*(k*a)**(1/3)) + 5 is used (relative
        truncation error below ~1e-8 up to k*a ~ 40).
    normalized : bool, optional
        If True, normalize each plane-wave steering vector so that
        sum(|D|^2) over the microphones equals 1 (default False).

    Returns
    -------
    D : ndarray (Nm,) or (Nm, Np)
        Complex total-pressure steering vector(s) at the microphones.
        Shape (Nm,) for a single plane wave, (Nm, Np) otherwise.

    Notes
    -----
    With a = diameter / 2 and the addition-theorem plane-wave expansion, the
    surface pressure for a plane wave from direction (theta_i, phi_i) is

        p(a, theta, phi) =
            sum_{n=0}^{N} 4*pi*i^n * R_n(k*a)
            * sum_{m=-n}^{n} Y_n^m(theta, phi) * conj(Y_n^m(theta_i, phi_i))

    where Y_n^m are the (scipy) spherical harmonics and

        R_n(ka) = j_n(ka) - ( j_n'(ka) / h_n^{(2)'}(ka) ) * h_n^{(2)}(ka)

    with j_n the spherical Bessel function and h_n^{(2)} the spherical Hankel
    function of the second kind (outgoing-wave convention, matching the rest
    of this module).  In the small-ka limit the result reduces to the plain
    incident plane wave.

    Raises
    ------
    ValueError
        If k * diameter / 2 <= 0 (degenerate zero-size / zero-frequency case;
        use the free-field plane wave instead).
    """
    a = diameter / 2.0
    ka = k * a
    if ka <= 0:
        raise ValueError(
            "pw_rigid_sphere requires k > 0 and diameter > 0 (ka > 0); "
            "for a zero-size sphere use the free-field plane wave instead."
        )

    if N is None:
        # The required order grows like ka + O(ka^(1/3)); this rule keeps
        # the relative truncation error below ~1e-8 for ka up to ~40.
        N = int(np.ceil(ka + 5.0 * ka ** (1.0 / 3.0))) + 5
    else:
        N = int(N)
    if N < 0:
        raise ValueError("N must be >= 0.")

    single_pw = (np.ndim(pw_elev) == 0) and (np.ndim(pw_az) == 0)

    mic_elev = np.atleast_1d(np.asarray(mic_elev, dtype=float))
    mic_az = np.atleast_1d(np.asarray(mic_az, dtype=float))
    if mic_elev.shape != mic_az.shape:
        mic_elev, mic_az = np.broadcast_arrays(mic_elev, mic_az)
    n_mics = mic_elev.shape[0]

    pw_elev = np.atleast_1d(np.asarray(pw_elev, dtype=float))
    pw_az = np.atleast_1d(np.asarray(pw_az, dtype=float))
    if pw_elev.shape != pw_az.shape:
        pw_elev, pw_az = np.broadcast_arrays(pw_elev, pw_az)
    n_pw = pw_elev.shape[0]

    # --- radial factors R_n(ka) for n = 0..N ---
    # h_n^(2)(x) using this module's outgoing-wave convention.
    h = np.empty(N + 1, dtype=np.complex128)
    for n in range(N + 1):
        h[n] = np.sqrt(np.pi / (2.0 * ka)) * hankel1(n + 0.5, ka).conj()

    # h_{-1}^{(2)}(x) = exp(-i x) / x, used by the derivative recurrence at n=0.
    h_m1 = np.exp(-1j * ka) / ka
    h_p = np.empty(N + 1, dtype=np.complex128)
    for n in range(N + 1):
        h_prev = h_m1 if n == 0 else h[n - 1]
        h_p[n] = h_prev - (n + 1) / ka * h[n]

    # --- accumulate over orders ---
    D = np.zeros((n_mics, n_pw), dtype=np.complex128)
    for n in range(N + 1):
        jn = spherical_jn(n, ka)
        jnp = spherical_jn(n, ka, derivative=True)
        Rn = jn - (jnp / h_p[n]) * h[n]
        coef = 4.0 * np.pi * (1j ** n) * Rn
        if coef == 0:
            continue

        # M_n: (n_mics, 2n+1) columns = Y_n^m evaluated at the microphones
        M_n = np.empty((n_mics, 2 * n + 1), dtype=np.complex128)
        # P_n: (n_pw, 2n+1) columns = conj(Y_n^m) evaluated at the source dirs
        P_n = np.empty((n_pw, 2 * n + 1), dtype=np.complex128)
        for j, m in enumerate(range(-n, n + 1)):
            M_n[:, j] = _sph_harm_safe(m, n, mic_az, mic_elev)
            P_n[:, j] = np.conj(_sph_harm_safe(m, n, pw_az, pw_elev))

        D += coef * (M_n @ P_n.T)

    if single_pw:
        D = D[:, 0]

    if normalized:
        norm = np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        norm = np.where(norm < np.finfo(float).eps, 1.0, norm)
        D = D / norm

    return D
