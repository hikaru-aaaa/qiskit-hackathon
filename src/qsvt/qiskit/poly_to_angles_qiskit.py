"""
Pure NumPy/SciPy implementation of poly_to_angles for QSVT
Based on PennyLane's implementation but without PennyLane dependencies

This module provides a complete implementation of the polynomial-to-angles conversion
for Quantum Singular Value Transformation (QSVT), using only NumPy and SciPy.

Key algorithms:
- Chebyshev polynomial transformation (numpy.polynomial.chebyshev)
- Complementary polynomial computation via root-finding
- QSP angle computation adapted from Generalized-QSP
- Angle transformation between QSP and QSVT conventions

References:
- arXiv:2105.02859 - A Grand Unification of Quantum Algorithms
- arXiv:2406.04246 - Generalized Quantum Signal Processing
- arXiv:2308.01501 - Complementary polynomial methods
- PennyLane: pennylane/templates/subroutines/qsvt.py

Author: Ported from PennyLane to pure NumPy/SciPy
License: Same as PennyLane (Apache License 2.0)
"""

import numpy as np
from numpy.polynomial import Polynomial, chebyshev


def _complementary_poly(poly_coeffs):
    """
    Computes the complementary polynomial Q given a polynomial P.

    The polynomial Q is complementary to P if it satisfies:
    |P(e^{iθ})|² + |Q(e^{iθ})|² = 1, ∀ θ ∈ [0, 2π]

    Based on arXiv:2308.01501

    Args:
        poly_coeffs: Coefficients of the complex polynomial P

    Returns:
        Coefficients of the complementary polynomial Q
    """
    poly_degree = len(poly_coeffs) - 1

    # Build R(z) = z^degree * (1 - conj(P(1/z)) * P(z))
    R = Polynomial.basis(poly_degree) - Polynomial(poly_coeffs) * Polynomial(
        np.conj(poly_coeffs[::-1])
    )
    r_roots = R.roots()

    inside_circle = [root for root in r_roots if np.abs(root) <= 1]
    outside_circle = [root for root in r_roots if np.abs(root) > 1]

    scale_factor = np.sqrt(np.abs(R.coef[-1] * np.prod(outside_circle)))
    Q_poly = scale_factor * Polynomial.fromroots(inside_circle)

    return Q_poly.coef


def _compute_ry_matrix(theta):
    """
    Compute RY rotation matrix.

    RY(θ) = [[cos(θ/2), -sin(θ/2)],
             [sin(θ/2),  cos(θ/2)]]

    Args:
        theta: Rotation angle

    Returns:
        2x2 rotation matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _compute_qsp_angle(poly_coeffs):
    """
    Computes QSP angles given polynomial coefficients.

    Based on arXiv:2406.04246 approach for Generalized-QSP.

    Args:
        poly_coeffs: Coefficients of the input polynomial F

    Returns:
        QSP angles corresponding to the input polynomial F
    """
    parity = (len(poly_coeffs) - 1) % 2

    # Convert to Chebyshev basis and reorder
    P = np.concatenate([
        np.zeros(len(poly_coeffs) // 2),
        chebyshev.poly2cheb(poly_coeffs)[parity::2]
    ]) * (1 - 1e-12)

    complementary = _complementary_poly(P)

    polynomial_matrix = np.array([P, complementary])
    num_terms = polynomial_matrix.shape[1]
    rotation_angles = np.zeros(num_terms)

    # Adaptation of Algorithm 1 of arXiv:2308.01501
    for idx in range(num_terms - 1, -1, -1):
        poly_a, poly_b = polynomial_matrix[:, idx]
        rotation_angles[idx] = np.arctan2(poly_b.real, poly_a.real)

        rotation_op = _compute_ry_matrix(-2 * rotation_angles[idx])

        updated_poly_matrix = rotation_op @ polynomial_matrix
        polynomial_matrix = np.array([
            updated_poly_matrix[0][1 : idx + 1],
            updated_poly_matrix[1][0:idx]
        ])

    return rotation_angles


def transform_angles(angles, routine1, routine2):
    """
    Converts angles between QSP and QSVT routines.

    Based on Appendix A.2 of arXiv:2105.02859.
    QSVT is equivalent to taking the reflection convention of QSP.

    Args:
        angles: Angles to be transformed
        routine1: Current routine ("QSP" or "QSVT")
        routine2: Target routine ("QSP" or "QSVT")

    Returns:
        Transformed angles as an array
    """
    angles = np.asarray(angles)

    if routine1 == routine2:
        return angles

    if routine1 == "QSP" and routine2 == "QSVT":
        # QSP to QSVT transformation
        num_angles = len(angles)
        update_vals = np.empty(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + num_angles % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4

        return angles + update_vals

    if routine1 == "QSVT" and routine2 == "QSP":
        # QSVT to QSP transformation
        num_angles = len(angles)
        update_vals = np.empty(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + num_angles % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4

        return angles - update_vals

    raise AssertionError(
        f"Invalid conversion. The conversion between {routine1} --> {routine2} is not defined."
    )


def poly_to_angles(poly, routine="QSVT", angle_solver="root-finding"):
    """
    Computes the angles needed to implement a polynomial transformation with QSVT.

    The polynomial P(x) = Σ aₙxⁿ must satisfy |P(x)| ≤ 1 for x ∈ [-1, 1].
    For QSVT, coefficients must be real and exponents must be all even or all odd.

    Based on arXiv:2105.02859

    Args:
        poly: Coefficients of the polynomial ordered from lowest to highest power
        routine: The routine ("QSP", "QSVT", or "GQSP")
        angle_solver: Method used to calculate angles ("root-finding" or "iterative")

    Returns:
        Computed angles for the specified routine

    Raises:
        AssertionError: If poly is not valid or routine is not supported

    Example:
        >>> poly = np.array([0, -1.5, 0, 2.5])  # P(x) = -1.5x + 2.5x³
        >>> qsvt_angles = poly_to_angles(poly, "QSVT")
        >>> print(qsvt_angles)
        [-2.356...  1.571...  0.912... -0.126...]
    """
    # Trim trailing zeros
    poly = np.trim_zeros(np.asarray(poly, dtype=float), trim='b')

    if len(poly) == 1:
        raise AssertionError("The polynomial must have at least degree 1.")

    # Check that |P(x)| ≤ 1 at critical points
    for x in [-1, 0, 1]:
        poly_val = np.abs(sum(coeff * x**i for i, coeff in enumerate(poly)))
        if poly_val > 1:
            raise AssertionError(
                f"The polynomial must satisfy |P(x)| ≤ 1 for all x in [-1, 1]. "
                f"Got |P({x})| = {poly_val}"
            )

    if routine in ["QSVT", "QSP"]:
        # Check parity: all odd or all even entries must be zero
        if not (np.isclose(np.sum(np.abs(poly[::2])), 0.0) or
                np.isclose(np.sum(np.abs(poly[1::2])), 0.0)):
            raise AssertionError(
                "The polynomial has no definite parity. "
                "All odd or even entries in the array must take a value of zero."
            )

        # Check that polynomial is real
        if not np.allclose(np.asarray(poly, dtype=np.complex128).imag, 0):
            raise AssertionError("Array must not have an imaginary part")

    if routine == "QSVT":
        if angle_solver != "root-finding":
            raise AssertionError(
                "Only 'root-finding' angle solver is currently implemented"
            )
        # Compute QSP angles then transform to QSVT
        qsp_angles = _compute_qsp_angle(poly)
        return transform_angles(qsp_angles, "QSP", "QSVT")

    if routine == "QSP":
        if angle_solver != "root-finding":
            raise AssertionError(
                "Only 'root-finding' angle solver is currently implemented"
            )
        return _compute_qsp_angle(poly)

    raise AssertionError("Invalid routine. Valid values are 'QSP' and 'QSVT'")

