"""Projective kernel multiples and one batch inversion per Velu quotient.

An implementation optimization of the same short-Weierstrass Velu formulas.
The differential-addition formula is valid even when x(P-Q)=0; it divides
only when all projective coordinates are normalized together at the end.
All denominators must be units. This condition holds for the requested first
(ell-1)/2 multiples of a nonzero odd-prime-order point, in characteristic
different from ell, provided the input lifts that residue point.

The module is independently callable on finite-field or capped-absolute Zq
elements.
"""

from __future__ import annotations


def projective_double(a, b, point):
    """(X:Z) of [2]P on y^2=x^3+a*x+b."""
    X, Z = point
    X2, Z2 = X * X, Z * Z
    Z3 = Z2 * Z
    return (X2 - a * Z2) ** 2 - 8 * b * X * Z3, 4 * Z * (X * X2 + a * X * Z2 + b * Z3)


def projective_diffadd(a, b, P, Q, difference):
    """Return x(P+Q) given P,Q and x(P-Q), all projectively."""
    Xp, Zp = P
    Xq, Zq = Q
    Xd, Zd = difference
    PZQZ = Zp * Zq
    xpzq, xqzp = Xp * Zq, Xq * Zp
    U = xpzq + xqzp
    V = Xp * Xq + a * PZQZ
    W2 = (xpzq - xqzp) ** 2
    return Zd * (2 * U * V + 4 * b * PZQZ**2) - Xd * W2, Zd * W2


def batch_inverse_units(values):
    """Invert a nonempty list of units using exactly one ring inversion."""
    if not values:
        return []
    one = values[0].parent().one()
    prefixes = []
    total = one
    for value in values:
        if not value.is_unit():
            raise ArithmeticError("A projective kernel denominator is not a unit")
        prefixes.append(total)
        total *= value
    inverse = ~total
    output = [None] * len(values)
    for index in range(len(values) - 1, -1, -1):
        output[index] = inverse * prefixes[index]
        inverse *= values[index]
    return output


def x_multiples(a, b, x, howmany):
    """x(P),...,x([howmany]P), normalized by a single batch inversion."""
    howmany = int(howmany)
    if howmany < 1:
        raise ValueError("At least one multiple is required")
    if x is None:
        return []
    if howmany == 1:
        return [x]
    one = x.parent().one()
    points = [(x, one), projective_double(a, b, (x, one))]
    while len(points) < howmany:
        # If the next index is n, P=(n-1)G, Q=G, P-Q=(n-2)G.
        points.append(projective_diffadd(a, b, points[-1], points[0], points[-2]))
    inverses = batch_inverse_units([Z for _, Z in points[1:]])
    return [x] + [point[0] * inverse for point, inverse in zip(points[1:], inverses)]


def quotient_and_sum(a, b, x, ell):
    """Normalized Velu quotient coefficients and sum of all nonzero kernel x."""
    ell = int(ell)
    if ell < 2 or (ell != 2 and ell % 2 == 0):
        raise ValueError("Expected degree two or an odd prime")
    v = w = S = a.parent().zero()
    for xQ in x_multiples(a, b, x, ell // 2):
        xQ2 = xQ * xQ
        gxQ = 3 * xQ2 + a
        uQ = 4 * ((xQ2 + a) * xQ + b)
        vQ = 2 * gxQ if ell >= 3 else gxQ
        v += vQ
        w += uQ + xQ * vQ
        S += 2 * xQ if ell >= 3 else xQ
    return a - 5 * v, b - 7 * w, S
