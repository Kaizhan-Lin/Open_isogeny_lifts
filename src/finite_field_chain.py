"""Evaluate isogeny chains on short Weierstrass curves over finite rings.

Compute the kernel polynomials once over F_(p^2) and reuse them over
extension fields or polynomial quotient rings via a base-field embedding.
For an affine point (x, y), return the image x-coordinate and the factor
that multiplies y, preserving the sign of the image point.

Raise ArithmeticError if a denominator is zero or otherwise noninvertible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from sage.all import GF, PolynomialRing, ZZ, pari

from batched_velu import x_multiples


def load_fixture(path):
    """Read the existing normalized fixture without importing lifting code."""
    fixture = json.loads(Path(path).read_text())
    p = ZZ(fixture["p"])
    F = GF(p * p, "i", modulus=[1, 0, 1])

    def decode(pair):
        return F(ZZ(pair[0])) + F.gen() * ZZ(pair[1])

    data = [
        (
            decode(row["a"]),
            decode(row["b"]),
            ZZ(row["ell"]),
            decode(row["u" if int(row["ell"]) == 1 else "x"]),
        )
        for row in fixture["steps"]
    ]
    final = tuple(decode(fixture["final"][key]) for key in ("a", "b"))
    if data[0][:2] != final:
        raise ValueError("Decoded chain does not close")
    degree = ZZ(math.prod(int(row[2]) for row in data))
    if degree != ZZ(fixture["degree"]):
        raise ValueError("Degree mismatch")
    return fixture, F, data, final, degree


def compact_chain(data, final):
    """Absorb intermediate scalings while retaining the signed final scaling."""
    F = data[0][0].parent()
    w = F.one()
    output = []
    for a, b, ell, xu in data:
        if ell == 1:
            w *= xu
        else:
            output.append((a * w**4, b * w**6, ell, xu * w**2))
    output.append((final[0] * w**4, final[1] * w**6, ZZ.one(), w))
    return output


def _product_tree(values, one):
    values = list(values)
    while len(values) > 1:
        values = [
            values[j] * values[j + 1] if j + 1 < len(values) else values[j]
            for j in range(0, len(values), 2)
        ]
    return values[0] if values else one


def _horner(coefficients, x):
    """Coefficients are descending and already embedded in x.parent()."""
    result = coefficients[0]
    for coefficient in coefficients[1:]:
        result = result * x + coefficient
    return result


def _pari_field_map(source, target, embedding):
    """Find a native map matching the caller's actual generator image."""
    if not target.is_field() or not target.is_finite():
        raise TypeError("The PARI backend requires a finite field target")
    generator = pari(source.gen())
    desired = pari(embedding(source.gen()))
    mapping = pari.ffembed(generator, pari(target.gen()))
    # ffembed chooses a conjugate. Preserve the explicit Sage embedding,
    # including its sign when source is F_(p^2).
    frobenius = pari.fffrobenius(generator)
    for _ in range(int(source.degree())):
        if pari.ffmap(mapping, generator) == desired:
            return mapping
        mapping = pari.ffcompomap(mapping, frobenius)
    raise ValueError("Native field map does not match the requested embedding")


class EdgeMap:
    """One normalized odd-prime Velu map with its signed y multiplier.

    For h=product(X-q), H=h'/h, J=h''/h, K=h'''/h,
      r = ell*X - 2*sum(q) - 2*f'*H + 4*f*(H^2-J),
      r'= ell - 12*X*H + 6*f'*(H^2-J)
                     + 4*f*(3*H*J-2*H^3-K).
    These follow from the paired-kernel Velu terms v_q/(X-q)+u_q/(X-q)^2,
    where v_q=2*f'(q), u_q=4*f(q).  Normalization gives y'=y*r'(x).
    """

    def __init__(self, a, b, ell, kernel_x, polynomial_ring=None, backend="horner"):
        ell = int(ell)
        if ell < 3 or ell % 2 != 1 or not ZZ(ell).is_prime():
            raise ValueError("Expected an odd prime isogeny degree")
        F = a.parent()
        if not F.is_finite() or not F.is_field():
            raise TypeError("Kernel preparation requires a finite field")
        if backend not in {"horner", "pari", "pari-horner"}:
            raise ValueError("Expected backend horner, pari, or pari-horner")
        self.backend = backend
        self.a, self.b, self.ell, self.kernel_x = a, b, ell, kernel_x
        R = polynomial_ring or PolynomialRing(F, "X")
        self._polynomial_ring = R
        xs = x_multiples(a, b, kernel_x, (ell - 1) // 2)
        self.kernel_x_sum = 2 * sum(xs, F.zero())
        self._kernel_polynomial = None
        self._pari_kernel = None
        self._coefficients = None
        if backend != "horner":
            self._pari_kernel = _product_tree(
                (pari([-pari(q), 1]).Polrev("X") for q in xs), pari(1)
            )
        else:
            X = R.gen()
            self._kernel_polynomial = _product_tree((X - q for q in xs), R.one())
            self._prepare_horner_coefficients()
        self._embedded = {}
        self._evaluation_backends = {}

    @property
    def kernel_polynomial(self):
        """Materialize a Sage polynomial only when a caller needs one."""
        if self._kernel_polynomial is None:
            F = self.a.parent()
            self._kernel_polynomial = self._polynomial_ring(
                [F(c) for c in self._pari_kernel.Vecrev()]
            )
        return self._kernel_polynomial

    def _prepare_horner_coefficients(self):
        derivatives = [self.kernel_polynomial]
        for _ in range(3):
            derivatives.append(derivatives[-1].derivative())
        F = self.a.parent()
        self._coefficients = tuple(
            tuple(reversed(poly.list())) or (F.zero(),) for poly in derivatives
        )

    def embed(self, target, embedding=None, backend=None, _native_map=None):
        """Cache coefficients under an explicit base-field embedding if needed."""
        embedding = embedding or target
        backend = backend or self.backend
        if backend in {"pari", "pari-horner"}:
            mapping = _native_map or _pari_field_map(self.a.parent(), target, embedding)
            if self._pari_kernel is None:
                self._pari_kernel = pari(self.kernel_polynomial)
            h = pari.ffmap(mapping, self._pari_kernel)
            derivatives = [h]
            for _ in range(3):
                derivatives.append(derivatives[-1].deriv("X"))
            a, b, kernel_sum = (
                target(c)
                for c in pari.ffmap(mapping, pari([self.a, self.b, self.kernel_x_sum]))
            )
            if backend == "pari":
                coefficients = pari(derivatives)
            else:
                coefficients = tuple(
                    tuple(target(c) for c in poly.Vec()) if poly else (target.zero(),)
                    for poly in derivatives
                )
            self._embedded[target] = (coefficients, a, b, kernel_sum, target(self.ell))
        elif backend == "horner":
            if self._coefficients is None:
                self._prepare_horner_coefficients()
            self._embedded[target] = (
                tuple(tuple(embedding(c) for c in cs) for cs in self._coefficients),
                embedding(self.a),
                embedding(self.b),
                embedding(self.kernel_x_sum),
                target(self.ell),
            )
        else:
            raise ValueError("Expected backend horner, pari, or pari-horner")
        self._evaluation_backends[target] = backend
        return self

    def eval_x_yfactor(self, x):
        target = x.parent()
        if target not in self._embedded:
            self.embed(target)
        coefficients, a, b, kernel_sum, ell = self._embedded[target]
        if self._evaluation_backends[target] == "pari":
            h, h1, h2, h3 = (target(c) for c in coefficients.subst("X", pari(x)))
        else:
            h, h1, h2, h3 = (_horner(cs, x) for cs in coefficients)
        if not h or (hasattr(h, "is_unit") and not h.is_unit()):
            raise ArithmeticError(
                "Evaluation meets a kernel point or nonunit denominator"
            )
        inverse_h = ~h
        H, J, K = h1 * inverse_h, h2 * inverse_h, h3 * inverse_h
        x2, H2 = x * x, H * H
        f = (x2 + a) * x + b
        f1 = 3 * x2 + a
        difference = H2 - J
        image_x = ell * x - kernel_sum - 2 * f1 * H + 4 * f * difference
        y_factor = (
            ell
            - 12 * x * H
            + 6 * f1 * difference
            + 4 * f * (3 * H * J - 2 * H2 * H - K)
        )
        return image_x, y_factor


class FiniteFieldChain:
    """A compact normalized chain followed by x/u^2,y/u^3 closure."""

    def __init__(self, data, final, progress=None, backend="horner"):
        started = time.perf_counter()
        self.data = compact_chain(data, final)
        self.base_field = self.data[0][0].parent()
        self.source = self.data[0][:2]
        self.final = final
        self.backend = backend
        R = PolynomialRing(self.base_field, "X")
        self.steps = []
        for index, (a, b, ell, x) in enumerate(self.data[:-1]):
            self.steps.append(EdgeMap(a, b, ell, x, R, backend=backend))
            if progress is not None:
                progress(
                    dict(
                        stage="prepare",
                        step=index + 1,
                        total=len(self.data) - 1,
                        ell=int(ell),
                        wall_seconds=time.perf_counter() - started,
                    )
                )
        self.closure_u = self.data[-1][3]
        if not self.closure_u:
            raise ValueError("Closure scaling must be nonzero")
        self.degree = ZZ(math.prod(edge.ell for edge in self.steps))
        self.prepare_seconds = time.perf_counter() - started
        self._closures = {}

    def embed(self, target, embedding=None, backend=None):
        """Prepare one target once; returns self for convenience."""
        embedding = embedding or target
        backend = backend or self.backend
        native_map = (
            _pari_field_map(self.base_field, target, embedding)
            if backend in {"pari", "pari-horner"}
            else None
        )
        for edge in self.steps:
            edge.embed(target, embedding, backend=backend, _native_map=native_map)
        u_inverse = ~embedding(self.closure_u)
        self._closures[target] = (u_inverse**2, u_inverse**3)
        return self

    def apply(self, x, yfactor=None, progress=None):
        """Return (tau_x(x), input_yfactor * tau_yfactor(x))."""
        target = x.parent()
        if target not in self._closures:
            self.embed(target)
        if yfactor is None:
            yfactor = target.one()
        for index, edge in enumerate(self.steps):
            try:
                x, step_factor = edge.eval_x_yfactor(x)
            except ArithmeticError as error:
                raise ArithmeticError(f"Chain step {index}: {error}") from error
            yfactor *= step_factor
            if progress is not None:
                progress(dict(stage="evaluate", step=index + 1, total=len(self.steps)))
        inverse_u2, inverse_u3 = self._closures[target]
        return x * inverse_u2, yfactor * inverse_u3

    def eval_x_yfactor(self, x):
        return self.apply(x)


def prepare_chain(data, final, *, progress=None, backend="horner"):
    return FiniteFieldChain(data, final, progress=progress, backend=backend)
