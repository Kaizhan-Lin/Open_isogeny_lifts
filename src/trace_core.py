"""Compute endomorphism traces on the same isogeny chain using PRS closed
lifting and open lifting with Gauss-Manin endpoint correction.

Both methods use the PRS p-adic arithmetic and torsion constraints.
The batched variants use batch inversion in the x-only Velu formulas.
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import sys
import time
import types

from sage.all import EllipticCurve, GF, ZZ
from sage.version import version as sage_version

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(
    os.environ.get("PRS_SOURCE", ROOT / "external" / "lifting_traces")
).resolve()
sys.path.insert(0, str(UPSTREAM))


def load_prs():
    path = UPSTREAM / "compute_trace.py"
    from sage.schemes.elliptic_curves.hom import EllipticCurveHom

    if sage_version != "10.10":
        raise RuntimeError(f"Expected SageMath 10.10, got {sage_version}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. See the PRS setup in README.md.")
    module = types.ModuleType("prs_native_sage1010")
    module.__file__ = str(path)
    module.EllipticCurveHom = EllipticCurveHom
    sys.modules[module.__name__] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


PRS = load_prs()
import hensel
import xonly


def timing():
    return time.perf_counter(), time.process_time()


def elapsed(start):
    return {
        "wall_seconds": time.perf_counter() - start[0],
        "cpu_seconds": time.process_time() - start[1],
    }


def progress(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}), flush=True)


def load_fixture(path):
    fixture = json.loads(Path(path).read_text())
    p = ZZ(fixture["p"])
    F = GF(p * p, "i", modulus=[1, 0, 1])

    def f(pair):
        return F(ZZ(pair[0])) + F.gen() * ZZ(pair[1])

    data = [
        (
            f(row["a"]),
            f(row["b"]),
            ZZ(row["ell"]),
            f(row["u" if int(row["ell"]) == 1 else "x"]),
        )
        for row in fixture["steps"]
    ]
    final = tuple(f(fixture["final"][key]) for key in ("a", "b"))
    if data[0][:2] != final:
        raise ValueError("Decoded chain does not close")
    degree = ZZ(math.prod(int(row[2]) for row in data))
    if degree != ZZ(fixture["degree"]):
        raise ValueError("Degree mismatch")
    return fixture, F, data, final, degree


def compact_chain(data, final):
    """Absorb intermediate scalings into a normalized chain, preserving its map."""
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


def make_ring(F, s, data, final):
    redc, lift = PRS.padic_ring(F, ZZ(s))
    vals = {}
    for idx, (a, b, ell, xu) in enumerate(data):
        vals[f"a{idx}"], vals[f"b{idx}"] = lift(a), lift(b)
        vals[f'{"u" if ell==1 else "x"}{idx}'] = lift(xu)
    vals[f"a{len(data)}"], vals[f"b{len(data)}"] = map(lift, final)
    return redc, lift, vals


def centered_trace(value, p, s):
    if value.precision_absolute() < s:
        raise ArithmeticError("Trace precision loss")
    coeffs = value.polynomial().list()
    if any(c.valuation() < s for c in coeffs[1:]):
        raise ArithmeticError("Trace did not land in Zp")
    modulus = int(p) ** s
    t = int(coeffs[0].lift()) % modulus if coeffs else 0
    return t if 2 * t <= modulus else t - modulus


def prs_closed(F, data, final, degree, s):
    start = timing()
    redc, lift, vals = make_ring(F, s, data, final)
    prep = timing()
    constraints = PRS._constraints(data, vals, redc)
    constraints += [
        PRS._equal(f"a{len(data)}", "a0"),
        PRS._equal(f"b{len(data)}", "b0"),
    ]
    setup_t = elapsed(prep)
    stages = []
    while True:
        (old,) = {int(v.precision_absolute()) for v in vals.values()}
        if old >= s:
            break
        clock = timing()
        vals = hensel.hensel(constraints, vals)
        row = {
            "from": old,
            "to": int(min(v.precision_absolute() for v in vals.values())),
            **elapsed(clock),
        }
        stages.append(row)
        progress("prs_stage", **row)
    u = next(iter(vals.values())).parent().one()
    for idx, (_, _, ell, _) in enumerate(data):
        if ell == 1:
            u *= vals[f"u{idx}"]
    trace = centered_trace(u + degree / u, F.characteristic(), s)
    return trace, {
        "method": "prs_released_sparse_closed",
        **elapsed(start),
        "constraint_setup": setup_t,
        "stages": stages,
        "variables": len(vals),
        "constraint_blocks": len(constraints),
    }


def mm(A, B):
    return (
        A[0] * B[0] + A[1] * B[2],
        A[0] * B[1] + A[1] * B[3],
        A[2] * B[0] + A[3] * B[2],
        A[2] * B[1] + A[3] * B[3],
    )


def ma(A, B):
    return tuple(a + b for a, b in zip(A, B))


def ms(c, A):
    return tuple(c * a for a in A)


def invm(A):
    d = A[0] * A[3] - A[1] * A[2]
    return (A[3] / d, -A[1] / d, -A[2] / d, A[0] / d)


def pmul(A, B):
    return [sum(A[j] * B[i - j] for j in range(i + 1)) for i in range(len(A))]


def padd(A, B):
    return [a + b for a, b in zip(A, B)]


def pscale(c, A):
    return [c * a for a in A]


def pinv(A):
    out = [1 / A[0]]
    for i in range(1, len(A)):
        out.append(-sum(A[j] * out[i - j] for j in range(1, i + 1)) / A[0])
    return out


def gm_convolution(a, b, at, bt, s):
    R = a.parent()
    p = R.prime()
    zero = R.zero()
    one = R.one()
    if not 1 < s < p - 1:
        raise ValueError("Unsupported Taylor precision")
    da, db = at - a, bt - b
    if min(da.valuation(), db.valuation()) < 1:
        raise ValueError("Different endpoint residues")
    aa = [a, da] + [zero] * (s - 2)
    bb = [b, db] + [zero] * (s - 2)
    a2 = pmul(aa, aa)
    a3 = pmul(a2, aa)
    b2 = pmul(bb, bb)
    ab = pmul(aa, bb)
    i2F = pscale(1 / R(2), pinv(padd(pscale(4, a3), pscale(27, b2))))
    c00 = pmul(padd(pscale(-2 * da, a2), pscale(-9 * db, bb)), i2F)
    c01 = pmul(padd(pscale(3 * da, ab), pscale(-2 * db, a2)), i2F)
    c10 = pmul(padd(pscale(9 * da, bb), pscale(-6 * db, aa)), i2F)
    C = [(c00[i], c01[i], c10[i], -c00[i]) for i in range(s)]
    X = [(one, zero, zero, one)]
    for n in range(s - 1):
        total = (zero, zero, zero, zero)
        for k in range(n + 1):
            total = ma(total, mm(X[k], C[n - k]))
        X.append(ms(1 / R(n + 1), total))
    out = (zero, zero, zero, zero)
    for value in X:
        out = ma(out, value)
    if (out[0] * out[3] - out[1] * out[2] - one).valuation() < s:
        raise ArithmeticError("Gauss-Manin determinant check failed")
    return out


def gm_compare(a, b, at, bt, s):
    """Solve q X' = X H with deg(q)<=3 and deg(H)<=2."""
    R = a.parent()
    if not 1 < s < R.prime() - 1:
        raise ValueError("Unsupported Taylor precision")
    z, o = R.zero(), R.one()
    da, db = at - a, bt - b
    if min(da.valuation(), db.valuation()) < 1:
        raise ValueError("Different endpoint residues")
    # Constant-size polynomial arithmetic, independent of requested s.
    aa, bb = [a, da, z, z], [b, db, z, z]
    a2 = pmul(aa, aa)
    a3, b2, ab = pmul(a2, aa), pmul(bb, bb), pmul(aa, bb)
    q = pscale(2, padd(pscale(4, a3), pscale(27, b2)))
    h00 = padd(pscale(-2 * da, a2), pscale(-9 * db, bb))
    h01 = padd(pscale(3 * da, ab), pscale(-2 * db, a2))
    h10 = padd(pscale(9 * da, bb), pscale(-6 * db, aa))
    H = [(h00[j], h01[j], h10[j], -h00[j]) for j in range(3)]
    iq0 = ~q[0]
    X = [(o, z, z, o)]
    out = X[0]
    for n in range(s - 1):
        total = (z, z, z, z)
        for j in range(min(2, n) + 1):
            total = ma(total, mm(X[n - j], H[j]))
        for j in range(1, min(3, n) + 1):
            total = ma(total, ms(-(n - j + 1) * q[j], X[n - j + 1]))
        value = ms(iq0 / R(n + 1), total)
        X.append(value)
        out = ma(out, value)
    if (out[0] * out[3] - out[1] * out[2] - o).valuation() < s:
        raise ArithmeticError("Short GM determinant failed")
    return out


def local_kernel_lift(a, b, x0, ell, s, lift):
    """Scalar finite-difference Newton, using the same torsion oracle as PRS."""
    R = a.parent()
    p = R.prime()
    x = lift(x0)
    k = 1
    while k < s:
        target = min(2 * k, s)
        x = hensel._lift(x, ZZ(target))
        E = EllipticCurve(R, [a.add_bigoh(target), b.add_bigoh(target)])
        value = E.division_polynomial(ell, x=x)
        shifted = E.division_polynomial(ell, x=x + (R.one() << k))
        derivative = (shifted - value) >> k
        if derivative.valuation() != 0:
            raise ArithmeticError("Nonunit kernel derivative")
        x -= ((value >> k) / derivative) << k
        if x.precision_absolute() < target:
            raise ArithmeticError("Kernel precision loss")
        k = target
    return x


def quotient_and_sum(a, b, x, ell):
    v = w = S = a.parent().zero()
    for xQ in xonly.x_multiples(a, b, x, ell // 2):
        xQ2 = xQ * xQ
        gxQ = 3 * xQ2 + a
        uQ = 4 * ((xQ2 + a) * xQ + b)
        vQ = 2 * gxQ if ell >= 3 else gxQ
        v += vQ
        w += uQ + xQ * vQ
        S += 2 * xQ if ell >= 3 else xQ
    return a - 5 * v, b - 7 * w, S


def open_gm(
    F,
    data,
    final,
    degree,
    s,
    batched=False,
    audit_kernels=False,
    *,
    compare=gm_compare,
    capture=None,
):
    start = timing()
    redc, lift, _ = make_ring(F, s, data, final)
    a, b = (hensel._lift(lift(v), ZZ(s)) for v in data[0][:2])
    a0, b0 = a, b
    R = a.parent()
    one = R.one()
    zero = R.zero()
    B = (one, zero, zero, one)
    kernel_wall = quotient_wall = iso_wall = kernel_check_wall = 0.0
    quotient = quotient_and_sum
    if batched:
        from batched_velu import quotient_and_sum as quotient
    checks = timing()
    for index, (ar, br, ell, xu) in enumerate(data):
        if (redc(a), redc(b)) != (ar, br):
            raise ValueError(f"Residue mismatch at step {index}")
        if ell == 1:
            clock = time.perf_counter()
            u = hensel._lift(lift(xu), ZZ(s))
            a, b = a / u**4, b / u**6
            B = (B[0] * u, B[1] / u, B[2] * u, B[3] / u)
            iso_wall += time.perf_counter() - clock
        else:
            clock = time.perf_counter()
            x = local_kernel_lift(a, b, xu, ell, s, lift)
            kernel_wall += time.perf_counter() - clock
            if audit_kernels:
                clock = time.perf_counter()
                residual = EllipticCurve(a.parent(), [a, b]).division_polynomial(
                    ell, x=x
                )
                if residual.valuation() < s:
                    raise ArithmeticError("Lifted kernel residual failed")
                kernel_check_wall += time.perf_counter() - clock
            clock = time.perf_counter()
            a, b, S = quotient(a, b, x, ell)
            B = (B[0], -B[0] * S + B[1] * ell, B[2], -B[2] * S + B[3] * ell)
            quotient_wall += time.perf_counter() - clock
        if index and index % 250 == 0:
            progress("open_progress", steps=index, total=len(data))
    chain_t = elapsed(checks)
    if (redc(a), redc(b)) != final:
        raise ValueError("Final residue mismatch")
    clock = timing()
    X = compare(a0, b0, a, b, s)
    A = mm(B, invm(X))
    gm_t = elapsed(clock)
    if (A[0] * A[3] - A[1] * A[2] - degree).valuation() < s:
        raise ArithmeticError("Endomorphism determinant mismatch")
    trace = centered_trace(A[0] + A[3], F.characteristic(), s)
    report = {
        "method": "open_gm_batched_Zq" if batched else "open_gm_shared_Zq",
        **elapsed(start),
        "chain": chain_t,
        "kernel_wall_seconds": kernel_wall,
        "velu_and_matrix_wall_seconds": quotient_wall,
        "full_kernel_residuals_checked": audit_kernels,
        "kernel_residual_checks_wall_seconds": kernel_check_wall,
        "isomorphisms_wall_seconds": iso_wall,
        "gm": gm_t,
        "gm_recurrence": compare.__name__,
        "trace_precision": int((A[0] + A[3]).precision_absolute()),
        "determinant_checked": True,
        "all_residues_checked": True,
    }
    if capture is not None:
        capture.update(endpoints=(a0, b0, a, b, s), X=X, A=A)
    return trace, report
