"""Verify a prescribed trace at reduced precision on Houben's base cycle.

Use the published base-curve group orders to sample points for the pairing
check. Compute Q = tau(R) along the given isogeny chain and report its
evaluation time separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from sage.all import EllipticCurve, ZZ, set_random_seed
from sage.version import version as sage_version

from finite_field_chain import load_fixture, prepare_chain


def prime_support(m, fixture):
    primes = sorted(
        {
            int(row["ell"])
            for row in fixture["steps"]
            if int(row["ell"]) > 1 and m % int(row["ell"]) == 0
        }
    )
    rest = m
    for q in primes:
        while rest % q == 0:
            rest //= q
    if rest != 1:
        raise ValueError("Missing published prime factor")
    return primes


def exact_order(P, m, primes):
    return (
        not P.is_zero()
        and (m * P).is_zero()
        and all(not ((m // q) * P).is_zero() for q in primes)
    )


def primitive(value, m, primes):
    return value**m == 1 and all(value ** (m // q) != 1 for q in primes)


def kernel_point(chain, x, edges):
    """Check actual first-block kernel membership, allowing infinity."""
    for edge in chain.steps[:edges]:
        if edge.kernel_polynomial(x) == 0:
            return True
        x, _ = edge.eval_x_yfactor(x)
    return False


def pairing_check(P, R, Q, m, target):
    start = time.perf_counter()
    base = P.weil_pairing(R, ZZ(m))
    image = P.weil_pairing(Q, ZZ(m))
    accepted = image == base ** ZZ(target % m)
    return accepted, time.perf_counter() - start, base, image


def verify(fixture_path, precision_reports, repeat=3):
    if sage_version != "10.10":
        raise RuntimeError(f"Expected SageMath 10.10, got {sage_version}")
    started = time.perf_counter()
    fixture, F, data, final, degree = load_fixture(fixture_path)
    p, M, target = int(fixture["p"]), int(fixture["M"]), int(fixture["expected_trace"])
    Ms, Mt = int(fixture["Ms"]), int(fixture["Mt"])
    if Ms * Mt != M or math.gcd(Ms, Mt) != 1 or math.gcd(M, p) != 1:
        raise ValueError("Invalid public CRT factorization")
    if int(fixture["r"]) != 13 or len(data) != 1353:
        raise ValueError("This pilot requires the published 13-block fixture")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    set_random_seed(20261018)
    decode = lambda pair: F(ZZ(pair[0])) + F.gen() * ZZ(pair[1])
    raw = list(map(decode, fixture["raw_cycle"][0]))
    A = raw[0]
    if data[0][:2] != (1 - A * A / 3, 2 * A**3 / 27 - A / 3):
        raise ValueError("First raw model is not the actual chain source")
    setup_start = time.perf_counter()
    chain = prepare_chain(data, final, backend="pari")
    chain.embed(F)
    setup_seconds = time.perf_counter() - setup_start
    a, b = data[0][:2]
    d = next(F(z) for z in (F.gen() + k for k in range(1, 100)) if not F(z).is_square())
    sides = []
    negative = {}
    for label, m, group_exponent, twist, raw_x in (
        ("curve", int(fixture["Ms"]), p + 1, F.one(), raw[1]),
        ("twist", int(fixture["Mt"]), p - 1, d, raw[2]),
    ):
        sample_start = time.perf_counter()
        E = EllipticCurve(F, [a * twist**2, b * twist**3])
        # Known only for the trusted published base fixture.
        E.set_order(ZZ(group_exponent) ** 2)
        xP = twist * (raw_x + A / 3)
        yP = (xP**3 + E.a4() * xP + E.a6()).sqrt()
        P = E(xP, yP)
        primes = prime_support(m, fixture)
        if not exact_order(P, m, primes):
            raise ArithmeticError("Published kernel point has wrong exact order")
        if not kernel_point(chain, xP / twist, 52):
            raise ArithmeticError("Published P is not killed by actual first block")
        attempts = 0
        while True:
            attempts += 1
            if attempts > 128:
                raise ArithmeticError("No independent torsion point found")
            R = ZZ(group_exponent // m) * E.random_point()
            if not exact_order(R, m, primes):
                continue
            e = P.weil_pairing(R, ZZ(m))
            if primitive(e, m, primes):
                break
        sampling_seconds = time.perf_counter() - sample_start
        eval_start = time.perf_counter()
        image_x, image_y = chain.apply(R[0] / twist, R[1])
        Q = E(twist * image_x, image_y)
        evaluation_seconds = time.perf_counter() - eval_start
        if not exact_order(Q, m, primes):
            raise ArithmeticError("Actual signed image Q has wrong exact order")
        timings = []
        for _ in range(repeat):
            ok, seconds, base, image = pairing_check(P, R, Q, m, target)
            if not ok or not primitive(base, m, primes):
                raise ArithmeticError("Actual-chain trace pairing check failed")
            timings.append(seconds)
        negative[label + "_opposite_signed_image_rejected"] = not pairing_check(
            P, R, -Q, m, target
        )[0]
        negative[label + "_t_plus_p7_rejected"] = image != base ** ZZ(
            (target + p**7) % m
        )
        negative[label + "_dependent_R_not_primitive"] = not primitive(
            P.weil_pairing(P, ZZ(m)), m, primes
        )
        row = dict(
            side=label,
            modulus=str(m),
            modulus_bits=m.bit_length(),
            exact_P_R_Q_orders_checked=True,
            primitive_pairing_checked=True,
            first_block_kernel_checked=True,
            Q_from_actual_signed_chain=True,
            sampling_attempts=attempts,
            sampling_wall_seconds=sampling_seconds,
            actual_chain_evaluation_wall_seconds=evaluation_seconds,
            two_pairings_and_exponent_wall_seconds=timings,
            pairing_check_median_seconds=statistics.median(timings),
        )
        if label == "curve":
            C = 553
            assert m % C == 0
            projection_start = time.perf_counter()
            Pc, Rc, Qc = ((m // C) * T for T in (P, R, Q))
            row["mod553_projection_wall_seconds"] = (
                time.perf_counter() - projection_start
            )
            small = []
            for _ in range(repeat):
                ok, seconds, ec, ic = pairing_check(Pc, Rc, Qc, C, target)
                if not ok or not primitive(ec, C, [7, 79]):
                    raise ArithmeticError("553 projection failed")
                small.append(seconds)
            row["mod553_pairing_check_wall_seconds"] = small
            row["mod553_median_seconds"] = statistics.median(small)
            negative["mod553_t_plus_p8_rejected"] = ic != ec ** ZZ((target + p**8) % C)
        sides.append(row)
        print(json.dumps(dict(event="side_complete", **row)), flush=True)
    H = math.isqrt(4 * int(degree)) + abs(target)
    strategies = []
    for report in precision_reports:
        if report.get("sage_version") != sage_version:
            raise ValueError(
                "Core timing and auxiliary check must both use SageMath 10.10"
            )
        s = int(report["precision"])
        C = {7: M, 8: 553, 9: 1}[s]
        if (
            report["fixture_sha256"]
            != hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest()
        ):
            raise ValueError("Precision report refers to a different fixture")
        if (
            not report["compact_chain"]
            or report["normalized_isogenies"] != 676
            or report["benchmark_sha256"]
            != hashlib.sha256(
                (Path(__file__).parent / "trace_core.py").read_bytes()
            ).hexdigest()
        ):
            raise ValueError(
                "Precision report has different arithmetic or input representation"
            )
        if not report["runs"] or not all(
            r["method"] == "open_gm_batched_Zq"
            and r["gm_recurrence"] == "gm_compare"
            and r["full_kernel_residuals_checked"]
            and r["all_residues_checked"]
            and r["determinant_checked"]
            and r["trace_precision"] >= s
            for r in report["runs"]
        ):
            raise ValueError("Precision report omits required core checks")
        if not all((int(r["trace"]) - target) % p**s == 0 for r in report["runs"]):
            raise ArithmeticError("Low-precision report has wrong trace residue")
        if C * p**s <= H:
            raise ArithmeticError("Insufficient exact CRT uniqueness bound")
        incremental = (
            sum(x["pairing_check_median_seconds"] for x in sides)
            if s == 7
            else sides[0]["mod553_median_seconds"] if s == 8 else 0.0
        )
        projection = sides[0]["mod553_projection_wall_seconds"] if s == 8 else 0.0
        core = statistics.median(r["wall_seconds"] for r in report["runs"])
        strategies.append(
            dict(
                precision=s,
                independent_congruence_modulus=str(C),
                exact_uniqueness_bound_H=str(H),
                exact_C_p_to_s_gt_H=True,
                trace_core_seconds=core,
                pairing_incremental_seconds=incremental,
                sum_of_separately_measured_core_and_pairings_seconds=core + incremental,
                projection_incremental_seconds=projection,
                verification_incremental_seconds=incremental + projection,
                total_seconds=core + incremental + projection,
            )
        )

    negative["t_plus_M_fails_p7_residue"] = M % p**7 != 0
    negative["t_plus_Mp7_outside_Hasse_interval"] = abs(
        target + M * p**7
    ) ** 2 > 4 * int(degree)
    if not all(negative.values()):
        raise ArithmeticError("A negative check did not reject")
    return dict(
        sage_version=sage_version,
        scope="Published decoded base fixture; signed actual-chain pairing plus recorded p-adic core. "
        "Not compressed-key integration; no new random public keys; sampling uses trusted base group orders. "
        "Core+pairing sum excludes independently measured sampling, chain preparation/evaluation and decompression.",
        fixture=Path(fixture_path).name,
        fixture_sha256=hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        affinity=sorted(os.sched_getaffinity(0)),
        seed=20261018,
        timer_excludes="Sage process startup, module imports, final output serialization",
        finite_field_chain_preparation_and_embedding_seconds=setup_seconds,
        sides=sides,
        strategies=strategies,
        negative_checks=negative,
        total_wall_seconds=time.perf_counter() - started,
    )
