"""Compute traces, compare endpoint recurrences and verify prescribed traces."""

import argparse
import hashlib
import json
import os
import time
import tempfile
from pathlib import Path
from statistics import median
from unittest.mock import patch

import batched_velu
import trace_core as core
from prs_checks import check_replacement

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HOUBEN = DATA / "houben_256.json"
METHODS = ("prs", "prs-batched", "open", "open-batched")
RECURRENCES = {"convolution": core.gm_convolution, "fixed_width": core.gm_compare}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink()


def setup():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    threads = {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if any(v != "1" for v in threads.values()):
        raise ValueError(
            "Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1 before starting Sage."
        )
    return dict(
        sage_version=core.sage_version,
        cpu=cpu,
        threads=threads,
        benchmark_sha256=sha(Path(core.__file__)),
        prs_sha256={
            p: sha(core.UPSTREAM / p)
            for p in ("compute_trace.py", "hensel.py", "xonly.py")
        },
    )


def evaluate(method, params, target, **kwargs):
    field, _, _, degree, s = params
    if method.startswith("open"):
        trace, row = core.open_gm(
            *params, batched=method == "open-batched", audit_kernels=True, **kwargs
        )
    elif method == "prs-batched":
        with patch.object(core.xonly, "x_multiples", batched_velu.x_multiples):
            trace, row = core.prs_closed(*params)
    else:
        trace, row = core.prs_closed(*params)
    modulus = int(field.characteristic()) ** s
    if (trace - target) % modulus or (
        modulus**2 > 16 * int(degree) and trace != target
    ):
        raise ArithmeticError("Reference trace check failed")
    return dict(**row, variant=method, trace=int(trace))


def load_chain(path):
    fixture, field, raw, final, degree = core.load_fixture(path)
    return int(fixture["expected_trace"]), (
        field,
        core.compact_chain(raw, final),
        final,
        degree,
    )


def endpoint_timings(endpoints):
    blocks = {name: [] for name in RECURRENCES}
    for fn in RECURRENCES.values():
        fn(*endpoints)
    for i in range(10):
        order = list(RECURRENCES) if i % 2 == 0 else list(reversed(RECURRENCES))
        for name in order:
            fn = RECURRENCES[name]
            start = time.perf_counter()
            for _ in range(30):
                fn(*endpoints)
            blocks[name].append((time.perf_counter() - start) / 30)
    return blocks


def paired(rounds):
    target, chain = load_chain(HOUBEN)
    cases = {}
    for s in (9, 8, 7):
        params = (*chain, s)
        matrices = {}
        for name, compare in RECURRENCES.items():
            capture = {}
            evaluate("open-batched", params, target, compare=compare, capture=capture)
            matrices[name] = (*capture["X"], *capture["A"])
        for a, b in zip(matrices["fixed_width"], matrices["convolution"]):
            if (
                min(a.precision_absolute(), b.precision_absolute(), (a - b).valuation())
                < s
            ):
                raise ArithmeticError("Endpoint or final matrix mismatch")
        blocks = endpoint_timings(capture["endpoints"])
        runs = []
        for i in range(rounds):
            order = list(RECURRENCES) if i % 2 == 0 else list(reversed(RECURRENCES))
            for name in order:
                row = evaluate(
                    "open-batched", params, target, compare=RECURRENCES[name]
                )
                runs.append(dict(**row, recurrence=name, round=i))
                print(
                    f"s={s} round={i+1} {name}: {row['wall_seconds']:.3f} s", flush=True
                )
        cases[s] = dict(endpoint_seconds_per_call=blocks, runs=runs)
    return dict(
        fixture_sha256=sha(HOUBEN),
        compact_chain=True,
        normalized_isogenies=sum(r[2] != 1 for r in chain[1]),
        cases=cases,
        expected_trace=target,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trace = commands.add_parser(
        "trace",
        help="Run a checked open or closed trace on a decoded Houben-format chain",
    )
    trace.add_argument("--input", type=Path, default=HOUBEN)
    trace.add_argument("--method", choices=METHODS, default="open-batched")
    trace.add_argument("--precision", type=int, default=9)
    trace.add_argument("--repeat", type=int, default=1)
    comp = commands.add_parser(
        "paired", help="Compare recurrences and retain the runs used for verification"
    )
    comp.add_argument("--rounds", type=int, default=4)
    ver = commands.add_parser(
        "verify",
        help="Add signed-chain auxiliary checks to the paired fixed-width runs",
    )
    ver.add_argument("--paired", type=Path, required=True)
    for cmd in (trace, comp, ver):
        cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} already exists; choose a new result file")
    if min(getattr(args, "rounds", 1), getattr(args, "repeat", 1)) < 1:
        parser.error("Repetition counts must be positive")
    meta = setup()
    if args.command == "trace":
        target, chain = load_chain(args.input)
        params = (*chain, args.precision)
        if args.method == "prs-batched":
            check_replacement(chain[0], chain[1], args.precision)
        runs = [evaluate(args.method, params, target) for _ in range(args.repeat)]
        result = dict(
            fixture_sha256=sha(args.input),
            precision=args.precision,
            expected_trace=target,
            runs=runs,
            median_wall_seconds=median(r["wall_seconds"] for r in runs),
        )
        print(
            f"Trace representative: {runs[-1]['trace']}; median: {result['median_wall_seconds']:.3f} s"
        )
    elif args.command == "paired":
        result = paired(args.rounds)
    else:
        from verification import verify

        prior = json.loads(args.paired.read_text())
        if (
            prior.get("command") != "paired"
            or prior["prs_sha256"] != meta["prs_sha256"]
        ):
            raise ValueError("Expected a paired report from the same PRS dependency")
        reports = []
        for s in (9, 8, 7):
            runs = [
                row
                for row in prior["cases"][str(s)]["runs"]
                if row["recurrence"] == "fixed_width"
            ]
            reports.append(
                dict(
                    sage_version=prior["sage_version"],
                    fixture_sha256=prior["fixture_sha256"],
                    compact_chain=prior["compact_chain"],
                    normalized_isogenies=prior["normalized_isogenies"],
                    benchmark_sha256=prior["benchmark_sha256"],
                    precision=s,
                    runs=runs,
                )
            )
        result = verify(HOUBEN, reports)
    save(args.out, {**meta, **result, "command": args.command})


if __name__ == "__main__":
    main()
