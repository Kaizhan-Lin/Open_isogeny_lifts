"""Check the batched quotient arithmetic before timing the closed PRS solver."""

import trace_core as bench
import batched_velu


def check_replacement(F, data, precision):
    """Check actual kernels and the perturbed inputs used by finite differences."""
    start = bench.timing()
    _, lift = bench.PRS.padic_ring(F, bench.ZZ(precision))
    selected = {}
    largest = max(int(row[2]) for row in data)
    for row in data:
        ell = int(row[2])
        if ell in {3, 5, largest} and ell not in selected:
            selected[ell] = row
    checks = []
    for ell, (ar, br, _, xr) in selected.items():
        a, b = (bench.hensel._lift(lift(v), bench.ZZ(precision)) for v in (ar, br))
        x = bench.local_kernel_lift(a, b, xr, bench.ZZ(ell), precision, lift)
        ring = a.parent()
        cases = [("kernel", (a, b, x))]
        k = 1
        while k < precision:
            for coordinate in range(3):
                values = [a, b, x]
                values[coordinate] += ring.one() << k
                cases.append(
                    (f"perturb_coordinate_{coordinate}_at_p^{k}", tuple(values))
                )
            k *= 2
        for label, values in cases:
            original = bench.xonly.x_multiples(*values, ell // 2)
            replacement = batched_velu.x_multiples(*values, ell // 2)
            if len(original) != len(replacement):
                raise ArithmeticError("Kernel-multiple count changed")
            for left, right in zip(original, replacement):
                if (
                    min(left.precision_absolute(), right.precision_absolute())
                    < precision
                ):
                    raise ArithmeticError("Kernel-multiple precision loss")
                if (left - right).valuation() < precision:
                    raise ArithmeticError("Kernel-multiple arithmetic mismatch")
            checks.append({"ell": ell, "case": label, "multiples": len(original)})
    return {"all_passed": True, "cases": checks, **bench.elapsed(start)}
