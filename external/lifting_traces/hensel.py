r"""
.. [PRS26]
    Lorenz Panny, Damien Robert, Alessandro Sferlazza:
    *Hensel-lifting black-box algorithms and fast trace computation for elliptic-curve endomorphisms*.
    ANTS XVII, 2026.

.. raw:: html

    <br>
"""

from sage.all import *

import collections

def solve_sparse(A, y):
    r"""
    Compute a solution :math:`x` of a sparse linear system :math:`A \cdot x = y`.
    If the system is underdetermined, the function returns a solution that depends
    deterministically on :math:`A, y`.

    :param A: :math:`m`-by-:math:`n` sparse matrix
    :type A: matrix
    :param y: length-:math:`n` vector
    :type y: vector

    :return: length-`m` vector :math:`x` solving the system
    :rtype: vector
    """
    m, n = A.dimensions()

    pivs = []
    col = -1
    for row in range(m):

        nz = collections.defaultdict(set)
        for i,j in A.nonzero_positions():
            nz[i].add(j)

        while True:
            col += 1
            try:
                piv = next(i for i in range(row,m) if col in nz[i] and not A[i,col].valuation())
            except StopIteration:
                pass
            else:
                pivs.append(col)
                break

        for j in nz[row] | nz[piv]:
            if j in nz[row] and j in nz[piv]:
                A[row,j], A[piv,j] = A[piv,j], A[row,j]
            else:
                assert j in nz[row] or j in nz[piv]
                u, v = (row, piv) if j in nz[row] else (piv, row)
                assert j in nz[u]
                A[v,j] = A[u,j]
                nz[v].add(j)
                A[u,j] = 0
                nz[u].remove(j)
        y[row], y[piv] = y[piv], y[row]

        inv = ~A[row,col]
        for j in nz[row]:
            A[row,j] *= inv
        y[row] *= inv
        assert A[row,col].is_one()

        for k in range(m):
            if k == row:
                continue
            if col not in nz[k]:
                continue
            mul = A[k,col]
            for j in nz[row]:
                A[k,j] -= mul * A[row,j]
            nz[k].remove(col)
            y[k] -= mul * y[row]

    # any solution
    x = vector(A.base_ring(), n, sparse=True)
    for i,j in enumerate(pivs):
        x[j] = y[i]
    assert A * x == y

    return x

def _lift(v, prec=None):
    r"""
    Lift :math:`v + O(p^k)` in the :math:`p`-adic ring :math:`\mathbb Z_q`
    to :math:`v + O(p^\mathtt{prec})`.
    If ``prec`` is not given, take by default :math:`\mathtt{prec} = 2k`.

    :param v: "capped-abs" :math:`p`-adic element in :math:`\mathbb Z_q`, with absolute precision `k`
    :param prec: positive integer
    :return: `
    """
    R = v.parent()
    if prec is None:
        prec = 2 * v.precision_absolute()
    f = v.polynomial()
    f = f.map_coefficients(lambda c: c.parent()(c.lift()).add_bigoh(prec))
    v = f(R.gen())
    return v

def hensel(funs_and_inpss, vals):
    r"""
    Lift a solution of a constraint system to double precision, following [PRS26]_, Algorithm 1.

    Given an algebraic constraint system given in ``funs_and_inpss``
    and a tuple of p-adic elements ``vals.values()`` that satisfies the constraints at precision :math:`k`
    (i.e., the ``funs`` in ``funs_and_inpss`` evaluated at ``vals.values()`` return :math:`0 + O(p^k)`)
    return a tuple that satisfies the constraints at precision :math:`2k`.

    :param funs_and_inpss: list of constraints of the form ``(fun, inps)``:
        In the notation of the paper, :math:`\mathtt{funs\_and\_inpss} = ((F_j, X_j))_j`.

        - ``inps`` is a tuple of variable names;
        - ``fun`` represents an algebraic constraint (an algebraic function depending on a subset of the variables in `vals`)
            - Input of ``fun``: a dictionary made like ``vals``.
              The keys are a subset of the keys of ``vals``.
              The values will be either all of precision :math:`k` or all of precision :math:`2k`.
            - Output of ``fun``: the evaluation of the algebraic constraint, as a tuple of :math:`p`-adic elements of the same precision as the input.

    :param vals:

        - Keys (:class:`str`): variable names.
        - Values: "capped-abs" :math:`p`-adic element at precision :math:`k`.
            All ``vals.values()`` should have the same precision.

    :type vals: dict

    EXAMPLE::

        >>> vals = {'a': 1 + O(p), 'b': 2 + O(p), 'c': 1 + O(p)}
        >>> fun_eq = lambda myvals: myvals['a'] - myvals['c'],
        >>> def fun_dbl_sq(myvals):
        ...     a, b = myvals['a'], myvals['b']
        ...     return b - 2 * a**3,
        >>> funs_and_inpss = [(fun_eq, ('a', 'c')), (fun_dbl_sq, ('a', 'b'))]
        >>> hensel(funs_and_inpss, vals)
        {'a': 1 + O(p^2), 'b': 2 + O(p^2), 'c': 1 + O(p^2)}


    The following example shows how the same algebraic functions can be used on different inputs::

        >>> vals = {'a': 1 + O(p), 'b': 1 + O(p), 'c': 1 + O(p)}
        >>> def make_fun_eq_on_inps(myinps):
        ...     def fun_eq(myvals):
        ...         inp1, inp2 = (myvals[i] for i in myinps)
        ...         return inp1 - inp2,
        ...     return fun_eq
        >>> eq_ab = make_fun_eq_on_inps('a', 'b')
        >>> eq_ac = make_fun_eq_on_inps('a', 'c')
        >>> funs_and_inpss = [(eq_ab, ('a', 'b')), (eq_ac, ('a', 'c'))]
        >>> hensel(funs_and_inpss, vals)
        {'a': 1 + O(p^2), 'b': 1 + O(p^2), 'c': 1 + O(p^2)}
    """
    R = next(v for v in vals.values()).parent()
    assert R == seq(vals.values()).universe()

    # preliminarily, increase the absolute precision of the values to 2k
    k, = {elt.precision_absolute() for elt in vals.values()}
    vals = {i: _lift(v) for i,v in vals.items()}

    # collect inputs actually used by funs_and_inpss. Should be a subset of the keys of vals.
    all_inps = []
    for _, inps in funs_and_inpss:
        for i in inps:
            if i not in all_inps:
                all_inps.append(i)

    # evaluate constraints
    res = vector(c for fun,_ in funs_and_inpss for c in fun(vals))
    assert all(c.valuation() >= k for c in res) # sanity check: initial inputs should evaluate to 0 mod p^k

    # save Jacobian matrix and constant term of the system.
    # the following code corresponds closely to Algorithm 1
    H = matrix(R, len(res), len(all_inps), sparse=True)
    b = vector(R, len(res), sparse=True)

    off = 0

    for fun,I in funs_and_inpss:
        ass = {i: vals[i] for i in I}
        y0 = vector(fun(ass))
        assert all(c.valuation() >= k for c in y0)

        for inp in I:
            ass = {i: vals[i] + (R(i == inp) << k) for i in I}
            yi = vector(fun(ass))
            assert all(c.valuation() >= k for c in yi)

            for j,c in enumerate(yi - y0, off):
                H[j, all_inps.index(inp)] = c >> k

        for j,c in enumerate(y0, off):
            b[j] = -c >> k

        off += len(y0)

    assert off == len(res)

    # equivalent to H.solve_right(b)
    δ = solve_sparse(H, b)

    vals = {i: vals[i] + (s << k) for i,s in zip(all_inps, δ)}

    if __debug__:
        ass = {i: vals[i] for i in all_inps}
        assert all(c.valuation() >= min(2*k, R.precision_cap())
                   for fun,_ in funs_and_inpss for c in fun(ass))

    return vals

__all__ = ['solve_sparse', '_lift', 'hensel']
