#!/usr/bin/env python3
from sage.all import *
proof.all(False)

from sage.misc.banner import require_version
if not require_version(10, 10, prerelease=True, print_message=True):
    exit(1)

################################################################

def padic_ring(F, maxprec):
    r"""
    Construct a reduction and a lifting map between a finite field :math:`F`
    and its ring of Witt vectors :math:`R`, i.e. a :math:`p`-adic ring :math:`\mathbb Z_q`
    with residue field :math:`F`.

    The constructed :math:`R` has capped absolute precision ``maxprec``.

    :param F: finite field
    :param maxprec: absolute capped precision of the output :math:`p`-adic field

    :return: a tuple containing..:

        - ``redc`` a :class:`ResidueReductionMap` from :math:`R` to :math:`F`
        - ``lift``: a :class:`ResidueLiftingMap` from :math:`F` to :math:`R`,
          mapping :math:`x + O(p) \in F` to :math:`x + O(p^\mathtt{maxprec}) \in R`
    """
    assert F in FiniteFields()
    p = F.characteristic()

    if F.degree().is_one():
        R = Zp(p, prec=maxprec, type='capped-abs')
    else:
        poly = F.modulus().change_ring(ZZ)
        R = Zq((p, F.degree()), prec=maxprec, type='capped-abs', modulus=poly, names='T', res_name=F.variable_name())

    from sage.rings.padics.padic_generic import ResidueReductionMap
    redc = ResidueReductionMap._create_(R, F)
    lift = redc.section()

    return redc, lift

################################################################

from sage.schemes.elliptic_curves.hom_composite import EllipticCurveHom_composite
from sage.schemes.elliptic_curves.weierstrass_morphism import WeierstrassIsomorphism

def padify_isogeny(phi: EllipticCurveHom_composite, lift, *, prefix='', verbose=False):
    r"""
    Given an isogeny chain :math:`\varphi` over a finite field,
    with :math:`\varphi_i: E_{i-1} \to E_i` for :math:`i=1,...,n`
    of degree :math:`\ell_i` either :math:`1` or prime,
    where all :math:`E_i` are short Weierstrass curves,
    for each step:

    - if :math:`\ell_i = 1`, the step is an isomorphism:
        recover the coefficients :math:`(a_i, b_i)` of :math:`E_i`
        and the differential scaling factor :math:`u` of the
        Weierstrass isomorphism;
    - if :math:`\ell_i` is a prime,
        recover the coefficients :math:`(a_i, b_i)` of :math:`E_i`
        and the x-coordinate :math:`x_i` of a kernel generator
        of :math:`\varphi_i`.

    Return these values and a lift of them via the lifting map ``lift``.

    :param phi: isogeny chain over a finite field :math:`\mathbb F_q`
    :param lift: :class:`ResidueLiftingMap` from :math:`\mathbb F_q` to :math:`p`-adic ring :math:`\mathbb Z_q`, as constructed by :func:`padic_ring`
    :param prefix: prefix for variable names :math:`a_i`, :math:`b_i`, :math:`x_i`, :math:`u_i`; used in output ``vals``
    :param verbose: :class:`bool` flag. Set to :const:`True` for debug printing (default: :const:`False`).

    :return: a tuple containing..:

        - ``data``: list of 4-tuples
          ``[(a_0, b_0, l_0, x_0|u_0), ..., (a_{n-1}, b_{n-1}, l_{n-1}, x_{n-1}|u_{n-1})]``
          as described above
        - ``vals``: dictionary used by :func:`hensel.hensel`:
          ``{"a0": lift(a_0), "b0": lift(b_0), ...}``
          with all lifts of the values in ``data``
          and of the Weierstrass coefficients of the codomain of :math:`\varphi`
    """
    pr = print if verbose else lambda *_: ()

    if not isinstance(phi, EllipticCurveHom_composite):
        phi = EllipticCurveHom_composite.from_factors([phi])

    def _process_WI(step: WeierstrassIsomorphism):
        r"""
        Given a :class:`WeierstrassIsomorphism` ``step`` from :math:`E_i` to :math:`E_{i+1}`,
        where the curves are assumed to be in short Weierstrass form, recover:

        - the Weierstrass coefficients :math:`a_i`, :math:`b_i` of the domain :math:`E_i`
        - the scaling factor :math:`u` of the isomorphism: :math:`(a_{i+1}, b_{i+1}) = (a_i / u^4, b_i / u^6)`

        :param step: a :class:`WeierstrassIsomorphism`

        :yield: tuple :math:`(a_i, b_i, 1, u)`, or nothing
        """
        if __debug__:
            nonlocal reconstructed  # holds the composition of the isogeny steps (factors of `phi`) processed so far
            assert reconstructed.codomain() == step.domain()
            reconstructed = step * reconstructed

        u = step.scaling_factor()
        assert step == step.domain().isomorphism(u)

        pr(f'have WeierstrassIsomorphism with {u = }')

        if not u.is_one():
            yield step.domain().a_invariants()[-2:] + (ZZ.one(), u)

    def _process_ECI_normalized(step: EllipticCurveIsogeny):
        r"""
        Given a cyclic EllipticCurveHom ``step`` of degree :math:`\ell_i` from :math:`E_i` to :math:`E_{i+1}`,
        where the isogeny is assumed to be normalized and the curves in short Weierstrass form, recover:

        - the Weierstrass coefficients :math:`a_i`, :math:`b_i` of the domain :math:`E_i`
        - the x-coordinate :math:`x_i` of a kernel generator

        :param step: _normalized_ cyclic isogeny

        :yield: tuple :math:`(a_i, b_i, \ell_i, x_i)`
        """
        if __debug__:
            nonlocal reconstructed  # holds the composition of the isogeny steps processed so far
            assert reconstructed.codomain() == step.domain()
            reconstructed = step * reconstructed

        assert step.is_normalized()

        deg = step.degree()

        # find kernel points depending on how the isogeny is represented
        if (x := getattr(step, '_hack_kernel_x', None)) is not None:
            pass
        else:
            if step._EllipticCurveIsogeny__algorithm == 'velu':
                kerlist = step._EllipticCurveIsogeny__kernel_mod_sign
                pts = (step.domain()(x,y) for x,(y,_,_,_,_) in kerlist.items())
            elif step._EllipticCurveIsogeny__algorithm == 'kohel':
                kerpoly = step.kernel_polynomial()
                assert step == step.domain().isogeny(kerpoly)
                pts = map(step.domain().lift_x, kerpoly.roots(multiplicities=False))
            else:
                raise NotImplementedError('should never happen')

            # find kernel generator `P`
            for P in pts:
                P.set_order(multiple=deg)
                if P.order() == deg:
                    break
            else:
                raise NotImplementedError('every step must be a cyclic isogeny')

            assert step == step.domain().isogeny(P, degree=deg)

            x = P.x()

        pr(f'have {step._EllipticCurveIsogeny__algorithm!r} EllipticCurveIsogeny with x = {x}')

        yield step.domain().a_invariants()[-2:] + (deg, x)

    def _process_ECI(step: EllipticCurveIsogeny):
        r"""
        Given a cyclic isogeny ``step`` of degree :math:`\ell_i` from :math:`E_i` to :math:`E_{i+1}`,
        where the curves are assumed to be in short Weierstrass form,
        split ``step`` into the composition of a normalized isogeny followed by an isomorphism,
        and process both parts individually.

        :param step: cyclic isogeny

        :yield: tuples :math:`(a, b, l, x|u)` for the factors of the given ``step``
        """
        if (post_iso := step._EllipticCurveIsogeny__post_isomorphism) is not None:
            step = ~post_iso * step

        yield from _process_ECI_normalized(step)

        if post_iso is not None:
            yield from _process_WI(post_iso)

    data = []
    if __debug__:
        reconstructed = phi.domain().identity_morphism()    #XXX

    # check input validity
    for step in phi.factors():
        if any(step.domain().a_invariants()[:-2]) or any(step.codomain().a_invariants()[:-2]):
            raise NotImplementedError('all curves must be in short Weierstraß form')
        if isinstance(step, WeierstrassIsomorphism):
            data.extend(_process_WI(step))
        elif isinstance(step, EllipticCurveIsogeny):
            data.extend(_process_ECI(step))
        else:
            raise NotImplementedError('every step must be an EllipticCurveIsogeny or WeierstrassIsomorphism')

    if __debug__:
        assert reconstructed == phi
        assert phi.scaling_factor() == prod(u for _,_,l,u in data if l.is_one())

    # prepare output ``vals`` to be well-formed input for function hensel.hensel()
    vals = dict()
    for i, (a, b, l, v) in enumerate(data):
        inps = [f'{prefix}a{i}', f'{prefix}b{i}', f'{prefix}{"xu"[l.is_one()]}{i}']
        vals[inps[0]] = lift(a)
        vals[inps[1]] = lift(b)
        vals[inps[2]] = lift(v)

    vals[f'{prefix}a{len(data)}'], vals[f'{prefix}b{len(data)}'] = map(lift, phi.codomain().a_invariants()[-2:])

    return data, vals

################################################################

import xonly

def _codomainWI(*myvars, pr=lambda *_: ()):
    r"""
    Create constraint function for lifting via :func:`hensel.hensel`:
    :math:`\mathtt{\_codomainWI}(...)(a, b, u, a_1, b_1)`
    vanishes modulo :math:`p^k` iff
    :math:`a_1 \equiv a / u^4` and :math:`b_1 \equiv b / u^6` modulo :math:`p^k`,
    i.e.,
    the curve with coefficients :math:`(a_1, b_1)`
    is the codomain of a Weierstrass isomorphism
    with scaling factor :math:`u`
    starting from the curve with coeffs :math:`(a, b)`.

    :param myvars: tuple of variable names (:class:`str`) to which the constraint function is applied; see :func:`hensel.hensel`
    :param pr: debug printing function, defaults do doing nothing

    :return: a tuple containing..:

        - the constraint function ``_codomainWI(...)``,
        - the tuple ``myvars`` of variable names received as input
    """
    def fun_aux(myvals):
        a, b, u, a1, b1 = (myvals[i] for i in myvars)
        pr(f'_codomainWI()', {i: myvals[i] for i in myvars})
        a_, b_ = a / u**4, b / u**6
        return a_ - a1, b_ - b1
    return fun_aux, myvars

def _ltorsion(l, *myvars, pr=lambda *_: ()):
    r"""
    Create constraint function for lifting via :func:`hensel.hensel`:
    :math:`\mathtt{\_ltorsion}(\ell, ...)(a, b, x)`
    evaluates the division polynomial of the Weierstrass curve with coefficients :math:`(a, b)` at the point :math:`x`,
    i.e.,
    vanishes iff :math:`x` is the x-coordinate of an :math:`\ell`-torsion point on ``EllipticCurve([a, b])``.

    :param l: positive integer
    :param myvars: tuple of variable names (:class:`str`) to which the constraint function is applied; see :func:`hensel.hensel`
    :param pr: debug printing function, defaults do doing nothing

    :return: a tuple containing..:
    - the constraint function ``_ltorsion(l, ...)``,
    - the tuple ``myvars`` of variable names received as input
    """
    def fun_aux(myvals):
        a, b, x = (myvals[i] for i in myvars)
        pr(f'_ltorsion({l=})', {i: myvals[i] for i in myvars})
        assert a.parent() is b.parent()
        R = a.parent()
        return EllipticCurve(R, [a,b]).division_polynomial(l, x=x),
    return fun_aux, myvars

def _codomainECI(l, *myvars, pr=lambda *_: ()):
    r"""
    Create constraint function for lifting via :func:`hensel.hensel`:
    :math:`\mathtt{\_codomainECI}(\ell, ...)(a, b, x, a_1, b_1)`
    evaluates the rational functions giving the codomain of an elliptic curve :math:`l`-isogeny from ``EllipticCurve([a, b])`` at :math:`x`,
    i.e., vanishes modulo :math:`p^k` iff
    ``EllipticCurve([a_1, b_1])`` is the codomain of a normalized :math:`l`-isogeny from ``EllipticCurve([a, b])`` modulo :math:`p^k`.

    :param l: positive integer
    :param myvars: tuple of variable names (:class:`str`) to which the constraint function is applied; see :func:`hensel.hensel`
    :param pr: debug printing function, defaults do doing nothing

    :return: a tuple containing..:

        - the constraint function ``_codomainECI(l, ...)`,
        - the tuple ``myvars`` of variable names received as input
    """
    def fun_aux(myvals):
        a, b, x, a1, b1 = (myvals[i] for i in myvars)
        pr(f'_codomainECI({l=})', {i: myvals[i] for i in myvars})
        # Vélu's formulas
        v = w = 0
        for xQ in xonly.x_multiples(a, b, x, l//2):
            xQ2 = xQ**2
            gxQ = 3*xQ2 + a
            uQ = 4 * ((xQ2 + a)*xQ + b)
            if l >= 3:
                vQ = 2 * gxQ
            else:
                vQ = gxQ
            v += vQ
            w += uQ + xQ*vQ
        a_ = a - 5*v
        b_ = b - 7*w
        #
        return a_ - a1, b_ - b1
    return fun_aux, myvars

def _equal(*myvars, pr=lambda *_: ()):
    r"""
    Create constraint function for lifting via :func:`hensel.hensel`
    to test equality between its input variables:
    :math:`\mathtt{\_equal}(...)(a, b)` vanishes iff :math:`a = b`.

    :param myvars: tuple of variable names (:class:`str`) to which the constraint function is applied; see :func:`hensel.hensel`
    :param pr: debug printing function, defaults do doing nothing

    :return: a tuple containing..:

        - the constraint function ``_equal(...)``,
        - the tuple ``myvars`` of variable names received as input
    """
    def fun_aux(myvals):
        a, b = (myvals[i] for i in myvars)
        pr(f'_equal()', {i: myvals[i] for i in myvars})
        return a - b,
    return fun_aux, myvars

def _constraints(data, vals, redc, *, prefix='', pr=lambda *_: ()):
    r"""
    Prepare constraint system for an elliptic curve isogeny chain
    following [PRS26]_, Corollary 4.7:
    The constraints are the ones in Proposition 4.8 except ``Y_endo``.

    The output is prepared for use in the lifting function :func:`hensel.hensel`.

    :param data: output of :func:`padify_isogeny`; tuple of elements of a finite field :math:`F` containing Weierstrass coefficients of all intermediate curves, scaling factors of the Weierstrass isomorphism steps, x-coordinates of kernel generators for the other steps
    :param vals: output of :func:`padify_isogeny`; contains a lift of ``data`` from :math:`F` to its associated :math:`p`-adic integer ring :math:`R`
    :param redc: output of :func:`padic_ring`; reduction from :math:`R` to :math:`F`
    :param prefix: see :func:`padify_isogeny`
    :param pr: debug printing function, defaults do doing nothing

    :return: list ``funs_and_inpss`` containing a constraint system for lifting the given isogeny, in the format required by :func:`hensel.hensel`.
    """
    funs_and_inpss = []

    for i,(_,_,l,_) in enumerate(data):
        pr(f'\x1b[35m{i = }\x1b[0m | {l = }')

        def prfx(*vs):
            """
            Take strings as input and prepend ``prefix`` to each.

            :param vs: tuple of (:class:`str`) variable names
            """
            return [prefix + v for v in vs]

        if l.is_one(): # step is a Weierstrass isomorphism
            fun, inps = _codomainWI(*prfx(f'a{i}', f'b{i}', f'u{i}', f'a{i+1}', f'b{i+1}'), pr=pr)
            funs_and_inpss.append((fun, inps))
            assert not any(fun({i: redc(vals[i]) for i in inps}))

        else: # step is a normalized (Vélu) l-isogeny
            fun, inps = _ltorsion(l, *prfx(f'a{i}', f'b{i}', f'x{i}'), pr=pr)
            funs_and_inpss.append((fun, inps))
            assert not any(fun({i: redc(vals[i]) for i in inps}))

            fun, inps = _codomainECI(l, *prfx(f'a{i}', f'b{i}', f'x{i}', f'a{i+1}', f'b{i+1}'), pr=pr)
            funs_and_inpss.append((fun, inps))
            assert not any(fun({i: redc(vals[i]) for i in inps}))

    return funs_and_inpss

################################################################

def trace(phi: EllipticCurveHom, *, verbose=False):
    r"""
    Given an endomorphism :math:`\varphi \in \mathrm{End}(E)`,
    compute :math:`\mathrm{tr}(\varphi)` following [PRS26]_, Algorithm 3.

    :param phi: endomorphism :math:`\varphi`

    :return: :math:`\mathrm{tr}(\varphi) \in \mathbb Z`
    """
    pr = print if verbose else lambda *_: ()

    F = phi.base_ring()
    p = F.characteristic()
    maxprec = ceil(log(4 * sqrt(phi.degree()), p))

    # shortcut: if maxprec == 1, we don't need to do any lifting. :^)
    if maxprec.is_one():
        u = phi.scaling_factor()
        t = Mod(u + phi.degree() / u, p).lift_centered()
        return t

    redc, lift = padic_ring(F, maxprec)

    # recover data for the isogeny representation
    data, vals = padify_isogeny(phi, lift)

    # create constraints from Proposition 4.8:
    # _constraints generates the ones relative to the chain steps
    # the last two are the endomorphism constraint
    funs_and_inpss = _constraints(data, vals, redc, pr=pr)
    funs_and_inpss.append(_equal(f'a{len(data)}', 'a0', pr=pr))
    funs_and_inpss.append(_equal(f'b{len(data)}', 'b0', pr=pr))

    # lift up to desired precision, doubling precision every time
    from hensel import hensel
    while True:
        prec, = {v.precision_absolute() for v in vals.values()}
        pr(f'\x1b[33m{prec = }\x1b[0m')
        if prec >= maxprec:
            break
        vals = hensel(funs_and_inpss, vals)

    # get scaling factor of the lifted isogeny
    u = prod(vals[f'u{i}'] for i,(_,_,l,_) in enumerate(data) if l.is_one())

    # recover trace
    t = Mod(u + phi.degree() / u, p**maxprec).lift_centered()

    return t

################################################################

def trace_pairing(phi: EllipticCurveHom, psi: EllipticCurveHom, *, verbose=False):
    r"""
    Given two isogenies :math:`\varphi, \psi \in \mathrm{Hom}(E, E')`,
    compute

    .. math::

        \deg(\varphi + \psi) - \deg(\varphi) - \deg(\psi) = \mathrm{tr}(\widehat\psi \circ \varphi)

    following [PRS26]_, § 5.

    :param phi: isogeny :math:`\varphi: E \to E'`
    :param psi: isogeny :math:`\psi: E \to E'`

    :return: the trace pairing :math:`\langle \varphi, \psi \rangle \in \mathbb Z`
    """
    pr = print if verbose else lambda *_: ()

    if phi.parent() != psi.parent():
        raise ValueError('phi and psi must be isogenies between the same pair of curves')

    dphi, dpsi = phi.degree(), psi.degree()

    F = phi.base_ring()
    p = F.characteristic()
    maxprec = ceil(log(4 * sqrt(dphi * dpsi), p))

    # shortcut: if maxprec == 1, we don't need to do any lifting. :^)
    if maxprec.is_one():
        u = phi.scaling_factor() * dpsi / psi.scaling_factor()
        t = Mod(u + (dphi * dpsi) / u, p).lift_centered()
        return t

    redc, lift = padic_ring(F, maxprec)

    # recover data for the isogeny representation
    data_phi, vals_phi = padify_isogeny(phi, lift, prefix='phi_')
    data_psi, vals_psi = padify_isogeny(psi, lift, prefix='psi_')
    vals = vals_phi | vals_psi; del vals_phi, vals_psi

    # create constraints from [PRS26, Section 5]
    # the calls to _constraints generate the isogeny-chain constraints with all the steps
    # the last two make sure (domain, codomain) is the same for phi and psi
    funs_and_inpss = []
    funs_and_inpss.extend(_constraints(data_phi, vals, redc, prefix='phi_', pr=pr))
    funs_and_inpss.extend(_constraints(data_psi, vals, redc, prefix='psi_', pr=pr))
    funs_and_inpss.append(_equal(f'phi_a0', f'psi_a0', pr=pr))
    funs_and_inpss.append(_equal(f'phi_b0', f'psi_b0', pr=pr))
    funs_and_inpss.append(_equal(f'phi_a{len(data_phi)}', f'psi_a{len(data_psi)}', pr=pr))
    funs_and_inpss.append(_equal(f'phi_b{len(data_phi)}', f'psi_b{len(data_psi)}', pr=pr))

    # lift up to desired precision, doubling precision every time
    from hensel import hensel
    while True:
        prec, = {v.precision_absolute() for v in vals.values()}
        pr(f'\x1b[33m{prec = }\x1b[0m')
        if prec >= maxprec:
            break
        vals = hensel(funs_and_inpss, vals)

    # compute scaling factor of psi^ * phi
    uphi = prod(vals[f'phi_u{i}'] for i,(_,_,l,_) in enumerate(data_phi) if l.is_one())
    upsi = prod(vals[f'psi_u{i}'] for i,(_,_,l,_) in enumerate(data_psi) if l.is_one())
    u = uphi * dpsi / upsi

    # recover trace
    t = Mod(u + (dphi * dpsi) / u, p**maxprec).lift_centered()

    return t

################################################################

################ TESTS ################

if __name__ in ('__main__', 'sage.all'):
    import collections
    p = 0xc89  # or, much slower: 0xcf89
    deg = p**2.5  # approximate degree of endomorphism to be constructed

    # random supersingular curve
    E = EllipticCurve(GF(p**2), [0,1])
    for _ in range(42):
        E = choice(E.isogenies_prime_degree(2)).codomain()

    # collision finding: find two isogeny chains from E to the same E1
    tab = collections.defaultdict(list)
    while sum(len(vs) >= 2 for vs in tab.values()) < 3:
        phi = E.identity_morphism()
        while phi.degree() < sqrt(deg):
            step = choice(phi.codomain().isogenies_prime_degree(random_prime(6)))
            phi = step * phi
        tab[phi.codomain().j_invariant()].append(phi)

    vss = [vs for vs in tab.values() if len(vs) >= 2]
    shuffle(vss)
    vs = vss[0]
    shuffle(vs)
    phi, psi = vs[:2]
    print(f'{phi = }')
    psi = psi.codomain().isomorphism_to(phi.codomain()) * psi
    print(f'{psi = }')

    # computing a trace
    endo = phi.dual() * psi
    print()
    print(f'{endo = }')
    print()
    t = trace(endo)
    print()
    print(f'trace:   {t = }')
    print()
    assert t == endo.trace()

    # computing a trace pairing
    assert t == trace_pairing(endo, E.identity_morphism())

    s = trace_pairing(phi, psi)
    print()
    print(f'pairing: {s = }')
    print()
    assert s == (phi * psi.dual()).trace()

__all__ = ['padic_ring', 'padify_isogeny', '_codomainWI', '_ltorsion', '_codomainECI', '_equal', '_constraints', 'trace', 'trace_pairing']
