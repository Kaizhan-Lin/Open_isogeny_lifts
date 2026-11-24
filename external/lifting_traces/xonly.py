
def xDBL(a, b, X1):
    r"""
    Given an x-coordinate :math:`X_1 = x(P)`
    on the curve :math:`y^2 = x^3 + ax + b`,
    returns the x-coordinate :math:`x([2]P)`.
    """
    # https://hyperelliptic.org/EFD/g1p/auto-code/shortw/xz/doubling/dbl-2002-it-2.op3
    # simplified using Z1=1
    T1 = X1**2
    T4 = T1-a
    T5 = T4**2
    T8 = b*X1
    T9 = 8*T8
    X3 = T5-T9
    T10 = T1+a
    T11 = X1*T10
    T13 = T11+b
    Z3 = 4*T13
    if not Z3:
        raise ZeroDivisionError
    return X3/Z3

def xADD(a, b, X1, X2, X3):
    r"""
    Given x-coordinates :math:`X_1 = x(P-Q),\, X_2 = x(P),\, X_3 = x(Q)`
    on the curve :math:`y^2 = x^3 + ax + b`,
    returns the x-coordinate :math:`x(P+Q)`.
    """
    if X1:
        # https://hyperelliptic.org/EFD/g1p/auto-code/shortw/xz/diffadd/mdadd-2002-it-3.op3
        # simplified using Z2=Z3=1
        T1 = X2*X3
        T6 = T1-a
        T7 = T6**2
        T9 = 4*b
        T10 = X2+X3
        T11 = T9*T10
        X5 = T7-T11
        T13 = X2-X3
        T14 = T13**2
        Z5 = X1*T14
    else:
        # https://hyperelliptic.org/EFD/g1p/auto-code/shortw/xz/diffadd/mdadd-2002-it-2.op3
        # simplified using Z2=Z3=1
        t2 = X2*X3
        t5 = X2+X3
        t6 = t2+a
        t11 = 4*b
        t12 = t5*t6
        t13 = 2*t12
        R = t13+t11
        t16 = X2-X3
        S = t16**2
        t17 = S*X1
        X5 = R-t17
        Z5 = S
    return X5/Z5

def x_multiples(a, b, x, howmany):
    r"""
    Given an x-coordinate :math:`x(P)`
    on the curve :math:`y^2 = x^3 + ax + b`,
    returns all the roots of the kernel polynomial
    corresponding to :math:`\langle P\rangle`.
    """
    assert howmany
    if x is None:
        return []
    xs = [x]
    try:
        x = xDBL(a, b, x)
    except ZeroDivisionError:    # point of order two
#        assert x**3 + a*x + b == 0
        return xs
    while len(xs) < howmany:
        xs.append(x)
        x = xADD(a, b, xs[-2], xs[-1], xs[0])
    return xs

