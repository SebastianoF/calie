from fractions import Fraction
from scipy.misc import factorial
from scipy.misc import comb as binomial


# ------------- auxiliary method for the BCH formula ----------------


def fact(n):
    """
    Covers the predefined function factorial from scipy.misc
    :param n:
    :return: factorial of n type float
    """
    return float(factorial(n, True))


def bern(n):
    """
    bern(n) \n
    :param n: integer n
    :return: nth Bernoulli number (type Fraction)
    (iterative algorithm, do not uses polynomials)
    """
    if n == 1:
        ans = Fraction(1, 2)
    elif n % 2 == 1:
        ans = Fraction(0, 1)
    else:
        n += 1
        a = [0] * (n + 1)
        for m in range(n + 1):
            a[m] = Fraction(1, m + 1)
            for j in range(m - 1, 0, -1):
                a[j - 1] = j * (a[j - 1] - a[j])
        ans = a[0]
    return ans  # type Fraction.
    # return float(ans) # type Float


def bernoulli_poly(x, n):
    """
    bernoulli_poly(x,n) \n
    :param x: value for the unknown.
    :param n: degree of the polynomial.
    :return: j-th bernoulli polynomial evaluate at x (unknown of the poly).
    """
    return sum([binomial(n, k) * bern(n - k) * (x ** k) for k in range(n)])


def bernoulli_numb_via_poly(n):
    """
    bernoulli_numb(n) \n
    :param n: integer n
    :return: nth Bernoulli number
    (uses first type bernoulli polynomials)
    """
    return bernoulli_poly(0, n)
