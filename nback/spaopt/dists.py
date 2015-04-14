import numpy as np

from nengo.dists import Distribution


class SqrtBeta(Distribution):
    """Distribution of the square root of a Beta distributed random variable.

    Given `n + m` dimensional random unit vectors, the length of subvectors
    with `m` elements will be distributed according to this distribution.

    Parameters
    ----------
    n, m : Number
        Shape parameters of the distribution.

    See also
    --------
    SubvectorLength
    """
    def __init__(self, n, m=1):
        super(SqrtBeta, self).__init__()
        self.n = n
        self.m = m

    def sample(self, num, d=None, rng=np.random):
        shape = (num,) if d is None else (num, d)
        return np.sqrt(rng.beta(self.m / 2.0, self.n / 2.0, size=shape))

    def pdf(self, x):
        """Probability distribution function.

        Requires Scipy.

        Parameters
        ----------
        x : ndarray
            Evaluation points in [0, 1].

        Returns
        -------
        ndarray
            Probability density at `x`.
        """
        from scipy.special import beta
        return (2 / beta(self.m / 2.0, self.n / 2.0) * x ** (self.m - 1) *
                (1 - x * x) ** (self.n / 2.0 - 1))

    def cdf(self, x):
        """Cumulative distribution function.

        Requires Scipy.

        Parameters
        ----------
        x : ndarray
            Evaluation points in [0, 1].

        Returns
        -------
        ndarray
            Probability that `X <= x`.
        """
        from scipy.special import betainc
        sq_x = x * x
        return np.where(
            sq_x < 1., betainc(self.m / 2.0, self.n / 2.0, sq_x),
            np.ones_like(x))


class SubvectorLength(SqrtBeta):
    """Distribution of the length of a subvectors of a unit vector.

    Parameters
    ----------
    dimensions : int
        Dimensionality of the complete unit vector.
    subdimensions : int, optional
        Dimensionality of the subvector.

    See also
    --------
    SqrtBeta
    """
    def __init__(self, dimensions, subdimensions=1):
        super(SubvectorLength, self).__init__(
            dimensions - subdimensions, subdimensions)
