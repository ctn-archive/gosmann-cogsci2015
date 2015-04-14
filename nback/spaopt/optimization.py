"""Utilities for optimizing neuron populations and networks."""

from __future__ import absolute_import

import numpy as np
from scipy.special import beta, betainc

import nengo
from .dists import SubvectorLength


class SubvectorRadiusOptimizer(object):
    """Class to find the optimal radius for an ensemble representing a
    subvector of a semantic pointer.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the ensemble optimizing for.
    dimensions : int
        Number of dimensions represented in the ensemble optimizing for. This
        might be larger than the dimensionality of the subvector if different
        semantic pointers are combined in the ensemble.
    seed : int, optional
        Seed to be used to estimate the distortion of the ensemble.
    ens_kwargs : dict
        Further parameter settings of the ensemble optimizing for. (Can also be
        set with the config system.)
    conn_kwargs : dict
        Further parameter settings of the connection optimizing for. (Can also
        be set with the config system.)
    """

    def __init__(
            self, n_neurons, dimensions, seed=None, ens_kwargs={},
            conn_kwargs={}):
        m = nengo.Network(seed=seed, add_to_container=False)
        with m:
            conn = nengo.Connection(
                nengo.Ensemble(
                    n_neurons, dimensions, radius=1.0, **ens_kwargs),
                nengo.Ensemble(
                    n_neurons=1, dimensions=dimensions,
                    neuron_type=nengo.Direct()), **conn_kwargs)
        sim = nengo.Simulator(m)
        self.distortion = np.mean(np.square(
            sim.model.params[conn].solver_info['rmses']))

    def find_optimal_radius(self, sp_dimensions, sp_subdimensions=1):
        """Determines the optimal radius for ensembles when splitting up a
        semantic pointer (unit vector) into subvectors.

        Requires Scipy.

        Parameters
        ----------
        sp_dimensions : int
            Dimensionality of the complete semantic pointer/unit vector.
        sp_subdimensions : int, optional
            Dimensionality of the subvectors represented by the ensembles.

        Returns
        -------
        float
            Optimal radius for the representing ensembles.
        """
        import scipy.optimize
        res = scipy.optimize.minimize(
            lambda x: self.sp_subvector_error(
                x, sp_dimensions, sp_subdimensions), 0.1, bounds=[(0., 1.)])
        return np.asscalar(res.x)

    def sp_subvector_error(self, radius, sp_dimensions, sp_subdimensions=1):
        """Estimate of representational error of a subvector of a semantic
        pointer (unit vector).

        Requires Scipy.

        Paramaters
        ----------
        radius : float or ndarray
            Radius of the representing ensemble.
        sp_dimensions : int
            Dimensionality of the complete semantic pointer/unit vector.
        sp_subdimensions : int, optional
            Dimensionality of the subvector represented by some ensemble.

        Returns
        -------
        Error estimates for representing a subvector with `subdimensions`
        dimensions of a `dimensions` dimensional unit vector with an ensemble
        initialized with of `radius`.
        """
        dist = SubvectorLength(sp_dimensions, sp_subdimensions)
        in_range = self._sp_subvector_error_in_range(radius, sp_subdimensions)
        out_of_range = self._sp_subvector_error_out_of_range(
            radius, sp_dimensions, sp_subdimensions)
        return dist.cdf(radius) * in_range + (
            1.0 - dist.cdf(radius)) * out_of_range

    def _sp_subvector_error_in_range(self, radius, subdimensions):
        return radius * radius * subdimensions * self.distortion

    def _sp_subvector_error_out_of_range(
            self, radius, dimensions, subdimensions):
        dist = SubvectorLength(dimensions, subdimensions)
        sq_r = radius * radius

        normalization = 1.0 - dist.cdf(radius)
        b = (dimensions - subdimensions) / 2.0
        aligned_integral = beta(subdimensions / 2.0 + 1.0, b) * (1.0 - betainc(
            subdimensions / 2.0 + 1.0, b, sq_r))
        cross_integral = beta((subdimensions + 1) / 2.0, b) * (1.0 - betainc(
            (subdimensions + 1) / 2.0, b, sq_r))

        numerator = (sq_r * normalization + (
            aligned_integral - 2.0 * radius * cross_integral) / beta(
            subdimensions / 2.0, b))
        with np.errstate(invalid='ignore'):
            return np.where(
                numerator > np.MachAr().eps,
                numerator / normalization, np.zeros_like(normalization))
