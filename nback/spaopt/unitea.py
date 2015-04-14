import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.params import Default
from nengo.utils.compat import is_number
from .optimization import SubvectorRadiusOptimizer


class UnitEA(EnsembleArray):

    def __init__(self, n_neurons, dimensions, n_ensembles, ens_dimensions=1,
                 label=None, radius=1.0, **ens_kwargs):
        if dimensions % n_ensembles != 0:
            raise ValueError(
                "'dimensions' has to be divisible by 'n_ensembles'.")

        optimizer = SubvectorRadiusOptimizer(
            n_neurons, ens_dimensions, ens_kwargs=ens_kwargs)
        scaled_r = radius * optimizer.find_optimal_radius(
            dimensions, dimensions // n_ensembles)

        super(UnitEA, self).__init__(
            n_neurons, n_ensembles, ens_dimensions, label=label,
            radius=scaled_r, **ens_kwargs)
