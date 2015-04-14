import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.compat import is_number
from nengo.dists import Choice
from .optimization import SubvectorRadiusOptimizer


class Product(nengo.Network):
    """Computes the element-wise product of two (scaled) unit vectors.

    Requires Scipy.
    """

    def __init__(
            self, n_neurons, dimensions, radius=1.0,
            encoders=nengo.Default, **ens_kwargs):
        super(Product, self).__init__(self)

        with self:
            self.config[nengo.Ensemble].update(ens_kwargs)
            self.A = nengo.Node(size_in=dimensions, label="A")
            self.B = nengo.Node(size_in=dimensions, label="B")
            self.dimensions = dimensions

            if encoders is nengo.Default:
                encoders = Choice([[1, 1], [1, -1], [-1, 1], [-1, -1]])

            optimizer = SubvectorRadiusOptimizer(
                n_neurons, 2, ens_kwargs=ens_kwargs)
            scaled_r = radius * optimizer.find_optimal_radius(dimensions, 1)

            self.product = EnsembleArray(
                n_neurons, n_ensembles=dimensions, ens_dimensions=2,
                radius=scaled_r, encoders=encoders, **ens_kwargs)

            nengo.Connection(
                self.A, self.product.input[::2], synapse=None)
            nengo.Connection(
                self.B, self.product.input[1::2], synapse=None)

            self.output = self.product.add_output(
                'product', lambda x: x[0] * x[1])

    def dot_product_transform(self, scale=1.0):
        """Returns a transform for output to compute the scaled dot product."""
        return scale * np.ones((1, self.dimensions))
