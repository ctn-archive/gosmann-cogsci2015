try:
    import faulthandler
    faulthandler.enable()
except:
    pass

import argparse
import random

import nengo
import spaopt as spa
from spaopt import CircularConvolution
import numpy as np

parser = argparse.ArgumentParser(description="n-back model")
parser.add_argument(
    '--symbols', nargs=1, type=str, help="The set of stimulus symbols",
    default=['bcdfghjklmnpqrstvwxz'])
parser.add_argument(
    '-n', '--n', nargs=1, type=int, help="n variable", required=True)
parser.add_argument(
    '-d', '--dimensions', nargs=1, type=int,
    help="Dimensionality of representation", default=[64])
parser.add_argument(
    '--subdimensions', nargs=1, type=int,
    help="Subdimensions per ensemble", default=[1])
parser.add_argument(
    '-N', '--neurons', nargs=1, type=int,
    help="Number of neurons per dimension.", default=[50])
parser.add_argument(
    '-i', '--infile', nargs=1, type=str, help="n-back input sequence",
    required=True)
parser.add_argument(
    '-o', '--outfile', nargs=1, type=str, help="Output file",
    required=True)
parser.add_argument(
    '-s', '--seed', nargs=1, type=int, help="Random number generator seed.")
parser.add_argument(
    '-p', '--progress', default=False, action='store_true',
    help="Show a progress bar.")
args = parser.parse_args()

symbols = [c.upper() for c in args.symbols[0]]
d = args.dimensions[0]
sd = args.subdimensions[0]
neurons_per_dimension = args.neurons[0]
input_filename = args.infile[0]
output_filename = args.outfile[0]
if args.seed is None:
    seed = None
else:
    seed = args.seed[0]
    random.seed(3 * seed)

vocab = spa.Vocabulary(d)
vocab.add('Ctx', vocab.create_pointer(unitary=True))
for s in symbols:
    vocab.parse(s)
conf = {
    'dimensions': d,
    'subdimensions': sd,
    'neurons_per_dimension': neurons_per_dimension,
    'vocab': vocab
}

state_vocab = spa.Vocabulary(d)
state_vocab.parse('Encode + Wait + Transfer')
state_conf = {
    'dimensions': d,
    'subdimensions': sd,
    'neurons_per_dimension': neurons_per_dimension,
    'vocab': state_vocab
}


def inhibit(pre, post):
    for e in post.ensembles:
        nengo.Connection(
            pre, e.neurons, transform=[[-5]] * e.n_neurons, synapse=tau_gaba)


class Control(object):
    def __init__(self, input_path):
        f = open(input_path, 'r')
        try:
            self.nback_seq = f.readline().strip().upper()
        finally:
            f.close()
        self.trial_duration = 2.5
        self.encode_duration = 0.5

    def trial_t(self, t):
        return t % self.trial_duration

    def stim_in(self, t):
        i = int(t // self.trial_duration)
        if i >= len(self.nback_seq) or self.trial_t(t) > self.encode_duration:
            return '0'
        return self.nback_seq[i]

    def cue(self, t):
        return '*'.join(args.n[0] * ['Ctx'])


class NBack(spa.SPA):
    def __init__(self, seed):
        super(NBack, self).__init__(seed=seed)
        x = 1.0 / np.sqrt(args.n[0])

        with self:
            self.ctrl = Control(input_filename)

            self.state = spa.Buffer(**conf)

            self.stim_in = spa.Buffer(**conf)
            self.stim_gate = spa.Buffer(**conf)
            self.stim = spa.Memory(synapse=0.1, **conf)

            self.wm_in = spa.Buffer(**conf)
            self.gate1 = spa.Buffer(**conf)
            self.wm1 = spa.Memory(synapse=0.1, **conf)

            self.gate2 = spa.Buffer(**conf)
            self.wm2 = spa.Memory(synapse=0.1, **conf)

            self.gate3 = spa.Buffer(**conf)
            self.wm3 = spa.Memory(synapse=0.1, **conf)

            self.cue = spa.Buffer(**conf)
            self.cc = CircularConvolution(
                n_neurons=200, dimensions=d, invert_b=True)
            self.comp = spa.Compare(
                dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                vocab=vocab)

            self.response = nengo.Ensemble(neurons_per_dimension, 1)
            nengo.Connection(self.response, self.response, synapse=0.1)
            self.rectify = nengo.Ensemble(
                neurons_per_dimension, 1, intercepts=nengo.dists.Uniform(-0., 1),
                encoders=nengo.dists.Choice([[1]]))
            nengo.Connection(
                self.comp.output, self.rectify,
                transform=[2 * vocab.parse('YES').v])
            nengo.Connection(self.rectify, self.response,
                function=lambda y: np.maximum(0, y))
            nengo.Connection(nengo.Node(output=-np.exp(-args.n[0] / 0.62) - 0.2), self.response)

            self.stim_in_dot = spa.Compare(
                dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                vocab=vocab)
            nengo.Connection(
                self.stim_in.state.output, self.stim_in_dot.inputA)
            nengo.Connection(
                self.stim_in.state.output, self.stim_in_dot.inputB)
            self.wm1_dot = spa.Compare(
                dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                vocab=vocab)
            nengo.Connection(self.wm1.state.output, self.wm1_dot.inputA)
            nengo.Connection(self.gate1.state.output, self.wm1_dot.inputB)
            self.wm2_dot = spa.Compare(
                dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                vocab=vocab)
            nengo.Connection(self.wm2.state.output, self.wm2_dot.inputA)
            nengo.Connection(self.gate1.state.output, self.wm2_dot.inputB)

            self.bg = spa.BasalGanglia(spa.Actions(
                '0.2 --> state = Encode',
                'dot(state, Encode) + dot(state, Wait) --> state = Wait',
                'dot(state, Transfer) --> state = Transfer',
            ))
            self.thalamus = spa.Thalamus(self.bg)

            nengo.Connection(
                self.stim_in_dot.output, self.bg.input[0],
                transform=[vocab.parse('YES').v])
            inhibit(self.thalamus.output[0], self.gate2.state)
            nengo.Connection(
                self.thalamus.output[0], self.response.neurons,
                transform=[[-5]] * neurons_per_dimension, synapse=tau_gaba)
            inhibit(self.thalamus.output[1], self.stim_gate.state)
            inhibit(self.thalamus.output[1], self.gate1.state)
            inhibit(self.thalamus.output[1], self.gate3.state)
            self.e1 = nengo.Ensemble(neurons_per_dimension, 1, intercepts=nengo.dists.Uniform(0.4, 1), encoders=nengo.dists.Choice([[1]]))
            self.e2 = nengo.Ensemble(neurons_per_dimension, 1, intercepts=nengo.dists.Uniform(0.8, 1), encoders=nengo.dists.Choice([[-1]]))
            nengo.Connection(self.response, self.e1)
            nengo.Connection(self.response, self.e2)
            nengo.Connection(self.e1, self.bg.input[2], function=lambda x: 1 if x > 0.5 else 0)
            nengo.Connection(self.e2, self.bg.input[2], function=lambda x: 1 if x < -0.9 else 0)
            inhibit(self.thalamus.output[2], self.gate1.state)
            inhibit(self.thalamus.output[2], self.stim_gate.state)
            nengo.Connection(
                self.thalamus.output[2], self.response.neurons,
                transform=[[-5]] * neurons_per_dimension, synapse=tau_gaba)

            y = np.sqrt((args.n[0] - 1) * x * x)
            self.cortical1 = spa.Cortical(spa.Actions(
                'wm_in = {x} * stim_in + {y} * wm2'.format(
                    x=x, y=y),
                'gate1 = wm_in - wm1',
                'wm1 = 3 * gate1',
                'gate2 = wm1 * Ctx - wm2',
                'wm2 = gate2',
                'gate3 = wm2 - wm3',
                'wm3 = 3 * gate3',
                'stim_gate = stim_in - stim',
                'stim = stim_gate',), synapse=0.005)

            nengo.Connection(self.wm3.state.output, self.cc.A)
            nengo.Connection(self.cue.state.output, self.cc.B)
            nengo.Connection(self.stim.state.output, self.comp.inputA)
            nengo.Connection(self.cc.output, self.comp.inputB)

            self.input = spa.Input(
                stim_in=self.ctrl.stim_in, cue=self.ctrl.cue)


tau_gaba = 0.00848
model = NBack(seed=seed)
with model:
    comp_out = nengo.Node(size_in=1)
    nengo.Connection(
        model.comp.output, comp_out, synapse=0.005,
        transform=[vocab.parse('YES').v])
    p_comp_out = nengo.Probe(comp_out)

    p_response = nengo.Probe(model.response, synapse=0.005)


sim = nengo.Simulator(model)
sim.run(
    model.ctrl.trial_duration * len(model.ctrl.nback_seq) + 0.01,
    progress_bar=args.progress)
np.savez(
    output_filename, trange=sim.trange(), comp_out=sim.data[p_comp_out],
    response=sim.data[p_response])
