import errno
import os
import os.path
import random
import subprocess

from nback.nbackgen import NBackBuilder


DATA_PATH = 'data'
CONFIG_FILENAME = 'conf.py'

with open(os.path.join(DATA_PATH, CONFIG_FILENAME), 'r') as f:
    conf = eval(f.read())


def create_n_back_seqs(n, targets):
    out_directory = os.path.dirname(targets[0])

    try:
        os.makedirs(out_directory)
    except OSError as err:
        if err.errno == errno.EEXIST:
            pass
        else:
            raise

    blocks_created = 0
    i = 0
    seeds = []
    while blocks_created < conf['blocks']:
        builder = NBackBuilder(conf['alphabet'])
        builder.n(n).trials(conf['trials']).match_trials(conf['match_trials'])
        builder.horizon(conf['horizon'])
        for lure, rate in conf['lure_rates'][n]:
            builder.lure_rate(lure, rate)
        i += 1
        random.seed(i)
        try:
            seq, conditions = builder.build_list()
        except NBackBuilder.MissingLuresError:
            continue

        output_path = os.path.join(out_directory, '{}.txt'.format(
            blocks_created))
        with open(output_path, 'w') as f:
            f.writelines((seq, '\n', conditions, '\n'))

        blocks_created += 1

    with open(os.path.join(out_directory, 'seeds'), 'w') as f:
        f.write(str(seeds))


def task_nback_gen():
    """Generate n-back task files."""

    for n in [1, 2, 3]:
        name = '{}back'.format(n)
        yield {
            'name': name,
            'file_dep': [os.path.join(DATA_PATH, CONFIG_FILENAME)],
            'actions': [(create_n_back_seqs, (n,))],
            'targets': [os.path.join(DATA_PATH, name, '{}.txt'.format(i))
                        for i in xrange(conf['blocks'])] +
                       [os.path.join(DATA_PATH, name, 'seeds')]
        }


class Trial(object):
    def __init__(self, n, seed):
        self.n = n
        self.seed = seed

        self.inpath = os.path.join(DATA_PATH, '{}back'.format(self.n))
        self.outpath = os.path.join(DATA_PATH, 'out_{}back'.format(self.n))
        self.input_file = os.path.abspath(
            os.path.join(self.inpath, str(seed) + '.txt'))
        self.output_file = os.path.abspath(
            os.path.join(self.outpath, str(seed) + '.npz'))

    def to_task(self):
        return {
            'name': '(n={}, seed={})'.format(self.n, self.seed),
            'file_dep': [self.input_file],
            'targets': [self.output_file],
            'actions': [(self, ())]
        }

    def __call__(self):
        try:
            os.mkdir(self.outpath)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

        python_call = ['python']
        model_call = [
            os.path.join(os.path.dirname(__file__), 'nback/model.py'),
            '--symbols', conf['alphabet'], '-i', self.input_file,
            '-o', self.output_file, '-s', str(self.seed), '-n', str(self.n)]

        rval = subprocess.call(python_call + model_call)
        if rval != 0:
            raise Exception()  # FIXME raise something helpful


def task_run_sim():
    """Run simulations."""

    for n in [1, 2, 3]:
        for seed in range(conf['blocks']):
            yield Trial(n, seed).to_task()
