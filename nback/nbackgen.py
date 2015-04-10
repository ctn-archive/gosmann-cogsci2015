#!/usr/bin/env python

from random import choice, randint, random, sample, seed, shuffle


class NBackBuilder(object):
    def __init__(self, alphabet='bcdfghjklmnpqrstvwxyz'):
        self.alphabet = alphabet
        self._horizon = 3
        self._lure_rates = {}
        self._match_trials = None
        self._n = None
        self._trials = None

    def horizon(self, value):
        """How many preceding characters are disallowed in a mismatch trial
        because they are considered to be a lure."""
        self._horizon = value
        return self

    def lure_rate(self, diff, value):
        self._lure_rates[diff] = value
        return self

    def match_trials(self, value):
        """Relative amount of match trials."""
        self._match_trials = value
        return self

    def n(self, value):
        self._n = value
        return self

    def trials(self, value):
        """Absolute number of overall trials."""
        self._trials = value
        return self

    def build_list(self):
        num_match_trials = int(self._match_trials * self._trials)
        num_mismatch_trials = self._trials - num_match_trials

        trial_types = ['match'] * num_match_trials + \
            ['mismatch'] * num_mismatch_trials
        shuffle(trial_types)

        lures = []
        for k, v in self._lure_rates.items():
            lures.extend([k] * int(v * num_mismatch_trials))

        l = sample(self.alphabet, self._n)
        conditions = len(l) * ['-']
        num_remaining_mismatches = num_mismatch_trials
        for t in trial_types:
            if t == 'match':
                l.append(self._match_trial(l))
                conditions.append('m')
            else:
                invalid = list(l[-min(len(l), self._horizon):])
                valid_lures = [
                    i for i, d in enumerate(lures)
                    if self._n + d < len(l) and
                    l[-self._n - d] not in invalid[:-self._n - d] and
                    l[-self._n - d] not in invalid[-self._n - d:][1:]]
                use_lure = len(valid_lures) > 0 and \
                    random() < float(len(lures)) / num_remaining_mismatches
                if use_lure:
                    i = choice(valid_lures)
                    l.append(self._lure_trial(lures[i], l))
                    del lures[i]
                    conditions.append('l')
                else:
                    l.append(self._mismatch_trial(l))
                    conditions.append('-')
                num_remaining_mismatches -= 1
        if len(lures) > 0:
            raise self.MissingLuresError(
                'Could not insert all lures. It might work with a different '
                'random number generator seed. If not the parameters have to '
                'be changed.')
        return ''.join(l), ''.join(conditions)

    def _match_trial(self, l):
        return l[-self._n]

    def _mismatch_trial(self, l):
        invalid = []
        horizon = min(len(l), self._horizon)
        if horizon > 0:
            invalid.extend(l[-horizon:])
        if len(l) >= self._n:
            invalid.append(l[-self._n])
        valid = [c for c in self.alphabet if c not in invalid]
        i = randint(0, len(valid) - 1)
        return valid[i]

    def _lure_trial(self, diff, l):
        v = l[-(self._n + diff)]
        return v

    class MissingLuresError(Exception):
        pass


if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser(
        description='Generates n-back task stimulus lists.')
    PARSER.add_argument(
        '-n', nargs=1, type=int, required=True,
        help='n, i.e. how many items one has to look back in the task.')
    PARSER.add_argument(
        '-t', '--trials', nargs=1, type=int, default=[45],
        help='Number of trials to generate.')
    PARSER.add_argument(
        '--match-trials', nargs=1, type=float, default=[1.0 / 3.0],
        help='Relative amount of match trials.')
    PARSER.add_argument(
        '--horizon', nargs=1, type=int, default=[3],
        help='Number of the last few characters to be considered to be a '
        'match or lure if they reoccur.')
    PARSER.add_argument(
        '--lure', type=float, nargs=2, action='append', default=[],
        help='Given a and b this defines a proportion of b lure trials of all '
        'mismatch trials at the n + a position')
    PARSER.add_argument(
        '--alphabet', type=str, nargs=1,
        help='Defines the valid characters in the n-back task.')
    PARSER.add_argument(
        '--seed', type=int, nargs=1, help='Random number generator seed.')

    args = PARSER.parse_args()

    if args.seed is not None:
        seed(args.seed[0])

    if args.alphabet is not None:
        builder = NBackBuilder(args.alphabet[0])
    else:
        builder = NBackBuilder()

    builder.n(args.n[0]).trials(args.trials[0])
    builder.match_trials(args.match_trials[0]).horizon(args.horizon[0])

    for delta, proportion in args.lure:
        builder.lure_rate(int(delta), proportion)

    built = builder.build_list()
    print built[0]
    print built[1]
