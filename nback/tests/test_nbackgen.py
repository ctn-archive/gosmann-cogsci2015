import random

import pytest

from ..nbackgen import NBackBuilder


class TestNBackBuilder(object):
    @pytest.fixture(autouse=True)
    def set_py_seed(self):
        random.seed(42)

    def test_produces_requested_number_of_match_trials(self):
        block, _ = NBackBuilder().trials(10).match_trials(.2).n(2).build_list()
        assert self._count_match_trials(block, 2) == 2

    def test_produces_no_unrequested_lures(self):
        # note: probabilistic test
        block, _ = NBackBuilder('abcd') \
            .trials(100).match_trials(0.2).n(1).horizon(3).build_list()
        assert self._count_lure_trials(block, 1, 1) == 0
        assert self._count_lure_trials(block, 1, 2) == 0

    def test_produces_requested_number_of_lures(self):
        block, _ = NBackBuilder() \
            .trials(30).match_trials(0.2).n(2) \
            .lure_rate(1, 0.1).lure_rate(-1, 0.05).build_list()
        assert self._count_lure_trials(block, 2, 1) == int(0.1 * 0.8 * 30)
        assert self._count_lure_trials(block, 2, -1) == int(0.05 * 0.8 * 30)

    def test_raises_exception_if_not_all_lures_could_be_inserted(self):
        with pytest.raises(NBackBuilder.MissingLuresError):
            NBackBuilder() \
                .trials(10).match_trials(0.2).n(2) \
                .lure_rate(1, 1.0).lure_rate(-1, 1.0).build_list()

    @pytest.mark.parametrize('n,seed', [(2, 42), (1, 11)])
    def test_produces_condition_string(self, n, seed):
        random.seed(seed)
        h = 3
        builder = NBackBuilder() \
            .horizon(h).trials(30).match_trials(0.2).n(n).lure_rate(1, 0.1)
        if n == 1:
            builder.lure_rate(2, 0.05)
        elif n == 2:
            builder.lure_rate(-1, 0.05)
        else:
            raise NotImplementedError()
        block, conditions = builder.build_list()
        for i, c in enumerate(conditions):
            assert (
                (i < 2 and c != '-') or
                (c == '-' and block[i] not in block[i - n:i]) or
                (c == 'm' and block[i] == block[i - n]) or
                (c == 'l' and block[i] in block[i - h:i] and
                    block[i] != block[i - n]))

    def test_returns_strings(self):
        a, b = NBackBuilder().trials(10).match_trials(.2).n(2).build_list()
        assert isinstance(a, str) and isinstance(b, str)

    def test_returned_strings_match_in_length(self):
        a, b = NBackBuilder().trials(10).match_trials(0.2).n(1).build_list()
        assert len(a) == len(b)

    def _count_match_trials(self, block, n):
        return sum(1 for i, t in enumerate(block[n:]) if block[i] == t)

    def _count_lure_trials(self, block, n, diff):
        return sum(
            1 for i, t in enumerate(block)
            if min(i - n, i - n - diff) > 0 and
            block[i] != block[i - n] and block[i] == block[i - n - diff])
