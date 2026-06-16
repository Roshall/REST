"""
Comprehensive comparison tests between old PatternPool (max_obj.py)
and new PatternPool (pattern_poolv2.py).

Both implementations are tested with identical inputs and their outputs
(from concatenate, pop_all, iter_until) are compared.

Design constraints:
- Only VALID patterns are provided to concatenate (objects always satisfy
  the label requirement).
- Result order is NOT significant; comparisons sort before asserting.
- label_count is adapted to each implementation's expected convention:
    old code: negative counters  ({label: -required_count})
    new code: positive requirement vector ({label: required_count})
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "search", "index_based"))

import pytest
from itertools import islice

from search.index_based.pattern_poolv2 import PatternPool as NewPatternPool


# ---------------------------------------------------------------------------
# Old PatternPool extracted from max_obj.py to avoid heavy import chain
# (search.rest -> search.verifier -> traj_seg C extension).
# ---------------------------------------------------------------------------
class OldPatternPool:
    """Exact copy of PatternPool from search/index_based/max_obj.py."""

    def __init__(self):
        self.patterns = []
        self.end = 0

    @staticmethod
    def _label_verifier(label_counter):
        for count in label_counter.values():
            if count < 0:
                return False
        return True

    def concatenate(self, objs_m, label_count, start, end):
        objs = set(objs_m)
        if not self.patterns:
            self.patterns = [(objs, start)]
            self.end = end
            return
        elif start > (end_l := self.end):
            for p_objs, s in self.patterns:
                yield p_objs, s, end_l
            self.patterns = [(objs, start)]
            self.end = end
            return

        new = []
        count = 0
        adopt = False
        cur = objs
        for p, ps in self.patterns:
            clen, plen = len(cur), len(p)

            if clen > plen:
                if cur > p:
                    new.append((cur, start))
                    adopt = True
                    break
            elif clen < plen:
                if cur < p:
                    start = ps
                    count += 1
                    continue
            elif clen == plen:
                if cur == p:
                    adopt = True
                    break

            new.append((cur, start))
            # update new pattern
            p_new = cur & p
            for o in cur - p_new:
                label_count[objs_m[o]] -= 1

            if self._label_verifier(label_count):
                start = ps
                cur = p_new
                count += 1
            else:
                break
        else:
            new.append((cur, start))

        p_iter = iter(self.patterns)
        for objs, s in islice(p_iter, count if adopt else None):
            yield list(objs), s, self.end
        new.extend(p_iter)
        self.patterns = new
        self.end = end

    def pop_all(self):
        for objs, start in self.patterns:
            yield list(objs), start, self.end
        self.patterns = []
        self._levels = {}

    def iter_until(self, max_start):
        for objs, start in self.patterns:
            if start <= max_start:
                yield list(objs), start, self.end
            else:
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neg(labels: dict[int, int]) -> dict[int, int]:
    """Negative convention for the old code."""
    return {l: -c for l, c in labels.items()}


def _pos(labels: dict[int, int]) -> dict[int, int]:
    """Positive requirement vector for the new code."""
    return dict(labels)


def collect_concatenate(old_pool, new_pool, objs_m, labels, start, end):
    """Run concatenate on both pools with adapted label_count conventions.

    `labels` is a positive requirement dict, e.g. {0: 2, 1: 1}.
    The old pool receives {-2, -1} and the new pool receives {2, 1}.
    Returns (old_results, new_results).
    """
    old_results = list(old_pool.concatenate(objs_m, _neg(labels).copy(), start, end))
    new_results = list(new_pool.concatenate(objs_m, _pos(labels).copy(), start, end))
    return old_results, new_results


def collect_pop_all(old_pool, new_pool):
    return list(old_pool.pop_all()), list(new_pool.pop_all())


def collect_iter_until(old_pool, new_pool, max_start):
    return list(old_pool.iter_until(max_start)), list(new_pool.iter_until(max_start))


def _normalize(pattern_tuple):
    objs, start, end = pattern_tuple
    return (tuple(sorted(objs)), start, end)


def sorted_results(results):
    """Normalize and sort results for order-insensitive comparison."""
    return sorted(_normalize(r) for r in results)


def assert_equal(old_results, new_results, msg=""):
    assert sorted_results(old_results) == sorted_results(new_results), msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicConcatenate:
    """Test basic concatenate behavior."""

    def test_empty_pool_first_insert(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        objs_m = {1: 0, 2: 0, 3: 1}
        labels = {0: 2, 1: 1}

        old_r, new_r = collect_concatenate(old_pool, new_pool, objs_m, labels, 0, 10)
        assert old_r == []
        assert new_r == []
        assert old_pool.end == new_pool.end == 10

    def test_empty_pool_pop_all(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        objs_m = {1: 0, 2: 0, 3: 1}
        labels = {0: 2, 1: 1}

        collect_concatenate(old_pool, new_pool, objs_m, labels, 0, 10)
        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_gap_between_windows(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 5)
        old_r, new_r = collect_concatenate(old_pool, new_pool, {2: 0}, labels, 10, 15)
        assert_equal(old_r, new_r)
        assert old_pool.end == new_pool.end == 15

    def test_consecutive_gap_pop_all(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 5)

        r1_old, r1_new = collect_concatenate(old_pool, new_pool, {2: 0}, labels, 10, 15)
        assert_equal(r1_old, r1_new)

        r2_old, r2_new = collect_concatenate(old_pool, new_pool, {3: 0}, labels, 20, 25)
        assert_equal(r2_old, r2_new)


class TestIdenticalObjects:
    """When current objects are identical to the existing pattern."""

    def test_same_objects_same_start(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        objs_m = {1: 0, 2: 0}
        labels = {0: 2}

        collect_concatenate(old_pool, new_pool, objs_m, labels, 0, 10)
        old_r, new_r = collect_concatenate(old_pool, new_pool, objs_m, labels, 5, 15)
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_same_objects_different_start(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        objs_m = {1: 0, 2: 0, 3: 1}
        labels = {0: 2, 1: 1}

        collect_concatenate(old_pool, new_pool, objs_m, labels, 0, 10)
        old_r, new_r = collect_concatenate(old_pool, new_pool, objs_m, labels, 3, 13)
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestObjectAddition:
    """When new objects are added (superset)."""

    def test_add_one_object(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        old_r, new_r = collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 3, 13)
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_add_multiple_objects(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        old_r, new_r = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0, 3: 0}, labels, 3, 13,
        )
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestObjectRemoval:
    """When some objects are lost (subset).  Always still valid."""

    def test_remove_one_object_still_valid(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 2}

        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0, 3: 0}, labels, 0, 10)
        old_r, new_r = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0}, labels, 5, 15,
        )
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_remove_multiple_objects(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0, 3: 0, 4: 0}, labels, 0, 10,
        )
        old_r, new_r = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0}, labels, 5, 15,
        )
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestMixedChanges:
    """When some objects are added and some are removed."""

    @pytest.mark.xfail(reason=(
        "Old code retains emitted patterns in storage as duplicates. "
        "V2 correctly discards them. pop_all results differ because "
        "the old code yields an extra stale pattern."
    ), strict=True)
    def test_add_and_remove(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 0, 10)
        old_r, new_r = collect_concatenate(
            old_pool, new_pool, {1: 0, 3: 0}, labels, 3, 13,
        )
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_complete_replacement(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 0, 10)
        old_r, new_r = collect_concatenate(
            old_pool, new_pool, {3: 0, 4: 0}, labels, 3, 13,
        )
        assert_equal(old_r, new_r)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestMultipleConcatenations:
    """Sequences of concatenations building up complex state."""

    def test_three_consecutive_supersets(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)

        r2_old, r2_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0}, labels, 3, 13,
        )
        assert_equal(r2_old, r2_new)

        r3_old, r3_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0, 3: 0}, labels, 6, 16,
        )
        assert_equal(r3_old, r3_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_grow_then_shrink(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 3, 13)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0, 3: 0}, labels, 6, 16)

        r_old, r_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0}, labels, 9, 19,
        )
        assert_equal(r_old, r_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_grow_shrink_grow(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0, 3: 0}, labels, 3, 13)

        r1_old, r1_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0}, labels, 6, 16,
        )
        assert_equal(r1_old, r1_new)

        r2_old, r2_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0, 4: 0}, labels, 9, 19,
        )
        assert_equal(r2_old, r2_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_multiple_labels_grow_shrink(self):
        """Multiple label types with grow/shrink — always valid."""
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1, 1: 1}

        # {1:L0, 2:L1}
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 1}, labels, 0, 10)

        # Add 3:L0 → {1:L0, 2:L1, 3:L0}
        r1_old, r1_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 1, 3: 0}, labels, 3, 13,
        )
        assert_equal(r1_old, r1_new)

        # Remove 2:L1 but add 4:L1 → {1:L0, 3:L0, 4:L1}  still valid
        r2_old, r2_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 3: 0, 4: 1}, labels, 6, 16,
        )
        assert_equal(r2_old, r2_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestIterUntil:
    """Test iter_until behavior."""

    @pytest.mark.xfail(reason=(
        "Old code stores patterns in DESCENDING order by start after "
        "growth concatenations. Its iter_until assumes ascending order "
        "and returns nothing. V2 stores groups in ascending order and "
        "correctly returns matching patterns."
    ), strict=True)
    def test_iter_until_basic(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 5, 15)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0, 3: 0}, labels, 10, 20)

        old_iter, new_iter = collect_iter_until(old_pool, new_pool, 5)
        assert_equal(old_iter, new_iter)

    def test_iter_until_all(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 5, 15)

        old_iter, new_iter = collect_iter_until(old_pool, new_pool, 100)
        assert_equal(old_iter, new_iter)

    def test_iter_until_none(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 5, 10)
        old_iter, new_iter = collect_iter_until(old_pool, new_pool, 3)
        assert_equal(old_iter, new_iter)


class TestPopAll:
    """Test pop_all behavior."""

    def test_pop_all_after_multiple_concat(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 3, 13)
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0, 3: 0}, labels, 6, 16)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_pop_all_empty_pool(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert old_pop == new_pop == []

    def test_pop_all_then_reuse(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        collect_pop_all(old_pool, new_pool)

        collect_concatenate(old_pool, new_pool, {5: 0}, labels, 20, 30)
        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_object(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_start_equals_end_prev(self):
        """start == self.end (boundary overlap, not a gap)."""
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0}, labels, 0, 10)
        r_old, r_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0}, labels, 10, 20,
        )
        assert_equal(r_old, r_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_many_objects(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        n = 50
        labels = {0: 1}

        objs1 = {i: 0 for i in range(n)}
        collect_concatenate(old_pool, new_pool, objs1, labels, 0, 10)

        objs2 = {i: 0 for i in range(n // 2)}
        r_old, r_new = collect_concatenate(old_pool, new_pool, objs2, labels, 5, 15)
        assert_equal(r_old, r_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_disjoint_objects_between_windows(self):
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0}, labels, 0, 10)
        r_old, r_new = collect_concatenate(
            old_pool, new_pool, {3: 0, 4: 0}, labels, 5, 15,
        )
        assert_equal(r_old, r_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestMultiStepScenarios:
    """Realistic multi-step scenarios mimicking actual usage."""

    @pytest.mark.xfail(reason=(
        "Same root cause as test_add_and_remove: old code retains "
        "emitted patterns as storage duplicates during mixed "
        "add+remove steps, producing extra stale patterns in pop_all."
    ), strict=True)
    def test_sliding_window_simulation(self):
        """Simulate a sliding window with objects entering and leaving."""
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1, 1: 1}

        # t=0: {1:L0, 2:L1} — valid
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 1}, labels, 0, 10)

        # t=3: add 3:L0 — valid
        r1_old, r1_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 1, 3: 0}, labels, 3, 13,
        )
        assert_equal(r1_old, r1_new)

        # t=6: 1 leaves, 4:L1 enters → {2:L1, 3:L0, 4:L1} — valid
        r2_old, r2_new = collect_concatenate(
            old_pool, new_pool, {2: 1, 3: 0, 4: 1}, labels, 6, 16,
        )
        assert_equal(r2_old, r2_new)

        # t=9: 2 leaves → {3:L0, 4:L1} — valid
        r3_old, r3_new = collect_concatenate(
            old_pool, new_pool, {3: 0, 4: 1}, labels, 9, 19,
        )
        assert_equal(r3_old, r3_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_complex_label_transitions(self):
        """Complex scenario with multiple label types and transitions."""
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 2, 1: 1}

        # exactly meets requirement
        collect_concatenate(old_pool, new_pool, {1: 0, 2: 0, 3: 1}, labels, 0, 10)

        # add 4:L0 — valid
        r1_old, r1_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0, 3: 1, 4: 0}, labels, 3, 13,
        )
        assert_equal(r1_old, r1_new)

        # remove 3:L1 but add 5:L1 → {1:L0, 2:L0, 4:L0, 5:L1} — valid
        r2_old, r2_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0, 4: 0, 5: 1}, labels, 6, 16,
        )
        assert_equal(r2_old, r2_new)

        # add 6:L0 — valid
        r3_old, r3_new = collect_concatenate(
            old_pool, new_pool, {1: 0, 2: 0, 4: 0, 5: 1, 6: 0}, labels, 9, 19,
        )
        assert_equal(r3_old, r3_new)

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)

    def test_alternating_growth_and_shrinkage(self):
        """Alternating between growing and shrinking, always valid."""
        old_pool, new_pool = OldPatternPool(), NewPatternPool()
        labels = {0: 1}

        for i in range(5):
            if i % 2 == 0:
                objs = {j: 0 for j in range(1, i + 3)}  # grow
            else:
                objs = {j: 0 for j in range(1, i + 1)}  # shrink (≥1 obj → valid)

            r_old, r_new = collect_concatenate(
                old_pool, new_pool, objs, labels, i * 3, i * 3 + 10,
            )
            assert_equal(r_old, r_new, f"Mismatch at step {i}, objs={objs}")

        old_pop, new_pop = collect_pop_all(old_pool, new_pool)
        assert_equal(old_pop, new_pop)


class TestV2Invariants:
    """Tests specific to v2's check_invariants."""

    def test_invariants_after_first_insert(self):
        new_pool = NewPatternPool()
        lc = _pos({0: 1})
        list(new_pool.concatenate({1: 0}, lc, 0, 10))
        new_pool.check_invariants()

    def test_invariants_after_multiple_operations(self):
        new_pool = NewPatternPool()
        lc = _pos({0: 1})

        list(new_pool.concatenate({1: 0}, lc.copy(), 0, 10))
        list(new_pool.concatenate({1: 0, 2: 0}, lc.copy(), 3, 13))
        list(new_pool.concatenate({1: 0, 2: 0, 3: 0}, lc.copy(), 6, 16))
        new_pool.check_invariants()

    def test_invariants_after_shrinkage(self):
        new_pool = NewPatternPool()
        lc = _pos({0: 1})

        list(new_pool.concatenate({1: 0, 2: 0, 3: 0}, lc.copy(), 0, 10))
        list(new_pool.concatenate({1: 0}, lc.copy(), 5, 15))
        new_pool.check_invariants()

    def test_invariants_after_pop_all(self):
        new_pool = NewPatternPool()
        lc = _pos({0: 1})

        list(new_pool.concatenate({1: 0}, lc.copy(), 0, 10))
        list(new_pool.pop_all())
        new_pool.check_invariants()

    def test_invariants_after_gap(self):
        new_pool = NewPatternPool()
        lc = _pos({0: 1})

        list(new_pool.concatenate({1: 0}, lc.copy(), 0, 5))
        list(new_pool.concatenate({2: 0}, lc.copy(), 10, 15))
        new_pool.check_invariants()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
