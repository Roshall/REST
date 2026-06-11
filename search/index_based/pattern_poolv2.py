from __future__ import annotations

from collections import defaultdict

from grouped_index_list import Entry, GroupedIndexedList


class PatternPool:
    """
    Optimized PatternPool using delta groups.

    Representation invariant:
    - self.patterns stores delta groups ordered by group_id/start.
    - The logical pattern at group i is the union of entries in groups [0, i].
    - Therefore pop_all reconstructs patterns by accumulating object ids.

    Assumption:
    - label_verifier is monotonic with respect to adding objects.
      If a prefix is valid, every larger prefix is valid too.
    """

    __slots__ = ("patterns", "end")

    def __init__(self):
        self.patterns: GroupedIndexedList[int] = GroupedIndexedList()
        self.end = 0

    def concatenate(self, objs_m, label_count, start, end):
        """
        Concatenate a new object map [start, end] into the pool.

        objs_m is expected to be a mapping:
            object_id -> label
        """
        if not self.patterns:
            self.patterns.insert_group(
                start,
                (Entry(oid, start) for oid in objs_m),
                assume_unique=True,
            )
            self.end = end
            return

        if start > self.end:
            yield from self.pop_all()
            self.patterns.insert_group(
                start,
                (Entry(oid, start) for oid in objs_m),
                assume_unique=True,
            )
            self.end = end
            return

        # Partition current objects by the old delta group where they already
        # appeared. Brand-new objects are assigned to the current start level.
        new: dict[int, list[Entry[int]]] = defaultdict(list)
        for oid in objs_m:
            found = self.patterns.find(oid)
            if found is not None:
                old_group_id, old_entry = found
                new[old_group_id].append(old_entry)
            else:
                new[start].append(Entry(oid, start))

        # Keep the longest unchanged prefix of old delta groups.
        remain_level_until = 0
        group_count = self.patterns.group_count

        while remain_level_until < group_count:
            group = self.patterns.group_at(remain_level_until)
            kept = new.get(group.group_id)

            if kept is None:
                break
            if len(kept) < len(group):
                break

            # Full old delta group is still present in current objects.
            # It remains in self.patterns, so remove it from the new suffix.
            del new[group.group_id]
            remain_level_until += 1

        # If some old suffix no longer remains, emit the old logical patterns
        # represented by that suffix, then physically drop that suffix.
        if remain_level_until < group_count:
            objects: list[int] = []

            if remain_level_until > 0:
                prefix_end = self.patterns.group_at(remain_level_until - 1).end
                objects.extend(
                    entry.key
                    for entry in self.patterns.iter_entries_until_index(prefix_end)
                )

            for i in range(remain_level_until, group_count):
                group = self.patterns.group_at(i)
                objects.extend(entry.key for entry in self.patterns.iter_group_idx(i))
                yield objects.copy(), group.group_id, self.end

            self.patterns.truncate_tail_at(remain_level_until)

        if new:
            sorted_new = sorted(new.items(), key=lambda x: x[0])

            if remain_level_until > 0:
                # A valid old prefix remains. With a monotonic verifier, every
                # larger prefix formed by appending new delta groups is valid.
                for group_id, entries in sorted_new:
                    self.patterns.insert_group(group_id, entries, assume_unique=True)
            else:
                # No valid old prefix remains. We must not discard invalid early
                # delta groups. Instead, merge them into the first cumulative
                # prefix that satisfies label_verifier.
                self._append_new_groups_from_empty_prefix(
                    sorted_new,
                    objs_m,
                    label_count,
                )

        self.end = end

    def _append_new_groups_from_empty_prefix(
        self,
        sorted_new: list[tuple[int, list[Entry[int]]]],
        objs_m,
        label_count,
    ) -> None:
        """
        Append new delta groups when no valid old prefix is retained.

        Assumptions:
        1. label_count is a requirement vector.
        Example: {0: 2, 1: 1} means a valid pattern needs at least
        two objects with label 0 and one object with label 1.

        2. self.patterns only stores valid logical patterns.

        3. sorted_new is ordered by group_id / pattern start.

        4. Validity is monotonic:
        once a pattern satisfies the label requirement, every superset
        also satisfies it.
        """
        running_count = label_count.copy()

        pending: list[Entry[int]] = []
        opened = False

        for group_id, entries in sorted_new:
            if opened:
                self.patterns.insert_group(group_id, entries, assume_unique=True)
                continue

            pending.extend(entries)

            for entry in entries:
                running_count[objs_m[entry.key]] -= 1

            # Requirement is satisfied when every remaining count is <= 0.
            if all(count <= 0 for count in running_count.values()):
                self.patterns.insert_group(group_id, pending, assume_unique=True)
                pending = []
                opened = True

        # If this fails, even the full current pattern is invalid.
        # That contradicts your assumption that concatenate() receives
        # a valid current object set.
        assert opened, "Expected at least the full pattern to satisfy label requirement"

    def pop_all(self):
        objects: list[int] = []
        for i in range(self.patterns.group_count):
            group = self.patterns.group_at(i)
            objects.extend(entry.key for entry in self.patterns.iter_group_idx(i))
            yield objects.copy(), group.group_id, self.end
        self.patterns.clear()

    def iter_until(self, max_start):
        objects: list[int] = []
        for i in range(self.patterns.group_count):
            group = self.patterns.group_at(i)
            if group.group_id > max_start:
                break
            objects.extend(entry.key for entry in self.patterns.iter_group_idx(i))
            yield objects.copy(), group.group_id, self.end

    def check_invariants(self) -> None:
        self.patterns.check_invariants()
