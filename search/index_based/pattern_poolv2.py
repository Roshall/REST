from collections import defaultdict

from grouped_index_list import Entry, GroupedIndexedList

from search.verifier import label_verifier


class PatternPool:
    def __init__(self):
        self.patterns: GroupedIndexedList[int] = GroupedIndexedList()
        self.end = 0

    def concatenate(self, objs_m, label_count, start, end):
        if not self.patterns:
            self.patterns.insert_group(start, (Entry(oid, start) for oid in objs_m))
            self.end = end
            return
        elif start > (end_l := self.end):
            self.end = end_l
            yield from self.pop_all()
            self.patterns.insert_group(start, (Entry(oid, start) for oid in objs_m))
            self.end = end
            return

        new = defaultdict(list)
        for o in objs_m:
            if (pat := self.patterns.find(o)) is not None:
                new[pat[0]].append(pat[1])
            else:
                new[start].append(Entry(o, start))

        remain_level_until = 0
        for group in self.patterns.groups():
            if group.group_id not in new:
                break

            if len(new[group.group_id]) < len(group):
                break
            else:
                new.pop(group.group_id)  # group has already been in self.patterns
            remain_level_until += 1
        else:
            remain_level_until = self.patterns.group_count

        if remain_level_until < self.patterns.group_count:
            # we need to yield the old patterns here
            if remain_level_until > 0:  # retrieve objects prefix from the old patterns
                group_meta = self.patterns.groups()[remain_level_until - 1]
                entries = self.patterns.entries()[: group_meta.end]
                objects = [entry.key for entry in entries]
            else:
                objects = []
            for i in range(remain_level_until, self.patterns.group_count):
                start, g_entries = self.patterns.get_entries_by_group_idx(i)
                objects.extend(entry.key for entry in g_entries)
                yield objects.copy(), start, self.end
            self.patterns.truncate_tail_at(remain_level_until)

        # now append the new patterns
        # sort new by level
        sorted_new = sorted(new.items(), key=lambda x: x[0])
        if remain_level_until > 0:
            for start, pattern_helpers in sorted_new:
                self.patterns.insert_group(start, (entry for entry in pattern_helpers))
        else:  # need to validate by label_count
            label_count = label_count.copy()
            satisfied = False
            for start, pattern_helpers in sorted_new:
                for ph in pattern_helpers:
                    label_count[objs_m[ph.key]] -= 1
                if not satisfied:
                    if label_verifier(label_count):
                        satisfied = True
                    else:
                        continue
                self.patterns.insert_group(start, (entry for entry in pattern_helpers))
        self.end = end

    def pop_all(self):
        objects = []
        for i in range(self.patterns.group_count):
            start, entries = self.patterns.get_entries_by_group_idx(i)
            objects.extend(entry.key for entry in entries)
            yield objects.copy(), start, self.end
        self.patterns.clear()

    def iter_until(self, max_end):
        objects = []
        for i in range(self.patterns.group_count):
            start, entries = self.patterns.get_entries_by_group_idx(i)
            if start <= max_end:
                objects.extend(entry.key for entry in entries)
                yield objects.copy(), start, self.end
            else:
                break
