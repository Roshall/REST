from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Entry(Generic[T]):
    key: int
    value: T


@dataclass(slots=True)
class GroupMeta:
    group_id: int
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class EntryRef:
    group_id: int
    offset: int


class GroupedIndexedList(Generic[T]):
    """
    Flat-list grouped container with fast key lookup.

    Invariants:
    1. All entries live in self._entries.
    2. Entries of the same group are adjacent.
    3. Each group has a border pointer: [start, end).
    4. Entries inside a group are sorted by key.
    5. Group key ranges may overlap.
    6. Lookup is handled by a global key index.
    """

    def __init__(self) -> None:
        self._entries: list[Entry[T]] = []
        self._groups: list[GroupMeta] = []
        self._group_pos: dict[int, int] = {}

        # Assumption: keys are globally unique.
        # If not, change this to dict[int, list[EntryRef]].
        self._key_index: dict[int, EntryRef] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def group_count(self) -> int:
        return len(self._groups)

    def insert_group(
        self,
        group_id: int,
        entries: Iterable[Entry[T] | tuple[int, T]],
        *,
        position: Optional[int] = None,
    ) -> None:
        """
        Insert a whole group.

        Entries inside the group are sorted by key.

        If position is None, append the group at the end.
        Otherwise, insert before the group at that position.
        """
        if group_id in self._group_pos:
            raise ValueError(f"group already exists: {group_id!r}")

        group_entries = self._normalize_entries(entries)

        if not group_entries:
            raise ValueError("cannot insert an empty group")

        group_entries.sort(key=lambda e: e.key)

        self._check_unique_keys_inside_group(group_entries)
        self._check_global_key_conflicts(group_entries)

        if position is None:
            position = len(self._groups)

        if not 0 <= position <= len(self._groups):
            raise IndexError("invalid group insertion position")

        start = self._groups[position - 1].end if position > 0 else 0
        count = len(group_entries)
        end = start + count

        self._entries[start:start] = group_entries

        # Shift following group borders.
        for meta in self._groups[position:]:
            meta.start += count
            meta.end += count

        meta = GroupMeta(group_id=group_id, start=start, end=end)

        self._groups.insert(position, meta)
        self._group_pos[group_id] = position

        self._rebuild_group_pos_from(position)

        # Build key -> local ref.
        for offset, entry in enumerate(group_entries):
            self._key_index[entry.key] = EntryRef(group_id, offset)

    def delete_group(self, group_id: int) -> list[Entry[T]]:
        """
        Delete a whole group.

        Key index cleanup only touches entries in this group.
        Group border shifting only touches later groups.
        """
        if group_id not in self._group_pos:
            raise KeyError(f"group not found: {group_id!r}")

        group_pos = self._group_pos[group_id]
        meta = self._groups[group_pos]

        removed = self._entries[meta.start : meta.end]
        count = len(removed)

        # Remove keys from global index.
        for entry in removed:
            del self._key_index[entry.key]

        del self._entries[meta.start : meta.end]
        del self._groups[group_pos]
        del self._group_pos[group_id]

        # Shift following group borders.
        for later_meta in self._groups[group_pos:]:
            later_meta.start -= count
            later_meta.end -= count

        self._rebuild_group_pos_from(group_pos)

        return removed

    def truncate_tail_at(self, idx: int) -> None:
        """
        Truncate a group by index.

        Keep the left part and remove the right part.
        """
        if not 0 <= idx < len(self._groups):
            raise IndexError(f"group index out of bounds: {idx}")

        meta = self._groups[idx]

        # Remove keys from global index.
        for entry in self._entries[meta.start :]:
            del self._key_index[entry.key]

        del self._entries[meta.start :]
        for later_meta in self._groups[idx:]:
            del self._group_pos[later_meta.group_id]
        del self._groups[idx:]

    def get_entries_by_group_idx(self, group_idx: int) -> tuple[int, list[Entry[T]]]:
        """
        Get entries of a group by group index.
        """
        if not 0 <= group_idx < len(self._groups):
            raise IndexError(f"group index out of bounds: {group_idx}")

        meta = self._groups[group_idx]
        return meta.group_id, self._entries[meta.start : meta.end]

    def find(self, key: int) -> tuple[int, Entry[T]] | None:
        """
        Find an entry by key.

        Average complexity: O(1)
        """
        ref = self._key_index.get(key)

        if ref is None:
            return None

        group_pos = self._group_pos[ref.group_id]
        meta = self._groups[group_pos]

        absolute_index = meta.start + ref.offset
        return ref.group_id, self._entries[absolute_index]

    def get(self, key: int, default=None):
        entry = self.find(key)
        return default if entry is None else entry[1].value

    def iter_group(self, group_id: int) -> Iterator[Entry[T]]:
        if group_id not in self._group_pos:
            raise KeyError(f"group not found: {group_id!r}")

        meta = self._groups[self._group_pos[group_id]]

        for i in range(meta.start, meta.end):
            yield self._entries[i]

    def group_meta(self, group_id: int) -> GroupMeta:
        if group_id not in self._group_pos:
            raise KeyError(f"group not found: {group_id!r}")

        meta = self._groups[self._group_pos[group_id]]

        return GroupMeta(
            group_id=meta.group_id,
            start=meta.start,
            end=meta.end,
        )

    def entries(self) -> list[Entry[T]]:
        return list(self._entries)

    def groups(self) -> list[GroupMeta]:
        return [GroupMeta(g.group_id, g.start, g.end) for g in self._groups]

    def _normalize_entries(
        self,
        entries: Iterable[Entry[T] | tuple[int, T]],
    ) -> list[Entry[T]]:
        result: list[Entry[T]] = []

        for item in entries:
            if isinstance(item, Entry):
                key = item.key
                value = item.value
            else:
                key, value = item

            if not isinstance(key, int):
                raise TypeError(f"entry key must be int, got {type(key).__name__}")

            result.append(Entry(key, value))

        return result

    def _check_unique_keys_inside_group(self, entries: list[Entry[T]]) -> None:
        for a, b in zip(entries, entries[1:]):
            if a.key == b.key:
                raise ValueError(f"duplicate key inside group: {a.key}")

    def _check_global_key_conflicts(self, entries: list[Entry[T]]) -> None:
        for entry in entries:
            if entry.key in self._key_index:
                raise ValueError(f"duplicate global key: {entry.key}")

    def _rebuild_group_pos_from(self, start: int) -> None:
        for i in range(start, len(self._groups)):
            self._group_pos[self._groups[i].group_id] = i

    def check_invariants(self) -> None:
        prev_end = 0

        for i, meta in enumerate(self._groups):
            if meta.start != prev_end:
                raise AssertionError("groups are not adjacent")

            if meta.end <= meta.start:
                raise AssertionError("empty group detected")

            if self._group_pos[meta.group_id] != i:
                raise AssertionError("wrong group position map")

            group_entries = self._entries[meta.start : meta.end]

            for a, b in zip(group_entries, group_entries[1:]):
                if a.key >= b.key:
                    raise AssertionError("entries inside group are not sorted")

            for offset, entry in enumerate(group_entries):
                ref = self._key_index.get(entry.key)

                if ref is None:
                    raise AssertionError(f"missing key index for key {entry.key}")

                if ref.group_id != meta.group_id or ref.offset != offset:
                    raise AssertionError(f"wrong key index for key {entry.key}")

            prev_end = meta.end

        if prev_end != len(self._entries):
            raise AssertionError("group borders do not cover all entries")

    def clear(self) -> None:
        self._entries.clear()
        self._groups.clear()
        self._group_pos.clear()
        self._key_index.clear()
