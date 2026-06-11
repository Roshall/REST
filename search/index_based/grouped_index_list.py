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
    Flat-list grouped container with global key lookup.

    Performance-oriented version:
    - base storage is one list
    - entries of the same group are adjacent
    - entries inside a group are sorted by key
    - group key ranges may overlap
    - keys must be globally unique
    - lookup uses key -> (group_id, local_offset)
    """

    __slots__ = ("_entries", "_groups", "_group_pos", "_key_index")

    def __init__(self) -> None:
        self._entries: list[Entry[T]] = []
        self._groups: list[GroupMeta] = []
        self._group_pos: dict[int, int] = {}

        # Assumption: keys are globally unique.
        # If not, change this to dict[int, list[EntryRef]].
        self._key_index: dict[int, EntryRef] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    @property
    def group_count(self) -> int:
        return len(self._groups)

    def insert_group(
        self,
        group_id: int,
        entries: Iterable[Entry[T] | tuple[int, T]],
        *,
        position: Optional[int] = None,
        assume_unique: bool = False,
    ) -> None:
        """
        Insert a whole group.

        If position is None, append the group.
        If assume_unique=True, duplicate key checks are skipped.
        """
        if group_id in self._group_pos:
            raise ValueError(f"group already exists: {group_id!r}")

        group_entries = self._normalize_entries(entries)
        if not group_entries:
            raise ValueError("cannot insert an empty group")

        group_entries.sort(key=lambda e: e.key)

        if not assume_unique:
            self._check_unique_keys_inside_group(group_entries)
            self._check_global_key_conflicts(group_entries)

        if position is None:
            position = len(self._groups)
        elif not 0 <= position <= len(self._groups):
            raise IndexError(f"invalid group insertion position: {position}")

        start = self._groups[position - 1].end if position > 0 else 0
        count = len(group_entries)
        end = start + count

        self._entries[start:start] = group_entries

        # Shift following group borders.
        for meta in self._groups[position:]:
            meta.start += count
            meta.end += count

        self._groups.insert(position, GroupMeta(group_id, start, end))
        self._rebuild_group_pos_from(position)

        # Build key -> local ref.
        for offset, entry in enumerate(group_entries):
            self._key_index[entry.key] = EntryRef(group_id, offset)

    def delete_group(self, group_id: int) -> list[Entry[T]]:
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

    def truncate_tail_at(self, group_idx: int) -> None:
        """
        Keep groups [0, group_idx), remove groups [group_idx, end).

        group_idx == group_count is allowed and is a no-op.
        """
        if not 0 <= group_idx <= len(self._groups):
            raise IndexError(f"group index out of bounds: {group_idx}")
        if group_idx == len(self._groups):
            return

        cut = self._groups[group_idx].start

        for entry in self._entries[cut:]:
            del self._key_index[entry.key]

        for meta in self._groups[group_idx:]:
            del self._group_pos[meta.group_id]

        del self._entries[cut:]
        del self._groups[group_idx:]

    def find(self, key: int) -> tuple[int, Entry[T]] | None:
        ref = self._key_index.get(key)
        if ref is None:
            return None

        meta = self._groups[self._group_pos[ref.group_id]]
        return ref.group_id, self._entries[meta.start + ref.offset]

    def get(self, key: int, default=None):
        found = self.find(key)
        return default if found is None else found[1].value

    def group_at(self, group_idx: int) -> GroupMeta:
        """Fast internal-style access. Do not mutate the returned meta."""
        return self._groups[group_idx]

    def iter_group_idx(self, group_idx: int) -> Iterator[Entry[T]]:
        meta = self._groups[group_idx]
        for i in range(meta.start, meta.end):
            yield self._entries[i]

    def iter_entries_until_index(self, end: int) -> Iterator[Entry[T]]:
        for i in range(end):
            yield self._entries[i]

    def get_entries_by_group_idx(self, group_idx: int) -> tuple[int, list[Entry[T]]]:
        meta = self._groups[group_idx]
        return meta.group_id, self._entries[meta.start : meta.end]

    def iter_group(self, group_id: int) -> Iterator[Entry[T]]:
        yield from self.iter_group_idx(self._group_pos[group_id])

    def group_meta(self, group_id: int) -> GroupMeta:
        meta = self._groups[self._group_pos[group_id]]
        return GroupMeta(meta.group_id, meta.start, meta.end)

    def entries(self) -> list[Entry[T]]:
        return list(self._entries)

    def groups(self) -> list[GroupMeta]:
        return [GroupMeta(g.group_id, g.start, g.end) for g in self._groups]

    def clear(self) -> None:
        self._entries.clear()
        self._groups.clear()
        self._group_pos.clear()
        self._key_index.clear()

    def _normalize_entries(
        self,
        entries: Iterable[Entry[T] | tuple[int, T]],
    ) -> list[Entry[T]]:
        result: list[Entry[T]] = []
        for item in entries:
            if isinstance(item, Entry):
                result.append(item)
            else:
                key, value = item
                result.append(Entry(key, value))
        return result

    @staticmethod
    def _check_unique_keys_inside_group(entries: list[Entry[T]]) -> None:
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
        seen_keys: set[int] = set()

        for i, meta in enumerate(self._groups):
            if meta.start != prev_end:
                raise AssertionError("groups are not adjacent")
            if meta.end <= meta.start:
                raise AssertionError("empty group detected")
            if self._group_pos.get(meta.group_id) != i:
                raise AssertionError("wrong group position map")

            prev_key: int | None = None
            for offset, entry in enumerate(self._entries[meta.start : meta.end]):
                if prev_key is not None and prev_key >= entry.key:
                    raise AssertionError("entries inside group are not sorted")
                prev_key = entry.key

                if entry.key in seen_keys:
                    raise AssertionError(f"duplicate global key: {entry.key}")
                seen_keys.add(entry.key)

                ref = self._key_index.get(entry.key)
                if ref is None:
                    raise AssertionError(f"missing key index for key {entry.key}")
                if ref.group_id != meta.group_id or ref.offset != offset:
                    raise AssertionError(f"wrong key index for key {entry.key}")

            prev_end = meta.end

        if prev_end != len(self._entries):
            raise AssertionError("group borders do not cover all entries")
        if len(self._key_index) != len(self._entries):
            raise AssertionError("key index size mismatch")
