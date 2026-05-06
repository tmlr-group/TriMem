"""
Dialogue Store - Stores raw dialogues for source context retrieval.

When memories are retrieved, their source_dialogue_ids point back to
the original dialogues stored here. This allows the answer generator
to access the full, uncompressed dialogue context.
"""
from typing import List, Dict, Optional
from models.memory_entry import Dialogue


class DialogueStore:
    """
    Simple in-memory store for raw dialogues, keyed by dialogue_id.
    Supports fetching individual dialogues and context windows (surrounding turns).
    """

    def __init__(self):
        self._store: Dict[int, Dialogue] = {}

    def add(self, dialogue: Dialogue):
        """Store a single dialogue."""
        self._store[dialogue.dialogue_id] = dialogue

    def add_batch(self, dialogues: List[Dialogue]):
        """Store a batch of dialogues."""
        for d in dialogues:
            self._store[d.dialogue_id] = d

    def get(self, dialogue_id: int) -> Optional[Dialogue]:
        """Fetch a single dialogue by ID."""
        return self._store.get(dialogue_id)

    def get_batch(self, dialogue_ids: List[int]) -> List[Dialogue]:
        """Fetch multiple dialogues by IDs, preserving order."""
        return [self._store[did] for did in dialogue_ids if did in self._store]

    def get_with_context(self, dialogue_ids: List[int], context_window: int = 2) -> List[Dialogue]:
        """
        Fetch dialogues plus surrounding context turns.

        For each dialogue_id, also fetches up to `context_window` turns
        before and after it. Results are deduplicated and sorted by ID.
        """
        all_ids = set()
        for did in dialogue_ids:
            for offset in range(-context_window, context_window + 1):
                all_ids.add(did + offset)

        # Filter to existing IDs and sort
        valid_ids = sorted(did for did in all_ids if did in self._store)
        return [self._store[did] for did in valid_ids]

    def get_all(self) -> List[Dialogue]:
        """Get all stored dialogues sorted by ID."""
        return [self._store[did] for did in sorted(self._store.keys())]

    def count(self) -> int:
        return len(self._store)

    def clear(self):
        self._store.clear()
