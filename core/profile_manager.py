"""
Profile Manager - Progressive entity profile building from extracted facts.

After each window of memories is stored, the ProfileManager groups entries
by person and updates each person's profile with a single LLM call.
At query time, it collects relevant profiles (zero LLM calls) and formats
them for the answer generator.
"""
from typing import List, Dict, Optional
from collections import defaultdict
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
from database.profile_store import ProfileStore


class ProfileManager:
    """
    Progressive profile builder.

    Write path: update_profiles(entries) after each window
    Read path: get_profiles_for_query(query, entries) at retrieval time
    """

    def __init__(self, llm_client: LLMClient, profile_store: ProfileStore):
        self.llm_client = llm_client
        self.profile_store = profile_store

    def update_profiles(self, entries: List[MemoryEntry]):
        """
        Called after each window's entries are stored.
        Groups entries by person, updates each profile with one LLM call.
        """
        # Group entries by person mentioned in persons field
        person_entries: Dict[str, List[MemoryEntry]] = defaultdict(list)
        for entry in entries:
            for person in entry.persons:
                person_entries[person].append(entry)

        # Update each person's profile
        for entity_name, entity_entries in person_entries.items():
            existing_profile = self.profile_store.get(entity_name)
            updated_profile = self._update_single_profile(
                entity_name, entity_entries, existing_profile
            )
            print(existing_profile)
            print(updated_profile)
            if updated_profile:
                self.profile_store.upsert(entity_name, updated_profile)
                print(f"[ProfileManager] Updated profile for {entity_name} ({len(entity_entries)} new facts)")

    def _update_single_profile(
        self,
        entity_name: str,
        new_entries: List[MemoryEntry],
        existing_profile: Optional[str]
    ) -> Optional[str]:
        """
        LLM call: existing profile + new facts -> updated profile.
        """
        facts = "\n".join(f"- {e.lossless_restatement}" for e in new_entries)

        prompt = f"""Update the persona profile for {entity_name} based on new information.

[Current Profile]
{existing_profile or "No profile yet."}

[New Facts]
{facts}

Rules:
- Update only sections affected by new info
- Preserve existing info not contradicted
- Infer personality traits from behavior patterns
- Synthesize preferences from specific examples
- For [How Others Describe Them]: quote the exact adjectives/phrases other speakers use about this person
- For [Beliefs/Spirituality]: note ANY signals — church mentions, faith symbols, religious encounters, spiritual language
- Keep each section to 1-3 lines
- Output the complete updated profile

Use this format:
Entity: {entity_name}
[Identity] Who they are (gender, age, core identity, origin/hometown)
[Personality] Inferred traits from behavior
[How Others Describe Them] Specific words/phrases others use about this person (e.g. "thoughtful", "driven")
[Interests] Hobbies, activities, things they enjoy
[Career] Job, education, professional goals
[Values] What they care about, causes, beliefs
[Beliefs/Spirituality] Religious views, faith, spiritual practices (even subtle signals like church involvement, faith symbols, or clashes with religious groups)
[Relationships] Key people in their life
[Life Events] Major events, milestones, experiences
[Preferences] Specific likes/dislikes, favorites

Only include sections that have information. Output ONLY the profile, no other text."""

        messages = [
            {
                "role": "system",
                "content": "You are a profile synthesis assistant. You build concise persona profiles from factual observations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.2,
            )
            # Strip think tags if present
            parts = response.split('</think>')
            profile = parts[-1].strip() if len(parts) > 1 else response.strip()
            return profile if profile else None
        except Exception as e:
            print(f"[ProfileManager] Failed to update profile for {entity_name}: {e}")
            return existing_profile

    def get_profiles_for_query(
        self,
        query: str,
        retrieved_entries: List[MemoryEntry]
    ) -> str:
        """
        Collect person names from query + retrieved entries, fetch and format profiles.
        Zero LLM calls - pure lookup.
        """
        # Collect all person names from retrieved entries
        person_names = set()
        for entry in retrieved_entries:
            for person in entry.persons:
                person_names.add(person)

        if not person_names:
            return ""

        # Fetch profiles
        profiles = self.profile_store.get_multiple(list(person_names))

        if not profiles:
            return ""

        # Format profiles
        formatted = []
        for name, profile_text in profiles.items():
            formatted.append(profile_text)

        return "\n\n".join(formatted)
