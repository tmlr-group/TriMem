"""
Memory Builder
Stage 1: Semantic Structured Compression (Section 3.1)
& Stage 2: Online Semantic Synthesis (Section 3.2)

Implements:
- Implicit semantic density gating: Φ_gate(W) → {m_k} (filters low-density windows)
- Sliding window processing for dialogue segmentation
- Generates compact memory units with resolved coreferences and absolute timestamps
"""
from typing import List, Optional, TYPE_CHECKING
from models.memory_entry import MemoryEntry, Dialogue
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config
import concurrent.futures

if TYPE_CHECKING:
    from core.profile_manager import ProfileManager


class MemoryBuilder:
    """
    Memory Builder - Semantic Structured Compression (Section 3.1)

    Core Functions:
    1. Sliding window segmentation
    2. Implicit semantic density gating: Φ_gate(W) → {m_k}
    3. Multi-view indexing: I(m_k) = {s_k, l_k, r_k}
    4. Intra-session consolidation during write (Section 3.2): by generating enough memory entries to ensure ALL information is captured
    """
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        window_size: int = None,
        enable_parallel_processing: bool = True,
        max_parallel_workers: int = 3,
        profile_manager: Optional['ProfileManager'] = None,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = window_size or config.WINDOW_SIZE
        self.overlap_size = getattr(config, 'OVERLAP_SIZE', 0)
        self.step_size = max(1, self.window_size - self.overlap_size)

        self.enable_parallel_processing = enable_parallel_processing if enable_parallel_processing is not None else getattr(config, 'ENABLE_PARALLEL_PROCESSING', True)
        self.max_parallel_workers = max_parallel_workers if max_parallel_workers is not None else getattr(config, 'MAX_PARALLEL_WORKERS', 4)

        # Dialogue buffer
        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0

        # Previous window entries (for context)
        self.previous_entries: List[MemoryEntry] = []

        # Profile manager (None = disabled)
        self.profile_manager = profile_manager

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True):
        """
        Add a dialogue to the buffer
        """
        self.dialogue_buffer.append(dialogue)

        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True):
        """
        Batch add dialogues with optional parallel processing
        """
        if self.enable_parallel_processing and len(dialogues) > self.window_size * 2:
            self.add_dialogues_parallel(dialogues)
        else:
            for dialogue in dialogues:
                self.add_dialogue(dialogue, auto_process=False)

            if auto_process:
                while len(self.dialogue_buffer) >= self.window_size:
                    self.process_window()

    def add_dialogues_parallel(self, dialogues: List[Dialogue]):
        """
        Add dialogues using parallel processing for better performance
        """
        pre_existing = list(self.dialogue_buffer)
        windows_to_process = []
        try:
            self.dialogue_buffer.extend(dialogues)

            pos = 0
            while pos + self.window_size <= len(self.dialogue_buffer):
                window = self.dialogue_buffer[pos:pos + self.window_size]
                windows_to_process.append(window)
                pos += self.step_size

            remaining = self.dialogue_buffer[pos:]
            if remaining:
                windows_to_process.append(remaining)
            self.dialogue_buffer = []

            if windows_to_process:
                print(f"\n[Parallel Processing] Processing {len(windows_to_process)} batches in parallel with {self.max_parallel_workers} workers")
                print(f"Batch sizes: {[len(w) for w in windows_to_process]}")
                self._process_windows_parallel(windows_to_process)

        except Exception as e:
            print(f"[Parallel Processing] Failed: {e}. Falling back to sequential processing...")
            if not self.dialogue_buffer:
                self.dialogue_buffer = pre_existing + list(dialogues)
            while len(self.dialogue_buffer) >= self.window_size:
                self.process_window()

    def process_window(self):
        """
        Process current window dialogues - Core logic
        """
        if not self.dialogue_buffer:
            return

        window = self.dialogue_buffer[:self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.step_size:]

        print(f"\nProcessing window: {len(window)} dialogues (processed {self.processed_count} so far)")

        # Stage 1: Call LLM to generate memory entries
        entries = self._generate_memory_entries(window)

        # Store to database
        if entries:
            self.vector_store.add_entries(entries)
            if self.profile_manager:
                self.profile_manager.update_profiles(entries)
            self.previous_entries = entries
            self.processed_count += len(window)

        print(f"Generated {len(entries)} memory entries")

    def process_remaining(self):
        """
        Process remaining dialogues (fallback method, normally handled in parallel)
        """
        if self.dialogue_buffer:
            print(f"\nProcessing remaining dialogues: {len(self.dialogue_buffer)} (fallback mode)")
            entries = self._generate_memory_entries(self.dialogue_buffer)
            if entries:
                self.vector_store.add_entries(entries)
                if self.profile_manager:
                    self.profile_manager.update_profiles(entries)
                self.processed_count += len(self.dialogue_buffer)
            self.dialogue_buffer = []
            print(f"Generated {len(entries)} memory entries")

    def _generate_memory_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        """
        Implicit Semantic Density Gating (Section 3.1)
        Φ_gate(W) → {m_k}, generates compact memory units from dialogue window
        """
        dialogue_text = "\n".join([str(d) for d in dialogues])
        dialogue_ids = [d.dialogue_id for d in dialogues]

        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:
                context += f"- {entry.lossless_restatement}\n"

        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)

        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                entries = self._parse_llm_response(response, dialogue_ids)
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed to parse LLM response: {e}")
                    print(f"Retrying...")
                else:
                    print(f"All {max_retries} attempts failed to parse LLM response: {e}")
                    print(f"Raw response: {response[:500] if 'response' in locals() else 'No response'}")
                    return []

    def _build_extraction_prompt(
        self,
        dialogue_text: str,
        dialogue_ids: List[int],
        context: str
    ) -> str:
        """
        Build LLM extraction prompt with source dialogue ID tagging.
        """
        return f"""
Your task is to extract all valuable FACTUAL information from the following dialogues and convert them into structured memory entries.

{context}

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. **Extract Facts, Not Social Gestures**: SKIP greetings, thank-yous, compliments, and generic praise. Only extract entries that contain novel factual information (events, activities, plans, preferences, relationships, specific details like names/titles/numbers).
2. **Source Dialogue IDs**: Each dialogue line starts with an ID in brackets like [ID:42]. For each memory entry, list the dialogue IDs that the entry was derived from in the "source_dialogue_ids" field. This is CRITICAL for tracing back to original context.
3. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that). Always use the person's actual name. Every memory MUST explicitly state WHO did/said/experienced the thing.
4. **Resolve Temporal References**: Convert ALL relative time expressions to absolute dates based on the dialogue timestamp:
   - "yesterday" on May 8 -> May 7
   - "last year" in 2023 -> 2022
   - "last week" -> compute the actual date
   - "next month" -> compute the actual month
   The "timestamp" field should be the EVENT time, NOT the conversation time.
5. **Atomic Facts**: Extract individual facts as SEPARATE entries. If one dialogue mentions 3 activities, create 3 entries. Do not merge unrelated facts into one summary.
6. **Preserve Specific Details**: Always capture exact names (people, pets, books, songs), exact numbers (durations, counts, ages), and specific entities.
7. **Identify Described-But-Unnamed Things Using World Knowledge**: When the dialogue describes something without naming it, IDENTIFY it by name in the memory entry. This is CRITICAL — future queries will search by name, not by description.
   - A study method like "25 minutes on, 5 minutes off" → identify as "Pomodoro technique"
   - A composer whose music is in a named movie → identify the composer (e.g. Harry Potter soundtrack → John Williams)
   - A game described by its mechanics → identify the game (e.g. "card game where you find the imposter" → Mafia, "game with colored cards you match" → UNO)
   - A location described by its features → identify it (e.g. "national park in northern Minnesota with lakes" → Voyageurs National Park)
   - A shop/brand described by what it does → identify it (e.g. "they made all the props for Harry Potter" → MinaLima / House of MinaLima)
   - A health condition implied by symptoms → identify it (e.g. frequent overeating + weight gain → obesity risk)
   - A geographic location implied by context → identify the state/country (e.g. a shelter in a named city → identify the state)
   Include BOTH the original description AND the identified name in the lossless_restatement, and add the identified name to keywords and entities.
8. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time of the EVENT in ISO 8601 format (resolved from relative expressions)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, book titles, song names, etc.
   - topic: The topic of this information

[Output Format]
Return a JSON array, each element is a memory entry:

```json
[
  {{
    "lossless_restatement": "Complete unambiguous restatement (must include WHO, WHAT, WHEN, WHERE)",
    "keywords": ["keyword1", "keyword2", ...],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", "name2", ...],
    "entities": ["entity1", "entity2", ...],
    "topic": "topic phrase",
    "source_dialogue_ids": [42, 43]
  }},
  ...
]
```

[Example]
Dialogues:
[ID:1] [2025-11-15T14:30:00] Alice: I just started working at Google! Bob, let's meet at Starbucks tomorrow at 2pm.
[ID:2] [2025-11-15T14:31:00] Bob: Congrats! I've been playing tennis a lot lately.
[ID:3] [2025-11-15T14:32:00] Alice: Nice! I've been studying with that method where you work 25 minutes then take a 5-minute break. It really helps!
[ID:4] [2025-11-15T14:33:00] Bob: I love playing that theme song from the wizard movie on piano. You know, the one with the boy who goes to magic school.

Output:
```json
[
  {{
    "lossless_restatement": "Alice started working at Google as of 2025-11-15.",
    "keywords": ["Alice", "Google", "employment"],
    "timestamp": "2025-11-15T14:30:00",
    "location": null,
    "persons": ["Alice"],
    "entities": ["Google"],
    "topic": "Alice's new job",
    "source_dialogue_ids": [1]
  }},
  {{
    "lossless_restatement": "Alice suggested meeting with Bob at Starbucks on 2025-11-16T14:00:00.",
    "keywords": ["Alice", "Bob", "Starbucks", "meeting"],
    "timestamp": "2025-11-16T14:00:00",
    "location": "Starbucks",
    "persons": ["Alice", "Bob"],
    "entities": [],
    "topic": "Meeting arrangement",
    "source_dialogue_ids": [1]
  }},
  {{
    "lossless_restatement": "Bob has been playing tennis frequently as of 2025-11-15.",
    "keywords": ["Bob", "tennis"],
    "timestamp": "2025-11-15T14:31:00",
    "location": null,
    "persons": ["Bob"],
    "entities": [],
    "topic": "Bob's hobby",
    "source_dialogue_ids": [2]
  }},
  {{
    "lossless_restatement": "Alice uses the Pomodoro technique (25 minutes work, 5-minute break) for studying.",
    "keywords": ["Alice", "Pomodoro technique", "studying", "time management"],
    "timestamp": "2025-11-15T14:32:00",
    "location": null,
    "persons": ["Alice"],
    "entities": ["Pomodoro technique"],
    "topic": "Alice's study method",
    "source_dialogue_ids": [3]
  }},
  {{
    "lossless_restatement": "Bob enjoys playing the Harry Potter theme song (composed by John Williams) on piano.",
    "keywords": ["Bob", "piano", "Harry Potter", "John Williams"],
    "timestamp": "2025-11-15T14:33:00",
    "location": null,
    "persons": ["Bob"],
    "entities": ["Harry Potter", "John Williams"],
    "topic": "Bob's piano playing",
    "source_dialogue_ids": [4]
  }}
]
```

Note how "tomorrow" was resolved to 2025-11-16, the greeting was skipped, described-but-unnamed things were identified (Pomodoro technique, John Williams), and Alice's job and meeting were extracted as separate atomic facts.

Now process the above dialogues. Return ONLY the JSON array, no other explanations.
"""

    def _parse_llm_response(
        self,
        response: str,
        dialogue_ids: List[int]
    ) -> List[MemoryEntry]:
        """
        Parse LLM response to MemoryEntry list
        """
        response = response.split('</think>')
        if len(response) > 1:
            response = response[-1]
        else:
            response = response[0]
        data = self.llm_client.extract_json(response)
        if isinstance(data, dict):
           data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array but got: {type(data)}")

        entries = []
        for item in data:
            raw_source_ids = item.get("source_dialogue_ids", [])
            source_ids = []
            for sid in raw_source_ids:
                try:
                    source_ids.append(int(sid))
                except (ValueError, TypeError):
                    pass
            if not source_ids:
                source_ids = list(dialogue_ids)

            entry = MemoryEntry(
                lossless_restatement=item["lossless_restatement"],
                keywords=item.get("keywords", []),
                timestamp=item.get("timestamp"),
                location=item.get("location"),
                persons=item.get("persons", []),
                entities=item.get("entities", []),
                topic=item.get("topic"),
                source_dialogue_ids=source_ids,
            )
            entries.append(entry)

        return entries

    def _process_windows_parallel(self, windows: List[List[Dialogue]]):
        """
        Process multiple windows in parallel using ThreadPoolExecutor
        """
        all_entries = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            future_to_window = {}
            for i, window in enumerate(windows):
                dialogue_ids = [d.dialogue_id for d in window]
                future = executor.submit(self._generate_memory_entries_worker, window, dialogue_ids, i+1)
                future_to_window[future] = (window, i+1)

            for future in concurrent.futures.as_completed(future_to_window):
                window, window_num = future_to_window[future]
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                    print(f"[Parallel Processing] Window {window_num} completed: {len(entries)} entries")
                except Exception as e:
                    print(f"[Parallel Processing] Window {window_num} failed: {e}")

        # Store all entries to database in batch
        if all_entries:
            print(f"\n[Parallel Processing] Storing {len(all_entries)} entries to database...")
            self.vector_store.add_entries(all_entries)
            if self.profile_manager:
                self.profile_manager.update_profiles(all_entries)
            self.processed_count += sum(len(window) for window in windows)

            if all_entries:
                self.previous_entries = all_entries[-10:]

        print(f"[Parallel Processing] Completed processing {len(windows)} windows")

    def _generate_memory_entries_worker(self, window: List[Dialogue], dialogue_ids: List[int], window_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel processing of a single batch
        """
        batch_size = len(window)
        batch_type = "full window" if batch_size == self.window_size else f"remaining batch"
        print(f"[Worker {window_num}] Processing {batch_type} with {batch_size} dialogues")

        dialogue_text = "\n".join([str(d) for d in window])

        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:
                context += f"- {entry.lossless_restatement}\n"

        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)

        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                entries = self._parse_llm_response(response, dialogue_ids)
                print(entries)
                print(f"[Worker {window_num}] Generated {len(entries)} entries")
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Worker {window_num}] Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"[Worker {window_num}] All {max_retries} attempts failed: {e}")
                    return []
