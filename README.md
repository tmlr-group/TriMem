<div align="center">

<img alt="TriMem logo" src="fig/trimem_logo.svg" width="180">

## Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory

</div>

## Project layout

```
TriMem/
├── main.py                      # entry — wires the three tiers together
├── config.py                    # all knobs (LLM, embedding, parallelism, profiles)
├── core/
│   ├── memory_builder.py        # window → atomic facts (compression + synthesis)
│   ├── hybrid_retriever.py      # planning + parallel multi-view retrieval
│   ├── answer_generator.py      # factual / inference prompt routing, profile-aware
│   └── profile_manager.py       # progressive per-entity profile builder
├── database/
│   ├── vector_store.py          # LanceDB-backed atomic-fact store
│   ├── profile_store.py         # SQLite KV store for persona profiles
│   └── dialogue_store.py        # raw-dialogue store with context windows
├── models/memory_entry.py       # MemoryEntry / Dialogue pydantic models
├── utils/                       # LLM client + embedding model
├── test_locomo10.py             # LoCoMo benchmark runner
└── polish_prompts_with_claude_code.md  # TextGrad-style prompt evolution loop
```

---

## Installation

Requirements: Python 3.10+, an OpenAI-compatible API key.

```bash
git clone https://github.com/tmlr-group/TriMem.git
cd TriMem

pip install -r requirements.txt
```

Edit `config.py` to set your provider, key, and model:

```python
OPENAI_API_KEY  = "sk-..."
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"   # or None for OpenAI default
LLM_MODEL       = "openai/gpt-4.1-mini"

EMBEDDING_MODEL     = "Qwen/Qwen3-Embedding-0.6B"  # local, no API call
EMBEDDING_DIMENSION = 1024

ENABLE_PROFILES     = True                         # turn on the persona tier
PROFILE_STORE_PATH  = "./profile_store.db"
```

---

## Quick start

```python
from main import TriMemSystem

system = TriMemSystem(clear_db=True)

# Ingest dialogues
system.add_dialogue("Alice", "Bob, let's meet at Starbucks tomorrow at 2pm",
                    "2025-11-15T14:30:00")
system.add_dialogue("Bob",   "Sure, I'll bring the market analysis report",
                    "2025-11-15T14:31:00")

# Flush the buffered window (atomic facts + profiles are written here)
system.finalize()

# Factual question — answered from atomic facts + source turns
print(system.ask("When and where will Alice and Bob meet?"))

# Inference question — also draws on Alice/Bob's persona profiles
print(system.ask("Would Bob enjoy a strategy board game?"))
```

What happens internally on `finalize()`:

1. The dialogue window is compressed into atomic memory entries and written to the vector store.
2. `ProfileManager.update_profiles(entries)` groups the new entries by `persons` and updates each persona profile with one LLM call.
3. Raw dialogues remain in `DialogueStore` and are reachable via `source_dialogue_ids`.

On `ask(...)`:

1. The hybrid retriever plans, generates queries, and merges semantic + lexical + symbolic results.
2. Persons appearing in the retrieved facts are looked up in the profile store (no LLM call).
3. The answer generator chooses **factual** or **inference** mode based on the question, then synthesizes the final answer using all three tiers.

---

## Configuration knobs

The most relevant settings (full list in `config.py`):

| Section | Setting | Purpose |
|:--------|:--------|:--------|
| LLM | `LLM_MODEL`, `OPENAI_BASE_URL`, `ENABLE_THINKING`, `USE_STREAMING` | Provider + decoding |
| Memory build | `WINDOW_SIZE`, `OVERLAP_SIZE`, `SOURCE_CONTEXT_WINDOW` | Sliding window + how many surrounding turns to pull when answering |
| Retrieval | `SEMANTIC_TOP_K`, `KEYWORD_TOP_K`, `STRUCTURED_TOP_K` | Top-k per view |
| Planning | `ENABLE_PLANNING`, `ENABLE_REFLECTION`, `MAX_REFLECTION_ROUNDS` | Intent-aware retrieval planning loop |
| Parallelism | `ENABLE_PARALLEL_PROCESSING`, `MAX_PARALLEL_WORKERS`, `ENABLE_PARALLEL_RETRIEVAL`, `MAX_RETRIEVAL_WORKERS` | Window/query concurrency |
| **Profiles** | `ENABLE_PROFILES`, `PROFILE_STORE_PATH` | Toggle the persona tier |

---

## Evaluation

LoCoMo benchmark runner lives in `test_locomo10.py`:

```bash
python test_locomo10.py                       # full benchmark
python test_locomo10.py --num-samples 5       # quick subset
python test_locomo10.py --result-file my.json # custom output path
```

---

## Prompt Evolution

End-to-end quality of TriMem is dominated by two prompts:

- **P_ext** — the fact-extraction prompt that turns a dialogue window into atomic memory units (in `core/memory_builder.py`).
- **P_prof** — the entity profile construction prompt that updates each persona profile from newly extracted facts (in `core/profile_manager.py`).

Rather than hand-tuning these indefinitely, TriMem refines them with a **TextGrad-style** loop. The evaluation result JSON plays the role of the *loss*, and a strong LLM (we use Claude Opus 4.6) plays the role of the **textual backward pass** — diagnosing systematic failure patterns and rewriting `P_ext` / `P_prof` in place. The full procedure lives in [`polish_prompts.md`](polish_prompts.md).

---

## 📄 Citation
If you find this environment useful, please consider citing our work:
```
@article{sun2026trimem,
  title   = {Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory},
  author  = {Jingwei Sun and Jianing Zhu and Jiangchao Yao and Tongliang Liu and Bo Han},
  journal = {arXiv preprint},
  year    = {2026}
}
```
---

## Acknowledgments

Dependencies: [Qwen3-Embedding](https://github.com/QwenLM/Qwen) for retrieval, [LanceDB](https://lancedb.com/) for vector storage, [LoCoMo](https://github.com/snap-research/locomo) for evaluation.

---
