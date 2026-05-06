"""
Analyze TriMem Memory Construction Process

For each window:
- Shows the raw dialogues in the window
- Shows the extracted memory entries from the LLM
- Shows all QA questions whose evidence falls within this window

This helps understand what information the memory builder captures vs. what
information is needed to answer questions, revealing gaps for improvement.
"""
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from main import TriMemSystem
from models.memory_entry import Dialogue, MemoryEntry
from core.memory_builder import MemoryBuilder
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config


def load_sample(dataset_path: str, sample_idx: int = 0):
    """Load a single sample from LoComo10 dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data[sample_idx]


def build_dialogues_and_mapping(sample: dict) -> Tuple[List[Dialogue], Dict[str, int]]:
    """
    Convert sample to ordered dialogues and build dia_id -> global_index mapping.
    Returns (dialogues_list, dia_id_to_global_index)
    """
    # Find all session keys
    session_ids = []
    for k in sample['conversation'].keys():
        if k.startswith('session_') and not k.endswith('_date_time') and isinstance(sample['conversation'][k], list):
            sid = int(k.split('_')[1])
            session_ids.append(sid)
    session_ids.sort()

    dialogues = []
    dia_id_to_idx = {}
    global_idx = 0
    dialogue_id = 1

    for sid in session_ids:
        session_key = f'session_{sid}'
        date_time = sample['conversation'].get(f'{session_key}_date_time', '')
        for turn in sample['conversation'][session_key]:
            # Build dialogue text (handle images)
            text = turn.get("text", "")
            if "img_url" in turn and "blip_caption" in turn:
                caption_text = f"[Image: {turn['blip_caption']}]"
                text = f"{caption_text} {text}" if text else caption_text

            dialogue = Dialogue(
                dialogue_id=dialogue_id,
                speaker=turn["speaker"],
                content=text,
                timestamp=date_time
            )
            dialogues.append(dialogue)
            dia_id_to_idx[turn['dia_id']] = global_idx
            global_idx += 1
            dialogue_id += 1

    return dialogues, dia_id_to_idx


def compute_windows(total_dialogues: int, window_size: int, overlap_size: int) -> List[Tuple[int, int]]:
    """Compute window boundaries as (start_idx, end_idx) pairs"""
    step_size = max(1, window_size - overlap_size)
    windows = []
    pos = 0
    while pos + window_size <= total_dialogues:
        windows.append((pos, pos + window_size))
        pos += step_size
    # Remaining dialogues
    if pos < total_dialogues:
        windows.append((pos, total_dialogues))
    return windows


def map_qa_to_windows(sample: dict, dia_id_to_idx: dict, windows: List[Tuple[int, int]]) -> Dict[int, List[dict]]:
    """
    Map QA questions to windows based on evidence.
    Returns: {window_idx: [list of QA dicts with evidence details]}
    """
    window_to_qas = defaultdict(list)

    for qi, qa in enumerate(sample['qa']):
        evidence_list = qa.get('evidence', [])
        evidence_in_windows = defaultdict(list)  # window_idx -> [evidence_ids in that window]

        for ev in evidence_list:
            # Handle compound evidence like "D8:6; D9:17"
            ev_parts = [e.strip() for e in ev.split(';')]
            for ev_part in ev_parts:
                if ev_part in dia_id_to_idx:
                    idx = dia_id_to_idx[ev_part]
                    for wi, (ws, we) in enumerate(windows):
                        if ws <= idx < we:
                            evidence_in_windows[wi].append(ev_part)

        # Add this QA to each relevant window
        for wi, ev_ids in evidence_in_windows.items():
            window_to_qas[wi].append({
                'qa_index': qi,
                'question': qa['question'],
                'answer': qa.get('answer'),
                'adversarial_answer': qa.get('adversarial_answer'),
                'category': qa.get('category'),
                'all_evidence': evidence_list,
                'evidence_in_this_window': ev_ids,
                'evidence_in_other_windows': [e for e in evidence_list if e not in ev_ids]
            })

    return dict(window_to_qas)


def extract_memories_per_window(
    dialogues: List[Dialogue],
    windows: List[Tuple[int, int]],
    llm_client: LLMClient,
    vector_store: VectorStore
) -> Dict[int, List[dict]]:
    """
    Run memory extraction on each window individually and capture the results.
    Returns: {window_idx: [list of memory entry dicts]}
    """
    window_memories = {}

    for wi, (ws, we) in enumerate(windows):
        window_dialogues = dialogues[ws:we]
        print(f"\n{'='*60}")
        print(f"Extracting memories for Window {wi}: dialogues [{ws+1}, {we}] ({we-ws} turns)")
        print(f"{'='*60}")

        # Create a fresh MemoryBuilder for each window (no cross-window context)
        builder = MemoryBuilder(
            llm_client=llm_client,
            vector_store=vector_store,
            window_size=len(window_dialogues),
            enable_parallel_processing=False
        )

        # Generate memory entries
        start_time = time.time()
        entries = builder._generate_memory_entries(window_dialogues)
        elapsed = time.time() - start_time

        print(f"Generated {len(entries)} memory entries in {elapsed:.2f}s")

        # Convert to dicts for JSON serialization
        entry_dicts = []
        for entry in entries:
            entry_dicts.append({
                'lossless_restatement': entry.lossless_restatement,
                'keywords': entry.keywords,
                'timestamp': entry.timestamp,
                'location': entry.location,
                'persons': entry.persons,
                'entities': entry.entities,
                'topic': entry.topic
            })

        window_memories[wi] = entry_dicts

    return window_memories


def build_analysis_output(
    sample: dict,
    dialogues: List[Dialogue],
    windows: List[Tuple[int, int]],
    window_memories: Dict[int, List[dict]],
    window_to_qas: Dict[int, List[dict]]
) -> dict:
    """Build the final analysis output structure"""
    analysis = {
        'metadata': {
            'sample_id': sample.get('sample_id', '0'),
            'speaker_a': sample['conversation']['speaker_a'],
            'speaker_b': sample['conversation']['speaker_b'],
            'total_dialogues': len(dialogues),
            'window_size': config.WINDOW_SIZE,
            'overlap_size': getattr(config, 'OVERLAP_SIZE', 0),
            'num_windows': len(windows),
            'total_qa': len(sample['qa']),
        },
        'windows': []
    }

    for wi, (ws, we) in enumerate(windows):
        window_dialogues = dialogues[ws:we]

        # Build dialogue text for this window
        dialogue_texts = []
        for d in window_dialogues:
            dialogue_texts.append({
                'dialogue_id': d.dialogue_id,
                'speaker': d.speaker,
                'content': d.content,
                'timestamp': d.timestamp
            })

        window_entry = {
            'window_index': wi,
            'dialogue_range': f"[{ws+1}, {we}]",
            'num_dialogues': we - ws,
            'dialogues': dialogue_texts,
            'extracted_memories': window_memories.get(wi, []),
            'num_memories': len(window_memories.get(wi, [])),
            'relevant_questions': window_to_qas.get(wi, []),
            'num_relevant_questions': len(window_to_qas.get(wi, []))
        }

        analysis['windows'].append(window_entry)

    # Add summary statistics
    total_memories = sum(len(m) for m in window_memories.values())
    qas_with_evidence = set()
    for qas in window_to_qas.values():
        for qa in qas:
            qas_with_evidence.add(qa['qa_index'])

    analysis['summary'] = {
        'total_memories_extracted': total_memories,
        'avg_memories_per_window': total_memories / len(windows) if windows else 0,
        'qas_with_evidence_in_windows': len(qas_with_evidence),
        'qas_without_evidence': len(sample['qa']) - len(qas_with_evidence),
    }

    return analysis


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze TriMem memory construction process')
    parser.add_argument('--dataset', type=str, default='./test_ref/locomo10.json',
                       help='Path to LoComo10 dataset')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample index to analyze (default: 0)')
    parser.add_argument('--output', type=str, default='memory_construction_analysis.json',
                       help='Output file path')
    parser.add_argument('--windows', type=str, default=None,
                       help='Comma-separated window indices to process (e.g., "0,1,2"). Default: all')

    args = parser.parse_args()

    print("="*60)
    print(" TriMem Memory Construction Analysis")
    print("="*60)

    # Load sample
    print(f"\nLoading sample {args.sample} from {args.dataset}...")
    sample = load_sample(args.dataset, args.sample)

    # Build dialogues and mapping
    dialogues, dia_id_to_idx = build_dialogues_and_mapping(sample)
    print(f"Total dialogues: {len(dialogues)}")

    # Compute windows
    window_size = config.WINDOW_SIZE
    overlap_size = getattr(config, 'OVERLAP_SIZE', 0)
    windows = compute_windows(len(dialogues), window_size, overlap_size)
    print(f"Window size: {window_size}, overlap: {overlap_size}")
    print(f"Number of windows: {len(windows)}")
    for wi, (ws, we) in enumerate(windows):
        print(f"  Window {wi}: dialogues [{ws+1}, {we}] ({we-ws} turns)")

    # Filter windows if specified
    if args.windows:
        selected = [int(x.strip()) for x in args.windows.split(',')]
        windows_to_process = [(wi, windows[wi]) for wi in selected if wi < len(windows)]
        print(f"\nProcessing only windows: {selected}")
    else:
        windows_to_process = list(enumerate(windows))

    # Map QA evidence to windows
    window_to_qas = map_qa_to_windows(sample, dia_id_to_idx, windows)
    print(f"\nQA-to-window mapping complete:")
    for wi in sorted(window_to_qas.keys()):
        cats = defaultdict(int)
        for qa in window_to_qas[wi]:
            cats[qa['category']] += 1
        cat_str = ', '.join(f'cat{k}:{v}' for k, v in sorted(cats.items()))
        print(f"  Window {wi}: {len(window_to_qas[wi])} questions ({cat_str})")

    # Initialize LLM client (no vector store needed for extraction only)
    print("\nInitializing LLM client...")
    llm_client = LLMClient()

    # Dummy vector store (not used for extraction)
    from utils.embedding import EmbeddingModel
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model=embedding_model, table_name="analysis_temp")

    # Extract memories per window
    print("\n" + "="*60)
    print(" Starting Memory Extraction Per Window")
    print("="*60)

    window_memories = {}
    for wi, (ws, we) in windows_to_process:
        window_dialogues = dialogues[ws:we]
        print(f"\n{'='*60}")
        print(f"Window {wi}: dialogues [{ws+1}, {we}] ({we-ws} turns)")
        print(f"{'='*60}")

        # Show first/last dialogue for context
        print(f"  First: [{window_dialogues[0].timestamp}] {window_dialogues[0].speaker}: {window_dialogues[0].content[:80]}")
        print(f"  Last:  [{window_dialogues[-1].timestamp}] {window_dialogues[-1].speaker}: {window_dialogues[-1].content[:80]}")

        # Show relevant questions for this window
        qas_for_window = window_to_qas.get(wi, [])
        print(f"  Relevant questions: {len(qas_for_window)}")
        for qa in qas_for_window[:5]:
            print(f"    Q{qa['qa_index']} [cat{qa['category']}]: {qa['question'][:70]}")
            print(f"      Evidence in window: {qa['evidence_in_this_window']}")
        if len(qas_for_window) > 5:
            print(f"    ... and {len(qas_for_window)-5} more")

        # Create a fresh MemoryBuilder
        builder = MemoryBuilder(
            llm_client=llm_client,
            vector_store=vector_store,
            window_size=len(window_dialogues),
            enable_parallel_processing=False
        )

        start_time = time.time()
        entries = builder._generate_memory_entries(window_dialogues)
        elapsed = time.time() - start_time

        print(f"\n  Generated {len(entries)} memory entries in {elapsed:.2f}s")
        for ei, entry in enumerate(entries):
            print(f"  [{ei}] {entry.lossless_restatement[:100]}")

        # Convert to dicts
        entry_dicts = []
        for entry in entries:
            entry_dicts.append({
                'lossless_restatement': entry.lossless_restatement,
                'keywords': entry.keywords,
                'timestamp': entry.timestamp,
                'location': entry.location,
                'persons': entry.persons,
                'entities': entry.entities,
                'topic': entry.topic
            })
        window_memories[wi] = entry_dicts

    # Build full analysis output (include all windows, even unprocessed ones)
    analysis = build_analysis_output(sample, dialogues, windows, window_memories, window_to_qas)

    # Save
    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f" Analysis saved to {output_path}")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Total windows: {len(windows)}")
    print(f"  Windows processed: {len(windows_to_process)}")
    print(f"  Total memories extracted: {sum(len(m) for m in window_memories.values())}")
    print(f"  QAs mapped to windows: {analysis['summary']['qas_with_evidence_in_windows']}")
    print(f"  QAs without evidence: {analysis['summary']['qas_without_evidence']}")


if __name__ == "__main__":
    main()
