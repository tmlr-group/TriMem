"""
Analyze coverage of extracted memories against QA questions.

Reads the output of analyze_memory_construction.py and checks whether
each question's evidence is captured in the extracted memories.

Usage:
    python scripts/analyze_coverage.py --analysis memory_construction_analysis.json
"""
import json
import argparse
from collections import defaultdict


def load_dia_id_mapping(dataset_path, sample_idx=0):
    """Build dia_id string (e.g. 'D1:3') -> global dialogue_id mapping."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    sample = dataset[sample_idx]

    dia_str_to_global = {}
    global_idx = 1
    session_ids = []
    for k in sample['conversation'].keys():
        if k.startswith('session_') and not k.endswith('_date_time') and isinstance(sample['conversation'][k], list):
            session_ids.append(int(k.split('_')[1]))
    session_ids.sort()
    for sid in session_ids:
        for turn in sample['conversation'][f'session_{sid}']:
            dia_str_to_global[turn['dia_id']] = global_idx
            global_idx += 1
    return dia_str_to_global


def analyze(analysis_path, dataset_path, sample_idx=0):
    with open(analysis_path, 'r') as f:
        data = json.load(f)

    dia_str_to_global = load_dia_id_mapping(dataset_path, sample_idx)

    # Build dialogue_id -> content lookup
    all_dialogues = {}
    for w in data['windows']:
        for d in w['dialogues']:
            all_dialogues[d['dialogue_id']] = d

    stop_words = {
        'the', 'a', 'an', 'is', 'was', 'were', 'are', 'in', 'on', 'at',
        'to', 'of', 'and', 'or', 'for', 'not', 'i', 'my', 'her', 'his',
        'she', 'he', 'it', 'they', 'we', 'me', 'that', 'this', 'with',
    }

    results_by_category = defaultdict(list)
    all_results = []

    for w in data['windows']:
        wi = w['window_index']
        memories = w['extracted_memories']
        memory_texts = [m['lossless_restatement'].lower() for m in memories]

        for qa in w['relevant_questions']:
            qi = qa['qa_index']
            question = qa['question']
            answer = str(qa.get('answer', '') or '')
            category = qa.get('category', 0)
            evidence_ids = qa.get('evidence_in_this_window', [])

            evidence_dialogues = []
            for ev_id in evidence_ids:
                parts = [e.strip() for e in ev_id.split(';')]
                for part in parts:
                    if part in dia_str_to_global:
                        gid = dia_str_to_global[part]
                        if gid in all_dialogues:
                            d = all_dialogues[gid]
                            evidence_dialogues.append({
                                'dia_id': part,
                                'global_id': gid,
                                'speaker': d['speaker'],
                                'content': d['content'],
                                'timestamp': d['timestamp'],
                            })

            answer_lower = answer.lower()
            answer_words = set(answer_lower.split()) - stop_words

            best_overlap = 0
            best_memory_idx = -1
            for mi, mt in enumerate(memory_texts):
                overlap = sum(1 for w in answer_words if w in mt)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_memory_idx = mi

            coverage_ratio = best_overlap / max(len(answer_words), 1)

            if coverage_ratio >= 0.5:
                status = 'COVERED'
            elif coverage_ratio > 0 and best_overlap >= 1:
                status = 'PARTIAL'
            else:
                status = 'MISSING'

            result = {
                'window': wi,
                'qa_index': qi,
                'category': category,
                'question': question,
                'answer': answer,
                'evidence_ids': evidence_ids,
                'evidence_text': [e['content'] for e in evidence_dialogues],
                'evidence_timestamp': evidence_dialogues[0]['timestamp'] if evidence_dialogues else '',
                'status': status,
                'coverage_ratio': coverage_ratio,
                'best_memory': memories[best_memory_idx]['lossless_restatement'] if best_memory_idx >= 0 else 'NONE',
            }
            all_results.append(result)
            results_by_category[category].append(result)

    # --- Reporting ---
    cat_name = {1: 'Factual', 2: 'Temporal', 3: 'Inferential', 4: 'Detail', 5: 'Adversarial'}

    print("=" * 100)
    print("COVERAGE ANALYSIS REPORT")
    print("=" * 100)

    status_counts = defaultdict(int)
    for r in all_results:
        status_counts[r['status']] += 1
    print(f"\nOverall: {len(all_results)} question-window pairs")
    for s in ['COVERED', 'PARTIAL', 'MISSING']:
        c = status_counts[s]
        print(f"  {s}: {c} ({100 * c / len(all_results):.1f}%)")

    print(f"\nBy Category:")
    for cat in sorted(results_by_category.keys()):
        items = results_by_category[cat]
        cat_status = defaultdict(int)
        for r in items:
            cat_status[r['status']] += 1
        total = len(items)
        print(f"  Cat {cat} ({cat_name.get(cat, '?')}, n={total}): {dict(sorted(cat_status.items()))}")

    print("\n" + "=" * 100)
    print("MISSING EXAMPLES (3 per category)")
    print("=" * 100)

    for cat in sorted(results_by_category.keys()):
        print(f"\n--- CATEGORY {cat} ({cat_name.get(cat, '?')}) ---")
        missing = [r for r in results_by_category[cat] if r['status'] == 'MISSING']
        for r in missing[:3]:
            print(f"\n  Q{r['qa_index']} (Window {r['window']}):")
            print(f"  Question:  {r['question']}")
            print(f"  Answer:    {r['answer']}")
            print(f"  Evidence:  {r['evidence_ids']}")
            for et in r['evidence_text']:
                print(f"  Dialogue:  \"{et[:200]}\"")
            print(f"  Timestamp: {r['evidence_timestamp']}")
            print(f"  Closest Memory: \"{r['best_memory'][:200]}\"")

    print("\n" + "=" * 100)
    print("COVERED EXAMPLES (5 total)")
    print("=" * 100)
    covered = [r for r in all_results if r['status'] == 'COVERED']
    for r in covered[:5]:
        print(f"\n  Q{r['qa_index']} [Cat{r['category']}] (Window {r['window']}):")
        print(f"  Question: {r['question']}")
        print(f"  Answer:   {r['answer']}")
        print(f"  Memory:   \"{r['best_memory'][:200]}\"")

    # Per-window summary
    print("\n" + "=" * 100)
    print("PER-WINDOW SUMMARY")
    print("=" * 100)
    window_stats = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        window_stats[r['window']][r['status']] += 1
        window_stats[r['window']]['total'] += 1
    for wi in sorted(window_stats.keys()):
        ws = window_stats[wi]
        total = ws['total']
        covered = ws.get('COVERED', 0)
        partial = ws.get('PARTIAL', 0)
        missing = ws.get('MISSING', 0)
        print(f"  Window {wi:2d}: {total:3d} questions | COVERED {covered:2d} | PARTIAL {partial:2d} | MISSING {missing:2d} | coverage {100 * (covered + partial) / total:.0f}%")

    # Memory quality per window
    print("\n" + "=" * 100)
    print("MEMORY QUALITY PER WINDOW (trivial vs substantive)")
    print("=" * 100)
    trivial_keywords = ['greeted', 'thanked', 'praised', 'expressed appreciation',
                        'commended', 'complimented', 'affirmed', 'acknowledged']
    for w in data['windows']:
        trivial = 0
        for m in w['extracted_memories']:
            text = m['lossless_restatement'].lower()
            if any(kw in text for kw in trivial_keywords):
                trivial += 1
        total = len(w['extracted_memories'])
        print(f"  Window {w['window_index']:2d}: {total:2d} memories | {trivial:2d} trivial ({100 * trivial / max(total, 1):.0f}%) | {total - trivial:2d} substantive")


def main():
    parser = argparse.ArgumentParser(description='Analyze memory extraction coverage')
    parser.add_argument('--analysis', type=str, default='memory_construction_analysis.json',
                        help='Path to analysis JSON from analyze_memory_construction.py')
    parser.add_argument('--dataset', type=str, default='./test_ref/locomo10.json',
                        help='Path to LoComo10 dataset')
    parser.add_argument('--sample', type=int, default=0, help='Sample index')
    args = parser.parse_args()

    analyze(args.analysis, args.dataset, args.sample)


if __name__ == '__main__':
    main()
