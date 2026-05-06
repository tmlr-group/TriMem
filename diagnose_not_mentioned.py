"""
Diagnose "Not mentioned" failures in profiles8 results.
For each failing question, check:
1. Is it routed to inference or non-inference prompt?
2. What memories are retrieved?
3. Do any retrieved memories contain relevant evidence?
4. What profile context is available?
"""
import json
import re
import sys

from main import TriMemSystem
from test_locomo10 import load_locomo_dataset

# The "Not mentioned" failures from profiles8
NOT_MENTIONED_FAILURES = [
    ("What nickname does Nate use for Joanna?", "Jo"),
    ("Based on Tim's collections, what is a shop that he would enjoy visiting in New York city?", "House of MinaLima"),
    ("Which popular music composer's tunes does Tim enjoy playing on the piano?", "John Williams"),
    ("Which US states might Tim be in during September 2023 based on his plans of visiting Universal Studios?", "California or Florida"),
    ("Which Star Wars-related locations would Tim enjoy during his visit to Ireland?", "Skellig Michael, Malin Head, Loop Head, Ceann Sibéal, and Brow Head"),
    ("Which national park could Audrey and Andrew be referring to in their conversations?", "Voyageurs National Park"),
    ("What are John's suspected health problems?", "Obesity"),
    ("Was James feeling lonely before meeting Samantha?", "Most likely yes"),
    ("Did John and James study together?", "Yes"),
    ("What additional country did James visit during his trip to Canada?", "Greenland"),
    ("Who is Jill?", "Most likely John's partner"),
    ("How old is Jolene?", "likely no more than 30; since she's in school"),
    ("Does Dave's shop employ a lot of people?", "Yes"),
]

# Inference detection patterns (from answer_generator.py)
INFERENCE_PATTERNS = [
    r"^would\b", r"^could\b", r"^might\b", r"^is it likely\b",
    r"^what might\b", r"^what would\b", r"^what could\b",
    r"^what fields would\b", r"^what personality\b", r"^what attributes\b",
    r"^what (?:advice|challenges|traits|characteristics)\b",
    r"^would .+ be considered\b", r"^would .+ likely\b", r"^would .+ enjoy\b",
    r"^would .+ want\b", r"^would .+ pursue\b", r"^would .+ be open\b",
    r"^does .+ (?:live|love|like|enjoy|prefer)\b",
    r"^is .+ (?:likely|considered|religious|patriotic|an? )\b",
    r"^are .+ fans?\b",
    r"^what (?:console|card game|technique|country|state)\b",
    r"^which (?:country|state|us state|console)\b",
    r"^in what (?:country|state)\b",
    r"\blikely\b", r"\bmight\b", r"\bpotentially\b",
    r"\bwouldn't\b", r"\bprobably\b",
    r"\bfinancial status\b", r"\bpolitical\b", r"\bleaning\b",
]
inference_re = re.compile("|".join(INFERENCE_PATTERNS), re.IGNORECASE)


def find_sample_for_question(dataset, question):
    """Find which sample a question belongs to."""
    for i, sample in enumerate(dataset):
        for qa in sample.qa:
            if qa.question == question:
                return i, sample, qa
    return None, None, None


def main():
    # Load dataset to find sample mapping
    dataset = load_locomo_dataset("test_ref/locomo10.json")

    # Group questions by sample
    sample_questions = {}
    for q, ref in NOT_MENTIONED_FAILURES:
        idx, sample, qa = find_sample_for_question(dataset, q)
        if idx is not None:
            if idx not in sample_questions:
                sample_questions[idx] = []
            sample_questions[idx].append((q, ref, qa))

    # Process each sample
    for sample_idx in sorted(sample_questions.keys()):
        sample = dataset[sample_idx]
        questions = sample_questions[sample_idx]

        print(f"\n{'='*80}")
        print(f"SAMPLE {sample_idx}: {sample.conversation.speaker_a} & {sample.conversation.speaker_b}")
        print(f"{'='*80}")

        # Initialize system for this sample (load existing DB)
        table_name = f"locomo_sample_{sample_idx}"
        system = TriMemSystem(
            db_path="./lancedb_data",
            table_name=table_name,
            clear_db=False,
        )

        for q, ref, qa in questions:
            is_inference = bool(inference_re.search(q))
            print(f"\n{'─'*70}")
            print(f"Q: {q}")
            print(f"Expected: {ref}")
            print(f"Inference detected: {is_inference}")
            print(f"Evidence IDs: {qa.evidence if qa else 'N/A'}")

            # Run retrieval only (no answer generation)
            contexts = system.hybrid_retriever.retrieve(q)
            print(f"\nRetrieved {len(contexts)} memories:")

            # Search for keywords from the reference answer in retrieved contexts
            ref_keywords = set(ref.lower().split())
            # Remove common words
            ref_keywords -= {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'or', 'and', 'to', 'of', 'in', 'for', 'on', 'at', 'by', 'with', 'from', 'that', 'this', 'it', 'they', 'he', 'she', 'his', 'her', 'their', 'yes', 'no', 'likely', 'most', 'because', 'since'}

            relevant_found = False
            for i, ctx in enumerate(contexts):
                content_lower = ctx.lossless_restatement.lower()
                matching_keywords = [kw for kw in ref_keywords if kw in content_lower]
                if matching_keywords:
                    relevant_found = True
                    print(f"  [{i+1}] *** RELEVANT (matches: {matching_keywords}) ***")
                    print(f"      {ctx.lossless_restatement[:200]}")
                    if ctx.persons:
                        print(f"      Persons: {ctx.persons}")

            if not relevant_found:
                print("  >>> NO relevant memories found in retrieval!")
                # Show first 5 for context
                for i, ctx in enumerate(contexts[:5]):
                    print(f"  [{i+1}] {ctx.lossless_restatement[:150]}")
                    if ctx.persons:
                        print(f"      Persons: {ctx.persons}")

            # Check profile context
            if system.profile_manager:
                profile_ctx = system.profile_manager.get_profiles_for_query(q, contexts)
                if profile_ctx:
                    # Check if reference keywords appear in profile
                    profile_lower = profile_ctx.lower()
                    matching_in_profile = [kw for kw in ref_keywords if kw in profile_lower]
                    if matching_in_profile:
                        print(f"\n  Profile has relevant keywords: {matching_in_profile}")
                    # Show relevant portion
                    for kw in ref_keywords:
                        idx = profile_lower.find(kw)
                        if idx >= 0:
                            start = max(0, idx - 50)
                            end = min(len(profile_ctx), idx + 50)
                            print(f"  Profile excerpt: ...{profile_ctx[start:end]}...")
                else:
                    print("\n  No profile context returned")

            print()


if __name__ == "__main__":
    main()
