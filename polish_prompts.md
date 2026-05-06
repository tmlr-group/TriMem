## Polishing Prompts

A lightweight loop for refining the extraction prompt and profile prompt of TriMem using Claude Code with Claude Opus 4.6 as an in-the-loop optimizer. No fine-tuning, no gradient code — one Claude Code session per round.

### The loop

```
                    eval results JSON
                            │
                            ▼
   prompts/   ───►   Claude Code   ───►   rewritten prompts
                     (one round)                 │
                          ▲                      │
                          │                      ▼
                          └──────────────── re-run eval 
```

One round = one Claude Code session that reads the latest eval JSON and the
current prompts, then writes new ones in place. Repeat until the metric
plateaus (typically 3–5 rounds).

### Quick start

1. **Run the baseline evaluation** to get a result JSON:

   ```bash
   python test_locomo10.py --result-file results/round_0.json --llm-judge
   ```

2. **Open Claude Code** in the project root:

   ```bash
   claude
   ```

3. **Paste this instruction**, replacing the three placeholders inline:
   - `<RESULT_JSON_PATH>` → path to the eval JSON (e.g. `results/round_0.json`)
   - `<PASTE CURRENT P_ext HERE>` → the current extraction prompt, verbatim
   - `<PASTE CURRENT P_prof HERE>` → the current profile prompt, verbatim

   ```
   You are a senior prompt engineer specializing in lifelong memory systems
   for LLM agents. Perform the "backward pass" of TextGrad: given the JSON
   dump of one full evaluation run, identify systematic failure patterns and
   directly produce updated versions of two upstream prompts — the fact
   extraction prompt P_ext and the entity profile construction prompt P_prof.

   [Current Extraction Prompt P_ext]
   <PASTE CURRENT P_ext HERE>

   [Current Profile Prompt P_prof]
   <PASTE CURRENT P_prof HERE>

   [Evaluation Result JSON]
   Read the file at <RESULT_JSON_PATH>. Each entry in `detailed_results`
   contains:
   {
     "question": str,
     "answer": str,             // system prediction
     "reference": str,          // gold answer
     "category": int,           // LoCoMo category id (1: multi-hop,
                                //   2: temporal, 3: open-domain, 4: single-hop)
     "metrics": {
       "f1": float, "rougeL_f": float, "bert_f1": float,
       "llm_judge_score": float (optional), "llm_reasoning": str (optional)
     }
   }

   Diagnose each failure based on whether the prediction loses a specific
   detail that should have been preserved (extraction issue) or fails to
   synthesize an entity-level judgment (profile issue), and locate the
   responsible prompt accordingly. Skip cases where the signal is unclear.

   Focus on systematic patterns rather than one-off failures. Per round,
   make only a small number of targeted additions; prefer adding new
   bullets, rules, or examples over rewriting existing content. Preserve
   the original output schema, field names, and existing examples, and
   keep the guidance domain-general. If a prompt does not need changes
   this round, leave it verbatim.

   Return a single JSON object with exactly three fields:

   {
     "rewritten_p_ext": "<full text of the updated P_ext, or verbatim copy
       of the input if unchanged>",
     "rewritten_p_prof": "<full text of the updated P_prof, or verbatim copy
       of the input if unchanged>",
     "change_summary": "<2-4 sentences explaining: (i) what failure pattern
       motivated each addition, (ii) which prompt and section was edited,
       (iii) what behavior change is expected. If a prompt was unchanged,
       state that explicitly and why no eligible pattern was found.>"
   }

   Return ONLY the JSON, no surrounding prose or code fences.
   ```

4. **Re-run the evaluation** with the new prompts:

   ```bash
   python test_locomo10.py --result-file results/round_1.json --llm-judge
   ```

   If the metric improved, commit and continue to round 2. If it regressed,
   roll back the prompts and skip this round.

### When to stop

Stop when *any* of these holds:

- The metric improves by less than ~0.5 absolute points for two rounds.
- A round regresses despite the failure batch still being non-empty.
- The change summary becomes vague ("minor refinements", "tightened
  wording") — Claude Code is out of distinct patterns to act on.

In our experiments 4 rounds was the sweet spot; beyond that the prompts
start to overfit and the JSON schema risks drift.
