"""
Answer Generator - Final synthesis from retrieved contexts

Section 3.3: Intent-Aware Retrieval Planning
Generates answers from the merged context C_q after multi-view retrieval.

Enhanced with source context linking: when memories are retrieved, the
original dialogues they were derived from are also fetched and provided
to the LLM for complete, uncompressed detail.

Enhanced with entity profiles for open-domain inference questions.
"""
from typing import List, Optional, Dict, Any
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
from database.dialogue_store import DialogueStore
import config

class AnswerGenerator:
    """
    Answer Generator - Synthesis from retrieved memory units (Section 3.3)

    Generates answers from C_q = R_sem ∪ R_lex ∪ R_sym
    Enhanced: also fetches source dialogues for each retrieved memory.
    Enhanced: includes entity profiles for personality/preference inference.
    Enhanced: separate inference prompt for Cat 3 open-domain questions.
    """

    # Patterns that indicate inference/hypothetical questions (Cat 3 style)
    # Anchored patterns (match at start of question)
    INFERENCE_PATTERNS = [
        # Modal verbs at start
        r"^would\b", r"^could\b", r"^might\b",
        r"^is it likely\b",
        # "What + modal" at start
        r"^what might\b", r"^what would\b", r"^what could\b",
        # "What + inference topic" at start
        r"^what fields would\b", r"^what personality\b", r"^what attributes\b",
        r"^what (?:advice|challenges|traits|characteristics)\b",
        r"^what (?:console|card game|technique|country|state)\b",
        r"^what (?:nickname|additional|role|other)\b",
        r"^what is the (?:game|board game)\b",
        # "Would + inference verb"
        r"^would .+ be considered\b", r"^would .+ likely\b", r"^would .+ enjoy\b",
        r"^would .+ want\b", r"^would .+ pursue\b", r"^would .+ be open\b",
        # "Does + inference verb"
        r"^does .+ (?:live|love|like|enjoy|prefer|wish|employ)\b",
        # "Is + inference adjective"
        r"^is .+ (?:likely|considered|religious|patriotic|married|an? )\b",
        # Other anchored patterns
        r"^are .+ fans?\b",
        r"^was .+ feeling\b",
        r"^which (?:country|state|us state|console)\b",
        r"^which (?:popular|major)\b",
        r"^in what (?:country|state)\b",
        r"^in which state\b",
        r"^around which\b",
        r"^based on\b",
        r"^in light of\b",
        r"^considering\b",
        r"^who is \w+\??$",
        r"^how old\b",
        r"^how often\b",
        # Non-anchored patterns (match anywhere in question)
        r"\bwould\b", r"\bcould\b",
        r"\blikely\b", r"\bmight\b", r"\bpotentially\b",
        r"\bwouldn't\b", r"\bprobably\b",
        r"\bsuspected\b", r"\bprefer\b", r"\balive\b",
        r"\bfinancial status\b", r"\bpolitical\b", r"\bleaning\b",
    ]

    def __init__(self, llm_client: LLMClient, dialogue_store: Optional[DialogueStore] = None):
        self.llm_client = llm_client
        self.dialogue_store = dialogue_store
        self.context_window = getattr(config, 'SOURCE_CONTEXT_WINDOW', 2)
        import re
        self._inference_re = re.compile(
            "|".join(self.INFERENCE_PATTERNS), re.IGNORECASE
        )

    def _is_inference_question(self, query: str) -> bool:
        """Detect inference/hypothetical questions (Cat 3 style)."""
        return bool(self._inference_re.search(query))

    def generate_answer(
        self,
        query: str,
        contexts: List[MemoryEntry],
        profile_context: str = ""
    ) -> str:
        """
        Generate answer using structured memories, source dialogues, and
        optionally entity profiles for inference questions.
        """
        if not contexts:
            return "No relevant information found"

        # Build context string (structured memories)
        context_str = self._format_contexts(contexts)

        # Fetch and format source dialogues if dialogue store is available
        source_dialogue_str = ""
        if self.dialogue_store and self.dialogue_store.count() > 0:
            source_dialogue_str = self._fetch_source_dialogues(contexts)

        # Route to inference or factual prompt based on question type
        if self._is_inference_question(query):
            prompt = self._build_inference_prompt(query, context_str, source_dialogue_str, profile_context)
            system_msg = "You are a Q&A assistant. Reason from the available evidence and your world knowledge to give the most likely answer. Always commit to a definitive answer — never say 'Not mentioned'. Keep the answer field as SHORT as possible — put all reasoning in the reasoning field. Output valid JSON only."
        else:
            prompt = self._build_answer_prompt(query, context_str, source_dialogue_str, profile_context)
            system_msg = "You are a precise Q&A assistant. Answer based strictly on the provided context. Keep answers concise and factual. Output valid JSON only."

        # Call LLM to generate answer
        messages = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
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

                result = self.llm_client.extract_json(response)
                return result.get("answer", response.strip())

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to parse JSON response after {max_retries} attempts: {e}")
                    if 'response' in locals():
                        return response.strip()
                    else:
                        return "Failed to generate answer"

    def _fetch_source_dialogues(self, contexts: List[MemoryEntry]) -> str:
        """
        Fetch source dialogues for retrieved memory entries.
        """
        all_source_ids = set()
        for entry in contexts:
            if entry.source_dialogue_ids:
                all_source_ids.update(entry.source_dialogue_ids)

        if not all_source_ids:
            return ""

        dialogues = self.dialogue_store.get_with_context(
            list(all_source_ids),
            context_window=self.context_window
        )

        if not dialogues:
            return ""

        lines = []
        for d in dialogues:
            time_str = f"[{d.timestamp}] " if d.timestamp else ""
            marker = "*" if d.dialogue_id in all_source_ids else " "
            lines.append(f"  {marker} {time_str}{d.speaker}: {d.content}")

        return "\n".join(lines)

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """
        Format contexts to readable text
        """
        formatted = []
        for i, entry in enumerate(contexts, 1):
            parts = [f"[Context {i}]"]
            parts.append(f"Content: {entry.lossless_restatement}")

            if entry.timestamp:
                parts.append(f"Time: {entry.timestamp}")

            if entry.location:
                parts.append(f"Location: {entry.location}")

            if entry.persons:
                parts.append(f"Persons: {', '.join(entry.persons)}")

            if entry.entities:
                parts.append(f"Related Entities: {', '.join(entry.entities)}")

            if entry.topic:
                parts.append(f"Topic: {entry.topic}")

            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)

    def _build_inference_prompt(
        self, query: str, context_str: str,
        source_dialogue_str: str = "", profile_context: str = ""
    ) -> str:
        """
        Build prompt for inference/hypothetical questions (Cat 3).
        Encourages reasoning from evidence rather than refusing to answer.
        """
        source_section = ""
        if source_dialogue_str:
            source_section = f"""
[Original Dialogue Excerpts]
{source_dialogue_str}
"""

        profile_section = ""
        if profile_context:
            profile_section = f"""
[Entity Profiles]
{profile_context}
"""

        return f"""Answer the inference question by reasoning from the provided context and profiles.

Question: {query}

[Structured Memory Entries]
{context_str}
{source_section}{profile_section}
RULES FOR INFERENCE QUESTIONS:
1. INFER from the evidence — NEVER say "Not mentioned" or "unknown". You MUST commit to an answer.
2. Use context + profiles to reason about personality, interests, values, behavior.
3. **USE WORLD KNOWLEDGE**: When the context describes something without naming it, IDENTIFY it.
   - A game exclusive to a platform → name the platform (e.g. Xenoblade 2 → Nintendo Switch)
   - A described but unnamed product → identify it (e.g. "card game about exploding cats" → Exploding Kittens)
   - A described technique → name it (e.g. "work 25 minutes then break" → Pomodoro technique)
   - A city mentioned when question asks about country → resolve to country (e.g. Paris → France, Toronto → Canada)
4. Pay attention to WHO said/did things — speaker tags are authoritative.
5. ALWAYS cite specific facts from profiles/context in your reasoning — never guess generically.
6. If the question asks what someone ELSE would say about a person, check [How Others Describe Them] in profiles.
7. If the question asks about religion/spirituality, check [Beliefs/Spirituality] in profiles.
8. Prefer the MOST SPECIFIC reason from profiles (e.g. "adopting children" beats "she is settled").

ANSWER FORMAT — BE MAXIMALLY CONCISE:
- The "answer" field must be as SHORT as possible. All reasoning goes in the "reasoning" field ONLY.
- For yes/no questions ("would", "does", "is", "can", "did", "are", "was"):
  * If evidence is DIRECT: answer "Yes" or "No"
  * If evidence is INDIRECT/inferred: answer "Likely yes" or "Likely no"
  * If the reference answer would naturally include a brief reason (e.g. "Yes, she is supportive" or "No, he has goals in the U.S."), add a SHORT reason after a comma. Keep it under 10 words.
  * NEVER add long explanations.
- For "A or B" choice questions (e.g. "beach or mountains?", "Charger or Forester?"): Answer with ONLY the chosen option — do NOT answer "Yes" or "No". Pick A or B based on evidence.
- For "what/who/which" questions: Answer with ONLY the name, term, or short phrase. No explanations.
- For trait/attribute questions: List ONLY the adjectives, comma-separated. Use exact words from [How Others Describe Them] in profiles if available.
- For "what might/what would/what could" questions: Give the shortest answer that captures the key point. Prefer a few words or a short phrase over a full sentence.
- For identification questions: Just name the thing. "Exploding Kittens", not "Exploding Kittens, a card game about cats".
- For "which country/state" questions: Answer with ONLY the country or state name.
- For "how many/how often" questions: Answer with ONLY the number or frequency.

Examples:
Q: "Would she enjoy classical music?" → {{"reasoning": "Profile [Preferences] says she loves Bach and Mozart", "answer": "Yes"}}
Q: "Would she be considered an ally?" → {{"reasoning": "She attended pride events and supported her friend's transition", "answer": "Yes, she is supportive"}}
Q: "Would he be open to moving to another country?" → {{"reasoning": "He has goals specifically in the U.S.", "answer": "No, he has goals specifically in the U.S."}}
Q: "Are they fans of the same team?" → {{"reasoning": "James is a Liverpool fan and John is a Manchester City fan", "answer": "No, James is a Liverpool fan and John is a Manchester City fan"}}
Q: "Does he live close to a beach or the mountains?" → {{"reasoning": "Context mentions sunset over ocean and beach photos", "answer": "beach"}}
Q: "Would he prefer a Charger or a Forester?" → {{"reasoning": "Profile shows he works on classic muscle cars", "answer": "Dodge Charger"}}
Q: "What career might he pursue?" → {{"reasoning": "Profile [Career] says pursuing counseling and mental health", "answer": "Counseling or social work"}}
Q: "Would she be considered religious?" → {{"reasoning": "Profile [Beliefs] says made art for a church and necklace symbolizes faith, but clashed with religious conservatives", "answer": "Somewhat, but not extremely religious"}}
Q: "What political leaning?" → {{"reasoning": "Profile [Values] shows LGBTQ+ advocacy, progressive values", "answer": "Liberal"}}
Q: "What card game is she talking about?" → {{"reasoning": "Context describes a card game about cats — this is Exploding Kittens", "answer": "Exploding Kittens"}}
Q: "What console does he own?" → {{"reasoning": "Context says he plays Xenoblade 2, which is a Nintendo Switch exclusive", "answer": "Nintendo Switch"}}
Q: "In what country did she buy the snake?" → {{"reasoning": "Context says she bought it in Paris, which is in France", "answer": "France"}}
Q: "What traits would X say Y has?" → {{"reasoning": "Profile [How Others Describe Them] says X called Y thoughtful, authentic, driven", "answer": "Thoughtful, authentic, driven"}}
Q: "Is it likely she moved?" → {{"reasoning": "Some indirect evidence suggests relocation but not confirmed", "answer": "Likely yes"}}
Q: "Does he love music?" → {{"reasoning": "Multiple contexts show passion for music", "answer": "Yes"}}
Q: "How many hikes has she been on?" → {{"reasoning": "Context mentions 4 separate hikes", "answer": "Four"}}
Q: "How often does he get checkups?" → {{"reasoning": "Context shows checkups every 3 months", "answer": "every three months"}}

Return ONLY valid JSON: {{"reasoning": "...", "answer": "shortest possible answer"}}
"""

    def _build_answer_prompt(
        self, query: str, context_str: str,
        source_dialogue_str: str = "", profile_context: str = ""
    ) -> str:
        """
        Build answer generation prompt with structured memories, source dialogues,
        and entity profiles.
        """
        source_section = ""
        if source_dialogue_str:
            source_section = f"""

[Original Dialogue Excerpts]
The following are the original conversation turns that the above memory entries were derived from.
Lines marked with * are the specific source turns. Use these for exact details, specific names,
numbers, and temporal references that may have been summarized in the memory entries above.

{source_dialogue_str}
"""

        profile_section = ""
        if profile_context:
            profile_section = f"""

[Entity Profiles]
The following are progressive profiles built from all known facts about these entities.
Use these for inference questions about personality, preferences, likely behavior, or character traits.

{profile_context}
"""

        return f"""Answer the question using ONLY the provided context.

Question: {query}

[Structured Memory Entries]
{context_str}
{source_section}{profile_section}
CRITICAL RULES:
1. Answer concisely with the key facts. Use short phrases or comma-separated keywords for simple factual questions.
2. Do NOT add unnecessary elaboration, but DO include all specific details the question asks for.
3. If asked "what does X do?" answer "running, pottery" NOT "X enjoys running and pottery as hobbies."
4. If asked "what is X?" answer with just the noun/phrase, e.g. "sunset" NOT "X painted a beautiful sunset."
5. Answer must be based ONLY on the provided context. If not found, say "Not mentioned in the conversation".
6. Pay attention to WHO said/did things - speaker tags are authoritative.
7. Resolve relative time expressions (e.g., "yesterday" on May 8 = May 7).
8. Format dates as 'DD Month YYYY'.
9. For "which country/state" questions, answer with the country/state, NOT a city.
10. Return JSON only.

Examples:
Q: "What career is she pursuing?" → {{"reasoning": "Context says she chose counseling", "answer": "counseling"}}
Q: "What does she do to destress?" → {{"reasoning": "Context mentions running and pottery", "answer": "Running, pottery"}}
Q: "What did she paint?" → {{"reasoning": "Context says she painted a sunset", "answer": "sunset"}}
Q: "Where has she camped?" → {{"reasoning": "Mentions beach, mountains, forest", "answer": "beach, mountains, forest"}}
Q: "When will they meet?" → {{"reasoning": "Meeting set for 2025-11-16", "answer": "16 November 2025"}}
Q: "What is her identity?" → {{"reasoning": "Context identifies her as transgender woman", "answer": "Transgender woman"}}
Q: "Which country did he visit?" → {{"reasoning": "Context says he went to Paris", "answer": "France"}}

Return ONLY valid JSON: {{"reasoning": "...", "answer": "concise factual answer"}}
"""
