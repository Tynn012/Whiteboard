from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QuizQuestion:
    question: str
    answer: str
    context: str
    choices: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "choices": list(self.choices),
        }


def build_choices(answer: str, answer_pool: Sequence[str], context: str, num_choices: int = 4) -> list[str]:
    """Build multiple choice options with challenging distractors.
    
    Prioritizes entity-based distractors from context that are more plausible wrong answers.
    """
    distractors: list[str] = []
    answer_lower = answer.lower()
    
    # Gather candidates from multiple sources, deduplicate
    candidates: list[str] = []
    for ent in _extract_context_entities(context):
        candidates.append(normalize_whitespace(ent))

    for s in _sample_answers(context, 20):
        candidates.append(normalize_whitespace(s))

    for a in answer_pool:
        candidates.append(normalize_whitespace(str(a)))

    # Clean and dedupe
    cleaned: list[str] = []
    for c in candidates:
        if not c:
            continue
        if c.lower() == answer_lower:
            continue
        if c in cleaned:
            continue
        cleaned.append(c)

    # Scoring: prefer candidates that are similar in length and share tokens
    def _score_candidate(cand: str) -> float:
        # length similarity
        len_diff = abs(len(cand) - len(answer))
        len_score = max(0.0, 1.0 - (len_diff / max(1, len(answer))))
        # token overlap
        ans_tokens = set(w.lower() for w in re.findall(r"\w+", answer))
        cand_tokens = set(w.lower() for w in re.findall(r"\w+", cand))
        if not ans_tokens:
            token_score = 0.0
        else:
            token_score = len(ans_tokens & cand_tokens) / max(1, len(ans_tokens))
        # numeric/alphabetic type match bonus
        def _is_numeric(s: str) -> bool:
            return bool(re.fullmatch(r"[\d\.,%]+", s))
        type_bonus = 0.2 if _is_numeric(cand) == _is_numeric(answer) else 0.0
        return len_score * 0.5 + token_score * 0.5 + type_bonus

    scored = [(c, _score_candidate(c)) for c in cleaned]
    scored.sort(key=lambda x: x[1], reverse=True)

    for c, _s in scored:
        if c not in distractors:
            distractors.append(c)
        if len(distractors) >= num_choices - 1:
            break

    # Fill remaining slots with context-aware fallbacks or generic fallbacks
    fallback_pool = _get_context_fallbacks(context) + _fallback_distractors()
    fi = 0
    while len(distractors) < num_choices - 1 and fi < len(fallback_pool):
        filler = fallback_pool[fi]
        fi += 1
        if filler.lower() != answer_lower and filler not in distractors:
            distractors.append(filler)

    choices = [answer] + distractors[: num_choices - 1]
    random.shuffle(choices)
    return choices


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sample_answers(text: str, limit: int) -> list[str]:
    if not text:
        return []

    patterns = [
        r"\b\d+(?:[.,]\d+)?(?:%|[A-Za-z]{1,3})?\b",
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
        r"\b[a-zA-Z][a-zA-Z\-]{4,}\b",
    ]
    seen: set[str] = set()
    candidates: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = normalize_whitespace(match.group(0)).strip(".,;:!?()[]{}\"'")
            if not candidate:
                continue
            if candidate.lower() in _QUESTION_WORDS:
                continue
            if candidate.lower() in seen:
                continue
            seen.add(candidate.lower())
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _extract_context_entities(context: str, limit: int = 10) -> list[str]:
    """Extract key entities and phrases from context that could serve as plausible distractors."""
    if not context:
        return []
    
    entities: list[str] = []
    
    # Extract capitalized phrases (proper nouns, organizations, etc.)
    proper_noun_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
    for match in re.finditer(proper_noun_pattern, context):
        entity = normalize_whitespace(match.group(0))
        if entity and len(entity) > 2:
            entities.append(entity)
            if len(entities) >= limit:
                return entities
    
    # Extract numbers with context
    number_pattern = r"\b\d+(?:[.,]\d+)?(?:%|\s*(?:million|billion|trillion|thousand|hundred|percent))?\b"
    for match in re.finditer(number_pattern, context):
        entity = normalize_whitespace(match.group(0))
        if entity:
            entities.append(entity)
            if len(entities) >= limit:
                return entities
    
    return entities


def _get_context_fallbacks(context: str) -> list[str]:
    """Generate context-aware fallback distractors."""
    # Determine if context seems to be about history, science, geography, etc.
    context_lower = context.lower()
    
    if any(word in context_lower for word in ["century", "year", "war", "president", "king"]):
        return [
            "An earlier period",
            "A later period",
            "During the same era",
            "In the previous century",
        ]
    elif any(word in context_lower for word in ["species", "animal", "plant", "organism", "cell"]):
        return [
            "A different species",
            "A related organism",
            "An unrelated species",
            "A extinct relative",
        ]
    elif any(word in context_lower for word in ["country", "city", "region", "capital", "located"]):
        return [
            "A neighboring region",
            "A different continent",
            "A nearby area",
            "The same general region",
        ]
    else:
        return [
            "Another relevant option",
            "A related concept",
            "A similar alternative",
            "A plausible but incorrect choice",
        ]


def _fallback_distractors() -> list[str]:
    return [
        "All of the above",
        "None of the above",
        "Insufficient information",
        "The main concept",
        "A supporting detail",
    ]


_QUESTION_WORDS = {
    "what",
    "when",
    "where",
    "which",
    "how",
    "who",
    "whose",
    "whom",
    "why",
}


def quiz_to_json(items: Sequence[QuizQuestion]) -> list[dict]:
    return [item.to_dict() for item in items]


def load_clapnq_sample(split: str = "train", max_items: int = 6, category: str | None = None, randomize: bool = True) -> list[QuizQuestion]:
    """Load sample questions from SQuAD v2 dataset for board-exam style review.
    
    Args:
        split: 'train' or 'validation' split
        max_items: number of items to load
        category: optional keyword to filter questions by topic/category
        randomize: whether to randomize question order (default: True)
        
    Returns:
        List of QuizQuestion objects ready for grading
    """
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("squad_v2", split=split)
        quiz_items: list[QuizQuestion] = []
        seen_questions: set[str] = set()
        candidates: list[QuizQuestion] = []
        
        # Randomize dataset scan so items are sampled from across the split
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Scan through dataset to find matching items
        for idx in indices:
            if len(candidates) >= max_items * 6:  # Over-sample to have room for filtering
                break
            try:
                item = dataset[idx]
                
                # SQuAD v2 schema: question, context, answers (list with text and answer_start)
                question = normalize_whitespace(str(item.get("question", "")))
                context = normalize_whitespace(str(item.get("context", "")))
                title = normalize_whitespace(str(item.get("title", "")))
                
                if not question or not context:
                    continue
                    
                fingerprint = question.lower()
                if fingerprint in seen_questions:
                    continue
                seen_questions.add(fingerprint)
                
                # Filter by category if provided
                if category:
                    category_lower = category.lower()
                    if not (category_lower in question.lower() or 
                            category_lower in context.lower() or 
                            category_lower in title.lower()):
                        continue
                
                # Extract answer from answers list
                answers = item.get("answers", {})
                answer = ""
                answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
                if isinstance(answer_texts, list) and len(answer_texts) > 0:
                    answer = normalize_whitespace(str(answer_texts[0]))
                
                # Fallback: sample from context if no answer
                if not answer:
                    sampled = _sample_answers(context, 1)
                    answer = sampled[0] if sampled else "Cannot determine"
                
                # Truncate answer if too long
                if len(answer) > 200:
                    answer = answer[:197] + "..."
                
                # collect answers so build_choices can use a richer answer pool later
                candidates.append(
                    QuizQuestion(
                        question=question,
                        answer=answer,
                        context=context,
                        choices=tuple(()),
                    )
                )
                    
            except Exception as item_err:
                continue
        
        # If requested, randomize candidate order before selecting final set
        if randomize:
            random.shuffle(candidates)

        # Build an answer pool from collected candidates (other answers provide good distractors)
        answer_pool = [c.answer for c in candidates]

        # Select up to max_items from candidates and build choices using broader pool
        quiz_items = []
        for c in candidates[: max_items]:
            choices = build_choices(c.answer, answer_pool, c.context)
            quiz_items.append(
                QuizQuestion(
                    question=c.question,
                    answer=c.answer,
                    context=c.context,
                    choices=tuple(choices),
                )
            )
        
        return quiz_items
        
    except Exception as exc:
        raise RuntimeError(f"Failed to load SQuAD v2 dataset: {exc}")
