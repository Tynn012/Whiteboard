from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# Default LLM model for optional generation mode
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


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


def build_choices(answer: str, answer_pool: Sequence[str], context: str, num_choices: int = 4, difficulty: float = 0.7) -> list[str]:
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
        base = len_score * 0.5 + token_score * 0.5 + type_bonus
        # difficulty scales preference for high-base scores; lower difficulty adds randomness
        noise = random.random() * (1.0 - difficulty) * 0.5
        return base * difficulty + noise

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
    # allow more fallback usage when difficulty is low
    max_fallbacks = max(0, int((1.0 - difficulty) * (num_choices - 1)))
    fallbacks_used = 0
    while len(distractors) < num_choices - 1 and fi < len(fallback_pool):
        filler = fallback_pool[fi]
        fi += 1
        if filler.lower() != answer_lower and filler not in distractors:
            # only accept fallback if difficulty allows or if we lack any distractors
            if difficulty < 0.85 or not distractors or fallbacks_used < max_fallbacks:
                distractors.append(filler)
                fallbacks_used += 1

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


class ModelBackend:
    """Light wrapper to load an instruction-tuned causal LLM and generate questions.

    This attempts to load the model with 8-bit (bitsandbytes) if available and falls back
    to a CPU/FP32 load if not.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, hf_token: str | None = None):
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers/torch not available in the environment")
        self.model_name = model_name
        self.hf_token = hf_token
        self.tokenizer = None
        self.model = None
        self._load_backend()

    def _load_backend(self):
        kwargs = {}
        if self.hf_token:
            kwargs['use_auth_token'] = self.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)

        # Prefer 8-bit load if bitsandbytes is available
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                load_in_8bit=True,
                torch_dtype=torch.float16,
                **kwargs,
            )
        except Exception:
            # fallback: normal load (may be large and require GPU/CPU memory)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        self.model.eval()

    def generate(self, passages: Sequence[str], answers: Sequence[str], num_questions_per_passage: int = 1, max_new_tokens: int = 64, temperature: float = 0.0) -> list[dict]:
        if not passages:
            return []

        results: list[dict] = []
        for idx, passage in enumerate(passages):
            answer = answers[idx] if idx < len(answers) else ""
            for _ in range(num_questions_per_passage):
                prompt = (
                    f"Generate a concise, exam-style question whose answer is '{answer}'.\nContext: {passage}\nQuestion:"
                )
                tokenized = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.model.device)
                gen = self.model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,
                    num_return_sequences=1,
                )
                q = self.tokenizer.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # heuristic: take text after 'Question:' if present
                if 'Question:' in q:
                    q = q.split('Question:')[-1].strip()
                results.append({
                    'context_id': f'passage-{idx+1}',
                    'context': passage,
                    'question': q,
                    'answer': answer,
                })
        return results


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
        "Insufficient information",
        "A related option",
        "A plausible alternative",
        "A similar concept",
        "A contextually relevant choice",
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


def load_clapnq_sample(
    split: str = "train",
    max_items: int = 6,
    category: str | None = None,
    randomize: bool = True,
    distractor_difficulty: float = 0.7,
    use_llm: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    hf_token: str | None = None,
    llm_questions_per_passage: int = 1,
) -> list[QuizQuestion]:
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
        
        # If using an LLM backend, generate questions from passages/answers
        if use_llm:
            if not _HAS_TRANSFORMERS:
                raise RuntimeError("transformers/torch are required for LLM generation mode")
            # Prepare inputs
            passages = [c.context for c in candidates]
            answers = [c.answer for c in candidates]
            backend = ModelBackend(model_name=model_name, hf_token=hf_token)
            raw_items = backend.generate(passages[: max_items], answers[: max_items], num_questions_per_passage=llm_questions_per_passage)
            # Build quiz items from generated content
            quiz_items = []
            for itm in raw_items[:max_items]:
                q = normalize_whitespace(str(itm.get('question', '')))
                a = normalize_whitespace(str(itm.get('answer', '')))
                ctx = normalize_whitespace(str(itm.get('context', '')))
                if not q or not a:
                    continue
                choices = build_choices(a, [x.answer for x in candidates], ctx, difficulty=distractor_difficulty)
                quiz_items.append(QuizQuestion(question=q, answer=a, context=ctx, choices=tuple(choices)))
            return quiz_items

        # If requested, randomize candidate order before selecting final set
        if randomize:
            random.shuffle(candidates)

        # Build an answer pool from collected candidates (other answers provide good distractors)
        answer_pool = [c.answer for c in candidates]

        # Select up to max_items from candidates and build choices using broader pool
        quiz_items = []
        for c in candidates[: max_items]:
            choices = build_choices(c.answer, answer_pool, c.context, difficulty=distractor_difficulty)
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
