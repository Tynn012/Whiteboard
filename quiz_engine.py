from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SPECIAL_SEPARATOR = "<<sep>>"


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


class PrimeQABackend:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._backend = self._load_backend(model_name)

    def _load_backend(self, model_name: str):
        try:
            from primeqa.qg.models.qg_model import QGModel

            return QGModel(model_name, modality="passage")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            return {
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
            }

    def generate(self, passages: Sequence[str], questions_per_passage: int = 2, num_beams: int = 4) -> list[dict]:
        if not passages:
            return []

        if hasattr(self._backend, "generate_questions"):
            ids = [f"passage-{index + 1}" for index in range(len(passages))]
            raw_items = self._backend.generate_questions(
                list(passages),
                num_questions_per_instance=questions_per_passage,
                num_beams=num_beams,
                id_list=ids,
            )
            return [_normalize_primeqa_item(item, passages) for item in raw_items]

        return self._generate_with_transformers(passages, questions_per_passage, num_beams)

    def _generate_with_transformers(self, passages: Sequence[str], questions_per_passage: int, num_beams: int) -> list[dict]:
        tokenizer = self._backend["tokenizer"]
        model = self._backend["model"]
        device = self._backend["device"]

        generated_items: list[dict] = []
        for passage_index, passage in enumerate(passages):
            answers = _sample_answers(passage, max(questions_per_passage * 3, 4))
            if not answers:
                continue

            chosen_answers = answers[:questions_per_passage]
            for answer in chosen_answers:
                prompt = f"{answer} {SPECIAL_SEPARATOR} {passage}"
                tokenized = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                generated_ids = model.generate(
                    **tokenized,
                    max_length=64,
                    num_beams=num_beams,
                    repetition_penalty=2.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                question = tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                if question:
                    generated_items.append(
                        {
                            "context_id": f"passage-{passage_index + 1}",
                            "context": passage,
                            "question": question,
                            "answer": answer,
                        }
                    )

        return generated_items


@lru_cache(maxsize=1)
def load_backend(model_name: str = DEFAULT_MODEL_NAME) -> PrimeQABackend:
    return PrimeQABackend(model_name)


def _normalize_primeqa_item(item: dict, passages: Sequence[str]) -> dict:
    question = normalize_whitespace(str(item.get("question", "")))
    answer = normalize_whitespace(str(item.get("answer", "")))
    context = normalize_whitespace(str(item.get("context", "")))

    if not context:
        context_id = str(item.get("context_id", ""))
        match = re.search(r"(\d+)$", context_id)
        if match:
            index = max(int(match.group(1)) - 1, 0)
            if index < len(passages):
                context = normalize_whitespace(passages[index])

    if not context and passages:
        context = normalize_whitespace(passages[0])

    return {
        "context_id": str(item.get("context_id", "")),
        "context": context,
        "question": question,
        "answer": answer,
    }


def split_into_passages(text: str, max_chars: int = 1000) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = normalize_whitespace(paragraph)
        if not paragraph:
            continue
        candidate = f"{current} {paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def generate_quiz(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    questions_per_chunk: int = 2,
    max_chunks: int = 4,
    num_beams: int = 4,
) -> list[QuizQuestion]:
    passages = split_into_passages(text)[:max_chunks]
    if not passages:
        return []

    backend = load_backend(model_name)
    raw_items = backend.generate(passages, questions_per_passage=questions_per_chunk, num_beams=num_beams)

    normalized_items: list[QuizQuestion] = []
    seen_questions: set[str] = set()
    for item in raw_items:
        question = normalize_whitespace(str(item.get("question", "")))
        answer = normalize_whitespace(str(item.get("answer", "")))
        context = normalize_whitespace(str(item.get("context", "")))
        if not question or not answer:
            continue
        fingerprint = question.lower()
        if fingerprint in seen_questions:
            continue
        seen_questions.add(fingerprint)
        normalized_items.append(
            QuizQuestion(
                question=question,
                answer=answer,
                context=context,
                choices=tuple(),
            )
        )

    answer_pool = [item.answer for item in normalized_items]
    final_items: list[QuizQuestion] = []
    for item in normalized_items:
        choices = build_choices(item.answer, answer_pool, item.context)
        final_items.append(
            QuizQuestion(
                question=item.question,
                answer=item.answer,
                context=item.context,
                choices=tuple(choices),
            )
        )

    return final_items


def build_choices(answer: str, answer_pool: Sequence[str], context: str, num_choices: int = 4) -> list[str]:
    """Build multiple choice options with challenging distractors.
    
    Prioritizes entity-based distractors from context that are more plausible wrong answers.
    """
    distractors: list[str] = []
    answer_lower = answer.lower()
    
    # Extract entities/phrases from context that could be plausible distractors
    context_entities = _extract_context_entities(context)
    
    # Prioritize context-based entities that aren't the answer
    for entity in context_entities:
        entity_normalized = normalize_whitespace(entity)
        if not entity_normalized or entity_normalized.lower() == answer_lower:
            continue
        if entity_normalized not in distractors:
            distractors.append(entity_normalized)
        if len(distractors) >= num_choices - 1:
            break
    
    # Fall back to sampling entities from context
    if len(distractors) < num_choices - 1:
        sampled = _sample_answers(context, 15)
        for candidate in sampled:
            candidate = normalize_whitespace(candidate)
            if not candidate or candidate.lower() == answer_lower:
                continue
            if candidate not in distractors:
                distractors.append(candidate)
            if len(distractors) >= num_choices - 1:
                break
    
    # Fill remaining slots with answer pool (if available from other questions)
    if len(distractors) < num_choices - 1:
        for candidate in answer_pool:
            candidate = normalize_whitespace(str(candidate))
            if not candidate or candidate.lower() == answer_lower:
                continue
            if candidate not in distractors:
                distractors.append(candidate)
            if len(distractors) >= num_choices - 1:
                break
    
    # Last resort: context-specific fallbacks
    while len(distractors) < num_choices - 1:
        filler = random.choice(_get_context_fallbacks(context))
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
        
        # Scan through dataset to find matching items
        for idx in range(len(dataset)):
            if len(candidates) >= max_items * 3:  # Over-sample to have room for filtering
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
                
                choices = build_choices(answer, [answer], context)
                
                candidates.append(
                    QuizQuestion(
                        question=question,
                        answer=answer,
                        context=context,
                        choices=tuple(choices),
                    )
                )
                    
            except Exception as item_err:
                continue
        
        # Select up to max_items from candidates
        quiz_items = candidates[:max_items]
        
        # Randomize order if requested
        if randomize and len(quiz_items) > 1:
            random.shuffle(quiz_items)
        
        return quiz_items
        
    except Exception as exc:
        raise RuntimeError(f"Failed to load SQuAD v2 dataset: {exc}")
