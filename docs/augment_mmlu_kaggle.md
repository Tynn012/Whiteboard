# Augment CAIS MMLU in Kaggle — Snippet & Checkpoint

This document contains a ready-to-copy Kaggle notebook snippet to augment the CAIS `mmlu` dataset with an external dataset (CSV or other), plus a small checkpoint you can run to verify the result.

---

## Overview

- Purpose: load CAIS MMLU (validation split), load an external CSV dataset, normalize both into a common schema, filter for rows with choices, and save a combined file to `/kaggle/working/mmlu_augmented.csv`.
- Output: CSV with columns `question`, `choices` (list-like string), `answer` (text), `context`.

## Requirements

- Kaggle notebook with internet access (to pip-install `datasets` if not present).

## Install (Kaggle cell)

```bash
pip install -q datasets pandas
```

## Notebook cell — augment and save

```python
# Kaggle cell - augment MMLU with another dataset (copy/paste)
from datasets import load_dataset
import pandas as pd

# 1) Load CAIS MMLU (validation split)
mmlu = load_dataset("cais/mmlu", name="all", split="validation")
mmlu_rows = [dict(x) for x in mmlu]

def normalize_mmlu_row(r):
    question = r.get("question", "") or r.get("prompt", "")
    context = r.get("context", "") if "context" in r else ""
    # common keys for choices/answer
    choices = None
    for k in ("choices", "options", "answer_choices", "candidates"):
        if k in r and isinstance(r[k], (list, tuple)):
            choices = list(r[k]); break
    answer = None
    for k in ("answer", "answer_key", "correct_answer", "label", "label_idx"):
        if k in r:
            answer = r[k]; break

    # map answer -> text
    answer_text = ""
    if isinstance(answer, int) and choices:
        if 0 <= answer < len(choices):
            answer_text = choices[answer]
    elif isinstance(answer, str) and choices:
        a = answer.strip()
        if len(a) == 1 and a.upper().isalpha():
            idx = ord(a.upper()) - 65
            if 0 <= idx < len(choices):
                answer_text = choices[idx]
            else:
                answer_text = answer
        else:
            match = next((c for c in choices if c.strip().lower() == a.lower()), None)
            answer_text = match or answer
    else:
        answer_text = answer or ""

    return {"question": question, "choices": choices or [], "answer": answer_text, "context": context}

mmlu_norm = [normalize_mmlu_row(r) for r in mmlu_rows]
mmlu_df = pd.DataFrame(mmlu_norm)

# 2) Load your external dataset
# Replace the path below with your CSV or use a HF dataset id instead.
# Example CSV path in Kaggle: '/kaggle/input/your-dataset/your.csv'
ext = load_dataset("csv", data_files="/kaggle/input/your-dataset/your.csv")["train"]
ext_rows = [dict(x) for x in ext]

def normalize_external_row(r):
    question = r.get("question") or r.get("prompt") or r.get("text") or ""
    context = r.get("context", "")
    # try to coerce 'choices' that might be stored as string or list
    choices = None
    for k in ("choices", "options", "answer_choices", "candidates"):
        if k in r:
            v = r[k]
            if isinstance(v, list):
                choices = v
            elif isinstance(v, str) and "||" in v:
                choices = [s.strip() for s in v.split("||") if s.strip()]
            elif isinstance(v, str) and ";" in v and len(v.split(";")) <= 10:
                choices = [s.strip() for s in v.split(";") if s.strip()]
    answer = r.get("answer") or r.get("label") or r.get("answer_key") or ""
    # map answer to text
    answer_text = ""
    if isinstance(answer, int) and choices:
        if 0 <= answer < len(choices):
            answer_text = choices[answer]
    elif isinstance(answer, str) and choices:
        a = answer.strip()
        if len(a) == 1 and a.upper().isalpha():
            idx = ord(a.upper()) - 65
            if 0 <= idx < len(choices):
                answer_text = choices[idx]
            else:
                answer_text = answer
        else:
            match = next((c for c in choices if c.strip().lower() == a.lower()), None)
            answer_text = match or answer
    else:
        answer_text = answer or ""

    return {"question": question, "choices": choices or [], "answer": answer_text, "context": context}

ext_norm = [normalize_external_row(r) for r in ext_rows]
ext_df = pd.DataFrame(ext_norm)

# 3) Filter only rows that have >= 2 choices (you can increase to exactly 4 if desired)
mmlu_df = mmlu_df[mmlu_df["choices"].apply(lambda x: isinstance(x, list) and len(x) >= 2)]
ext_df = ext_df[ext_df["choices"].apply(lambda x: isinstance(x, list) and len(x) >= 2)]

# 4) Concatenate and save
combined_df = pd.concat([mmlu_df, ext_df], ignore_index=True)
out_path = "/kaggle/working/mmlu_augmented.csv"
combined_df.to_csv(out_path, index=False)
print(f"Saved {len(combined_df)} rows to {out_path}")
```

## Checkpoint — quick verification

After running the notebook cell above, verify the following:

- The file `/kaggle/working/mmlu_augmented.csv` exists.
- It contains a non-zero number of rows (print `len(combined_df)` to confirm).
- Each row has a `choices` column with a parsed list-like string or consistent delimiter. If you need a proper JSON list in CSV, consider saving as JSONL instead:

```python
combined_df.to_json('/kaggle/working/mmlu_augmented.jsonl', orient='records', lines=True)
```

- Optionally filter for exactly 4-choice rows before saving:

```python
combined_df = combined_df[combined_df['choices'].apply(lambda x: isinstance(x, list) and len(x) == 4)]
```

## Next steps

- Upload the saved file as a Kaggle dataset or copy to local for reuse in your Streamlit app.
- If you want, I can add a small loader to the Streamlit app that reads a local JSONL/CSV file in the same schema and uses it as the quiz source.

---

Generated by the workspace assistant — file created as `docs/augment_mmlu_kaggle.md`.
