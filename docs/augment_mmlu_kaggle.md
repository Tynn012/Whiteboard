# Augment CAIS MMLU in Kaggle — Cell-by-cell Notebook

This document is a step-by-step, copy-paste-ready Kaggle notebook recipe to:

- Load CAIS `mmlu` (validation split)
- Load an external dataset (CSV / JSON / JSONL) from `/kaggle/input/`
- Normalize both sources to a common schema: `question`, `choices` (list), `answer` (text), `context`
- Filter for 4-choice items (optional)
- Save a combined `mmlu_augmented.csv` (or JSONL) to `/kaggle/working/`

Each step below is presented as a single Kaggle cell you can paste into the notebook. Follow the cells in order.

---

## Cell 1 — Install dependencies

Paste into the first Kaggle cell and run once.

```bash
pip install -q datasets pandas
```

## Cell 2 — Imports + small helpers

```python
from datasets import load_dataset
import pandas as pd
import json
import ast

def parse_choices_field(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        # try JSON-like literal
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except Exception:
            pass
        # common delimiters
        for d in ("||", "|", ";", "\t", " / "):
            if d in s:
                return [p.strip() for p in s.split(d) if p.strip()]
        return [s] if s else []
    return []

def resolve_answer_text(ans, choices):
    if ans is None:
        return ""
    if isinstance(ans, int) and 0 <= ans < len(choices):
        return choices[ans]
    if isinstance(ans, str) and len(ans.strip()) == 1 and ans.strip().isalpha():
        idx = ord(ans.strip().upper()) - 65
        if 0 <= idx < len(choices):
            return choices[idx]
        return ans
    # try matching by text
    for c in choices:
        if str(c).strip().lower() == str(ans).strip().lower():
            return c
    return str(ans)
```

## Cell 3 — Load CAIS MMLU (validation split)

```python
# Load CAIS MMLU (validation) and normalize
ds = load_dataset("cais/mmlu", name="all", split="validation")
mmlu_rows = [dict(x) for x in ds]

def normalize_mmlu_row(r):
    q = r.get("question") or r.get("prompt") or ""
    ctx = r.get("context", "") or ""
    # try common keys for choices
    choices = None
    for k in ("choices", "options", "answer_choices", "candidates", "options_list"):
        if k in r:
            choices = r[k]
            break
    if isinstance(choices, dict):
        choices = [v for _, v in sorted(choices.items())]
    choices = parse_choices_field(choices)

    # find answer
    ans = None
    for k in ("answer", "correct_answer", "label", "answer_key", "gold", "correct_idx", "answer_idx"):
        if k in r:
            ans = r[k]; break
    ans_text = resolve_answer_text(ans, choices) if choices else ""

    return {"question": q, "choices": choices or [], "answer": ans_text, "context": ctx}

mmlu_norm = [normalize_mmlu_row(r) for r in mmlu_rows]
mmlu_df = pd.DataFrame(mmlu_norm)
print(f"Loaded MMLU rows: {len(mmlu_df)}; sample:\n", mmlu_df.head(2).to_dict(orient='records'))
```

## Cell 4 — Load your external dataset (CSV / JSON / JSONL)

Replace `/kaggle/input/your-dataset/your.csv` with the path shown in the Kaggle notebook data selector.

```python
# Example CSV path: '/kaggle/input/your-dataset/your.csv'
external_path = "/kaggle/input/your-dataset/your.csv"

ext_rows = []
if external_path.lower().endswith('.csv'):
    df = pd.read_csv(external_path)
    ext_rows = df.to_dict(orient='records')
else:
    # try JSON array or JSONL
    txt = open(external_path, 'r', encoding='utf-8').read()
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            ext_rows = parsed
        elif isinstance(parsed, dict):
            ext_rows = [parsed]
    except Exception:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ext_rows.append(json.loads(line))
            except Exception:
                continue

print(f"External rows loaded: {len(ext_rows)}; sample:\n", ext_rows[:2])
```

## Cell 5 — Normalize external rows to the same schema

```python
def normalize_external_row(r):
    q = r.get('question') or r.get('prompt') or r.get('text') or ''
    ctx = r.get('context', '') or ''
    # look for choices
    c = None
    for k in ('choices', 'options', 'answer_choices', 'candidates', 'options_list'):
        if k in r:
            c = r[k]; break
    choices = parse_choices_field(c)
    # fallback to per-column choices
    if not choices:
        chs = []
        for col in ('choice_a','choice_b','choice_c','choice_d','A','B','C','D'):
            if col in r and r[col]:
                chs.append(r[col])
        choices = chs

    ans = r.get('answer') or r.get('answer_text') or r.get('answer_key') or r.get('label') or ''
    ans_text = resolve_answer_text(ans, choices) if choices else ''
    return {'question': q, 'choices': choices or [], 'answer': ans_text, 'context': ctx}

ext_norm = [normalize_external_row(r) for r in ext_rows]
ext_df = pd.DataFrame(ext_norm)
print(f"Normalized external rows: {len(ext_df)}; sample:\n", ext_df.head(2).to_dict(orient='records'))
```

## Cell 6 — Filter for quality and 4-choice rows (optional)

```python
# Keep rows that have at least 2 choices (or change to ==4 for strict 4-choice only)
mmlu_df = mmlu_df[mmlu_df['choices'].apply(lambda x: isinstance(x, list) and len(x) >= 2)]
ext_df = ext_df[ext_df['choices'].apply(lambda x: isinstance(x, list) and len(x) >= 2)]

# Optional: enforce exactly 4 choices
# mmlu_df = mmlu_df[mmlu_df['choices'].apply(lambda x: len(x) == 4)]
# ext_df = ext_df[ext_df['choices'].apply(lambda x: len(x) == 4)]

print(f"MMLU kept: {len(mmlu_df)}; External kept: {len(ext_df)}")
```

## Cell 7 — Concatenate and save (CSV + optional JSONL)

```python
combined_df = pd.concat([mmlu_df, ext_df], ignore_index=True)
out_csv = '/kaggle/working/mmlu_augmented.csv'
combined_df.to_csv(out_csv, index=False)
out_jsonl = '/kaggle/working/mmlu_augmented.jsonl'
combined_df.to_json(out_jsonl, orient='records', lines=True)
print(f"Saved {len(combined_df)} rows to:\n  {out_csv}\n  {out_jsonl}")
```

## Checkpoint — Quick verification (Cell 8)

```python
import os
print('Files in /kaggle/working (tail):')
print([f for f in os.listdir('/kaggle/working') if 'mmlu_augmented' in f])

# Basic sanity checks
print('Row count:', len(combined_df))
display(combined_df.sample(3))

# Verify choices column type for first few rows
for i, row in combined_df.head(5).iterrows():
    print(i, 'choices_len=', len(row['choices']), 'answer=', row['answer'])
```

## Example CSV format hints

- Preferred columns: `question`, `choices` (JSON list or delimited string), `answer`, `context`.
- Example row (CSV cell for `choices`):

  "[\"Option A\", \"Option B\", \"Option C\", \"Option D\"]"

- Or delimited: `Option A||Option B||Option C||Option D`

## Troubleshooting

- If `load_dataset("cais/mmlu")` fails, confirm Kaggle internet access and that `datasets` is installed.
- If rows are dropped, print `ext_rows[:5]` to inspect raw input and adapt parsing.
- If choices are parsed as a single string, update the delimiters tried in `parse_choices_field`.

## Next steps

- Use the saved CSV/JSONL as input for downstream tools or upload it as a Kaggle dataset.
- To import into your Streamlit app: upload the CSV/JSONL to the app via its sidebar `Upload quiz CSV/JSONL` control (no code changes required).

---

## Short README snippet you can paste

Add this to your `README.md` to explain the notebook and the local import:

```markdown
Augment MMLU with your own data:

- Run the cell-by-cell notebook in `docs/augment_mmlu_kaggle.md` on Kaggle.
- The notebook writes `/kaggle/working/mmlu_augmented.csv` and `mmlu_augmented.jsonl`.
- Upload the CSV/JSONL to the app using the sidebar `Upload quiz CSV/JSONL` control and click `Load uploaded quiz file`.
```

Generated by the workspace assistant — updated `docs/augment_mmlu_kaggle.md` with cell-by-cell instructions.
