# Kaggle Cells: Fine-tuning PrimeQA for Board Exam Review Questions

Copy each cell below into Kaggle in order. This workflow fine-tunes the same checkpoint used by the app: `PrimeQA/mt5-base-tydi-question-generator`.

## Cell 1 - Install dependencies

```python
!pip -q install transformers datasets accelerate sentencepiece evaluate rouge_score pandas
# Optional, only if your Kaggle environment uses Python 3.10 or lower and you want the PrimeQA toolkit:
# !pip -q install primeqa==0.14.2
```

## Cell 2 - Imports and seed

```python
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

## Cell 3 - Configuration

```python
MODEL_NAME = "PrimeQA/mt5-base-tydi-question-generator"
SPECIAL_SEPARATOR = "<<sep>>"

# Update these paths to match your Kaggle dataset.
TRAIN_PATH = "/kaggle/input/board-exam-data/train.csv"
VALID_PATH = "/kaggle/input/board-exam-data/valid.csv"

MAX_SOURCE_LENGTH = 384
MAX_TARGET_LENGTH = 64
```

## Cell 4 - Load data

```python
from pathlib import Path

train_path = Path(TRAIN_PATH)
valid_path = Path(VALID_PATH)

if train_path.exists() and valid_path.exists():
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
else:
    # Small fallback sample so the notebook still runs end-to-end.
    sample_rows = [
        {
            "context": "The Philippine Constitution defines the structure of government and guarantees fundamental rights.",
            "answer": "Philippine Constitution",
            "question": "What document defines the structure of government and guarantees fundamental rights?",
        },
        {
            "context": "Mitosis produces two genetically identical daughter cells from one parent cell.",
            "answer": "two genetically identical daughter cells",
            "question": "What does mitosis produce from one parent cell?",
        },
        {
            "context": "Insulin lowers blood glucose by helping cells absorb glucose from the bloodstream.",
            "answer": "Insulin",
            "question": "What hormone lowers blood glucose by helping cells absorb glucose?",
        },
    ]
    train_df = pd.DataFrame(sample_rows)
    valid_df = pd.DataFrame(sample_rows[:2])

train_df = train_df.dropna(subset=["context", "answer", "question"]).reset_index(drop=True)
valid_df = valid_df.dropna(subset=["context", "answer", "question"]).reset_index(drop=True)

train_df.head()
```

## Cell 5 - Build the input text

```python
def build_source_text(row):
    return f"{row['answer']} {SPECIAL_SEPARATOR} {row['context']}"

train_df["source_text"] = train_df.apply(build_source_text, axis=1)
valid_df["source_text"] = valid_df.apply(build_source_text, axis=1)

train_df[["source_text", "question"]].head()
```

## Cell 6 - Convert to Hugging Face datasets

```python
train_dataset = Dataset.from_pandas(train_df[["source_text", "question"]])
valid_dataset = Dataset.from_pandas(valid_df[["source_text", "question"]])
raw_datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
raw_datasets
```

## Cell 7 - Load tokenizer and model

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
```

## Cell 8 - Tokenize the dataset

```python
def tokenize_batch(batch):
    model_inputs = tokenizer(
        batch["source_text"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["question"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

encoded_datasets = raw_datasets.map(
    tokenize_batch,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
encoded_datasets
```

## Cell 9 - Data collator and training arguments

```python
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/primeqa-board-exam-qg",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

## Cell 10 - Trainer

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

## Cell 11 - Fine-tune the model

```python
train_result = trainer.train()
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
train_result
```

## Cell 12 - Evaluate

```python
eval_result = trainer.evaluate()
eval_result
```

## Cell 13 - Test inference on a new passage

```python
def generate_question(answer, context, max_new_tokens=64):
    prompt = f"{answer} {SPECIAL_SEPARATOR} {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LENGTH).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_length=max_new_tokens,
        num_beams=4,
        repetition_penalty=2.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

sample_context = "The Supreme Court interprets the Constitution and reviews the constitutionality of laws."
sample_answer = "Supreme Court"
print(generate_question(sample_answer, sample_context))
```

## Cell 14 - Save a lightweight checkpoint note

```python
print(f"Saved files to: {training_args.output_dir}")
print("You can upload the folder to Hugging Face or reuse it in the Streamlit app by swapping MODEL_NAME to your checkpoint path.")
```
