# Kaggle / Colab: Fine-tune question-generation checkpoint

This notebook-style guide contains step-by-step cells to prepare data and fine-tune a question-generation model (answer-conditioned) on Kaggle or Colab.

Notes

- This guide intentionally uses Hugging Face Transformers and the `datasets` library.
- For GPU-backed runs prefer `p3`/`T4` kernels on Kaggle or a GPU runtime on Colab.

---

## Cell 1 — Install dependencies

```bash
# Run in a Kaggle/Colab cell
apt-get update -y && apt-get install -y git-lfs
python -m pip install --upgrade pip
python -m pip install transformers datasets accelerate evaluate sentencepiece
```

---

## Cell 2 — Prepare dataset (CSV -> Hugging Face Dataset)

- Expect a CSV/TSV with columns: `context`, `answer`, `question`.

```python
# Python cell: load CSV and convert to HF dataset
from datasets import load_dataset, Dataset
import pandas as pd

# Upload your CSV to the environment, e.g. '/kaggle/working/data.csv'
df = pd.read_csv('/kaggle/working/data.csv')
# Optional: basic cleaning
df = df.dropna(subset=['context', 'answer', 'question']).reset_index(drop=True)

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)
# Optional: train/validation split
dataset = dataset.train_test_split(test_size=0.05)
print(dataset)
```

---

## Cell 3 — Tokenize and prepare inputs

Use a seq2seq tokenizer compatible with the chosen model (e.g., `t5-small`, `mt5`, or your preferred text2text model).

```python
from transformers import AutoTokenizer

model_name = 't5-small'  # choose a base to fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_input_length = 512
max_target_length = 64

def preprocess(example):
    # input format: "<answer> <<sep>> <context>" (answer-conditioned QG)
    input_text = example['answer'] + ' <<sep>> ' + example['context']
    model_inputs = tokenizer(input_text, truncation=True, max_length=max_input_length)
    labels = tokenizer(example['question'], truncation=True, max_length=max_target_length)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized = dataset.map(preprocess, batched=False)
```

---

## Cell 4 — Training (Hugging Face Trainer) [Note: Add ka ng checkpoint dito men para in case na mahaba yung training may checkpoints na mnababalikan mo in case na magkaerrors (Prompt mo nalang to) - JL]

```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir='./qg_finetuned',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    num_train_epochs=3,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    tokenizer=tokenizer,
)

trainer.train()
```

---

## Cell 5 — Save and export

```python
trainer.save_model('./qg_finetuned')
# Optionally push to the HF hub
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(repo_id='your-username/qg-finetuned', folder_path='./qg_finetuned')
```

---

## Cell 6 — Minimal inference example

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('./qg_finetuned')
tokenizer = AutoTokenizer.from_pretrained(model_name)

context = "<your passage here>"
answer = "<answer span>"
prompt = answer + ' <<sep>> ' + context
inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
outputs = model.generate(**inputs, max_length=64, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Tips

- Monitor evaluation metrics and sample generations to ensure output quality.
- If using a large model, reduce batch sizes or use gradient accumulation.
- For production/edge inference, export to an optimized format (ONNX, TorchScript) if needed.
