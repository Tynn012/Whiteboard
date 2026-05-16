# Board Exam Quiz Forge — Version 0.2.0

A lightweight reviewer app that turns your study notes into a quiz. This workspace uses the SQuAD v2 dataset for sample questions and a streamlined Streamlit UI focused on a minimal white/black theme.

## What's changed in this version

- Uses SQuAD v2 as the primary sample dataset for quicker, reliable question loading.
- Improved question sampling and randomization to avoid repeated top-of-split items.
- Smarter distractor generation with a `Distractor difficulty` slider in the UI.
- Minimal, high-contrast white/black theme for Streamlit UI and radio-based choice inputs.

## What it does

- Load sample questions from the SQuAD v2 split.
- Let you filter by a keyword-based category and randomize sampling.
- Adjust distractor difficulty to make multiple-choice options easier or harder.
- Download the generated quiz as JSON.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Dependencies

See `requirements.txt` for the minimal runtime dependencies required by the Streamlit app (`streamlit`, `datasets`, `pandas`).

This app runs in dataset-only mode: it loads sample questions from SQuAD v2 and generates multiple-choice options using built-in distractor logic. LLM generation support has been removed to keep the app lightweight and deployable.

## Fine-tuning notes & notebook

If you want to fine-tune a question-generation checkpoint on your own dataset (Kaggle or Colab), follow the step-by-step guide in:

`notebooks/kaggle_SQuAD v2_board_exam_finetune_cells.md`

This file contains a cell-by-cell Kaggle/Colab workflow for preparing data, training a Hugging Face text-to-text model, and exporting a lightweight checkpoint for inference.

## Suggested dataset format for fine-tuning

- `context`: the study passage or reviewer note
- `answer`: the span that should be asked about
- `question`: the target board-exam style question

These fields match an answer-conditioned question generation setup and can be converted into the HF `datasets` format for training.

## How this app was built (step-by-step)

This section documents the exact steps and prompts used to create this application so your friend can reproduce it and learn from the process.

1. Project bootstrap

   - Create a new GitHub repo (or local folder) and initialize git:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Board Exam Quiz Forge"
   git branch -M main
   # add remote and push (after creating repo on GitHub)
   git remote add origin https://github.com/<your-username>/<repo>.git
   git push -u origin main
   ```

   - (Optional) Create the repo via GitHub CLI:

   ```bash
   gh repo create <your-username>/<repo> --public --source=. --remote=origin --push
   ```

2. Local environment

   - Create a virtual environment and install runtime deps:

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Key implementation steps with exact assistant prompts

   Use the following prompts with an AI assistant (or follow manually) to implement the main features. Paste the prompts verbatim when asking an assistant.

   - Change dataset and model (initial request):

   ```text
   Update the project to use SQuAD v2 as the primary dataset (replace any PrimeQA CLAPNQ loading code) and set the default model name to meta-llama/Meta-Llama-3-8B-Instruct. Adjust parsing logic to read `question`, `context`, and `answers.text` from SQuAD v2. Update error messages and README accordingly.
   ```

   - Add randomized sampling and category filter:

   ```text
   Improve the sample loader to randomly sample indices across the dataset split, add an optional `category` keyword filter (checks question/context/title), and a `randomize` flag. Over-sample candidates to build a broader answer pool before selecting final questions.
   ```

   - Improve distractors (scoring logic):

   ```text
   Replace the simple distractor picker with a scored ranking approach: extract entities and number-like tokens from context, sample candidate answers from other items, compute a score based on length similarity and token overlap (and a numeric type match bonus), sort candidates by score, and pick top-ranked distractors. Add a `difficulty` parameter that increases/decreases randomness when selecting distractors.
   ```

   - UI improvements (Streamlit):

   ```text
   Change the Streamlit UI to a minimal white/black theme: simplify CSS, use high contrast buttons, replace selectboxes with radio buttons for choices, add a sidebar slider called 'Distractor difficulty' (0.0–1.0), and add a category text input and randomize checkbox. Ensure quiz JSON download remains available.
   ```

   - Cleanup and requirements:

   ```text
   Remove heavy ML dependencies that aren't required for simple SQuAD sampling (transformers, torch) from `requirements.txt`, and update README to reflect new setup. Remove the PrimeQA-specific classes and code paths not used anymore.
   ```

4. Files and structure to create

   - `app.py`: Streamlit UI and main runner.
   - `quiz_engine.py`: data loader, distractor logic, and helpers.
   - `requirements.txt`: keep minimal runtime deps (`streamlit`, `datasets`, `pandas`).
   - `notebooks/kaggle_primeqa_board_exam_finetune_cells.md`: fine-tuning guide.

5. GitHub setup and CI (recommended)

   - Push your repository to GitHub. Use the repository's `Settings` → `Secrets` if adding CI.
   - (Optional) Add a GitHub Action to run flake8/pytest on pushes. Example `.github/workflows/ci.yml`:

   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.10'
         - name: Install deps
           run: python -m pip install -r requirements.txt
         - name: Run lint
           run: |
             python -m pip install flake8
             flake8 --max-line-length=120
   ```

6. Deploying to Streamlit Community Cloud

   - Sign in at https://share.streamlit.io and connect your GitHub account.
   - Click "New app", select your GitHub repo, branch (e.g., `main`), and the path to `app.py`.
   - Deploy — Streamlit will build the app and provide a public link.

   Example share URL format (replace with your values after deploy):

   `https://share.streamlit.io/<github-username>/<repo>/main/app.py`

7. Reproducing the app locally (quick commands)

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
python -m venv .venv
.venv\Scripts\Activate.ps1   # on Windows PowerShell
python -m pip install -r requirements.txt
streamlit run app.py
```

8. Example prompts to iterate and improve (copy-paste)

   - Tuning distractor difficulty:

   ```text
   Make the distractor scoring prefer candidates that share tokens with the answer and have similar length; add a noise term controlled by a `difficulty` float (0.0 easy → high noise, 1.0 hard → deterministic). Provide code for `_score_candidate` and its integration into `build_choices` in `quiz_engine.py`.
   ```

   - Add a distractor preview mode to Streamlit:

   ```text
   Add an optional debug panel in `app.py` that shows for each question the scored candidate distractors and their scores (for tuning). This should be behind a checkbox in the sidebar: 'Show distractor preview'.
   ```

9. Teaching notes

   - Use this repo to demonstrate: dataset formats, prompt engineering for assistants, model fine-tuning basics, and deployment to Streamlit.
   - Walk through the `notebooks/kaggle_primeqa_board_exam_finetune_cells.md` with your friend to show real fine-tuning steps.

If you want, I will finish this `How this app was built` section by adding small code snippets for the exact functions we edited (for example, the final `build_choices` implementation) and I can open a pull request with those snippets included as separate helper files.

UI notes

- The app includes a sidebar checkbox: **Show expanded choices under each question**. Enabling this renders A/B/C/D style choice blocks under each question for improved readability.
- After grading, the correct answer is shown with a high-contrast badge so it's immediately visible.
- If you ever see choice text hard to read (browser selection or theme interaction), toggle the expanded choices option — those blocks force readable colors.

## Turnover checklist (readying for your friend)

- **Code:** `app.py`, `quiz_engine.py`, `notebooks/*` are present and functional for dataset-only mode (SQuAD v2).
- **Secrets:** The app reads `HF_TOKEN` from `Streamlit secrets` or the `HF_TOKEN` environment variable. Do not paste tokens into the app UI for long-term storage — use environment variables or `secrets.toml`.
- **Run:** Install dependencies and run locally:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

- **Quick HF token setup (local):** create a file named `.env` with `HF_TOKEN=...` or set `HF_TOKEN` in your environment.
- **Streamlit deploy:** add `HF_TOKEN` to your app's Secrets on share.streamlit.io or use GitHub Actions/Secrets in your repository.

## Added packaging & CI

- `.gitignore` added to ignore local environments and secrets.
- `LICENSE` (MIT) added for handoff.
- `.env.example` included to show expected env variables.
- A lightweight GitHub Action workflow is included at `.github/workflows/ci.yml` that installs minimal deps and runs a smoke test.

