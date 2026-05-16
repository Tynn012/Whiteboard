from __future__ import annotations

import json

import streamlit as st

from quiz_engine import load_mmlu_sample, quiz_to_json

# Known CAIS MMLU configs (keeps the UI friendly and avoids an extra Hub query)
MMLU_CONFIGS = [
    "abstract_algebra",
    "all",
    "anatomy",
    "astronomy",
    "auxiliary_train",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


st.set_page_config(
    page_title="Board Exam Quiz Forge",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
"""
<style>
:root {
    --bg: #f6f7fb;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e5e7eb;
    --primary: #111827;
    --accent: #4f46e5;
    --correct: #16a34a;
    --wrong: #dc2626;
}

/* GLOBAL */
html, body, [class*='css'] {
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
    background: var(--bg);
    color: var(--text);
}

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
}

/* LAYOUT */
.block-container {
    padding: 1.5rem 1rem 2rem;
    max-width: 1250px;
}

/* HERO */
.hero {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.5rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
}

.hero h1 {
    font-size: 2rem;
    margin: 0;
    font-weight: 850;
    letter-spacing: -0.03em;
}

.hero p {
    margin-top: 0.4rem;
    color: var(--muted);
    font-size: 0.98rem;
}

/* BADGES */
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.badge {
    font-size: 0.78rem;
    background: #eef2ff;
    border: 1px solid #dbeafe;
    color: #1e293b;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-weight: 700;
}

/* PANELS */
.panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}

.panel h3 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 800;
}

/* QUESTION CARD */
.question-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 25px rgba(2, 6, 23, 0.05);
}

.question-topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.question-badge {
    background: #0f172a;
    color: white;
    font-size: 0.75rem;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    font-weight: 700;
}

.question-meta {
    font-size: 1rem;
    font-weight: 650;
    color: var(--text);
    line-height: 1.55;
}

.question-subtext {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* CHOICES */
.choice-card {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 12px;
    margin-top: 10px;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: #fff;
    transition: all 0.15s ease-in-out;
}

.choice-card:hover {
    transform: translateY(-1px);
    border-color: #c7d2fe;
    box-shadow: 0 8px 18px rgba(79, 70, 229, 0.08);
}

.choice-label {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: #111827;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.choice-text {
    font-size: 0.97rem;
    line-height: 1.5;
    color: var(--text);
    font-weight: 600;
}

/* RESULT GRID */
.result-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin: 1rem 0;
}

.result-tile {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}

.result-tile .label {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 700;
    text-transform: uppercase;
}

.result-tile .value {
    font-size: 1.5rem;
    font-weight: 850;
}

/* BUTTONS */
.stButton>button {
    background: var(--primary);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-weight: 700;
    border: none;
    box-shadow: 0 10px 20px rgba(17, 24, 39, 0.2);
}

.stButton>button:hover {
    background: #0b1220;
}

/* RADIO */
.stRadio > div {
    gap: 0.5rem;
}

.stRadio label {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.6rem 0.75rem;
    font-weight: 600;
}

/* MOBILE */
@media (max-width: 900px) {
    .result-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<style>
.question-meta, .question-card, .stRadio, .stMarkdown {
    color: #0f172a !important;
}

/* Improve readability of selected radio */
.stRadio input[type="radio"] {
    accent-color: #111827;
    transform: scale(1.1);
}

/* Remove ugly default spacing */
.stMarkdown p {
    margin-bottom: 0.4rem;
}
</style>
""",
unsafe_allow_html=True
)

st.markdown(
    """
<div class="hero">
    <h1>Board Exam Quiz Forge</h1>
    <p>
        Load multiple-choice questions from the CAIS MMLU dataset (pre-built 4-choice MCQs) for review and practice.
    </p>
    <div class="badge-row">
        <span class="badge">CAIS MMLU Dataset</span>
        <span class="badge">Validation split</span>
        <span class="badge">4-choice questions</span>
        <span class="badge">High-contrast UI</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if "quiz_items" not in st.session_state:
    st.session_state.quiz_items = []
if "graded" not in st.session_state:
    st.session_state.graded = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "total" not in st.session_state:
    st.session_state.total = 0

with st.sidebar:
    st.markdown("## Quiz settings")
    num_questions = st.slider("Number of questions", min_value=1, max_value=50, value=6)
    # Removed split and category filter to simplify the UI; we default to validation split
    mmlu_config_select = st.selectbox("MMLU subset", options=MMLU_CONFIGS + ["Other (type...)"], index=MMLU_CONFIGS.index("all"))
    if mmlu_config_select == "Other (type...)":
        mmlu_config = st.text_input("MMLU config (custom)", value="all", help="Type a config name from the dataset list")
    else:
        mmlu_config = mmlu_config_select
    randomize_questions = st.checkbox("Randomize question order", value=True)
    show_expanded_choices = st.checkbox("Show expanded choices under each question", value=False)

col_main, col_info = st.columns([1.12, 0.88], gap="large")

with col_main:
    st.markdown('<div class="panel"><h3>Load Quiz</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Pick a subject subset, then load a quiz. The app uses the validation split automatically so the interface stays simple.</div>', unsafe_allow_html=True)
    
    load_col_1, load_col_2 = st.columns([0.55, 0.45])
    with load_col_1:
        load_mmlu_clicked = st.button("Load MMLU questions (4-choice)", use_container_width=True, type="primary")
    with load_col_2:
        clear_clicked = st.button("Clear quiz", use_container_width=True)

    if clear_clicked:
        st.session_state.quiz_items = []
        st.session_state.graded = False
        st.session_state.score = 0
        st.session_state.total = 0
        for key in list(st.session_state.keys()):
            if key.startswith("choice_"):
                del st.session_state[key]
        st.rerun()

    if load_mmlu_clicked:
        with st.spinner("Loading multiple-choice questions from CAIS MMLU dataset..."):
            try:
                # Fixed: always load validation split for simplicity
                quiz_items = load_mmlu_sample(
                    split="validation",
                    max_items=num_questions,
                    randomize=randomize_questions,
                    config=mmlu_config,
                )
                st.session_state.quiz_items = quiz_items
                st.session_state.graded = False
                st.session_state.score = 0
                st.session_state.total = len(quiz_items)
                for key in list(st.session_state.keys()):
                    if key.startswith("choice_"):
                        del st.session_state[key]
                if not quiz_items:
                    st.warning("Could not load MMLU quiz items. The dataset config may be incorrect or the split may be unavailable.")
                else:
                    st.success(f"✅ Loaded {len(quiz_items)} questions from CAIS MMLU dataset (config={mmlu_config}).")
            except Exception as exc:
                st.error(f"❌ MMLU dataset loading failed: {exc}")

    if st.session_state.quiz_items:
        export_payload = json.dumps(quiz_to_json(st.session_state.quiz_items), indent=2, ensure_ascii=False)
        st.download_button(
            "Download quiz JSON",
            data=export_payload,
            file_name="board_exam_quiz.json",
            mime="application/json",
            use_container_width=True,
        )

with col_info:
    st.markdown('<div class="panel"><h3>How it works</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="small-note">
        1. <strong>Choose a subject:</strong> Select an `MMLU config` such as `all` or `machine_learning`.
        2. <strong>Load quiz:</strong> Click "Load MMLU questions (4-choice)" to fetch pre-built multiple-choice items.
        3. <strong>Answer clearly:</strong> Each choice is displayed as a bordered card with a visible A/B/C/D label.
        4. <strong>Review results:</strong> The grading view highlights the correct answer and your selection.
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.quiz_items:
    st.markdown('<div class="panel"><h3>Your quiz</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Each question is shown in a dedicated card. Use the radio button row below each question to select one answer.</div>', unsafe_allow_html=True)

    with st.form("quiz_form"):
        for index, item in enumerate(st.session_state.quiz_items):
            labels = ["A", "B", "C", "D", "E"]
            question_tag = f"Question {index + 1}"

            st.markdown(
                f"""
                <div class="question-card">
                    <div class="question-topline">
                        <span class="question-badge">{question_tag}</span>
                        <span class="muted-pill">MMLU • 4 choices</span>
                    </div>
                    <div class="question-meta">{item.question}</div>
                    <div class="question-subtext">Read the prompt carefully, then choose the best answer below.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            display_options = [f"{labels[i]}. {ch}" for i, ch in enumerate(item.choices)]
            st.radio(
                "Select an answer",
                options=["Select an answer"] + display_options,
                key=f"choice_{index}",
                label_visibility="visible",
                horizontal=False,
                index=0,
            )

            if show_expanded_choices:
                choices_html = ""
                for i, ch in enumerate(item.choices):
                    prefix = labels[i] if i < len(labels) else str(i + 1)
                    choices_html += (
                        f"<div class='choice-card'><div class='choice-label'>{prefix}</div>"
                        f"<div class='choice-text'>{ch}</div></div>"
                    )
                st.markdown(choices_html, unsafe_allow_html=True)

        submitted = st.form_submit_button("Grade my quiz", use_container_width=True)

        if submitted:
            correct = 0
            answered = 0
            results = []
            for index, item in enumerate(st.session_state.quiz_items):
                selected = st.session_state.get(f"choice_{index}", "Select an answer")
                is_answered = selected != "Select an answer"
                if is_answered:
                    answered += 1
                # selected is formatted like 'A. choice' — extract the choice text for comparison
                if is_answered and ". " in selected:
                    selected_text = selected.split(". ", 1)[1]
                else:
                    selected_text = selected if is_answered else ""
                is_correct = selected_text == item.answer
                if is_correct:
                    correct += 1
                results.append((item, selected_text, is_correct, is_answered))

            st.session_state.graded = True
            st.session_state.score = correct
            st.session_state.total = len(st.session_state.quiz_items)

            st.markdown(
                f"""
                <div class="result-grid">
                    <div class="result-tile"><div class="label">Correct</div><div class="value">{correct}</div></div>
                    <div class="result-tile"><div class="label">Answered</div><div class="value">{answered}</div></div>
                    <div class="result-tile"><div class="label">Score</div><div class="value">{correct}/{len(st.session_state.quiz_items)}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for index, (item, selected, is_correct, is_answered) in enumerate(results):
                state_label = "✓ Correct" if is_correct else "✗ Incorrect"
                state_class = "correct" if is_correct else "incorrect"
                # Build choices display with highlight for correct answer
                choice_rows = ""
                labels = ["A", "B", "C", "D", "E"]
                for i, ch in enumerate(item.choices):
                    prefix = labels[i] if i < len(labels) else str(i + 1)
                    # highlight the correct answer and indicate the user's selection
                    if ch == item.answer:
                        # correct answer
                        choice_rows += f"<div class='choice-card' style='background:#0f172a;color:#fff;border-color:#0f172a;'><div class='choice-label' style='background:#fff;color:#0f172a;'>{prefix}</div><div class='choice-text' style='color:#fff;'>{ch}</div></div>"
                    elif ch == selected:
                        # user's (incorrect) selection
                        choice_rows += f"<div class='choice-card' style='border:2px solid #b42318;background:#fff7f7;'><div class='choice-label' style='background:#b42318;'>?</div><div class='choice-text'>{ch}</div></div>"
                    else:
                        choice_rows += f"<div class='choice-card'><div class='choice-label' style='background:#64748b;'>{prefix}</div><div class='choice-text'>{ch}</div></div>"

                st.markdown(
                    f"""
                    <div class="question-card">
                        <div class="question-topline">
                            <span class="question-badge">Question {index + 1}</span>
                            <span class="{state_class}">{state_label}</span>
                        </div>
                        <div class="question-meta">{item.question}</div>
                        <div style="margin-top:8px;">{choice_rows}</div>
                        <div class="answer-state">Your answer: <strong>{selected if is_answered else 'Not answered'}</strong></div>
                        <div class="question-subtext" style="margin-top:0.45rem;">Context: {item.context}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    st.info("Click 'Load MMLU questions (4-choice)' to start a quiz.")
