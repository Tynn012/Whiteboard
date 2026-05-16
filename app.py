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
html, body, [class*='css'] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    background: #ffffff;
    color: #000000;
}

.stApp {
    background: #ffffff;
    color: #000000;
}

.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }

.hero { padding: 1rem; background: transparent; border-bottom: 1px solid #e6e6e6; margin-bottom: 1rem; }
.hero h1 { font-size: 1.6rem; margin: 0; font-weight: 700; }
.hero p { margin: 4px 0 0 0; color: #222; font-size: 0.95rem; }

.badge-row { display:flex; gap:0.4rem; margin-top:0.6rem; }
.badge { font-size:0.8rem; color:#111; background:#f4f4f4; padding:0.25rem 0.6rem; border-radius:999px; }

.panel { background: transparent; border: 1px solid #efefef; border-radius:8px; padding:0.8rem; margin-bottom:1rem; }
.panel h3 { margin:0; font-size:1.05rem; font-weight:700; }

.question-card { padding:0.85rem; border-radius:8px; background:#fff; border:1px solid #f0f0f0; margin-bottom:0.9rem; }
.question-title { font-weight:700; margin-bottom:0.25rem; }
.question-meta { color:#111; font-size:0.95rem; }
.small-note { color:#444; font-size:0.92rem; }

/* New: card-style choices */
.choice-card { margin:8px 0; padding:10px 12px; border-radius:8px; background:#fbfbfb; border:1px solid #e9e9e9; display:flex; align-items:flex-start; gap:10px; }
.choice-label { min-width:34px; height:34px; border-radius:6px; background:#111; color:#fff; display:inline-flex; align-items:center; justify-content:center; font-weight:700; }
.choice-text { color:#111; flex:1; }
.choice-selected { border:2px solid #007a3d; }

.correct { color: #007a3d; font-weight:700; }
.incorrect { color: #b42318; font-weight:700; }

/* Reduce visual clutter on widgets */
.stButton>button, .stSelectbox>div, .stTextInput>div { border-radius:6px; }
/* Make primary buttons black with white text for high contrast */
.stButton>button { background:#000000; color:#ffffff; border:none; }
.stRadio>div[role="radiogroup"] > label > div { padding: 0.45rem 0.6rem; border-radius:6px; }
</style>
""",
    unsafe_allow_html=True,
)
# Ensure radio and choice text is always readable (override selection/contrast issues)
st.markdown(
    """
    <style>
    /* Force choice text to dark color and prevent text-selection highlight hiding text */
    .stRadio div[role="radiogroup"] > label > div, .question-card, .question-meta {
        color: #000 !important;
        user-select: none;
    }
    /* Make expanded choice blocks explicit and readable */
    .question-card div[style] { color: #111 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <h1>Board Exam Quiz Forge</h1>
    <p>
        Load multiple-choice questions from the CAIS MMLU dataset (pre-built 4-choice MCQs) for review and practice.
    </p>
    <div class="badge-row">
        <span class="badge">SQuAD v2 Dataset</span>
        <span class="badge">CAIS MMLU Dataset</span>
        <span class="badge">Randomized questions</span>
        <span class="badge">Category filtering</span>
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
    quiz_split = st.selectbox("Dataset split", options=["validation", "train"], index=0)
    mmlu_config_select = st.selectbox("MMLU subset", options=MMLU_CONFIGS + ["Other (type...)"] , index=MMLU_CONFIGS.index("all"))
    if mmlu_config_select == "Other (type...)":
        mmlu_config = st.text_input("MMLU config (custom)", value="all", help="Type a config name from the dataset list")
    else:
        mmlu_config = mmlu_config_select
    category_filter = st.text_input("Filter by category (optional)", placeholder="e.g., history, science, geography...")
    randomize_questions = st.checkbox("Randomize question order", value=True)
    show_expanded_choices = st.checkbox("Show expanded choices under each question", value=False)

col_main, col_info = st.columns([1.12, 0.88], gap="large")

with col_main:
    st.markdown('<div class="panel"><h3>Load Quiz</h3></div>', unsafe_allow_html=True)
    
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
                quiz_items = load_mmlu_sample(
                    split=quiz_split,
                    max_items=num_questions,
                    category=category_filter if category_filter else None,
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
        1. <strong>Configure settings:</strong> Adjust number of questions, pick a split, set `MMLU config` (e.g., 'all' or a subject), and optionally filter by category.
        2. <strong>Dataset split:</strong> Use `validation` to sample held-out items suitable for practice/evaluation; `train` contains training items. For most use-cases pick `validation`.
        3. <strong>Load questions:</strong> Click "Load MMLU questions (4-choice)" to fetch pre-built multiple-choice items.
        4. <strong>Answer:</strong> Select from multiple-choice options.
        5. <strong>Grade:</strong> Click "Grade my quiz" to see your score and review with explanations.
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.quiz_items:
    st.markdown('<div class="panel"><h3>Your quiz</h3></div>', unsafe_allow_html=True)

    with st.form("quiz_form"):
        for index, item in enumerate(st.session_state.quiz_items):
            st.markdown(
                f"""
                <div class="question-card">
                    <div class="question-title">Question {index + 1}</div>
                    <div class="question-meta">{item.question}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # show choices as radio buttons with A/B prefixes for clarity
            labels = ["A", "B", "C", "D", "E"]
            display_options = [f"{labels[i]}. {ch}" for i, ch in enumerate(item.choices)]
            st.radio(
                f"Answer for question {index + 1}",
                options=["Select an answer"] + display_options,
                key=f"choice_{index}",
                label_visibility="collapsed",
            )
            if show_expanded_choices:
                # Render choices explicitly (A/B/C/D) so they're always visible
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

        score_col_1, score_col_2, score_col_3 = st.columns(3)
        score_col_1.metric("Correct", correct)
        score_col_2.metric("Answered", answered)
        score_col_3.metric("Score", f"{correct}/{len(st.session_state.quiz_items)}")

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
                    choice_rows += f"<div class='choice-card' style='background:#000;color:#fff;'><div class='choice-label'>{prefix}</div><div class='choice-text'>{ch}</div></div>"
                elif ch == selected:
                    # user's (incorrect) selection
                    choice_rows += f"<div class='choice-card' style='border:2px solid #b42318;background:#fff0f0;'><div class='choice-label' style='background:#b42318;'>?</div><div class='choice-text'>{ch}</div></div>"
                else:
                    choice_rows += f"<div class='choice-card'><div class='choice-label' style='background:#777;'> </div><div class='choice-text'>{ch}</div></div>"

            st.markdown(
                f"""
                <div class="question-card">
                    <div class="question-title">Question {index + 1} - <span class="{state_class}">{state_label}</span></div>
                    <div class="question-meta">{item.question}</div>
                    <div style="margin-top:8px;">{choice_rows}</div>
                    <div style="margin-top:8px;">Your answer: <strong>{selected if is_answered else 'Not answered'}</strong></div>
                    <div style="margin-top:8px; color:#666;">Context: {item.context}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("Click 'Load MMLU questions (4-choice)' to start a quiz.")
