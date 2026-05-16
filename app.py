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
    font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: #f5f7fb;
    color: #111827;
}

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    color: #111827;
}

.block-container { padding-top: 1.25rem; padding-bottom: 1.6rem; max-width: 1320px; }

.hero { padding: 1.2rem 1.25rem; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 18px; margin-bottom: 1rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06); }
.hero h1 { font-size: 1.9rem; margin: 0; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; }
.hero p { margin: 6px 0 0 0; color: #334155; font-size: 0.98rem; line-height: 1.5; }

.badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.85rem; }
.badge { font-size:0.8rem; color:#0f172a; background:#eef2ff; border:1px solid #c7d2fe; padding:0.35rem 0.7rem; border-radius:999px; font-weight:700; }

.panel { background: #ffffff; border: 1px solid #e5e7eb; border-radius:16px; padding:1rem; margin-bottom:1rem; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); }
.panel h3 { margin:0; font-size:1.05rem; font-weight:800; color:#0f172a; }

.question-card { padding:1rem 1.05rem; border-radius:16px; background:#ffffff; border:1px solid #dbe2ea; margin-bottom:1rem; box-shadow: 0 8px 20px rgba(2, 6, 23, 0.04); }
.question-topline { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:0.75rem; }
.question-title { font-weight:800; margin:0; font-size:1rem; color:#0f172a; }
.question-badge { display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.65rem; border-radius:999px; background:#111827; color:#fff; font-size:0.8rem; font-weight:700; letter-spacing:0.02em; }
.question-meta { color:#111827; font-size:1rem; line-height:1.55; font-weight:600; }
.question-subtext { color:#475569; font-size:0.92rem; margin-top:0.35rem; }
.small-note { color:#334155; font-size:0.94rem; line-height:1.6; }

/* New: card-style choices */
.choice-card { margin:10px 0; padding:12px 12px; border-radius:14px; background:#ffffff; border:1px solid #dbe2ea; display:flex; align-items:flex-start; gap:12px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03); }
.choice-label { min-width:42px; height:42px; border-radius:12px; background:#0f172a; color:#fff; display:inline-flex; align-items:center; justify-content:center; font-weight:800; font-size:0.95rem; box-shadow: inset 0 -1px 0 rgba(255,255,255,0.08); }
.choice-text { color:#0f172a; flex:1; font-size:0.98rem; line-height:1.5; font-weight:600; }
.choice-selected { border:2px solid #16a34a; }

.correct { color: #15803d; font-weight:800; }
.incorrect { color: #b42318; font-weight:800; }
.muted-pill { display:inline-flex; align-items:center; padding:0.22rem 0.55rem; border-radius:999px; font-size:0.78rem; font-weight:700; background:#eef2f7; color:#334155; border:1px solid #d7dee8; }
.answer-state { margin-top:0.9rem; font-size:0.92rem; color:#334155; }

/* Reduce visual clutter on widgets */
.stButton>button, .stSelectbox>div, .stTextInput>div { border-radius:10px; }
/* Make primary buttons black with white text for high contrast */
.stButton>button { background:#111827; color:#ffffff; border:1px solid #111827; font-weight:700; box-shadow: 0 6px 16px rgba(17, 24, 39, 0.18); }
.stButton>button:hover { background:#0b1220; border-color:#0b1220; }
.stSelectbox>div, .stTextInput>div { background:#ffffff; border:1px solid #cbd5e1; }
.stRadio>div[role="radiogroup"] { padding-left:0.35rem; }
.stRadio>div[role="radiogroup"] > label { display:flex; align-items:flex-start; gap:0.65rem; margin-bottom:0.65rem; }
.stRadio>div[role="radiogroup"] > label > div { padding: 0.85rem 0.95rem; border-radius:14px; background:#fff; border:1px solid #dbe2ea; font-weight:600; color:#111827 !important; transition: all 0.18s ease; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.03); }
.stRadio>div[role="radiogroup"] > label:has(input:checked) > div { background:#111827; color:#ffffff !important; border-color:#111827; box-shadow: 0 10px 24px rgba(17, 24, 39, 0.18); transform: translateX(2px); }
.stRadio>div[role="radiogroup"] > label:has(input:checked) { margin-left:0.15rem; }
/* Ensure native radio inputs are visible across themes */
.stRadio input[type="radio"] { display:inline-block !important; opacity:1 !important; width:20px; height:20px; margin-right:10px; vertical-align:middle; accent-color:#111827; flex:0 0 auto; margin-top:0.95rem; }

/* Streamlit form labels and markdown readability */
.stMarkdown, .stText, .stSelectbox, .stRadio, .stSlider, .stCheckbox { color:#111827 !important; }

/* Sidebar header visibility */
div[data-testid="stSidebar"] .stMarkdown h2,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] .css-10trblm {
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
}

div[data-testid="stSidebar"] .stMarkdown,
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] p {
    color: #f8fafc !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
}

div[data-testid="stSidebar"] [data-testid="stSlider"] label,
div[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
div[data-testid="stSidebar"] [data-testid="stTextInput"] label,
div[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Result area */
.result-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:0.75rem; margin:0.75rem 0 1rem 0; }
.result-tile { background:#fff; border:1px solid #dbe2ea; border-radius:16px; padding:0.85rem 1rem; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04); }
.result-tile .label { font-size:0.8rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; }
.result-tile .value { font-size:1.45rem; color:#0f172a; font-weight:800; margin-top:0.15rem; }

@media (max-width: 900px) {
    .result-grid { grid-template-columns: 1fr; }
    .question-topline { flex-direction:column; align-items:flex-start; }
}
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
    st.markdown('<div style="color: #FFFFFF;"><h2>QUIZ SETTINGS</h2></div>',unsafe_allow_html=True)
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
    st.markdown('<div style="padding-bottom: 1rem;" class="small-note">Pick a subject subset, then load a quiz. The app uses the validation split automatically so the interface stays simple.</div>', unsafe_allow_html=True)
    
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
        <div class="small-note">\n
        1. <strong>Choose a subject:</strong> Select an `MMLU config` such as `all` or `machine_learning`.\n
        2. <strong>Load quiz:</strong> Click "Load MMLU questions (4-choice)" to fetch pre-built multiple-choice items.\n
        3. <strong>Answer clearly:</strong> Each choice is displayed as a bordered card with a visible A/B/C/D label.\n
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
