from __future__ import annotations

import json

import streamlit as st

from quiz_engine import load_clapnq_sample, quiz_to_json


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

st.markdown(
    """
<div class="hero">
    <h1>Board Exam Quiz Forge</h1>
    <p>
        Powered by SQuAD v2 dataset: Stanford Question Answering Dataset. 
        Load board-exam style questions with improved randomization, category filtering, 
        and challenging multiple-choice options with better distractors.
    </p>
    <div class="badge-row">
        <span class="badge">SQuAD v2 Dataset</span>
        <span class="badge">Randomized questions</span>
        <span class="badge">Category filtering</span>
        <span class="badge">Smart distractors</span>
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
    num_questions = st.slider("Number of questions", min_value=3, max_value=20, value=6)
    quiz_split = st.selectbox("Dataset split", options=["train", "validation"], index=0)
    category_filter = st.text_input("Filter by category (optional)", placeholder="e.g., history, science, geography...")
    randomize_questions = st.checkbox("Randomize question order", value=True)
    distractor_difficulty = st.slider("Distractor difficulty", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Higher = harder distractors")
    use_llm = st.checkbox("Use LLM for question generation (heavy)", value=False)
    model_name_input = st.text_input("LLM model name", value="meta-llama/Meta-Llama-3-8B-Instruct")
    hf_token = st.text_input("Hugging Face token (optional)", type="password")
    llm_qpp = st.slider("LLM questions per passage", min_value=1, max_value=3, value=1)

col_main, col_info = st.columns([1.12, 0.88], gap="large")

with col_main:
    st.markdown('<div class="panel"><h3>Load Quiz</h3></div>', unsafe_allow_html=True)
    
    load_col_1, load_col_2 = st.columns([0.55, 0.45])
    with load_col_1:
        load_clicked = st.button("Load SQuAD v2 questions", use_container_width=True, type="primary")
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

    if load_clicked:
        with st.spinner("Loading board exam questions from SQuAD v2 dataset..."):
            try:
                quiz_items = load_clapnq_sample(
                    split=quiz_split,
                    max_items=num_questions,
                    category=category_filter if category_filter else None,
                    randomize=randomize_questions,
                    distractor_difficulty=float(distractor_difficulty),
                    use_llm=bool(use_llm),
                    model_name=model_name_input if model_name_input else None,
                    hf_token=hf_token if hf_token else None,
                    llm_questions_per_passage=int(llm_qpp),
                )
                st.session_state.quiz_items = quiz_items
                st.session_state.graded = False
                st.session_state.score = 0
                st.session_state.total = len(quiz_items)
                for key in list(st.session_state.keys()):
                    if key.startswith("choice_"):
                        del st.session_state[key]
                if not quiz_items:
                    st.warning("Could not load quiz items from the dataset. Please try again or adjust your category filter.")
                else:
                    st.success(f"✅ Loaded {len(quiz_items)} questions from SQuAD v2 dataset.")
            except Exception as exc:
                st.error(f"❌ Dataset loading failed: {exc}")

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
        1. <strong>Configure settings:</strong> Adjust number of questions, pick a split, optionally filter by category, and choose if you want randomization.
        2. <strong>Load questions:</strong> Click "Load SQuAD v2 questions" to fetch board-exam style questions with improved distractors.
        3. <strong>Answer:</strong> Select from multiple-choice options (distractors are more challenging and realistic).
        4. <strong>Grade:</strong> Click "Grade my quiz" to see your score and review with explanations.
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
            # show choices as radio buttons for better visibility
            st.radio(
                f"Answer for question {index + 1}",
                options=["Select an answer"] + list(item.choices),
                key=f"choice_{index}",
                label_visibility="collapsed",
            )

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
            is_correct = selected == item.answer
            if is_correct:
                correct += 1
            results.append((item, selected, is_correct, is_answered))

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
            st.markdown(
                f"""
                <div class="question-card">
                    <div class="question-title">Question {index + 1} - <span class="{state_class}">{state_label}</span></div>
                    <div class="question-meta">{item.question}</div>
                    <div class="question-meta">Your answer: <strong>{selected if is_answered else 'Not answered'}</strong></div>
                    <div class="question-meta">Correct answer: <strong>{item.answer}</strong></div>
                    <div class="question-meta"><strong>Context:</strong> {item.context}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("Click 'Load from CLAPNQ' to start a quiz.")
