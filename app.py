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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Merriweather:wght@700&display=swap');

:root {
    --bg-1: #f8f1e6;
    --bg-2: #efe4d1;
    --ink: #182033;
    --muted: #5b657a;
    --card: rgba(255, 255, 255, 0.82);
    --border: rgba(24, 32, 51, 0.08);
    --accent: #c75b12;
    --accent-soft: rgba(199, 91, 18, 0.12);
    --good: #0f7b52;
    --bad: #b42318;
}

html, body, [class*='css'] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(199, 91, 18, 0.12), transparent 26%),
        radial-gradient(circle at bottom right, rgba(15, 123, 82, 0.08), transparent 22%),
        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
    color: var(--ink);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 1.4rem 1.5rem;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid var(--border);
    border-radius: 26px;
    box-shadow: 0 24px 60px rgba(24, 32, 51, 0.08);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

.hero h1 {
    font-family: 'Merriweather', serif;
    font-size: 2.2rem;
    margin-bottom: 0.35rem;
    color: var(--ink);
}

.hero p {
    margin: 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
}

.badge-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 0.85rem;
    font-weight: 700;
}

.panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1rem 1.05rem;
    box-shadow: 0 20px 45px rgba(24, 32, 51, 0.06);
    margin-bottom: 1rem;
}

.panel h3 {
    margin-top: 0;
    margin-bottom: 0.6rem;
    font-family: 'Merriweather', serif;
    color: var(--ink);
}

.question-card {
    padding: 1rem 1rem 0.25rem;
    margin-bottom: 1rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--border);
    box-shadow: 0 14px 30px rgba(24, 32, 51, 0.05);
}

.question-title {
    font-size: 1rem;
    font-weight: 800;
    color: var(--ink);
    margin-bottom: 0.45rem;
}

.question-meta {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 0.65rem;
}

.correct {
    color: var(--good);
    font-weight: 700;
}

.incorrect {
    color: var(--bad);
    font-weight: 700;
}

.small-note {
    color: var(--muted);
    font-size: 0.92rem;
}
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
                    randomize=randomize_questions
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
            st.selectbox(
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
