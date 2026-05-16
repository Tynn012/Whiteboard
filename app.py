from __future__ import annotations

import json

import streamlit as st

from quiz_engine import DEFAULT_MODEL_NAME, generate_quiz, quiz_to_json


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
        Paste your reviewer notes, let PrimeQA generate practice questions, and turn one long study block into a scored quiz.
        The app prefers the PrimeQA passage question-generation pipeline and falls back to the same Hugging Face checkpoint when needed.
    </p>
    <div class="badge-row">
        <span class="badge">PrimeQA / mT5 reviewer</span>
        <span class="badge">Board-exam quiz mode</span>
        <span class="badge">Offline-friendly fallback</span>
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
    model_name = st.text_input("Model checkpoint", value=DEFAULT_MODEL_NAME)
    questions_per_chunk = st.slider("Questions per passage", min_value=1, max_value=4, value=2)
    max_chunks = st.slider("Passage chunks to use", min_value=1, max_value=8, value=4)
    num_beams = st.slider("Beam search width", min_value=2, max_value=6, value=4)
    st.caption("PrimeQA's TyDi passage question generator works best with clean study notes and paragraph-sized chunks.")

col_left, col_right = st.columns([1.12, 0.88], gap="large")

with col_left:
    st.markdown('<div class="panel"><h3>Study material</h3></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"], label_visibility="collapsed")
    pasted_text = st.text_area(
        "Paste your board exam reviewer notes here",
        height=360,
        placeholder="Example: The Philippine Constitution establishes ...",
        label_visibility="collapsed",
    )

    input_text = pasted_text.strip()
    if uploaded is not None:
        uploaded_text = uploaded.read().decode("utf-8", errors="ignore")
        if uploaded_text.strip():
            input_text = uploaded_text.strip()

    action_col_1, action_col_2 = st.columns([0.55, 0.45])
    with action_col_1:
        generate_clicked = st.button("Generate quiz", use_container_width=True, type="primary")
    with action_col_2:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.quiz_items = []
        st.session_state.graded = False
        st.session_state.score = 0
        st.session_state.total = 0
        for key in list(st.session_state.keys()):
            if key.startswith("choice_"):
                del st.session_state[key]
        st.rerun()

    if generate_clicked:
        if not input_text:
            st.warning("Paste some notes or upload a text file first.")
        else:
            with st.spinner("Generating reviewer questions with PrimeQA..."):
                try:
                    quiz_items = generate_quiz(
                        input_text,
                        model_name=model_name,
                        questions_per_chunk=questions_per_chunk,
                        max_chunks=max_chunks,
                        num_beams=num_beams,
                    )
                    st.session_state.quiz_items = quiz_items
                    st.session_state.graded = False
                    st.session_state.score = 0
                    st.session_state.total = len(quiz_items)
                    for key in list(st.session_state.keys()):
                        if key.startswith("choice_"):
                            del st.session_state[key]
                    if not quiz_items:
                        st.warning("The model did not produce quiz items from the current text. Try cleaner paragraphs or more detailed notes.")
                    else:
                        st.success(f"Generated {len(quiz_items)} quiz questions.")
                except Exception as exc:
                    st.error(f"Quiz generation failed: {exc}")

    if st.session_state.quiz_items:
        export_payload = json.dumps(quiz_to_json(st.session_state.quiz_items), indent=2, ensure_ascii=False)
        st.download_button(
            "Download quiz JSON",
            data=export_payload,
            file_name="board_exam_quiz.json",
            mime="application/json",
            use_container_width=True,
        )

with col_right:
    st.markdown('<div class="panel"><h3>Reviewer workflow</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="small-note">
        1. Paste your reviewer notes or upload a text file.
        2. PrimeQA samples answer spans from the passage and turns them into questions.
        3. The app builds multiple-choice options, grades your responses, and shows the answer key.
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
                    <div class="question-meta"><strong>Source answer:</strong> {item.answer}</div>
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
            state_label = "Correct" if is_correct else "Incorrect"
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
    st.info("Add reviewer notes, generate the quiz, then answer the questions here.")
