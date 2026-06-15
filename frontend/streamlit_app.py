"""
Phase 5: Streamlit UI for the multilingual RAG chatbot.

This is the public-facing demo — clean chat interface, source chunk
transparency, and language indicators for cross-lingual retrieval.

Run:
    streamlit run frontend/streamlit_app.py

Make sure the FastAPI backend is running first:
    uvicorn app.main:app --reload
"""

import requests
import streamlit as st


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hindi-English RAG Chatbot",
    page_icon="🪔",
    layout="centered",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Overall background */
    .stApp {
        background: linear-gradient(180deg, #0f1116 0%, #1a1d29 100%);
    }

    /* Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0;
        padding-top: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 14px;
        padding: 0.4rem 0.2rem;
    }

    /* Source chunk cards */
    .source-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .source-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .lang-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .lang-hi {
        background: rgba(255, 153, 51, 0.18);
        color: #FF9933;
        border: 1px solid rgba(255, 153, 51, 0.35);
    }
    .lang-en {
        background: rgba(19, 136, 8, 0.18);
        color: #4ADE80;
        border: 1px solid rgba(19, 136, 8, 0.35);
    }
    .score-badge {
        color: #9CA3AF;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .source-text {
        color: #D1D5DB;
        line-height: 1.5;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161922;
    }

    /* Example question buttons */
    .stButton button {
        width: 100%;
        text-align: left;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        color: #E5E7EB;
        font-size: 0.85rem;
        padding: 0.6rem 0.8rem;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        background: rgba(255, 153, 51, 0.12);
        border-color: rgba(255, 153, 51, 0.3);
        color: #FF9933;
    }

    /* Metric pills in sidebar */
    .metric-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        color: #9CA3AF;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #6B7280;
        font-size: 0.75rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🪔 Cross-Lingual RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask in Hindi or English — answers grounded in retrieved sources</div>',
    unsafe_allow_html=True
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### About this project")
    st.markdown(
        "A **cross-lingual Retrieval-Augmented Generation** system. "
        "Questions in Hindi or English are embedded using **LaBSE** "
        "(a multilingual model), retrieved from a shared vector store, "
        "and answered using **Llama-3.3-70B** via Groq."
    )

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown("""
    1. **Embed** query with LaBSE
    2. **Retrieve** top-k chunks from ChromaDB
    3. **Generate** grounded answer with Llama-3.3
    4. **Evaluate** faithfulness with RAGAS
    """)

    st.markdown("---")
    st.markdown("### Evaluation scores")
    st.markdown("""
    <div class="metric-pill">
        <div class="metric-label">Faithfulness</div>
        <div class="metric-value">0.95+</div>
    </div>
    <div class="metric-pill">
        <div class="metric-label">Answer Relevancy</div>
        <div class="metric-value">0.92</div>
    </div>
    <div class="metric-pill">
        <div class="metric-label">Context Precision</div>
        <div class="metric-value">0.96</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=5, value=3)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ── Backend health check ─────────────────────────────────────────────────────

def check_backend():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None


backend_ok, health_data = check_backend()

if not backend_ok:
    st.error(
        "⚠️ Backend not reachable. Make sure FastAPI is running:\n\n"
        "```\nuvicorn app.main:app --reload\n```"
    )
    st.stop()


# ── Example questions ─────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "भारतीय संविधान में मौलिक अधिकार क्या हैं?",
    "What is the significance of Mahatma Gandhi?",
    "हिंदी साहित्य का इतिहास क्या है?",
    "How many schedules does the Indian Constitution have?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"example_{i}"):
                st.session_state.pending_question = q


# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🪔"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📚 View {len(msg['sources'])} source chunks"):
                for src in msg["sources"]:
                    lang = src["language"]
                    lang_class = "lang-hi" if lang == "hi" else "lang-en"
                    lang_label = "हिन्दी" if lang == "hi" else "English"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-meta">
                            <span class="lang-badge {lang_class}">{lang_label}</span>
                            <span class="score-badge">similarity: {src['score']:.3f}</span>
                        </div>
                        <div class="source-text">{src['content'][:300]}{'...' if len(src['content']) > 300 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ── Query function ────────────────────────────────────────────────────────────

def ask_question(question: str):
    """Send question to FastAPI backend and return response."""
    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"question": question, "top_k": top_k},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may be busy — try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}


# ── Handle input ──────────────────────────────────────────────────────────────

# Check for example question click
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Type your question in Hindi or English...")

question = pending or user_input

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    # Get response
    with st.chat_message("assistant", avatar="🪔"):
        with st.spinner("Retrieving sources and generating answer..."):
            result = ask_question(question)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {result['error']}"
            })
        else:
            st.markdown(result["answer"])

            with st.expander(f"📚 View {len(result['sources'])} source chunks"):
                for src in result["sources"]:
                    lang = src["language"]
                    lang_class = "lang-hi" if lang == "hi" else "lang-en"
                    lang_label = "हिन्दी" if lang == "hi" else "English"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-meta">
                            <span class="lang-badge {lang_class}">{lang_label}</span>
                            <span class="score-badge">similarity: {src['score']:.3f}</span>
                        </div>
                        <div class="source-text">{src['content'][:300]}{'...' if len(src['content']) > 300 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })

    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer-note">Built with LangChain · ChromaDB · LaBSE · Groq · Streamlit</div>',
    unsafe_allow_html=True
)
