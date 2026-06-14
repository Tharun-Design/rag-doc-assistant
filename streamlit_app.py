"""
streamlit_app.py — Chat UI for RAG Documentation Assistant
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import uuid

API_BASE = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Documentation Assistant",
    layout="wide",
)

# ── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }

    /* Header */
    .doc-header {
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .doc-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1A1D23;
        margin-bottom: 0.25rem;
        letter-spacing: -0.01em;
    }
    .doc-header p {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.8rem;
        color: #6B7280;
        margin: 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F7F8FA;
        border-right: 1px solid #E5E7EB;
    }
    .sidebar-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1A1D23;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }
    .sidebar-section-label {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.35rem 0.65rem;
        border-radius: 4px;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.online { background-color: #16A34A; }
    .status-dot.offline { background-color: #DC2626; }

    /* Chunk count */
    .chunk-count {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 1.75rem;
        font-weight: 600;
        color: #1A1D23;
        line-height: 1;
        margin-top: 0.5rem;
    }
    .chunk-label {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.7rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.15rem;
    }

    /* Trace footer on assistant messages */
    .trace {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.72rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
        margin-top: 0.85rem;
        padding-top: 0.6rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem;
        align-items: center;
    }
    .trace-item {
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .trace-key {
        color: #9CA3AF;
    }
    .trace-val {
        color: #374151;
        font-weight: 500;
    }
    .trace-val.flag-web {
        color: #B45309;
    }
    .trace-val.flag-local {
        color: #15803D;
    }

    /* Source list */
    .source-item {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.78rem;
        color: #374151;
        padding: 0.3rem 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .source-item:last-child {
        border-bottom: none;
    }

    /* Notice / error banner */
    .notice {
        font-size: 0.85rem;
        color: #92400E;
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 4px;
        padding: 0.6rem 0.85rem;
        margin-top: 0.75rem;
    }

    /* Buttons */
    .stButton button {
        border-radius: 4px;
        font-weight: 500;
        border: 1px solid #E5E7EB;
    }
    .stButton button[kind="primary"] {
        background-color: #1A1D23;
        border-color: #1A1D23;
    }

    /* Feedback rating confirmation */
    .rated {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.72rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def send_feedback(question: str, answer: str, rating: str):
    try:
        requests.post(f"{API_BASE}/feedback", json={
            "question": question,
            "answer": answer,
            "rating": rating,
        }, timeout=5)
    except Exception:
        pass


def render_trace(meta: dict):
    web_used = meta.get("web_search_used", False)
    source_flag = "web search" if web_used else "local index"
    flag_class = "flag-web" if web_used else "flag-local"

    st.markdown(f"""
    <div class="trace">
        <div class="trace-item">
            <span class="trace-key">source</span>
            <span class="trace-val {flag_class}">{source_flag}</span>
        </div>
        <div class="trace-item">
            <span class="trace-key">type</span>
            <span class="trace-val">{meta.get('query_type', 'general')}</span>
        </div>
        <div class="trace-item">
            <span class="trace-key">retries</span>
            <span class="trace-val">{meta.get('retry_count', 0)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Documentation Assistant</div>', unsafe_allow_html=True)

    # Server status
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        data = r.json()
        st.markdown(
            '<div class="status-pill"><span class="status-dot online"></span>server online</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="chunk-count">{data.get("total_chunks", 0)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="chunk-label">indexed chunks</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown(
            '<div class="status-pill"><span class="status-dot offline"></span>server offline</div>',
            unsafe_allow_html=True,
        )
        st.caption("Start the API with `python main.py`")

    # Ingest documents
    st.markdown('<div class="sidebar-section-label">Add documents</div>', unsafe_allow_html=True)

    url_input = st.text_area(
        "URLs, one per line",
        placeholder="https://docs.example.com/guide",
        height=90,
        label_visibility="collapsed",
    )

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["md", "txt", "html"],
        label_visibility="collapsed",
    )

    if st.button("Ingest", use_container_width=True, type="primary"):
        with st.spinner("Indexing documents..."):
            try:
                files_data = []
                for f in (uploaded_files or []):
                    files_data.append(("files", (f.name, f.read(), "text/plain")))

                data = {}
                if url_input.strip():
                    urls = ",".join(
                        u.strip() for u in url_input.strip().splitlines() if u.strip()
                    )
                    data["urls"] = urls

                response = requests.post(
                    f"{API_BASE}/ingest",
                    data=data if data else None,
                    files=files_data if files_data else None,
                    timeout=60,
                )
                result = response.json()
                st.success(f"Indexed {result.get('chunks_added', '?')} new chunks")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    # Session info
    st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
    st.code(st.session_state.session_id[:8], language=None)
    turns = len(st.session_state.messages) // 2
    st.caption(f"{turns} exchange{'s' if turns != 1 else ''} in this session")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback_given = {}
        try:
            requests.delete(f"{API_BASE}/memory/{st.session_state.session_id}")
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="doc-header">
    <h1>Ask your documentation</h1>
    <p>retrieval-augmented · graded · grounded</p>
</div>
""", unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            render_trace(meta)

            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(f'<div class="source-item">{s}</div>', unsafe_allow_html=True)

            feedback_key = f"feedback_{i}"
            if feedback_key not in st.session_state.feedback_given:
                col_a, col_b, _ = st.columns([0.08, 0.08, 0.84])
                with col_a:
                    if st.button("Helpful", key=f"up_{i}"):
                        send_feedback(msg.get("question", ""), msg["content"], "thumbs_up")
                        st.session_state.feedback_given[feedback_key] = "Marked helpful"
                        st.rerun()
                with col_b:
                    if st.button("Not helpful", key=f"down_{i}"):
                        send_feedback(msg.get("question", ""), msg["content"], "thumbs_down")
                        st.session_state.feedback_given[feedback_key] = "Marked not helpful"
                        st.rerun()
            else:
                st.markdown(
                    f'<span class="rated">{st.session_state.feedback_given[feedback_key]}</span>',
                    unsafe_allow_html=True,
                )

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documentation"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            try:
                response = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )
                result = response.json()
                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])
                query_type = result.get("query_type", "general")
                web_used = result.get("web_search_used", False)
                retry_count = result.get("retry_count", 0)
                error = result.get("error")

                st.markdown(answer)

                meta = {
                    "sources": sources,
                    "query_type": query_type,
                    "web_search_used": web_used,
                    "retry_count": retry_count,
                }
                render_trace(meta)

                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for s in sources:
                            st.markdown(f'<div class="source-item">{s}</div>', unsafe_allow_html=True)

                if error:
                    st.markdown(f'<div class="notice">{error}</div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "question": prompt,
                    "meta": meta,
                })

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server. Start it with `python main.py`.")
            except Exception as e:
                st.error(f"Request failed: {e}")