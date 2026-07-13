"""
RAG Document Assistant — Streamlit UI

Run with:  streamlit run app.py
"""

import html
import logging

import streamlit as st
from config import validate_config
from rag_engine import (
    save_uploaded_file,
    cleanup_temp_file,
    validate_upload,
    load_and_split_document,
    create_vector_store,
    get_retriever,
    build_qa_chain,
    ask_question,
    format_sources,
    summarize_document,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import Google Font ──────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main header gradient ────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
}
.main-header h1 {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0.4rem 0 0 0;
}

/* ── Status badges ───────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.status-ready {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-empty {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

/* ── Doc-info card ───────────────────────────────────── */
.doc-info-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%);
    border: 1px solid rgba(102,126,234,0.18);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-top: 0.6rem;
}
.doc-info-card .metric {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 0.9rem;
}
.doc-info-card .metric .label {
    color: #94a3b8;
}
.doc-info-card .metric .value {
    font-weight: 600;
}

/* ── Source citation expander ────────────────────────── */
.source-box {
    background: rgba(102,126,234,0.06);
    border-left: 3px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ── Sidebar polish ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1rem;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(102,126,234,0.35);
}

/* ── File Uploader styling inside Dark Sidebar ───────── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"],
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background-color: #1e293b !important;
    border: 2px dashed #64748b !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Общие правила для всех текстов внутри зоны загрузки */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {
    color: #000000 !important;
    opacity: 1 !important;
}

/* Специфично для инструкции 'Limit 200MB per file' и форматов */
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzoneInstructions"],
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzoneInstructions"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #000000 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}

/* Кнопка 'Browse files' / 'Upload' внутри зоны */
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {
    background-color: #3b82f6 !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}

/* ── Подложка (карточка) уже загруженного файла ──────── */
section[data-testid="stSidebar"] [data-testid="stUploadedFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div > div > div {
    background-color: #000000 !important; /* Очень тёмный, почти чёрный фон подложки */
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 0.6rem !important;
}
section[data-testid="stSidebar"] [data-testid="stUploadedFile"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div > div > div * {
    color: #000000 !important; /* Яркий, чистый белый цвет для имени файла и размера */
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────

defaults = {
    "messages": [],
    "vector_store": None,
    "qa_chain": None,
    "doc_info": None,
    "indexed": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Document Upload")

    config_problems = validate_config()
    if config_problems:
        st.warning(
            "⚠️ **Configuration incomplete.**\n\n"
            + "\n".join(f"- {p}" for p in config_problems)
            + "\n\nCopy `.env.example` to `.env` and fill in the values."
        )

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Max recommended size: 50 MB",
    )

    # ── Index button ───────────────────────────────────
    if uploaded_file is not None:
        if st.button("🚀 Index Document", type="primary"):
            with st.spinner("Parsing & indexing…"):
                file_path = None
                try:
                    validate_upload(uploaded_file)
                    file_path = save_uploaded_file(uploaded_file)
                    chunks = load_and_split_document(
                        file_path, source_name=uploaded_file.name
                    )
                    vector_store = create_vector_store(chunks)
                    retriever = get_retriever(vector_store)
                    qa_chain = build_qa_chain(retriever)

                    st.session_state.vector_store = vector_store
                    st.session_state.qa_chain = qa_chain
                    st.session_state.indexed = True

                    pages = set()
                    for c in chunks:
                        pages.add(c.metadata.get("page", 0))
                    st.session_state.doc_info = {
                        "name": uploaded_file.name,
                        "pages": len(pages),
                        "chunks": len(chunks),
                        "size_kb": round(uploaded_file.size / 1024, 1),
                    }
                    st.session_state.messages = []
                    st.success("✅ Document indexed successfully!")

                except ValueError as e:
                    st.error(f"❌ {e}")
                except Exception:
                    logger.exception("Indexing failed")
                    st.error(
                        "❌ Indexing failed due to an internal error. "
                        "Please try again or check the application logs."
                    )
                finally:
                    if file_path:
                        cleanup_temp_file(file_path)

    # ── Status & info ──────────────────────────────────
    st.markdown("---")

    if st.session_state.indexed and st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown(
            '<span class="status-badge status-ready">● Ready</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
<div class="doc-info-card">
    <div class="metric"><span class="label">File</span><span class="value">{html.escape(str(info['name']))}</span></div>
    <div class="metric"><span class="label">Size</span><span class="value">{info['size_kb']} KB</span></div>
    <div class="metric"><span class="label">Pages</span><span class="value">{info['pages']}</span></div>
    <div class="metric"><span class="label">Chunks</span><span class="value">{info['chunks']}</span></div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="status-badge status-empty">○ No document</span>',
            unsafe_allow_html=True,
        )

    # ── Quick actions ──────────────────────────────────
    if st.session_state.indexed:
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")

        if st.button("📋 Summarize Document"):
            with st.spinner("Generating summary…"):
                summary = summarize_document(st.session_state.qa_chain)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"**📋 Document Summary**\n\n{summary}"}
                )
                st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📄 RAG Document Assistant</h1>
    <p>Upload a document and ask questions — AI answers with source citations</p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Sources"):
                st.markdown(
                    f'<div class="source-box">{msg["sources"]}</div>',
                    unsafe_allow_html=True,
                )

# ── Chat input ─────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your document…"):
    if not st.session_state.indexed:
        st.warning("Please upload and index a document first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching document…"):
                try:
                    result = ask_question(st.session_state.qa_chain, prompt)
                    answer = result["result"]
                    sources_text = format_sources(result.get("source_documents", []))

                    st.markdown(answer)
                    with st.expander("📎 Sources"):
                        st.markdown(
                            f'<div class="source-box">{sources_text}</div>',
                            unsafe_allow_html=True,
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_text,
                    })
                except Exception:
                    logger.exception("Query failed")
                    error_msg = (
                        "❌ Something went wrong while answering. Please try again."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

# ── Empty state ────────────────────────────────────────────────────────────────

if not st.session_state.messages and not st.session_state.indexed:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📤 Upload")
        st.markdown("Upload a PDF or TXT document using the sidebar.")
    with col2:
        st.markdown("### 🔍 Index")
        st.markdown("Click **Index Document** to process and embed the content.")
    with col3:
        st.markdown("### 💬 Chat")
        st.markdown("Ask any question — the AI answers with page citations.")
