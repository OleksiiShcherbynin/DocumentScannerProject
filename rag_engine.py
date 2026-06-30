"""
RAG Engine — core logic for document loading, indexing, retrieval and Q&A.

Pipeline:
  1. Load PDF / TXT  →  2. Split into chunks  →  3. Embed & store in Qdrant
  4. Retrieve relevant chunks  →  5. Generate answer with citations via LLM
"""

import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding


class LocalEmbeddings(Embeddings):
    """Lightweight wrapper around FastEmbed for local, free text embedding."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [emb.tolist() for emb in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self.model.embed([text]))[0].tolist()

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    RETRIEVAL_K,
)


# ── 1. Load & split ────────────────────────────────────────────────────────────

def load_and_split_document(file_path: str) -> list:
    """
    Load a PDF or TXT file and split it into overlapping text chunks.

    Each chunk retains metadata (source filename, page number) so that
    answers can later cite the exact location in the original document.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")

    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(raw_docs)

    # Normalise metadata so every chunk has a human-friendly source name
    source_name = Path(file_path).name
    for chunk in chunks:
        chunk.metadata["source"] = source_name
        chunk.metadata.setdefault("page", 0)

    return chunks


# ── 2. Vector store ────────────────────────────────────────────────────────────

def create_vector_store(documents: list) -> QdrantVectorStore:
    """
    Embed all document chunks and store them in Qdrant Cloud.
    If the collection already exists it will be overwritten.
    """
    # Free local embedding model — runs on your PC, no API needed
    embeddings = LocalEmbeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True,          # fresh index for each upload
    )

    return vector_store


# ── 3. Retriever ───────────────────────────────────────────────────────────────

def get_retriever(vector_store: QdrantVectorStore):
    """Return a retriever that fetches the top-k most relevant chunks."""
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})


# ── 4. QA Chain ────────────────────────────────────────────────────────────────

_QA_PROMPT_TEMPLATE = """You are a helpful document assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information in the document to answer this question."

Always cite the source page number when possible, using the format (Page X).

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=_QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_qa_chain(retriever) -> RetrievalQA:
    """
    Build a RetrievalQA chain that:
      - retrieves relevant chunks via the retriever
      - feeds them + the user question into the LLM
      - returns both the answer text and the source documents
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )

    return qa_chain


# ── 5. Ask ─────────────────────────────────────────────────────────────────────

def ask_question(qa_chain: RetrievalQA, question: str) -> dict:
    """
    Send a question to the QA chain.

    Returns
    -------
    dict with keys:
        "result"           – the LLM-generated answer text
        "source_documents" – list of LangChain Document objects used as context
    """
    response = qa_chain.invoke({"query": question})
    return response


def format_sources(source_documents: list) -> str:
    """
    Build a human-readable citation block from retrieved documents.

    Example output:
        📄 Source: report.pdf, Page 3
        📄 Source: report.pdf, Page 7
    """
    seen = set()
    lines = []

    for doc in source_documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            lines.append(f"📄 Source: {source}, Page {int(page) + 1}")

    return "\n".join(lines) if lines else "No source information available."


# ── 6. Summarise ───────────────────────────────────────────────────────────────

def summarize_document(qa_chain: RetrievalQA) -> str:
    """Ask the LLM to produce a structured summary of the indexed document."""
    summary_prompt = (
        "Provide a comprehensive summary of this document. "
        "Structure your response as 5-7 key points, each on a new line "
        "starting with a bullet (•). Be specific and cite page numbers."
    )
    response = ask_question(qa_chain, summary_prompt)
    return response["result"]


# ── Helper: save uploaded file to a temporary path ─────────────────────────────

def save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to a temp directory and return the path.
    The caller is responsible for cleanup (or relies on OS temp-dir GC).
    """
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
