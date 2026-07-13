# RAG Document Assistant

A retrieval-augmented generation (RAG) system for document analysis, designed for zero-cost operation using local embeddings and free-tier LLM endpoints.

## Architecture & Technology Stack

| Component               | Technology        | Specification                                               |
| :---------------------- | :---------------- | :---------------------------------------------------------- |
| **Language**      | Python            | 3.10+                                                       |
| **Orchestration** | LangChain         | Pipeline management & chunking                              |
| **Embeddings**    | FastEmbed         | `BAAI/bge-small-en-v1.5` (Local execution, zero API cost) |
| **Vector DB**     | Qdrant Cloud      | Remote managed cluster                                      |
| **LLM**           | Nous Research API | `stepfun/step-3.7-flash:free`                             |
| **UI**            | Streamlit         | Web interface                                               |

## Workflow

1. **Upload & Split**: Documents (PDF/TXT) are chunked (`CHUNK_SIZE = 800`, `OVERLAP = 200`).
2. **Local Vectorization**: Chunks are embedded locally via FastEmbed without network call overhead.
3. **Cloud Indexing**: Embeddings are persisted in a clean Qdrant Cloud vector collection.
4. **Q&A Retrieval**: Queries match top 4 relevant chunks to generate responses with page citations.

## Setup & Execution

### 1. Configuration

Create a `.env` file in the root directory:

```ini
OPENAI_API_KEY=your_nous_research_api_key
OPENAI_API_BASE=https://portal.nousresearch.com/v1
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
```

### 2. Launch (Windows)

Run the automated launcher:

```Shell
.\run.bat
```

*App opens locally at `http://localhost:8501`.*

## Security

This project was built with a documented threat model and defence-in-depth
controls: uploaded files are validated (extension, size, magic bytes) and stored
under sanitised paths, untrusted filenames are HTML-escaped to prevent XSS,
untrusted document text is isolated from LLM instructions to limit prompt
injection, internal errors are never leaked to the UI, secrets are loaded only
from a git-ignored `.env`, and the app binds to `127.0.0.1`.

A CI pipeline (`.github/workflows/security.yml`) runs on every push:

- **SAST** — `bandit` static analysis of the source
- **SCA** — `pip-audit` against pinned dependencies for known CVEs
- **Secret scanning** — `gitleaks` across the git history

See [SECURITY.md](SECURITY.md) for the full threat model, mitigations, and
residual risks.