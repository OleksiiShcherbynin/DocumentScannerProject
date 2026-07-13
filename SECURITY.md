# Security

This document describes the threat model of the RAG Document Assistant, the
trust boundaries in its data flow, the controls implemented in the code, and
the residual risks that remain by design.

## Deployment assumption

The application is designed to run **locally** on a single user's machine
(`streamlit run app.py`, bound to `127.0.0.1`). The threat model below reflects
that assumption. Exposing the app to a network or the public internet changes
the model significantly — see [Hardening required before public deployment](#hardening-required-before-public-deployment).

## Data flow and trust boundaries

```
[User] --upload PDF/TXT--> [Streamlit app]  (trust boundary: untrusted file bytes + filename)
                               |
                               v
                     [validate_upload]  size / extension / magic-byte checks
                               |
                               v
                 [temp dir, UUID filename]  (path is never derived from user input)
                               |
                               v
                       [pypdf / TextLoader]  parses untrusted document
                               |
                               v
                    [FastEmbed - LOCAL]  embeddings, no network
                               |
                               v
        =========== trust boundary: data leaves the machine ===========
                               |
              +----------------+-----------------+
              v                                  v
     [Qdrant Cloud]                     [LLM API - Nous/OpenAI]
   stores document chunks          receives question + retrieved context
```

Two trust boundaries matter:

1. **Untrusted input** enters via the uploaded file's **bytes**, its **filename**,
   and the **chat query**.
2. **Data egress**: document chunks are sent to Qdrant Cloud, and the question
   plus retrieved context are sent to the LLM endpoint. Uploaded content leaves
   the local machine. Embeddings are computed locally and never leave.

## Threats and mitigations

| # | Threat | Vector | Mitigation | Status |
|---|--------|--------|------------|--------|
| 1 | Secret exposure | API keys in source / VCS | Secrets loaded only from `.env` (git-ignored); `.env.example` holds placeholders; `validate_config()` fails closed; CI secret scanning (gitleaks) | Mitigated |
| 2 | Stored XSS | Malicious filename rendered into HTML via `unsafe_allow_html` | All user/document-derived strings passed through `html.escape()` before HTML interpolation (`app.py`, `format_sources`) | Mitigated |
| 3 | Path traversal | `../` or absolute path in uploaded filename | On-disk name is a random UUID + validated extension; the client filename never reaches the write path | Mitigated |
| 4 | Malicious / oversized upload | File type mismatch, decompression bombs, huge files | Extension allow-list + magic-byte sniffing (`%PDF-`, UTF-8 decode) + `MAX_FILE_SIZE` limit, enforced server-side and in `.streamlit/config.toml` | Mitigated |
| 5 | Information disclosure | Raw exception text (endpoints, status) shown in UI | Internal errors are logged server-side; users see a generic message. Validation errors show safe, intentional messages | Mitigated |
| 6 | Local data retention | Uploaded files linger in temp dir | Temp directory is deleted in a `finally` block after indexing | Mitigated |
| 7 | Supply-chain / known CVEs | Vulnerable or tampered dependency | Dependencies pinned to exact versions; CI runs `pip-audit` (SCA) and `bandit` (SAST) on every push | Mitigated |
| 8 | Network exposure | App reachable from LAN/internet | Bound to `127.0.0.1`, XSRF protection enabled | Mitigated (local scope) |
| 9 | Prompt injection | Instructions embedded in the document (indirect) or in the query (direct) | Context is wrapped in delimiters and explicitly marked as untrusted data in the system prompt; the LLM is instructed to ignore embedded commands | Partially mitigated — see below |

## Residual risks

- **Prompt injection is not fully solvable.** A crafted document can still steer
  the model's answer text. Impact is contained because the LLM has **no tools or
  side effects** — the worst case is a misleading answer, and the XSS path that
  could have amplified it (threat #2) is closed. Do not build automated actions
  on top of model output without re-validating.
- **Third-party data egress.** Document content is sent to Qdrant Cloud and the
  LLM provider. Do not upload confidential documents unless those providers'
  terms are acceptable to you. Self-hosting Qdrant and a local LLM removes this.
- **PDF parser surface.** `pypdf` parses untrusted files. It is pinned and
  audited in CI, but keep it patched.

## Hardening required before public deployment

The current build is safe for local single-user use. Before exposing it to
multiple users or the internet, add:

- **Authentication** — there is none; anyone who reaches the port can spend the
  API credentials.
- **Per-user data isolation** — all uploads share one Qdrant collection
  (`documents`) and every index run recreates it (`force_recreate=True`). One
  user's upload would overwrite another's.
- **Rate limiting / quotas** on uploads and queries.
- **A reverse proxy** (TLS, request limits) instead of exposing Streamlit
  directly.

## Reporting a vulnerability

If you find a security issue, please open a private report / contact the
maintainer directly rather than filing a public issue. Include steps to
reproduce and the affected version.
