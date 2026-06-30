# RAG AI Document Assistant — План реализации

## Цель проекта

Создать веб-приложение на Python, которое позволяет загрузить PDF/TXT документ, проиндексировать его содержимое в векторной базе данных и вести с документом диалог на естественном языке. ИИ отвечает на вопросы с цитированием конкретных фрагментов документа.

**Целевая аудитория резюме:** Вакансия AI Engineering Intern в Swiss Re.
**Демонстрируемые навыки:** Python, GenAI/NLP integration, RAG architecture, Vector DB, API engineering.

---

## Технологический стек

| Компонент | Технология | Версия | Назначение |
|---|---|---|---|
| Язык | Python | 3.11+ | Основной язык |
| Оркестрация | LangChain + langchain-openai + langchain-qdrant | latest | RAG pipeline |
| Векторная БД | Qdrant (local mode) | latest | Хранение эмбеддингов |
| Эмбеддинги | OpenAI `text-embedding-3-small` | — | Векторизация текста ($0.02/1M токенов) |
| LLM | OpenAI `gpt-4o-mini` | — | Генерация ответов (дёшево и быстро) |
| Парсинг PDF | PyPDF2 / pypdf | latest | Извлечение текста из PDF |
| UI | Streamlit | latest | Веб-интерфейс с чатом |
| Env vars | python-dotenv | latest | Управление API ключами |

---

## Архитектура приложения

```
┌─────────────────────────────────────────────────┐
│                  STREAMLIT UI                    │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐  │
│  │ File      │  │ Chat       │  │ Source     │  │
│  │ Upload    │  │ Interface  │  │ Citations  │  │
│  └─────┬─────┘  └─────┬──────┘  └────────────┘  │
│        │              │                          │
├────────┼──────────────┼──────────────────────────┤
│        ▼              ▼                          │
│  ┌───────────────────────────────────────────┐   │
│  │           RAG ENGINE (rag_engine.py)      │   │
│  │                                           │   │
│  │  1. Document Loader (PDF/TXT)             │   │
│  │  2. Text Splitter (RecursiveCharacter)    │   │
│  │  3. Embedding Model (OpenAI)              │   │
│  │  4. Vector Store (Qdrant local)           │   │
│  │  5. Retriever (similarity search, k=4)    │   │
│  │  6. QA Chain (prompt + LLM + sources)     │   │
│  └───────────────────────────────────────────┘   │
│                      │                           │
│        ┌─────────────┼─────────────┐             │
│        ▼             ▼             ▼             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ OpenAI   │  │ Qdrant   │  │ Config   │       │
│  │ API      │  │ Local DB │  │ (.env)   │       │
│  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────┘
```

---

## Структура файлов проекта

```
d:\Antigravity_Projects\Resume\rag-doc-assistant\
├── app.py                  # Streamlit UI (точка входа)
├── rag_engine.py           # Ядро RAG: загрузка, индексация, поиск, генерация
├── config.py               # Настройки и константы
├── requirements.txt        # Зависимости
├── .env.example            # Шаблон переменных окружения
├── .gitignore              # Игнор .env, __pycache__, qdrant_data/
├── README.md               # Описание проекта для GitHub
├── sample_docs/            # Тестовые документы
│   └── sample_report.pdf
└── qdrant_data/            # Локальное хранилище Qdrant (создаётся автоматически)
```

---

## Пошаговая реализация

### Фаза 1: Настройка окружения

1. Создать папку проекта `rag-doc-assistant`
2. Создать виртуальное окружение: `python -m venv venv`
3. Активировать: `.\venv\Scripts\activate` (Windows)
4. Создать `requirements.txt`:
   ```
   streamlit>=1.38.0
   langchain>=0.3.0
   langchain-openai>=0.2.0
   langchain-qdrant>=0.2.0
   langchain-community>=0.3.0
   qdrant-client>=1.12.0
   pypdf>=5.0.0
   python-dotenv>=1.0.0
   ```
5. Установить: `pip install -r requirements.txt`
6. Создать `.env` с ключом: `OPENAI_API_KEY=sk-...`

---

### Фаза 2: config.py — Конфигурация

Файл с константами и загрузкой переменных окружения:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Qdrant
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "documents"

# Retrieval
RETRIEVAL_K = 4
```

---

### Фаза 3: rag_engine.py — Ядро RAG

Этот модуль содержит всю логику: загрузка документов, разбивка на чанки, создание эмбеддингов, сохранение в Qdrant, поиск по запросу и генерация ответа с цитатами.

**Ключевые функции:**

#### 3.1 `load_and_split_document(file_path: str) -> list[Document]`
- Определяет тип файла (PDF или TXT)
- Для PDF: использует `PyPDFLoader` из `langchain_community.document_loaders`
- Для TXT: использует `TextLoader`
- Разбивает на чанки через `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)`
- Каждый чанк сохраняет metadata: `source` (имя файла), `page` (номер страницы)
- Возвращает список объектов `Document`

#### 3.2 `create_vector_store(documents: list[Document]) -> QdrantVectorStore`
- Создаёт экземпляр `OpenAIEmbeddings(model="text-embedding-3-small")`
- Инициализирует `QdrantVectorStore.from_documents()` с параметрами:
  - `documents=documents`
  - `embedding=embeddings`
  - `path=QDRANT_PATH` (локальный режим, без Docker)
  - `collection_name=COLLECTION_NAME`
- Возвращает объект vector store

#### 3.3 `get_retriever(vector_store) -> VectorStoreRetriever`
- Вызывает `vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})`
- Возвращает retriever для использования в цепочке

#### 3.4 `build_qa_chain(retriever) -> RetrievalQA`
- Создаёт системный промпт:
  ```
  You are a helpful document assistant. Answer the user's question 
  based ONLY on the provided context. If the answer is not in the 
  context, say "I don't have enough information to answer this."
  
  Always cite the source page number when possible.
  
  Context: {context}
  Question: {question}
  ```
- Создаёт `ChatOpenAI(model="gpt-4o-mini", temperature=0.2)`
- Собирает цепочку через `RetrievalQA.from_chain_type()` с `return_source_documents=True`
- Возвращает готовую QA chain

#### 3.5 `ask_question(qa_chain, question: str) -> dict`
- Вызывает `qa_chain.invoke({"query": question})`
- Возвращает словарь с ключами:
  - `"result"` — текст ответа от LLM
  - `"source_documents"` — список Document объектов с metadata (page, source)
- Форматирует цитаты: `📄 Источник: {filename}, стр. {page}`

#### 3.6 `summarize_document(qa_chain) -> str`
- Вызывает `ask_question()` с запросом: "Provide a comprehensive summary of this document in 5-7 key points."
- Возвращает структурированную выжимку документа

---

### Фаза 4: app.py — Streamlit UI

#### 4.1 Конфигурация страницы
```python
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)
```

#### 4.2 Боковая панель (Sidebar)
- **Загрузка файла:** `st.file_uploader("Upload document", type=["pdf", "txt"])`
- **Кнопка "Index Document"** — запускает индексацию загруженного файла
- **Кнопка "Summarize"** — генерирует краткую выжимку документа
- **Статус индексации:** индикатор (✅ Indexed / ⏳ Processing / ❌ No document)
- **Информация о документе:** имя файла, количество страниц, количество чанков

#### 4.3 Основная область — Чат
- Инициализация `st.session_state.messages = []`
- Отображение истории через цикл `for msg in st.session_state.messages`
- Каждое сообщение через `st.chat_message(role)` с аватарами: 👤 user, 🤖 assistant
- Ввод через `st.chat_input("Ask a question about your document...")`
- При получении вопроса:
  1. Добавить вопрос в историю
  2. Показать спиннер `st.spinner("Searching document...")`
  3. Вызвать `ask_question(qa_chain, question)`
  4. Отобразить ответ
  5. В `st.expander("📎 Sources")` показать цитаты с номерами страниц
  6. Добавить ответ в историю

#### 4.4 Session State (ключевые переменные)
```python
st.session_state.messages      # История чата
st.session_state.vector_store  # Проиндексированный vector store
st.session_state.qa_chain      # Готовая QA chain
st.session_state.doc_info      # Метаданные документа (имя, страницы, чанки)
st.session_state.indexed       # bool — проиндексирован ли документ
```

#### 4.5 Кастомные стили (CSS)
- Тёмная тема с акцентными цветами (синий/фиолетовый градиент)
- Стилизация чат-пузырей
- Анимация загрузки при индексации
- Inject через `st.markdown("<style>...</style>", unsafe_allow_html=True)`

---

### Фаза 5: Дополнительные фичи (для впечатления)

1. **Кнопка "Summarize Document"** — мгновенная выжимка документа в 5-7 пунктов
2. **Отображение источников (Sources)** — под каждым ответом раскрывающийся блок с цитатами и номерами страниц
3. **Очистка чата** — кнопка "Clear Chat" в sidebar
4. **Обработка ошибок:**
   - Отсутствие API ключа → понятное сообщение
   - Пустой PDF → предупреждение
   - Превышение лимита → graceful fallback
5. **Поддержка нескольких файлов** (опционально): возможность загрузить 2-3 документа и искать по всем

---

### Фаза 6: README.md для GitHub

```markdown
# 📄 RAG Document Assistant

An AI-powered document analysis tool built with Python, LangChain, 
Qdrant and Streamlit. Upload any PDF or text document and have an 
intelligent conversation with its contents.

## Features
- 📤 Upload PDF/TXT documents
- 🔍 Semantic search across document content  
- 💬 Natural language Q&A with source citations
- 📋 One-click document summarization
- 🗄️ Local vector storage (Qdrant) — no external DB needed

## Tech Stack
- **Python 3.11+**
- **LangChain** — RAG orchestration
- **Qdrant** (local mode) — Vector database
- **OpenAI API** — Embeddings (text-embedding-3-small) & LLM (gpt-4o-mini)
- **Streamlit** — Web interface

## Architecture
[RAG architecture diagram]

## Quick Start
1. Clone: `git clone <repo-url> && cd rag-doc-assistant`
2. Install: `pip install -r requirements.txt`
3. Configure: `cp .env.example .env` and add your OpenAI API key
4. Run: `streamlit run app.py`

## How It Works
1. Document is parsed and split into overlapping chunks (800 chars, 200 overlap)
2. Each chunk is converted to a vector embedding via OpenAI
3. Embeddings are stored in a local Qdrant vector database
4. User questions are embedded and matched against stored chunks (top-4)
5. Retrieved chunks + question are sent to GPT-4o-mini for answer generation
6. Response includes source page citations
```

---

## План верификации

### Автоматическая проверка
1. `pip install -r requirements.txt` — убедиться что все зависимости ставятся
2. `streamlit run app.py` — убедиться что UI запускается без ошибок
3. Загрузить тестовый PDF → проверить индексацию
4. Задать 3-5 вопросов → проверить релевантность ответов и цитат
5. Нажать Summarize → проверить качество выжимки

### Ручная проверка
- Загрузить PDF на 10+ страниц и убедиться что поиск работает по всему документу
- Задать вопрос, ответа на который нет в документе → убедиться что модель честно отвечает "I don't have enough information"
- Проверить что `.env` не попадает в Git (`.gitignore`)

---

## Порядок написания кода

| Шаг | Файл | Что делаем |
|-----|------|------------|
| 1 | Создание папки и venv | Инфраструктура |
| 2 | `requirements.txt` | Зависимости |
| 3 | `.env.example`, `.gitignore` | Конфигурация |
| 4 | `config.py` | Константы |
| 5 | `rag_engine.py` | Ядро RAG (все 6 функций) |
| 6 | `app.py` | Streamlit UI + интеграция с rag_engine |
| 7 | `README.md` | Документация для GitHub |
| 8 | Тестирование | Загрузка PDF, вопросы, проверка цитат |

---

## Open Questions

> [!IMPORTANT]
> **Нужен OpenAI API ключ.** У тебя уже есть ключ OpenAI API (`sk-...`)? Если нет, нужно зарегистрироваться на [platform.openai.com](https://platform.openai.com) и получить ключ. Стоимость для этого проекта будет минимальной (менее $1 на десятки документов).

> [!NOTE]
> **Язык интерфейса.** Сделать UI на английском (лучше для GitHub и резюме) или на русском? Рекомендую английский, так как проект для резюме в международную компанию.
