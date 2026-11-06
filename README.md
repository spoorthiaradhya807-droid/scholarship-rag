# 🎓 Scholarship & Financial Aid Assistant (RAG)

A chat assistant that answers questions about scholarships in India using **Retrieval-Augmented Generation (RAG)**. It searches official scholarship documents (FAQs, guidelines, policies), then uses Google Gemini to write an answer based only on what it found, and shows links so you can verify the answer on the official website.

**Live demo:** <PASTE YOUR DEPLOYED LINK HERE>

## Features

- Chat interface built with Streamlit
- Answers grounded in official documents (central, Karnataka state and institution scholarships)
- "Verify on the official website" links under each answer
- Handles PDFs, with OCR support for scanned documents
- Admin panel (password protected) to upload or delete documents
- Unit tests for chunking, loaders, links, security and config

## How it works

1. **Load** documents from `data/raw/` (PDF, DOCX).
2. **Chunk** the text into small, searchable pieces.
3. **Embed** each chunk with Gemini embeddings and store it in **ChromaDB**.
4. **Retrieve** the most relevant chunks for a user question.
5. **Generate** an answer with Gemini using only the retrieved text, and attach source links.

## Tech stack

- Python 3.12
- Streamlit (UI)
- ChromaDB (vector database)
- Google Gemini via `google-genai` (embeddings and answer generation)
- pypdf, python-docx (document loading)
- pytest (tests)

## Project structure

```
scholarship-rag/
├── app.py                # Streamlit app
├── check_setup.py        # Quick check that the API key works
├── requirements.txt
├── .env.example          # Template for environment variables
├── src/
│   ├── config.py         # Settings loaded from environment
│   ├── loaders.py        # PDF/DOCX loading
│   ├── chunking.py       # Text chunking
│   ├── embeddings.py     # Gemini embeddings
│   ├── vectorstore.py    # ChromaDB operations
│   ├── rag.py            # Retrieval + answer generation
│   ├── links.py          # Official website links for sources
│   ├── security.py       # Upload validation
│   ├── documents.py      # Add/delete documents (admin)
│   ├── index_documents.py# Build the index from data/raw
│   └── ocr_scanned.py    # OCR for scanned PDFs
├── data/
│   ├── raw/              # Source documents (central, karnataka, institution)
│   └── chroma/           # Vector database
└── tests/                # pytest tests
```

## Run locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/scholarship-rag.git
   cd scholarship-rag
   ```

2. **Create a virtual environment and install packages**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Set up environment variables.** Copy `.env.example` to `.env` and fill it in:

   ```
   GEMINI_API_KEY=your_key_here
   GEN_MODEL=gemini-3.5-flash-lite
   EMBED_MODEL=gemini-embedding-001
   ADMIN_PASSWORD=choose_a_password
   ```

   Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

4. **Build the index** (only needed if `data/chroma/` is missing or you added documents):

   ```bash
   python -m src.index_documents
   ```

   Useful options: `--list` (show indexed documents), `--dry-run` (preview changes), `--query "your question"` (test a search).

5. **Start the app**

   ```bash
   streamlit run app.py
   ```

## Run the tests

```bash
pytest
```

## Deployment

Deployed on Streamlit Community Cloud. Secrets (`GEMINI_API_KEY`, `ADMIN_PASSWORD`, `GEN_MODEL`, `EMBED_MODEL`) are set in the app's **Secrets** settings and are never stored in the repository.

## Disclaimer

This assistant is for general guidance only. Scholarship rules, dates and amounts change, so always confirm details on the official portal before applying.

## Author

Spoorthi
