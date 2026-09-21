# 📑 Contract Audit & Legal Risk Analysis Engine

An enterprise-grade FastAPI backend that automates procurement contract ingestion, text chunking, vector embedding generation, and semantic RAG (Retrieval-Augmented Generation) audit queries.

---

## 🏗️ System Architecture

```text
                              [ NEXT.JS FRONTEND ]
                                       │
                              ( HTTP / JSON / File )
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 FASTAPI BACKEND                                     │
│                                                                                     │
│   ┌──────────────────────────┐                  ┌───────────────────────────────┐   │
│   │   Ingestion Pipeline     │                  │     Semantic Query Engine     │   │
│   ├──────────────────────────┤                  ├───────────────────────────────┤   │
│   │ 1. PDF Text Extraction   │                  │ 1. Prompt Embedding           │   │
│   │ 2. Text Chunking         │                  │ 2. Vector Similarity Search   │   │
│   │ 3. FastEmbed Generation  │                  │ 3. Context Construction       │   │
│   └────────────┬─────────────┘                  └──────────────┬────────────────┘   │
└────────────────│───────────────────────────────────────────────│────────────────────┘
                 │                                               │
                 ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│        VECTOR DATABASE          │             │            GROQ LLM             │
│  (PgVector / SQL Vector Index)  │             │      (llama-3.3-70b-versatile)  │
└─────────────────────────────────┘             └─────────────────────────────────┘
```

### 🔄 End-to-End Workflow

1. **Document Ingestion (`/contracts/upload`)**:
   - Accepts a PDF document alongside metadata (Title, Vendor, Value, Tenant ID).
   - Extracts raw text page-by-page using PyPDF/pdfplumber.
   - Splits long text into overlapping chunks.
   - Computes dense vector embeddings locally via **FastEmbed**.
   - Persists document metadata and embeddings into PostgreSQL using **PgVector** / **SQL Server**.

2. **Semantic Risk Querying (`/contracts/query`)**:
   - Takes a user prompt and Tenant ID to preserve strict multi-tenant isolation.
   - Converts the prompt into a vector query embedding.
   - Performs cosine similarity matching against stored contract chunks.
   - Sends retrieved top matches to **Groq API** to synthesize an auditable risk evaluation with citations.

---

## 🛠️ Tech Stack & Dependencies

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
- **Server**: Uvicorn
- **Embeddings**: [FastEmbed](https://github.com/qdrant/fastembed) (Local ONNX-optimized embedding generation)
- **Database**: PostgreSQL with `pgvector` or SQL Server with Vector extensions
- **LLM Integration**: [Groq Cloud API](https://groq.com/)
- **PDF Processing**: `pypdf` / `pdfplumber`
- **Validation**: Pydantic v2

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites

- Python 3.10+
- PostgreSQL database instance with `pgvector` enabled

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/contract_audit
GROQ_API_KEY=gsk_your_groq_api_key_here
PORT=8000
```

### 3. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/contract-audit-backend.git
cd contract-audit-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Development Server

```bash
uvicorn main:app --reload --port 8000
```

Access the interactive API documentation (Swagger UI) at <http://localhost:8000/docs>.

---

## 📡 API Endpoints Summary

### `POST /contracts/upload`

Uploads and indexes a contract.

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: PDF file blob
  - `title`: string
  - `vendor`: string
  - `value`: float
  - `tenant_id`: UUID / string

### `POST /contracts/query`

Executes semantic vector search and generates AI analysis.

- **Content-Type**: `application/json`
- **Request Body**:

```json
{
  "prompt": "What are the termination conditions and liability caps?",
  "tenant_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
}
```
