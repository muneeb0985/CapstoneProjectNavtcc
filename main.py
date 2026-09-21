import os
import json
import re
from datetime import date
from contextlib import asynccontextmanager
from typing import Optional, List 
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import pypdf
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = None
pool: Optional[asyncpg.Pool] = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, model, llm
    
    print("Loading SentenceTransformer model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            llm = ChatGroq(
                model_name="openai/gpt-oss-120b",
                api_key=groq_key
            )
            print("Groq LLM successfully initialized.")
        except Exception as e:
            print(f"Failed to initialize Groq LLM: {e}")
            llm = None
    else:
        print("WARNING: GROQ_API_KEY environment variable is not set in .env")
        llm = None

    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:123@localhost:5432/Contract"
    )
    
    try:
        pool = await asyncpg.create_pool(dsn=DATABASE_URL)
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS contracts (
                    id SERIAL PRIMARY KEY,
                    tenant_id VARCHAR(255) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    vendor VARCHAR(255) NOT NULL,
                    value NUMERIC NOT NULL,
                    content TEXT,
                    safety_score INT,
                    rationale TEXT,
                    expiration_date VARCHAR(50) DEFAULT '2026-12-31',
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS expiration_date VARCHAR(50) DEFAULT '2026-12-31';")
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS contract_chunks (
                    id SERIAL PRIMARY KEY,
                    contract_id INT REFERENCES contracts(id) ON DELETE CASCADE,
                    tenant_id VARCHAR(255) NOT NULL,
                    page_number INT NOT NULL DEFAULT 1,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384)
                );
            """)
            
            print("Database pool and tables initialized successfully.")
    except Exception as e:
        print("Database connection error:", e)
        pool = None

    yield

    if pool:
        await pool.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    tenant: Optional[str] = "tenant-1"

class CompareRequest(BaseModel):
    contract_ids: List[int]
    tenant: Optional[str] = "tenant-1"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


@app.get("/contracts")
@app.get("/api/contracts")
async def get_contracts(tenant: str = Query("tenant-1")):
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool is not initialized.")
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, tenant_id, title, vendor, value, content, safety_score, rationale, expiration_date, created_at
            FROM contracts
            WHERE tenant_id = $1
            ORDER BY created_at DESC;
            """,
            tenant
        )
        
        contracts = []
        for r in rows:
            safety_score = r["safety_score"]
            rationale = r["rationale"]
            expiration_date = r["expiration_date"] or "2026-12-31"

            if safety_score is None:
                text_content = (r["content"] or "").lower()
                if len(text_content) > 20:
                    critical_risks = ["unlimited liability", "severe penalty", "indemnify without limit", "forfeiture"]
                    found_critical = sum(1 for cr in critical_risks if cr in text_content)
                    
                    if found_critical > 0:
                        safety_score = 55
                        rationale = "⚠ High risk liability or penalty clauses detected."
                    else:
                        safety_score = 88
                        rationale = "✓ No critical anomalies detected"
                else:
                    safety_score = 88
                    rationale = "✓ No critical anomalies detected"

                await conn.execute(
                    "UPDATE contracts SET safety_score = $1, rationale = $2 WHERE id = $3",
                    safety_score, rationale, r["id"]
                )

            contracts.append({
                "id": str(r["id"]),
                "name": r["title"],
                "vendor": r["vendor"],
                "value": float(r["value"]),
                "safety_score": safety_score,
                "risk_score": 100 - safety_score,
                "risk_category": "Low" if safety_score >= 70 else "High",
                "rationale": rationale,
                "expiration_date": expiration_date,
                "renewal_notice_days": 30
            })
            
        return {"contracts": contracts}


@app.post("/contracts/upload")
@app.post("/api/upload")
async def upload_contract(
    title: str = Query(..., description="Contract title"),
    vendor: str = Query(..., description="Vendor name"),
    value: float = Query(..., description="Contract value"),
    file: UploadFile = File(...),
    x_tenant_id: Optional[str] = Header("tenant-1", alias="X-Tenant-ID"),
    tenant: Optional[str] = Query("tenant-1")
):
    active_tenant = x_tenant_id or tenant
    if pool is None or model is None:
        raise HTTPException(status_code=500, detail="Database pool or model is not initialized.")

    try:
        pdf_reader = pypdf.PdfReader(file.file)
        pages_data = []
        full_extracted_text = ""
        
        for idx, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_data.append((idx + 1, text))
                full_extracted_text += text + "\n"

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

    safety_score = 88
    rationale = "✓ No critical anomalies detected"
    expiration_date = "2026-12-31"

    if llm and full_extracted_text.strip():
        try:
            analysis_prompt = f"""
You are an expert corporate legal and procurement analyst. Analyze the following contract text extracted from an uploaded PDF. 
Extract the following details and return them strictly as a JSON object:
1. "safety_score": an integer from 0 to 100 (default 85-95 for standard contracts; below 70 only if severe risks exist).
2. "rationale": a short one-sentence explanation of findings.
3. "expiration_date": the expiration, termination, or end date written in the contract text (formatted as YYYY-MM-DD or string format as written in the text, e.g., '2027-10-15'). If no date is found, default to '2026-12-31'.

Contract Text:
{full_extracted_text[:4000]}

Return ONLY valid JSON format:
{{"safety_score": 90, "rationale": "✓ No critical anomalies detected", "expiration_date": "2027-12-31"}}
"""
            ai_result = llm.invoke(analysis_prompt).content
            json_match = re.search(r"\{.*\}", ai_result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                safety_score = int(parsed.get("safety_score", 88))
                rationale = parsed.get("rationale", rationale)
                expiration_date = parsed.get("expiration_date", "2026-12-31")
        except Exception as e:
            print(f"LLM Parsing Error: {e}")

    doc_embedding_list = model.encode(full_extracted_text[:3500] if full_extracted_text else title).tolist()
    doc_embedding_str = f"[{','.join(map(str, doc_embedding_list))}]"

    async with pool.acquire() as conn:
        async with conn.transaction():
            contract_id = await conn.fetchval(
                """
                INSERT INTO contracts (tenant_id, title, vendor, value, content, safety_score, rationale, expiration_date, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)
                RETURNING id;
                """,
                active_tenant, title, vendor, value, full_extracted_text, safety_score, rationale, expiration_date, doc_embedding_str
            )

            contract_id_int = int(contract_id)

            for page_num, page_text in pages_data:
                chunks = chunk_text(page_text)
                for chunk in chunks:
                    chunk_emb = model.encode(chunk).tolist()
                    chunk_emb_str = f"[{','.join(map(str, chunk_emb))}]"
                    await conn.execute(
                        """
                        INSERT INTO contract_chunks (contract_id, tenant_id, page_number, chunk_text, embedding)
                        VALUES ($1::int, $2, $3, $4, $5::vector);
                        """,
                        contract_id_int, active_tenant, page_num, chunk, chunk_emb_str
                    )

    return {
        "status": "success", 
        "id": contract_id_int, 
        "title": title, 
        "vendor": vendor, 
        "value": value, 
        "safety_score": safety_score,
        "rationale": rationale,
        "expiration_date": expiration_date
    }


@app.post("/query")
@app.post("/api/query")
async def query_contracts(
    request: QueryRequest,
    tenant: Optional[str] = Query("tenant-1"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    active_tenant = x_tenant_id or request.tenant or tenant
    if pool is None or model is None:
        raise HTTPException(status_code=500, detail="Database pool is not initialized.")

    query_vector_list = model.encode(request.query).tolist()
    query_vector_str = f"[{','.join(map(str, query_vector_list))}]"

    async with pool.acquire() as conn:
        chunk_rows = await conn.fetch(
            """
            SELECT 
                c.title, c.vendor, c.value, 
                chk.page_number, chk.chunk_text, 
                (chk.embedding <=> $1::vector) as distance
            FROM contract_chunks chk
            JOIN contracts c ON chk.contract_id = c.id
            WHERE chk.tenant_id = $2
            ORDER BY distance ASC
            LIMIT 3;
            """,
            query_vector_str, active_tenant
        )

    if not chunk_rows:
        return {"found": False, "answer": "No relevant contracts found for your query in this tenant partition."}

    top_row = chunk_rows[0]
    citations = []
    context_chunks = []

    for r in chunk_rows:
        context_chunks.append(f"[Page {r['page_number']}]\n{r['chunk_text']}")
        citations.append({
            "page": r["page_number"],
            "excerpt": (r["chunk_text"][:180] + "...") if r["chunk_text"] else "",
            "similarity_score": round(1 - float(r["distance"]), 2)
        })

    combined_context = "\n\n".join(context_chunks)
    ai_response = None
    current_date_str = date.today().isoformat()

    if llm:
        try:
            prompt_text = f"""
You are an expert AI procurement analyst and legal risk assessor. 
Current Reference Date: {current_date_str}.

Analyze the provided contract context carefully to:
1. Accurately answer the user's question.
2. Actively scan for and flag any legal, financial, or operational risks.

User Question: {request.query}

Contract Context:
{combined_context}
"""
            ai_response = llm.invoke(prompt_text).content
        except Exception as e:
            print(f"Groq LLM Execution Error: {e}")

    if not ai_response:
        ai_response = f"**Summary for {top_row['title']}**\nVendor: {top_row['vendor']}\nValue: ${float(top_row['value']):,.2f}"

    return {
        "found": True,
        "answer": ai_response,
        "citations": citations
    }


@app.post("/compare")
@app.post("/api/compare")
async def compare_contracts(request: CompareRequest):
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool is not initialized.")
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, vendor, value, content 
            FROM contracts 
            WHERE id = ANY($1::int[]);
            """,
            request.contract_ids
        )
        
    if len(rows) < 2:
        return {"ai_summary": "Selected contracts could not be found for delta comparison."}
        
    contract_a = rows[0]
    contract_b = rows[1]

    comparison_text = f"""
Contract A: {contract_a['title']} (Vendor: {contract_a['vendor']}, Value: ${contract_a['value']})
Content Excerpt: {(contract_a['content'] or '')[:1500]}

Contract B: {contract_b['title']} (Vendor: {contract_b['vendor']}, Value: ${contract_b['value']})
Content Excerpt: {(contract_b['content'] or '')[:1500]}
"""
    
    ai_summary = f"Comparing {contract_a['title']} vs {contract_b['title']}"
    if llm:
        try:
            prompt = f"""
You are an expert legal and procurement risk analyst. Perform a detailed legal and financial variance audit comparing these two contracts. 

CRITICAL FORMATTING RULES:
- DO NOT use markdown table syntax. Do NOT use pipe characters (`|`).
- Use clear Markdown headings (###) and bullet points (*).
- For each dimension (e.g., Jurisdiction, Termination, Liability), format it cleanly as:
  * **Dimension Name:** Contract A details vs Contract B details.
- Structure your response into clear sections: Executive Summary, Key Financial Differences, Risk & Liability Variance, and Bottom-line Recommendation.

Data to compare:
{comparison_text}
"""
            res = llm.invoke(prompt)
            ai_summary = res.content
        except Exception as e:
            print(f"Comparison LLM Error: {e}")
            
    return {"ai_summary": ai_summary}