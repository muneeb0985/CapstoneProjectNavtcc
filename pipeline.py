import io
from typing import List
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


encoder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_and_chunk_pdf(file_bytes: bytes, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    pdf = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    chunk_index = 0

    full_text_pages = []
    for page_num, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            full_text_pages.append((page_num + 1, text))

    for page_num, text in full_text_pages:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "page": page_num
                })
                chunk_index += 1

    return chunks

def generate_embedding(text: str) -> List[float]:
    """Generates a 384-dimensional vector embedding for text."""
    return encoder.encode(text).tolist()
