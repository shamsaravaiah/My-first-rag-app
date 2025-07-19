import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load your .env file
load_dotenv()
api_key = os.getenv("GEMINI_KEY")
assert api_key, "GEMINI_KEY missing in .env"

# File locations
jsonl_file = "/Users/sham_sara/Desktop/aprobo-chatbot/data/aprobo_docs/aprobo_embeddings_ready_20250719_132149.jsonl"  # change if needed
vectorstore_dir = "data/vector_store/aprobo_en"

# Step 1: Load JSONL
documents = []
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("text"):
            documents.append(Document(
                page_content=item["text"],
                metadata={
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "type": item.get("type", "")
                }
            ))

# Step 2: Chunk the content
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Step 3: Embed with Gemini
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Step 4: Store in FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local(vectorstore_dir)

print(f"✅ Stored {len(chunks)} chunks into FAISS at: {vectorstore_dir}")
