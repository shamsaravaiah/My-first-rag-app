# 📁 app/chains/qa_chain.py

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

# Load Gemini API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_KEY")

# Embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Load FAISS vector store
vectorstore = FAISS.load_local(
    "data/vector_store/aprobo_en",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Optional: Customize prompt to match your assistant’s voice
CUSTOM_PROMPT = PromptTemplate.from_template("""
You are Aprobo's intelligent product assistant, designed to help customers—whether beginners or professionals—find the ideal flooring solution from Aprobo's catalog.

Use the context provided (retrieved from our product database and website) to:
- Understand the customer's needs, even if they are vague, confused, or unfamiliar with flooring terms.
- Ask clarifying follow-up questions **only if absolutely necessary**.
- Recommend the most suitable product(s) from Aprobo’s catalog based on their needs.
- Explain your recommendations clearly, highlighting key product features, differences, and benefits in simple terms.
- Link directly to the recommended product(s) on https://aprobo.com/en/produkter/.
- Avoid guessing or making up information. Base your answers only on the provided context.

Context:
{context}

Customer Question:
{question}

Response:
""")

# LLM (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

# Expose it as a function
def answer_question(query: str) -> str:
    result = qa_chain.invoke({"query": query})
    return result["result"]




#Loadenv – Load Gemini API key

#Embedder – Set up embedding model

#Vectorstore – Load FAISS index

#Retriever – Enable vector search

#Prompt – Customize assistant behavior

#LLM – Set up Gemini Pro

#Chain – Create RetrievalQA pipeline

#Function – Wrap it as answer_question()

