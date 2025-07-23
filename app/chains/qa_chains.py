# 📁 app/chains/qa_chains.py

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_KEY")

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Shared prompt template (can customize per company)
def get_prompt(company_name):
    return PromptTemplate.from_template(f"""
You are {company_name}'s intelligent assistant.

Use the context to help users with product or service questions. Be honest. Do not make up facts.

Context:
{{context}}

Customer Question:
{{question}}

Response:
""")

# Load vectorstore + chain with memory and limit tracking
def load_company_chain(company_id: str):
    vector_path = f"data/vector_store/{company_id}"
    vs = FAISS.load_local(vector_path, embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    prompt = get_prompt(company_id.capitalize())  # this expects 'context' and 'question'

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt}  # ✅ correct place to inject your prompt
    )

    return {"chain": chain, "memory": memory, "question_count": 0}

# Dict to track sessions per company
company_chains = {
    "aprobo": load_company_chain("aprobo_en"),
    "stim": load_company_chain("stim"),
    "youngstival": load_company_chain("youngstival"),
}

# Main function with limit enforcement
def answer_company_question(company: str, question: str) -> str:
    session = company_chains[company]
    session["question_count"] += 1

    if session["question_count"] > 5:
        return "You've reached the free limit. Want more? Contact us."

    result = session["chain"].invoke({"question": question})
    return result["answer"] if "answer" in result else result["result"]
