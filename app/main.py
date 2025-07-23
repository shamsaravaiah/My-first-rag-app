# 📁 main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.chains.qa_chains import answer_company_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask/aprobo/")
async def ask_aprobo(query: Query):
    answer = answer_company_question("aprobo", query.question)
    return {"answer": answer}

@app.post("/ask/stim/")
async def ask_company_b(query: Query):
    answer = answer_company_question("stim", query.question)
    return {"answer": answer}

@app.post("/ask/youngstival/")
async def ask_company_c(query: Query):
    answer = answer_company_question("youngstival", query.question)
    return {"answer": answer}
