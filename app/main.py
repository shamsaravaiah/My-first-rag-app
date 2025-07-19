from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.chains.qa_chains import answer_question  # ✅ This works now after your fix

app = FastAPI()

# ✅ Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # For dev — lock to domain in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask/")
async def ask(query: Query):
    answer = answer_question(query.question)
    return {"answer": answer}
