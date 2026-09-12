import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.pipeline import CustomerSupportPipeline

app = FastAPI(
    title="RAG-Based E-commerce Customer Support Chatbot API",
    description="End-to-end NLP Customer Support Pipeline featuring Language Detection, Sentiment Analysis, Intent Classification, and Grounded RAG with Groq LLM.",
    version="1.0.0"
)

# Initialize pipeline as singleton
pipeline: Optional[CustomerSupportPipeline] = None

@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = CustomerSupportPipeline(models_dir="models")

class ChatRequest(BaseModel):
    message: str = Field(..., example="Where is my order #12345?")

class AnalyzeRequest(BaseModel):
    text: str = Field(..., example="I want to return this broken item!")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "RAG-Based E-commerce Customer Support Chatbot",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "analyze": "/analyze (POST)"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "device": str(pipeline.device) if pipeline else "unknown"
    }

@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    lang, lang_conf = pipeline.detect_language(req.text)
    sentiment, sent_conf, sent_probs = pipeline.detect_sentiment(req.text)
    intent, intent_conf = pipeline.detect_intent(req.text)
    
    return {
        "text": req.text,
        "language": {"code": lang, "confidence": lang_conf},
        "sentiment": {"label": sentiment, "confidence": sent_conf, "probabilities": sent_probs},
        "intent": {"label": intent, "confidence": intent_conf}
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    result = pipeline.process_message(req.message)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
