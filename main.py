from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from config import settings
from api.models import ChatRequest, ChatResponse, SummarizeRequest, SummarizeResponse, EmbedRequest, EmbedResponse, HealthResponse
from services import llm_service

app = FastAPI(title="FastAPI LLM Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
security = HTTPBearer()

def verify_key(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != settings.service_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return creds.credentials

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", openai_configured=bool(settings.openai_api_key),
                          anthropic_configured=bool(settings.anthropic_api_key))

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _=Depends(verify_key)):
    model = req.model or settings.default_model
    result = await llm_service.chat_completion(req.message, model, req.system_prompt, req.temperature, req.max_tokens)
    return ChatResponse(response=result["text"], model=model, tokens_used=result.get("tokens"))

@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest, _=Depends(verify_key)):
    model = req.model or settings.default_model
    async def gen():
        async for t in llm_service.stream_completion(req.message, model, req.system_prompt, req.temperature, req.max_tokens):
            yield f"data: {t}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/v1/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest, _=Depends(verify_key)):
    styles = {"concise": "Summarise in 2-3 sentences.", "detailed": "Detailed summary of all key points.",
              "bullet": "Bullet-point list of key facts."}
    system = styles.get(req.style, styles["concise"]) + f" Respond in {req.language}."
    result = await llm_service.chat_completion(req.text, settings.default_model, system, 0.3, 512)
    return SummarizeResponse(summary=result["text"], original_length=len(req.text),
                             summary_length=len(result["text"]), compression_ratio=round(len(result["text"])/len(req.text), 3))

@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, _=Depends(verify_key)):
    embs = await llm_service.get_embeddings(req.texts)
    return EmbedResponse(embeddings=embs, model="text-embedding-3-small", dimensions=len(embs[0]) if embs else 0)
