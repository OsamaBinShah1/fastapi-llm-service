from typing import Optional, List
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful AI assistant."
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: Optional[int] = None

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10)
    style: Optional[str] = "concise"
    language: Optional[str] = "English"

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    openai_configured: bool
    anthropic_configured: bool
