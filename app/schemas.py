from typing import List, Optional

from pydantic import BaseModel, Field, constr


NonEmptyText = constr(strip_whitespace=True, min_length=1)


class Message(BaseModel):
    role: NonEmptyText
    content: NonEmptyText
    timestamp: Optional[int] = None


class AddRequest(BaseModel):
    request_id: NonEmptyText
    messages: List[Message] = Field(..., min_items=1)
    user_id: NonEmptyText
    session_id: NonEmptyText


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: NonEmptyText
    options: Optional[List[str]] = None
    user_id: NonEmptyText
    top_k: int = Field(100, ge=1, le=100)


class MemoryResult(BaseModel):
    id: str
    content: str
    score: float
    created_at: str


class SearchResponse(BaseModel):
    data: List[MemoryResult]
