from pydantic import BaseModel, ConfigDict
from datetime import datetime
from uuid import UUID


class ConversationResponse(BaseModel):
    id: UUID
    title: str | None
    summary: str | None
    created_at: datetime
    updated_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationHistoryResponse(BaseModel):
    id: UUID
    title: str | None
    summary: str | None
    created_at: datetime
    updated_at: datetime | None
    messages: list[MessageResponse]

    model_config = ConfigDict(from_attributes=True)