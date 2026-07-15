from pydantic import BaseModel, model_validator, ConfigDict
from datetime import datetime
from uuid import UUID


class ConversationResponse(BaseModel):
    id: UUID
    title: str | None
    summary: str | None
    created_at: datetime
    updated_at: datetime | None

    @model_validator(mode='before')
    @classmethod
    def populate_defaults(cls, data):
        if isinstance(data, dict):
            if not data.get("title"):
                conv_id = str(data.get("id", ""))
                data["title"] = f"Conversation {conv_id[:8]}" if conv_id else "New Conversation"
            if not data.get("summary"):
                data["summary"] = "No summary available yet."
            if data.get("updated_at") is None:
                data["updated_at"] = data.get("created_at")
        else:
            conv_id = str(getattr(data, "id", ""))
            return {
                "id": getattr(data, "id"),
                "title": getattr(data, "title") or (f"Conversation {conv_id[:8]}" if conv_id else "New Conversation"),
                "summary": getattr(data, "summary") or "No summary available yet.",
                "created_at": getattr(data, "created_at"),
                "updated_at": getattr(data, "updated_at") or getattr(data, "created_at")
            }
        return data

    model_config = ConfigDict(from_attributes=True)