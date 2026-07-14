from sqlalchemy import Column, String, ForeignKey, DateTime, UUID
from sqlalchemy.sql import func
from uuid import uuid4
from app.core.database import Base


class UserFact(Base):
    __tablename__ = "user_facts"

    id = Column(UUID, primary_key=True, default=uuid4)
    conversation_id = Column(UUID, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    fact = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
