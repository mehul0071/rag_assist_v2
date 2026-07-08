from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document


class GraphState(TypedDict):    
    question: str
    conversation_id: Optional[str]
    chat_history: str
    intent: str
    retrieved_docs: List[Document]
    answer: Optional[str]
    sources: List[Dict]
    metadata: Dict[str, Any]
    error: Optional[str]

