from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate


class PromptManager:
    
    def __init__(self):
        self.templates_dir = Path("app/core/prompts/templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)


    def get_rag_prompt(self) -> ChatPromptTemplate:
        file_path = self.templates_dir / "rag_answer.md"
        
        if not file_path.exists():
            template = """
                You are an intelligent assistant.

                Previous Conversation:
                {chat_history}

                Relevant Information:
                {context}

                Question: {question}

                Answer concisely using only the provided information.
            """
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                template = f.read()

        return ChatPromptTemplate.from_template(template)

    