import os
from dotenv import load_dotenv

load_dotenv()

tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def setup_tracing():
    if tracing_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-assistant-prod")
        print("LangSmith tracing enabled and configured")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("LangSmith tracing is disabled (set LANGCHAIN_TRACING_V2=true in .env to enable)")