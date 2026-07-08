from dataclasses import dataclass
from app.config.settings import settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    context_window: int
    response_tokens: int
    tokenizer: str | None = None


class LLMConfig:
    MODELS = {
        "llama-3.3-70b-versatile": ModelSpec(
            name="llama-3.3-70b-versatile",
            provider="groq",
            context_window=131072,
            response_tokens=4096,
            tokenizer="llama",
        ),

        "llama-3.1-8b-instant": ModelSpec(
            name="llama-3.1-8b-instant",
            provider="groq",
            context_window=131072,
            response_tokens=4096,
            tokenizer="llama",
        ),

        "deepseek-r1-distill-llama-70b": ModelSpec(
            name="deepseek-r1-distill-llama-70b",
            provider="groq",
            context_window=131072,
            response_tokens=4096,
            tokenizer="llama",
        ),

        "qwen-qwq-32b": ModelSpec(
            name="qwen-qwq-32b",
            provider="groq",
            context_window=131072,
            response_tokens=4096,
            tokenizer="qwen",
        ),
    }

    @classmethod
    def get(cls, model_name: str | None = None):
        model_name = model_name or settings.LLM_MODEL
        return cls.MODELS.get(model_name, cls.MODELS["llama-3.3-70b-versatile"])


llm_config = LLMConfig()