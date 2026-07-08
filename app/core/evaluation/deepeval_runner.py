from deepeval import evaluate
from deepeval.metrics import (HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric, GEval)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import List, Dict
from langchain_groq import ChatGroq
from app.config.settings import settings


class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name=None):
        self.model_name = model_name or settings.LLM_MODEL
        self.llm = ChatGroq(
            model=self.model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=0.0
        )
        
    def load_model(self):
        return self.llm
        
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name


async def run_deepeval(examples: List[Dict]) -> Dict:
    test_cases = []

    for example in examples:
        test_case = LLMTestCase(
            input=example["question"],
            actual_output=example.get("answer", ""),
            expected_output=example.get("ground_truth", ""),
            retrieval_context=example.get("contexts", [])
        )
        test_cases.append(test_case)

    eval_model = GroqDeepEvalModel()

    metrics = [
        HallucinationMetric(threshold=0.7, model=eval_model),
        AnswerRelevancyMetric(threshold=0.7, model=eval_model),
        FaithfulnessMetric(threshold=0.7, model=eval_model),
        GEval(
            name="Coherence",
            criteria="The response is coherent, well-structured, and flows naturally.",
            evaluation_params=["actual_output"],
            model=eval_model
        )
    ]

    result = evaluate(test_cases, metrics)

    scores = {}
    for metric in result:
        scores[metric.__class__.__name__] = {
            "score": metric.score,
            "reason": metric.reason
        }

    return {
        "framework": "DeepEval",
        "total_test_cases": len(test_cases),
        "scores": scores,
        "passed": sum(1 for m in result if m.score >= 0.7)
    }