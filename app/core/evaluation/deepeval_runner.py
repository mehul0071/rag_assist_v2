from deepeval import evaluate
from deepeval.metrics import (HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric, GEval)
from deepeval.test_case import LLMTestCase
from typing import List, Dict


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

    metrics = [
        HallucinationMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        GEval(
            name="Coherence",
            criteria="The response is coherent, well-structured, and flows naturally.",
            evaluation_params=["actual_output"]
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