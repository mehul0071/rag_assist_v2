from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from typing import List, Dict
import statistics
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings


async def run_ragas_evaluation(examples: List[Dict]) -> Dict:
    if not examples:
        return {"error": "No examples provided"}

    dataset = Dataset.from_list(examples)

    eval_llm = ChatGroq(
        model=settings.LLM_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.0
    )
    eval_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings
    )

    scores = result.scores

    faithfulness_scores = [s["faithfulness"] for s in scores]
    relevancy_scores = [s["answer_relevancy"] for s in scores]
    precision_scores = [s["context_precision"] for s in scores]

    return {
        "total_examples": len(examples),
        "aggregate": {
            "faithfulness": {
                "mean": round(statistics.mean(faithfulness_scores), 4),
                "median": round(statistics.median(faithfulness_scores), 4),
                "min": round(min(faithfulness_scores), 4),
                "max": round(max(faithfulness_scores), 4),
                "std_dev": round(statistics.stdev(faithfulness_scores), 4) if len(faithfulness_scores) > 1 else 0
            },
            "answer_relevancy": {
                "mean": round(statistics.mean(relevancy_scores), 4),
                "median": round(statistics.median(relevancy_scores), 4),
            },
            "context_precision": {
                "mean": round(statistics.mean(precision_scores), 4),
            }
        },
        "individual_scores": scores,
        "recommendation": get_recommendation(faithfulness_scores, relevancy_scores)
    }


def get_recommendation(faith: List[float], relev: List[float]) -> str:
    avg_faith = statistics.mean(faith)
    avg_rel = statistics.mean(relev)

    if avg_faith > 0.9 and avg_rel > 0.9:
        return "Excellent"
    elif avg_faith > 0.8 and avg_rel > 0.8:
        return "Good"
    elif avg_faith > 0.7 and avg_rel > 0.7:
        return "Fair"
    else:
        return "Needs Improvement"