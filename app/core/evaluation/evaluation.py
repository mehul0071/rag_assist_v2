from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
from app.core.evaluation.dataset import load_benchmark_dataset
from app.core.evaluation.ragas_runner import run_ragas_evaluation
from app.core.evaluation.report import generate_evaluation_report

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])


@router.post("/run")
async def run_evaluation(examples: List[Dict] = None):
    if not examples:
        raise HTTPException(status_code=400, detail="Examples are required")

    result = await run_ragas_evaluation(examples)
    return result


@router.post("/run-benchmark")
async def run_benchmark_evaluation():
    try:
        examples = await load_benchmark_dataset()
        result = await run_ragas_evaluation(examples)
        
        return {
            "status": "success",
            "dataset": "benchmark",
            "total_questions": len(examples),
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/report")
async def get_evaluation_report():
    sample_results = [
        {"faithfulness": 0.92, "answer_relevancy": 0.88, "context_precision": 0.91},
        {"faithfulness": 0.95, "answer_relevancy": 0.91, "context_precision": 0.93},
        {"faithfulness": 0.89, "answer_relevancy": 0.85, "context_precision": 0.87}
    ]

    return generate_evaluation_report(sample_results)


@router.get("/benchmark")
async def get_benchmark_info():
    return {
        "available_datasets": ["benchmark"],
        "total_questions": 4,
        "description": "Standard Machine Learning concepts benchmark"
    }