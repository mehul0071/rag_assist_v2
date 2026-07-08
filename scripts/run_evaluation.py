import asyncio
import json
from pathlib import Path
from datetime import datetime
from app.core.evaluation.ragas_runner import run_ragas_evaluation
from app.core.evaluation.deepeval_runner import run_deepeval
from app.core.evaluation.dataset import load_benchmark_dataset


async def main():
    print("Starting RAG System Evaluation...\n")

    examples = await load_benchmark_dataset()
    print(f"Loaded {len(examples)} benchmark questions\n")

    print("Running RAGAS Evaluation...")
    ragas_result = await run_ragas_evaluation(examples)
    print(f"RAGAS - Faithfulness: {ragas_result['aggregate']['faithfulness']['mean']:.3f}")

    print("\nRunning DeepEval...")
    deepeval_result = await run_deepeval(examples)
    print(f"DeepEval - Hallucination Score: {deepeval_result['scores'].get('HallucinationMetric', {}).get('score', 0):.3f}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "ragas": ragas_result,
        "deepeval": deepeval_result,
        "total_questions": len(examples)
    }

    reports_dir = Path("evaluation_reports")
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete! Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())