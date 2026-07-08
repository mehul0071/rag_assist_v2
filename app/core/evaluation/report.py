from typing import Dict, List
from datetime import datetime
from pathlib import Path
from datetime import datetime
import json



def generate_evaluation_report(results: List[Dict]) -> Dict:
    if not results:
        return {"status": "no_data", "message": "No evaluation data yet"}

    total = len(results)
    avg_faithfulness = sum(r.get("faithfulness", 0) for r in results) / total
    avg_relevancy = sum(r.get("answer_relevancy", 0) for r in results) / total

    return {
        "report_generated_at": datetime.now().isoformat(),
        "total_evaluations": total,
        "average_scores": {
            "faithfulness": round(avg_faithfulness, 3),
            "answer_relevancy": round(avg_relevancy, 3),
        },
        "recent_results": results[-5:],
        "recommendation": "Good" if avg_faithfulness > 0.85 else "Needs Improvement"
    }


def save_evaluation_report(report: dict, name: str = None):
    reports_dir = Path("evaluation_reports")
    reports_dir.mkdir(exist_ok=True)

    filename = name or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = reports_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return path