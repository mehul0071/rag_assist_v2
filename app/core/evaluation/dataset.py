from pathlib import Path
import json
from typing import List, Dict


async def load_benchmark_dataset() -> List[Dict]:
    path = Path("app/core/evaluation/benchmark.json")
    
    if not path.exists():
        raise FileNotFoundError("Benchmark dataset not found. Please create it.")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)