from __future__ import annotations

from typing import Dict, List

from inference import run_task
from tasks import TASK_ORDER


def summarize(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def run_baseline() -> List[float]:
    scores = []
    for task_id in TASK_ORDER:
        score = run_task(task_id)
        scores.append(score)
    return scores


def main() -> None:
    print("[EVAL] running baseline heuristic policy", flush=True)
    baseline_scores = run_baseline()
    baseline_summary = summarize(baseline_scores)

    print(
        f"[RESULT] baseline_avg={baseline_summary['avg']:.4f} "
        f"baseline_min={baseline_summary['min']:.4f} "
        f"baseline_max={baseline_summary['max']:.4f}",
        flush=True,
    )

    print(
        "[NOTE] add trained-model inference hook here after onsite fine-tuning "
        "to compare baseline vs trained oversight policy.",
        flush=True,
    )


if __name__ == "__main__":
    main()
