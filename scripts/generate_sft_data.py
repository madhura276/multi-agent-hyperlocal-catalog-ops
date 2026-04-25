from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import choose_action
from server.environment import MultiAgentHyperlocalCatalogOpsEnvironment
from tasks import TASK_ORDER


OUTPUT_PATH = PROJECT_ROOT / "data" / "oversight_sft.jsonl"
TARGET_AGENT = "oversight_agent"


def build_prompt(env: MultiAgentHyperlocalCatalogOpsEnvironment) -> str:
    state = env.state
    task = TASK_ORDER[0] if state.task_id is None else state.task_id

    payload = {
        "task_id": task,
        "active_agent": state.current_agent.value,
        "remaining_steps": state.remaining_steps,
        "records": [
            {
                "record_id": record.record_id,
                "store_id": record.store_id,
                "raw_title": record.raw_title,
                "normalized_title": record.normalized_title,
                "category": record.category.value if record.category else None,
                "quantity_value": record.quantity_value,
                "quantity_unit": record.quantity_unit.value if record.quantity_unit else None,
                "pack_count": record.pack_count,
                "price": record.price,
                "status": record.status.value,
                "notes": record.notes,
            }
            for record in state.records
        ],
        "pending_proposals": [
            {
                "proposal_id": proposal.proposal_id,
                "proposer": proposal.proposer.value,
                "action_type": proposal.action_type.value,
                "record_id": proposal.record_id,
                "secondary_record_id": proposal.secondary_record_id,
                "field_name": proposal.field_name,
                "value": proposal.value,
                "reason": proposal.reason,
                "status": proposal.status.value,
            }
            for proposal in state.pending_proposals
        ],
    }

    return (
        "You are the oversight agent in a multi-agent hyperlocal catalog ops environment.\n"
        "Choose exactly one safe JSON action.\n"
        "Return only minified JSON.\n\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def build_completion(action) -> str:
    payload = {
        "agent_role": action.agent_role.value,
        "action_type": action.action_type.value,
        "proposal_id": action.proposal_id,
        "record_id": action.record_id,
        "secondary_record_id": action.secondary_record_id,
        "field_name": action.field_name,
        "value": action.value,
        "reason": action.reason,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for task_id in TASK_ORDER:
            env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id=task_id)
            observation = env.reset()

            while not observation.done:
                if env.state.current_agent.value == TARGET_AGENT:
                    prompt = build_prompt(env)
                    action = choose_action(env)
                    completion = build_completion(action)
                    handle.write(
                        json.dumps(
                            {"prompt": prompt, "completion": completion},
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    rows_written += 1
                else:
                    action = choose_action(env)

                observation = env.step(action)

    print(f"Wrote {rows_written} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
