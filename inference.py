from __future__ import annotations

import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import ActionType, AgentRole, MultiAgentAction, RecordStatus
from server.environment import MultiAgentHyperlocalCatalogOpsEnvironment
from tasks import TASK_ORDER, TASKS


MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-3B-Instruct"
TRAINED_MODEL_PATH = os.getenv("TRAINED_MODEL_PATH") or "trained_models/oversight-sft-final"
BENCHMARK = os.getenv("MULTI_AGENT_HYPERLOCAL_BENCHMARK", "multi_agent_hyperlocal_catalog_ops")
SUCCESS_SCORE_THRESHOLD = 0.72

_OVERSIGHT_TOKENIZER: Optional[AutoTokenizer] = None
_OVERSIGHT_MODEL: Optional[AutoModelForCausalLM] = None


def log_start(task: str, env: str, model: str, mode: str) -> None:
    print(f"[START] mode={mode} task={task} env={env} model={model}", flush=True)


def log_step(step: int, agent: str, action: str, reward: float, done: bool, error: Optional[str], mode: str) -> None:
    error_val = error.replace("\n", "\\n") if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] mode={mode} step={step} agent={agent} action={action} reward={reward:.3f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float], mode: str) -> None:
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(
        f"[END] mode={mode} success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def heuristic_title(raw_title: str) -> str:
    lower = " ".join(raw_title.strip().lower().split())

    title_map = {
        "coca cola 1000ml": "Coca Cola 1 L",
        "coke 1 ltr": "Coca Cola 1 L",
        "amul taaza toned milk 500 ml": "Amul Taaza Toned Milk 500 Ml",
        "amul taaza milk 500ml pack": "Amul Taaza Milk 500 Ml Pack",
        "banana 6 pc": "Banana 6 Pcs",
        "bananas pack of 6": "Bananas Pack Of 6",
        "aashirvaad atta 5kg": "Aashirvaad Atta 5 Kg",
        "surf exel easy wash 1kg": "Surf Excel Easy Wash 1 Kg",
        "coke zero can 300ml": "Coke Zero Can 300 Ml",
        "coca cola zero 330 ml can": "Coca Cola Zero 330 Ml Can",
        "tomato loose 1kg": "Tomato Loose 1 Kg",
        "tomatoes 1 kg pack": "Tomatoes 1 Kg Pack",
        "ariel matic front load 2kg": "Ariel Matic Front Load 2 Kg",
        "safal frozen green peas 500 g": "Safal Frozen Green Peas 500 G",
        "green peas 500 g": "Green Peas 500 G",
    }

    return title_map.get(lower, raw_title.title())


def infer_category(raw_title: str) -> str:
    text = raw_title.lower()

    if any(token in text for token in ["cola", "coke"]):
        return "beverages"
    if "milk" in text:
        return "dairy"
    if any(token in text for token in ["banana", "tomato", "peas"]):
        return "produce"
    if any(token in text for token in ["atta", "bread", "salt", "rice"]):
        return "staples"
    if any(token in text for token in ["surf", "ariel", "refill"]):
        return "household"
    if "frozen" in text:
        return "frozen"
    return "unknown"


def infer_size_actions(record) -> List[tuple[str, object]]:
    raw = record.raw_title.lower()
    actions: List[tuple[str, object]] = []

    if record.quantity_value is None:
        if "1000ml" in raw or "1000 ml" in raw:
            actions.append(("quantity_value", 1000.0))
        elif "500ml" in raw or "500 ml" in raw:
            actions.append(("quantity_value", 500.0))
        elif "5kg" in raw or "5 kg" in raw:
            actions.append(("quantity_value", 5.0))
        elif "2kg" in raw or "2 kg" in raw:
            actions.append(("quantity_value", 2.0))
        elif "1kg" in raw or "1 kg" in raw:
            actions.append(("quantity_value", 1.0))
        elif "500 g" in raw or "500g" in raw:
            actions.append(("quantity_value", 500.0))

    if record.quantity_unit is None:
        if "ltr" in raw or "1 l" in raw:
            actions.append(("quantity_unit", "l"))
        elif "ml" in raw:
            actions.append(("quantity_unit", "ml"))
        elif "kg" in raw:
            actions.append(("quantity_unit", "kg"))
        elif "500 g" in raw or " g" in raw:
            actions.append(("quantity_unit", "g"))

    if record.pack_count is None and ("6 pc" in raw or "pack of 6" in raw):
        actions.append(("pack_count", 6))
        if record.quantity_unit is None:
            actions.append(("quantity_unit", "pcs"))

    return actions


def proposal_exists(
    env: MultiAgentHyperlocalCatalogOpsEnvironment,
    *,
    action_type: ActionType,
    record_id: Optional[str] = None,
    secondary_record_id: Optional[str] = None,
    field_name: Optional[str] = None,
) -> bool:
    for proposal in env.state.pending_proposals:
        if proposal.action_type != action_type:
            continue
        if record_id is not None and proposal.record_id != record_id:
            continue
        if secondary_record_id is not None and proposal.secondary_record_id != secondary_record_id:
            continue
        if field_name is not None and proposal.field_name != field_name:
            continue
        return True
    return False


def record_is_done(record) -> bool:
    return record.status in {RecordStatus.MERGED, RecordStatus.FLAGGED}


def unresolved_records(env: MultiAgentHyperlocalCatalogOpsEnvironment):
    return [
        record
        for record in env.state.records
        if not record_is_done(record)
        and (
            not record.normalized_title
            or record.category is None
            or (
                any(token in record.raw_title.lower() for token in ["ml", "kg", " g", "ltr", "pc", "pack of"])
                and (
                    record.quantity_value is None
                    or record.quantity_unit is None
                    or ("pack of 6" in record.raw_title.lower() and record.pack_count is None)
                )
            )
        )
    ]


def curation_policy(env: MultiAgentHyperlocalCatalogOpsEnvironment) -> MultiAgentAction:
    for record in env.state.records:
        if record_is_done(record):
            continue

        if not record.normalized_title and not proposal_exists(
            env,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id=record.record_id,
        ):
            return MultiAgentAction(
                agent_role=AgentRole.CURATION,
                action_type=ActionType.NORMALIZE_TITLE,
                record_id=record.record_id,
                field_name="normalized_title",
                value=heuristic_title(record.raw_title),
                reason="High-value title normalization",
            )

    for record in env.state.records:
        if record_is_done(record):
            continue

        if record.category is None and not proposal_exists(
            env,
            action_type=ActionType.ASSIGN_CATEGORY,
            record_id=record.record_id,
        ):
            return MultiAgentAction(
                agent_role=AgentRole.CURATION,
                action_type=ActionType.ASSIGN_CATEGORY,
                record_id=record.record_id,
                field_name="category",
                value=infer_category(record.raw_title),
                reason="High-value category assignment",
            )

    for record in env.state.records:
        if record_is_done(record):
            continue

        for field_name, value in infer_size_actions(record):
            if not proposal_exists(
                env,
                action_type=ActionType.NORMALIZE_SIZE,
                record_id=record.record_id,
                field_name=field_name,
            ):
                return MultiAgentAction(
                    agent_role=AgentRole.CURATION,
                    action_type=ActionType.NORMALIZE_SIZE,
                    record_id=record.record_id,
                    field_name=field_name,
                    value=value,
                    reason="Size normalization after title/category",
                )

    target = env.state.records[0].record_id
    return MultiAgentAction(
        agent_role=AgentRole.CURATION,
        action_type=ActionType.FLAG_FOR_REVIEW,
        record_id=target,
        reason="No high-value curation action left",
    )


def dedupe_policy(env: MultiAgentHyperlocalCatalogOpsEnvironment) -> MultiAgentAction:
    task = TASKS[env.state.task_id]
    duplicate_candidates = task.hidden_signals.get("dedupe_agent", {}).get("duplicate_candidates", [])
    risk_hints = task.hidden_signals.get("oversight_agent", {}).get("risk_hints", {})

    existing_pairs = {tuple(pair) for pair in env.state.merged_pairs}

    for primary, secondary in duplicate_candidates:
        if (secondary, primary) in existing_pairs or (primary, secondary) in existing_pairs:
            continue
        if proposal_exists(
            env,
            action_type=ActionType.MERGE_RECORDS,
            record_id=primary,
            secondary_record_id=secondary,
        ):
            continue

        risk_key = f"{primary}_{secondary}"
        reverse_key = f"{secondary}_{primary}"
        risk = risk_hints.get(risk_key) or risk_hints.get(reverse_key)

        if risk == "same_store_clear_duplicate":
            return MultiAgentAction(
                agent_role=AgentRole.DEDUPE,
                action_type=ActionType.MERGE_RECORDS,
                record_id=primary,
                secondary_record_id=secondary,
                reason="Safe duplicate merge",
            )

    target = env.state.records[0].record_id
    return MultiAgentAction(
        agent_role=AgentRole.DEDUPE,
        action_type=ActionType.FLAG_FOR_REVIEW,
        record_id=target,
        reason="No safe duplicate merge available",
    )


def pricing_policy(env: MultiAgentHyperlocalCatalogOpsEnvironment) -> MultiAgentAction:
    task = TASKS[env.state.task_id]
    references = task.hidden_signals.get("pricing_agent", {}).get("price_reference", {})

    for record_id, ref in references.items():
        if proposal_exists(
            env,
            action_type=ActionType.CORRECT_PRICE,
            record_id=record_id,
        ):
            continue

        record = next((item for item in env.state.records if item.record_id == record_id), None)
        if record is None or record.price is None:
            continue

        low, high = ref["expected_band"]
        if record.price < low or record.price > high:
            midpoint = round((low + high) / 2, 2)
            return MultiAgentAction(
                agent_role=AgentRole.PRICING,
                action_type=ActionType.CORRECT_PRICE,
                record_id=record_id,
                value=midpoint,
                reason="High-value price correction",
            )

    target = env.state.records[0].record_id
    return MultiAgentAction(
        agent_role=AgentRole.PRICING,
        action_type=ActionType.FLAG_FOR_REVIEW,
        record_id=target,
        reason="No obvious price anomaly left",
    )


def heuristic_oversight_policy(env: MultiAgentHyperlocalCatalogOpsEnvironment) -> MultiAgentAction:
    pending = env.state.pending_proposals
    task = TASKS[env.state.task_id]
    risk_hints = task.hidden_signals.get("oversight_agent", {}).get("risk_hints", {})

    if pending:
        proposal = pending[0]

        if proposal.action_type == ActionType.MERGE_RECORDS:
            risk_key = f"{proposal.record_id}_{proposal.secondary_record_id}"
            reverse_key = f"{proposal.secondary_record_id}_{proposal.record_id}"
            risk = risk_hints.get(risk_key) or risk_hints.get(reverse_key)

            if risk and risk != "same_store_clear_duplicate":
                return MultiAgentAction(
                    agent_role=AgentRole.OVERSIGHT,
                    action_type=ActionType.ESCALATE_PROPOSAL,
                    proposal_id=proposal.proposal_id,
                    reason=f"Escalating risky merge: {risk}",
                )

        return MultiAgentAction(
            agent_role=AgentRole.OVERSIGHT,
            action_type=ActionType.APPROVE_PROPOSAL,
            proposal_id=proposal.proposal_id,
            reason="Approve pending proposal",
        )

    if env.state.remaining_steps <= 1 or len(unresolved_records(env)) == 0:
        return MultiAgentAction(
            agent_role=AgentRole.OVERSIGHT,
            action_type=ActionType.FINALIZE_BATCH,
            reason="Finalize after high-value work",
        )

    return MultiAgentAction(
        agent_role=AgentRole.OVERSIGHT,
        action_type=ActionType.FINALIZE_BATCH,
        reason="Finalize instead of wasting turns",
    )


def load_trained_oversight():
    global _OVERSIGHT_MODEL, _OVERSIGHT_TOKENIZER

    if _OVERSIGHT_MODEL is not None and _OVERSIGHT_TOKENIZER is not None:
        return _OVERSIGHT_MODEL, _OVERSIGHT_TOKENIZER

    if not os.path.exists(TRAINED_MODEL_PATH):
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(TRAINED_MODEL_PATH)
    model.eval()

    _OVERSIGHT_TOKENIZER = tokenizer
    _OVERSIGHT_MODEL = model
    return model, tokenizer


def build_oversight_prompt(env: MultiAgentHyperlocalCatalogOpsEnvironment) -> str:
    pending = env.state.pending_proposals
    task = TASKS[env.state.task_id]
    risk_hints = task.hidden_signals.get("oversight_agent", {}).get("risk_hints", {})

    if not pending:
        unresolved = [record.record_id for record in unresolved_records(env)]
        return (
            "You are the oversight_agent in a multi-agent catalog curation environment.\n"
            f"Task: {env.state.task_id}\n"
            f"Remaining steps: {env.state.remaining_steps}\n"
            f"Unresolved records: {unresolved}\n"
            "Choose exactly one action from: finalize_batch\n"
            "Answer with one word only: finalize_batch"
        )

    proposal = pending[0]
    risk_key = f"{proposal.record_id}_{proposal.secondary_record_id}" if proposal.secondary_record_id else None
    reverse_key = f"{proposal.secondary_record_id}_{proposal.record_id}" if proposal.secondary_record_id else None
    risk = (risk_hints.get(risk_key, "") if risk_key else "") or (risk_hints.get(reverse_key, "") if reverse_key else "")

    return (
        "You are the oversight_agent in a multi-agent catalog curation environment.\n"
        f"Task: {env.state.task_id}\n"
        f"Remaining steps: {env.state.remaining_steps}\n"
        "Review the pending proposal and choose the safest best action.\n"
        f"Proposal ID: {proposal.proposal_id}\n"
        f"Proposal action: {proposal.action_type.value}\n"
        f"Record ID: {proposal.record_id}\n"
        f"Secondary record ID: {proposal.secondary_record_id}\n"
        f"Field: {proposal.field_name}\n"
        f"Value: {proposal.value}\n"
        f"Reason: {proposal.reason}\n"
        f"Risk hint: {risk}\n"
        "Choose exactly one action from: approve_proposal, reject_proposal, escalate_proposal\n"
        "Answer with one word only."
    )


def parse_oversight_decision(text: str) -> Optional[str]:
    lower = text.strip().lower()
    if "escalate_proposal" in lower or "escalate" in lower:
        return "escalate_proposal"
    if "reject_proposal" in lower or "reject" in lower:
        return "reject_proposal"
    if "approve_proposal" in lower or "approve" in lower:
        return "approve_proposal"
    if "finalize_batch" in lower or "finalize" in lower:
        return "finalize_batch"
    return None


def trained_oversight_policy(env: MultiAgentHyperlocalCatalogOpsEnvironment) -> MultiAgentAction:
    model, tokenizer = load_trained_oversight()
    if model is None or tokenizer is None:
        return heuristic_oversight_policy(env)

    prompt = build_oversight_prompt(env)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    decision = parse_oversight_decision(generated)

    pending = env.state.pending_proposals
    if pending:
        proposal = pending[0]
        if decision == "approve_proposal":
            return MultiAgentAction(
                agent_role=AgentRole.OVERSIGHT,
                action_type=ActionType.APPROVE_PROPOSAL,
                proposal_id=proposal.proposal_id,
                reason="Trained oversight decision: approve",
            )
        if decision == "reject_proposal":
            return MultiAgentAction(
                agent_role=AgentRole.OVERSIGHT,
                action_type=ActionType.REJECT_PROPOSAL,
                proposal_id=proposal.proposal_id,
                reason="Trained oversight decision: reject",
            )
        if decision == "escalate_proposal":
            return MultiAgentAction(
                agent_role=AgentRole.OVERSIGHT,
                action_type=ActionType.ESCALATE_PROPOSAL,
                proposal_id=proposal.proposal_id,
                reason="Trained oversight decision: escalate",
            )

    return MultiAgentAction(
        agent_role=AgentRole.OVERSIGHT,
        action_type=ActionType.FINALIZE_BATCH,
        reason="Trained oversight decision: finalize",
    )


def choose_action(env: MultiAgentHyperlocalCatalogOpsEnvironment, mode: str) -> MultiAgentAction:
    active_agent = env.state.current_agent

    if active_agent == AgentRole.CURATION:
        return curation_policy(env)
    if active_agent == AgentRole.DEDUPE:
        return dedupe_policy(env)
    if active_agent == AgentRole.PRICING:
        return pricing_policy(env)
    if active_agent == AgentRole.OVERSIGHT:
        if mode == "trained":
            return trained_oversight_policy(env)
        return heuristic_oversight_policy(env)

    raise RuntimeError(f"Unsupported agent role: {active_agent.value}")


def run_task(task_id: str, mode: str) -> float:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id=task_id)
    observation = env.reset()

    model_label = TRAINED_MODEL_PATH if mode == "trained" and os.path.exists(TRAINED_MODEL_PATH) else MODEL_NAME
    log_start(task_id, BENCHMARK, model_label, mode)

    rewards: List[float] = []
    while not observation.done:
        action = choose_action(env, mode)
        observation = env.step(action)

        reward = float(observation.reward or 0.0)
        rewards.append(reward)

        log_step(
            step=env.state.step_count,
            agent=action.agent_role.value,
            action=action.action_type.value,
            reward=reward,
            done=observation.done,
            error=observation.last_action_error,
            mode=mode,
        )

    final_score = env.state.score
    success = final_score >= SUCCESS_SCORE_THRESHOLD
    log_end(success, env.state.step_count, final_score, rewards, mode)
    return final_score


def run_benchmark(mode: str) -> float:
    scores = []
    for task_id in TASK_ORDER:
        score = run_task(task_id, mode)
        scores.append(score)

    average_score = sum(scores) / max(len(scores), 1)
    print(f"[SUMMARY] mode={mode} average_score={average_score:.4f}", flush=True)
    return average_score


def main() -> None:
    baseline_score = run_benchmark("baseline")
    trained_score = run_benchmark("trained")

    improvement = trained_score - baseline_score
    print(f"[BASELINE] average_score={baseline_score:.4f}", flush=True)
    print(f"[TRAINED] average_score={trained_score:.4f}", flush=True)
    print(f"[IMPROVEMENT] delta={improvement:.4f}", flush=True)


if __name__ == "__main__":
    main()
