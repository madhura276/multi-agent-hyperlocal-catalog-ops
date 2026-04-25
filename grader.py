from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from models import (
    ExpectedRecordOutcome,
    MultiAgentReward,
    MultiAgentState,
    InventoryRecord,
    RecordStatus,
    RewardComponent,
    TaskDefinition,
)

TITLE_WEIGHT = 0.18
SIZE_WEIGHT = 0.14
CATEGORY_WEIGHT = 0.14
DUPLICATE_WEIGHT = 0.16
PRICE_WEIGHT = 0.14
ESCALATION_WEIGHT = 0.10
COORDINATION_WEIGHT = 0.08
OVERSIGHT_WEIGHT = 0.06

SCORE_EPS = 1e-3


@dataclass(frozen=True)
class GradeBreakdown:
    total_score: float
    progress_score: float
    components: List[RewardComponent]


def _clip_score(value: float) -> float:
    return max(SCORE_EPS, min(1.0 - SCORE_EPS, value))


def _strict_ratio(hits: int, total: int) -> float:
    if total <= 0:
        return 1.0 - SCORE_EPS
    return _clip_score(hits / total)


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _float_equal(a: Optional[float], b: Optional[float], tolerance: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


def _build_record_map(records: Iterable[InventoryRecord]) -> Dict[str, InventoryRecord]:
    return {record.record_id: record for record in records}


def _build_expected_map(task: TaskDefinition) -> Dict[str, ExpectedRecordOutcome]:
    return {outcome.record_id: outcome for outcome in task.expected_outcomes}


def score_title_normalization(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.normalized_title is not None
    ]
    if not expected_ids:
        return 1.0 - SCORE_EPS

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and _normalize_text(actual.normalized_title) == _normalize_text(expected.normalized_title):
            hits += 1

    return _strict_ratio(hits, len(expected_ids))


def score_unit_normalization(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.quantity_value is not None or outcome.quantity_unit is not None or outcome.pack_count is not None
    ]
    if not expected_ids:
        return 1.0 - SCORE_EPS

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if not actual:
            continue

        quantity_ok = True
        unit_ok = True
        pack_ok = True

        if expected.quantity_value is not None:
            quantity_ok = _float_equal(actual.quantity_value, expected.quantity_value)

        if expected.quantity_unit is not None:
            unit_ok = actual.quantity_unit == expected.quantity_unit

        if expected.pack_count is not None:
            pack_ok = actual.pack_count == expected.pack_count

        if quantity_ok and unit_ok and pack_ok:
            hits += 1

    return _strict_ratio(hits, len(expected_ids))


def score_category_assignment(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.category is not None
    ]
    if not expected_ids:
        return 1.0 - SCORE_EPS

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and actual.category == expected.category:
            hits += 1

    return _strict_ratio(hits, len(expected_ids))


def score_duplicate_resolution(task: TaskDefinition, state: MultiAgentState) -> float:
    expected_pairs = {
        (outcome.record_id, outcome.merged_into)
        for outcome in task.expected_outcomes
        if outcome.merged_into is not None
    }
    actual_pairs = {
        (pair[0], pair[1])
        for pair in state.merged_pairs
        if len(pair) == 2
    }

    if not expected_pairs:
        return 1.0 - SCORE_EPS if not actual_pairs else SCORE_EPS

    correct_pairs = expected_pairs & actual_pairs
    extra_pairs = actual_pairs - expected_pairs

    base = len(correct_pairs) / len(expected_pairs)
    penalty = len(extra_pairs) / max(len(expected_pairs), 1)
    return _clip_score(base - 0.5 * penalty)


def score_price_handling(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.price is not None
    ]
    if not expected_ids:
        return 1.0 - SCORE_EPS

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and _float_equal(actual.price, expected.price):
            hits += 1

    return _strict_ratio(hits, len(expected_ids))


def score_escalation_quality(task: TaskDefinition, state: MultiAgentState) -> float:
    expected_flagged = {
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.should_flag
    }
    actual_flagged = set(state.flagged_records)
    actual_flagged.update(state.escalated_records)

    record_map = _build_record_map(state.records)
    for record_id, record in record_map.items():
        if record.status == RecordStatus.FLAGGED:
            actual_flagged.add(record_id)

    if not expected_flagged:
        return 1.0 - SCORE_EPS if not actual_flagged else SCORE_EPS

    correct_flags = expected_flagged & actual_flagged
    extra_flags = actual_flagged - expected_flagged

    base = len(correct_flags) / len(expected_flagged)
    penalty = len(extra_flags) / max(len(expected_flagged), 1)
    return _clip_score(base - 0.5 * penalty)


def score_status_alignment(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.status is not None
    ]
    if not expected_ids:
        return 1.0 - SCORE_EPS

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and actual.status == expected.status:
            hits += 1

    return _strict_ratio(hits, len(expected_ids))


def score_coordination_quality(state: MultiAgentState) -> float:
    resolved = len(state.approved_proposals) + len(state.rejected_proposals)
    pending = len(state.pending_proposals)
    total = resolved + pending

    if total == 0:
        return 1.0 - SCORE_EPS

    resolution_ratio = resolved / max(total, 1)
    repeated_error_penalty = 0.0
    if len(state.action_history) >= 2:
        repeated_errors = sum(1 for item in state.action_history if item.error)
        repeated_error_penalty = min(0.25, repeated_errors * 0.05)

    return _clip_score(resolution_ratio - repeated_error_penalty)


def score_oversight_quality(state: MultiAgentState) -> float:
    oversight_actions = [
        item for item in state.action_history
        if item.agent_role.value == "oversight_agent"
    ]
    if not oversight_actions:
        return 0.5

    approvals = sum(1 for item in oversight_actions if item.action_type.value == "approve_proposal")
    rejections = sum(1 for item in oversight_actions if item.action_type.value == "reject_proposal")
    escalations = sum(1 for item in oversight_actions if item.action_type.value == "escalate_proposal")
    errors = sum(1 for item in oversight_actions if item.error)

    raw_score = 0.45
    raw_score += min(0.25, approvals * 0.04)
    raw_score += min(0.20, (rejections + escalations) * 0.05)
    raw_score -= min(0.30, errors * 0.08)

    return _clip_score(raw_score)


def grade_state(task: TaskDefinition, state: MultiAgentState) -> GradeBreakdown:
    title_score = _clip_score(score_title_normalization(task, state.records))
    size_score = _clip_score(score_unit_normalization(task, state.records))
    category_score = _clip_score(score_category_assignment(task, state.records))
    duplicate_score = _clip_score(score_duplicate_resolution(task, state))
    price_score = _clip_score(score_price_handling(task, state.records))
    escalation_score = _clip_score(score_escalation_quality(task, state))
    coordination_score = _clip_score(score_coordination_quality(state))
    oversight_score = _clip_score(score_oversight_quality(state))
    status_score = _clip_score(score_status_alignment(task, state.records))

    components = [
        RewardComponent(
            name="title_normalization",
            score=title_score,
            weight=TITLE_WEIGHT,
            contribution=TITLE_WEIGHT * title_score,
        ),
        RewardComponent(
            name="size_normalization",
            score=size_score,
            weight=SIZE_WEIGHT,
            contribution=SIZE_WEIGHT * size_score,
        ),
        RewardComponent(
            name="category_assignment",
            score=category_score,
            weight=CATEGORY_WEIGHT,
            contribution=CATEGORY_WEIGHT * category_score,
        ),
        RewardComponent(
            name="duplicate_resolution",
            score=duplicate_score,
            weight=DUPLICATE_WEIGHT,
            contribution=DUPLICATE_WEIGHT * duplicate_score,
        ),
        RewardComponent(
            name="price_handling",
            score=price_score,
            weight=PRICE_WEIGHT,
            contribution=PRICE_WEIGHT * price_score,
        ),
        RewardComponent(
            name="escalation_quality",
            score=escalation_score,
            weight=ESCALATION_WEIGHT,
            contribution=ESCALATION_WEIGHT * escalation_score,
        ),
        RewardComponent(
            name="coordination_quality",
            score=coordination_score,
            weight=COORDINATION_WEIGHT,
            contribution=COORDINATION_WEIGHT * coordination_score,
        ),
        RewardComponent(
            name="oversight_quality",
            score=oversight_score,
            weight=OVERSIGHT_WEIGHT,
            contribution=OVERSIGHT_WEIGHT * oversight_score,
        ),
    ]

    total_score = sum(component.contribution for component in components)
    progress_score = _clip_score((total_score + status_score * 0.10) / 1.10)

    return GradeBreakdown(
        total_score=_clip_score(total_score),
        progress_score=progress_score,
        components=components,
    )


def build_reward(
    task: TaskDefinition,
    previous_state: MultiAgentState,
    current_state: MultiAgentState,
    penalty: float = 0.0,
    submitted: bool = False,
) -> MultiAgentReward:
    previous_breakdown = grade_state(task, previous_state)
    current_breakdown = grade_state(task, current_state)

    delta = current_breakdown.progress_score - previous_breakdown.progress_score - penalty

    if submitted and current_breakdown.total_score < 0.65:
        penalty += 0.08
        delta -= 0.08

    explanation = (
        f"Progress moved from {previous_breakdown.progress_score:.3f} "
        f"to {current_breakdown.progress_score:.3f} with penalty {penalty:.3f}."
    )

    return MultiAgentReward(
        delta=delta,
        total_score=current_breakdown.total_score,
        progress_score=current_breakdown.progress_score,
        penalty=penalty,
        components=current_breakdown.components,
        explanation=explanation,
    )
