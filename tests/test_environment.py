import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grader import grade_state
from models import (
    ActionType,
    AgentRole,
    MultiAgentAction,
    ProposalStatus,
    RecordStatus,
)
from server.environment import MultiAgentHyperlocalCatalogOpsEnvironment
from tasks import TASKS


def test_reset_loads_requested_task() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="medium_multistore_conflict")
    observation = env.reset()

    assert observation.task_id == "medium_multistore_conflict"
    assert observation.done is False
    assert observation.active_agent == AgentRole.CURATION
    assert len(observation.records) == len(TASKS["medium_multistore_conflict"].records)


def test_reset_clears_state() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="easy_single_store_cleanup")
    env.reset()

    env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
            reason="Obvious cleanup",
        )
    )

    reset_obs = env.reset()

    record = next(r for r in reset_obs.records if r.record_id == "easy_1")
    assert reset_obs.action_history == []
    assert reset_obs.last_action_error is None
    assert reset_obs.pending_proposals == []
    assert record.normalized_title is None


def test_grader_score_is_strictly_between_zero_and_one() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="easy_single_store_cleanup")
    env.reset()

    breakdown = grade_state(TASKS["easy_single_store_cleanup"], env.state)

    assert 0.0 < breakdown.total_score < 1.0
    assert 0.0 < breakdown.progress_score < 1.0


def test_curation_agent_creates_pending_proposal() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="easy_single_store_cleanup")
    env.reset()

    observation = env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
            reason="Obvious title normalization",
        )
    )

    assert observation.last_action_error is None
    assert len(env.state.pending_proposals) == 1
    assert env.state.pending_proposals[0].status == ProposalStatus.PENDING
    assert env.state.current_agent == AgentRole.DEDUPE


def test_wrong_turn_gets_penalized() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="easy_single_store_cleanup")
    env.reset()

    observation = env.step(
        MultiAgentAction(
            agent_role=AgentRole.PRICING,
            action_type=ActionType.CORRECT_PRICE,
            record_id="easy_1",
            value=49.0,
            reason="Trying out of turn",
        )
    )

    assert observation.last_action_error is not None
    assert "turn" in observation.last_action_error.lower()
    assert float(observation.reward or 0.0) < 0.0


def test_oversight_can_approve_and_apply_proposal() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="easy_single_store_cleanup")
    env.reset()

    env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
            reason="Obvious title normalization",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.DEDUPE,
            action_type=ActionType.FLAG_FOR_REVIEW,
            record_id="easy_2",
            reason="Dummy action to advance turn",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.PRICING,
            action_type=ActionType.FLAG_FOR_REVIEW,
            record_id="easy_3",
            reason="Dummy action to advance turn",
        )
    )

    proposal_id = next(
        p.proposal_id for p in env.state.pending_proposals if p.record_id == "easy_1"
    )

    observation = env.step(
        MultiAgentAction(
            agent_role=AgentRole.OVERSIGHT,
            action_type=ActionType.APPROVE_PROPOSAL,
            proposal_id=proposal_id,
            reason="Safe to approve",
        )
    )

    record = next(r for r in env.state.records if r.record_id == "easy_1")
    assert observation.last_action_error is None
    assert record.normalized_title == "Coca Cola 1 L"
    assert proposal_id in env.state.approved_proposals


def test_oversight_can_reject_proposal() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="medium_multistore_conflict")
    env.reset()

    env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="med_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
            reason="Obvious title cleanup",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.DEDUPE,
            action_type=ActionType.MERGE_RECORDS,
            record_id="med_1",
            secondary_record_id="med_2",
            reason="Potential duplicate",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.PRICING,
            action_type=ActionType.CORRECT_PRICE,
            record_id="med_3",
            value=320.0,
            reason="Obvious price correction",
        )
    )

    merge_proposal = next(
        p for p in env.state.pending_proposals if p.action_type == ActionType.MERGE_RECORDS
    )

    observation = env.step(
        MultiAgentAction(
            agent_role=AgentRole.OVERSIGHT,
            action_type=ActionType.REJECT_PROPOSAL,
            proposal_id=merge_proposal.proposal_id,
            reason="Rejected for testing",
        )
    )

    assert observation.last_action_error is None
    assert merge_proposal.proposal_id in env.state.rejected_proposals


def test_escalation_flags_record_in_hard_task() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="hard_ambiguous_oversight_batch")
    env.reset()

    env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="hard_4",
            field_name="normalized_title",
            value="Tomatoes 1 Kg Pack",
            reason="Basic cleanup",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.DEDUPE,
            action_type=ActionType.MERGE_RECORDS,
            record_id="hard_3",
            secondary_record_id="hard_4",
            reason="Ambiguous duplicate candidate",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.PRICING,
            action_type=ActionType.CORRECT_PRICE,
            record_id="hard_5",
            value=320.0,
            reason="Obvious outlier correction",
        )
    )

    merge_proposal = next(
        p for p in env.state.pending_proposals if p.record_id == "hard_3"
    )

    before_score = env.state.progress_score

    observation = env.step(
        MultiAgentAction(
            agent_role=AgentRole.OVERSIGHT,
            action_type=ActionType.ESCALATE_PROPOSAL,
            proposal_id=merge_proposal.proposal_id,
            reason="Loose vs packed tomatoes should be escalated",
        )
    )

    after_score = env.state.progress_score
    flagged_record = next(r for r in env.state.records if r.record_id == "hard_3")

    assert observation.last_action_error is None
    assert after_score >= before_score
    assert "hard_3" in env.state.escalated_records
    assert flagged_record.status == RecordStatus.FLAGGED


def test_repeated_identical_action_gets_repeat_penalty() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="easy_single_store_cleanup")
    env.reset()

    first = env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
            reason="Repeat check",
        )
    )

    env.state.current_agent = AgentRole.CURATION

    second = env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
            reason="Repeat check",
        )
    )

    assert float(first.reward or 0.0) != 0.0
    assert float(second.reward or 0.0) <= float(first.reward or 0.0)



def test_medium_task_progresses_on_known_good_sequence() -> None:
    env = MultiAgentHyperlocalCatalogOpsEnvironment(task_id="medium_multistore_conflict")
    env.reset()

    env.step(
        MultiAgentAction(
            agent_role=AgentRole.CURATION,
            action_type=ActionType.NORMALIZE_TITLE,
            record_id="med_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
            reason="Known good normalization",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.DEDUPE,
            action_type=ActionType.MERGE_RECORDS,
            record_id="med_1",
            secondary_record_id="med_2",
            reason="Known good duplicate merge",
        )
    )
    env.step(
        MultiAgentAction(
            agent_role=AgentRole.PRICING,
            action_type=ActionType.CORRECT_PRICE,
            record_id="med_3",
            value=320.0,
            reason="Known good price correction",
        )
    )

    pending_ids = [proposal.proposal_id for proposal in env.state.pending_proposals]
    for proposal_id in pending_ids:
        env.step(
            MultiAgentAction(
                agent_role=AgentRole.OVERSIGHT,
                action_type=ActionType.APPROVE_PROPOSAL,
                proposal_id=proposal_id,
                reason="Approve known good proposal",
            )
        )
        if not env.state.done and env.state.current_agent != AgentRole.OVERSIGHT:
            while not env.state.done and env.state.current_agent != AgentRole.OVERSIGHT:
                env.step(
                    MultiAgentAction(
                        agent_role=env.state.current_agent,
                        action_type=ActionType.FLAG_FOR_REVIEW,
                        record_id=env.state.records[0].record_id,
                        reason="Advance turn for test flow",
                    )
                )

    assert env.state.progress_score > 0.2
