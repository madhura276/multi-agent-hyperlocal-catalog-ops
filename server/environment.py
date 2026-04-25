from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..grader import build_reward
    from ..models import (
        ActionRecord,
        ActionType,
        AgentProposal,
        AgentRole,
        Category,
        MultiAgentAction,
        MultiAgentObservation,
        MultiAgentState,
        ProposalStatus,
        RecordStatus,
        Unit,
    )
    from ..tasks import DEFAULT_TASK_ID, TASKS
except ImportError:
    from grader import build_reward
    from models import (
        ActionRecord,
        ActionType,
        AgentProposal,
        AgentRole,
        Category,
        MultiAgentAction,
        MultiAgentObservation,
        MultiAgentState,
        ProposalStatus,
        RecordStatus,
        Unit,
    )
    from tasks import DEFAULT_TASK_ID, TASKS


TURN_ORDER = [
    AgentRole.CURATION,
    AgentRole.DEDUPE,
    AgentRole.PRICING,
    AgentRole.OVERSIGHT,
]


class MultiAgentHyperlocalCatalogOpsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = DEFAULT_TASK_ID):
        self._default_task_id = task_id
        self._task = TASKS[task_id]
        self._state = MultiAgentState()
        self._seen_action_signatures: set[str] = set()
        self.reset()

    def reset(self) -> MultiAgentObservation:
        self._task = TASKS[self._default_task_id]
        self._state = MultiAgentState(
            episode_id=str(uuid4()),
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            step_count=0,
            max_steps=self._task.max_steps,
            done=False,
            current_agent=TURN_ORDER[0],
            records=deepcopy(self._task.records),
            pending_proposals=[],
            approved_proposals=[],
            rejected_proposals=[],
            escalated_records=[],
            merged_pairs=[],
            flagged_records=[],
            action_history=[],
            last_action_error=None,
            last_reward=None,
            score=0.0,
            progress_score=0.0,
        )
        self._seen_action_signatures = set()
        return self._build_observation()

    def step(self, action: MultiAgentAction) -> MultiAgentObservation:  # type: ignore[override]
        if self._state.done:
            self._state.last_action_error = "Episode already finished. Reset the environment."
            return self._build_observation(reward_override=-0.05)

        previous_state = deepcopy(self._state)
        penalty = 0.0
        error: str | None = None

        try:
            if action.agent_role != self._state.current_agent:
                raise ValueError(
                    f"It is {self._state.current_agent.value}'s turn, not {action.agent_role.value}"
                )

            penalty += self._check_repeat_penalty(action)

            if action.agent_role == AgentRole.OVERSIGHT:
                penalty += self._resolve_proposal(action)
            else:
                penalty += self._create_proposal(action)

        except ValueError as exc:
            error = str(exc)
            penalty += 0.08

        self._state.step_count += 1

        submitted = action.action_type == ActionType.FINALIZE_BATCH
        if submitted or self._state.step_count >= self._state.max_steps:
            self._state.done = True
        else:
            self._state.current_agent = self._next_agent()

        reward_model = build_reward(
            task=self._task,
            previous_state=previous_state,
            current_state=self._state,
            penalty=penalty,
            submitted=submitted,
        )
        self._state.last_reward = reward_model
        self._state.last_action_error = error
        self._state.score = reward_model.total_score
        self._state.progress_score = reward_model.progress_score

        self._state.action_history.append(
            ActionRecord(
                step=self._state.step_count,
                agent_role=action.agent_role,
                action_type=action.action_type,
                proposal_id=action.proposal_id,
                record_id=action.record_id,
                secondary_record_id=action.secondary_record_id,
                field_name=action.field_name,
                value=action.value,
                reward=reward_model.delta,
                error=error,
            )
        )

        return self._build_observation()

    @property
    def state(self) -> MultiAgentState:
        return deepcopy(self._state)

    def _next_agent(self) -> AgentRole:
        idx = TURN_ORDER.index(self._state.current_agent)
        return TURN_ORDER[(idx + 1) % len(TURN_ORDER)]

    def _build_observation(self, reward_override: float | None = None) -> MultiAgentObservation:
        reward_value = reward_override
        if reward_value is None and self._state.last_reward is not None:
            reward_value = self._state.last_reward.delta

        return MultiAgentObservation(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            active_agent=self._state.current_agent,
            objective=self._task.objective,
            records=deepcopy(self._state.records),
            policy_snippets=self._build_policy_view(self._state.current_agent),
            pending_proposals=deepcopy(self._state.pending_proposals),
            action_history=deepcopy(self._state.action_history),
            remaining_steps=max(self._state.max_steps - self._state.step_count, 0),
            last_action_error=self._state.last_action_error,
            reward_details=deepcopy(self._state.last_reward),
            done=self._state.done,
            reward=reward_value if reward_value is not None else 0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "task_title": self._task.title,
                "score": self._state.score,
                "active_agent": self._state.current_agent.value,
                "hidden_signals": self._task.hidden_signals.get(self._state.current_agent.value, {}),
            },
        )

    def _build_policy_view(self, agent_role: AgentRole) -> list[str]:
        policy = list(self._task.policy_snippets)
        hidden = self._task.hidden_signals.get(agent_role.value, {})
        if hidden:
            policy.append(f"role_context={hidden}")
        return policy

    def _check_repeat_penalty(self, action: MultiAgentAction) -> float:
        signature = (
            f"{action.agent_role.value}|{action.action_type.value}|{action.proposal_id}|"
            f"{action.record_id}|{action.secondary_record_id}|{action.field_name}|"
            f"{action.value}|{action.reason}"
        )
        if signature in self._seen_action_signatures and action.action_type != ActionType.FINALIZE_BATCH:
            return 0.03

        self._seen_action_signatures.add(signature)
        return 0.0

    def _create_proposal(self, action: MultiAgentAction) -> float:
        if action.action_type in {
            ActionType.APPROVE_PROPOSAL,
            ActionType.REJECT_PROPOSAL,
            ActionType.ESCALATE_PROPOSAL,
        }:
            raise ValueError("Only oversight agent can approve, reject, or escalate proposals")

        if action.action_type == ActionType.FINALIZE_BATCH:
            raise ValueError("Only oversight agent can finalize the batch")

        if action.action_type in {
            ActionType.NORMALIZE_TITLE,
            ActionType.NORMALIZE_SIZE,
            ActionType.ASSIGN_CATEGORY,
            ActionType.CORRECT_PRICE,
            ActionType.FLAG_FOR_REVIEW,
        }:
            self._require_record(action.record_id)

        if action.action_type == ActionType.MERGE_RECORDS:
            primary = self._require_record(action.record_id)
            secondary = self._require_record(action.secondary_record_id)
            if primary.record_id == secondary.record_id:
                raise ValueError("cannot merge a record into itself")

        proposal = AgentProposal(
            proposal_id=str(uuid4()),
            proposer=action.agent_role,
            action_type=action.action_type,
            record_id=action.record_id,
            secondary_record_id=action.secondary_record_id,
            field_name=action.field_name,
            value=action.value,
            reason=action.reason,
            confidence=0.75,
            status=ProposalStatus.PENDING,
        )
        self._state.pending_proposals.append(proposal)
        return 0.0

    def _resolve_proposal(self, action: MultiAgentAction) -> float:
        if action.action_type == ActionType.FINALIZE_BATCH:
            self._state.done = True
            return 0.0

        proposal = self._require_proposal(action.proposal_id)

        if action.action_type == ActionType.APPROVE_PROPOSAL:
            proposal.status = ProposalStatus.APPROVED
            self._apply_approved_proposal(proposal)
            proposal.status = ProposalStatus.APPLIED
            self._state.approved_proposals.append(proposal.proposal_id)

        elif action.action_type == ActionType.REJECT_PROPOSAL:
            proposal.status = ProposalStatus.REJECTED
            self._state.rejected_proposals.append(proposal.proposal_id)

        elif action.action_type == ActionType.ESCALATE_PROPOSAL:
            proposal.status = ProposalStatus.ESCALATED
            if proposal.record_id and proposal.record_id not in self._state.escalated_records:
                self._state.escalated_records.append(proposal.record_id)

            record = self._find_record(proposal.record_id)
            if record is not None:
                record.status = RecordStatus.FLAGGED
                if proposal.record_id not in self._state.flagged_records:
                    self._state.flagged_records.append(proposal.record_id)

        else:
            raise ValueError("Oversight agent can only approve, reject, escalate, or finalize")

        self._state.pending_proposals = [
            item for item in self._state.pending_proposals if item.proposal_id != proposal.proposal_id
        ]
        return 0.0

    def _apply_approved_proposal(self, proposal: AgentProposal) -> None:
        if proposal.action_type == ActionType.NORMALIZE_TITLE:
            record = self._require_record(proposal.record_id)
            if not isinstance(proposal.value, str) or len(proposal.value.strip()) < 3:
                raise ValueError("normalize_title requires a non-empty string value")
            record.normalized_title = proposal.value.strip()
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif proposal.action_type == ActionType.NORMALIZE_SIZE:
            record = self._require_record(proposal.record_id)
            if not proposal.field_name:
                raise ValueError("normalize_size requires field_name")

            if proposal.field_name == "quantity_value":
                if not isinstance(proposal.value, (int, float)) or float(proposal.value) <= 0:
                    raise ValueError("quantity_value must be > 0")
                record.quantity_value = float(proposal.value)

            elif proposal.field_name == "quantity_unit":
                if not isinstance(proposal.value, str):
                    raise ValueError("quantity_unit must be a string")
                try:
                    record.quantity_unit = Unit(proposal.value.lower())
                except ValueError as exc:
                    raise ValueError(f"invalid unit: {proposal.value}") from exc

            elif proposal.field_name == "pack_count":
                if not isinstance(proposal.value, (int, float)) or int(proposal.value) < 1:
                    raise ValueError("pack_count must be >= 1")
                record.pack_count = int(proposal.value)

            else:
                raise ValueError(
                    "normalize_size field_name must be quantity_value, quantity_unit, or pack_count"
                )

            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif proposal.action_type == ActionType.ASSIGN_CATEGORY:
            record = self._require_record(proposal.record_id)
            if not isinstance(proposal.value, str):
                raise ValueError("assign_category requires a string value")
            try:
                record.category = Category(proposal.value.lower())
            except ValueError as exc:
                raise ValueError(f"invalid category: {proposal.value}") from exc
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif proposal.action_type == ActionType.MERGE_RECORDS:
            primary = self._require_record(proposal.record_id)
            secondary = self._require_record(proposal.secondary_record_id)

            if primary.store_id != secondary.store_id:
                raise ValueError("cannot merge records from different stores")
            if secondary.status == RecordStatus.MERGED:
                raise ValueError("secondary record is already merged")

            secondary.status = RecordStatus.MERGED
            note = f"Merged into {primary.record_id}"
            if note not in secondary.notes:
                secondary.notes.append(note)

            pair = [secondary.record_id, primary.record_id]
            if pair not in self._state.merged_pairs:
                self._state.merged_pairs.append(pair)

        elif proposal.action_type == ActionType.CORRECT_PRICE:
            record = self._require_record(proposal.record_id)
            if not isinstance(proposal.value, (int, float)) or float(proposal.value) < 0:
                raise ValueError("correct_price requires a non-negative numeric value")
            record.price = float(proposal.value)
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif proposal.action_type == ActionType.FLAG_FOR_REVIEW:
            record = self._require_record(proposal.record_id)
            record.status = RecordStatus.FLAGGED
            if proposal.record_id not in self._state.flagged_records:
                self._state.flagged_records.append(proposal.record_id)
            if proposal.reason:
                note = f"Flagged: {proposal.reason}"
                if note not in record.notes:
                    record.notes.append(note)

        else:
            raise ValueError(f"Unsupported proposal action_type: {proposal.action_type.value}")

    def _find_record(self, record_id: str | None):
        if not record_id:
            return None
        for record in self._state.records:
            if record.record_id == record_id:
                return record
        return None

    def _require_record(self, record_id: str | None):
        if not record_id:
            raise ValueError("record_id is required")
        record = self._find_record(record_id)
        if record is None:
            raise ValueError(f"record not found: {record_id}")
        return record

    def _require_proposal(self, proposal_id: str | None) -> AgentProposal:
        if not proposal_id:
            raise ValueError("proposal_id is required")
        for proposal in self._state.pending_proposals:
            if proposal.proposal_id == proposal_id:
                return proposal
        raise ValueError(f"proposal not found: {proposal_id}")
