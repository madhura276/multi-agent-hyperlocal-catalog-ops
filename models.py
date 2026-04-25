from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State as OpenEnvState
except ImportError:
    class OpenEnvAction(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class OpenEnvObservation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class OpenEnvState(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = None
        step_count: int = 0


ActionValue = Union[str, float, int, bool]


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AgentRole(str, Enum):
    CURATION = "curation_agent"
    DEDUPE = "dedupe_agent"
    PRICING = "pricing_agent"
    OVERSIGHT = "oversight_agent"


class ActionType(str, Enum):
    NORMALIZE_TITLE = "normalize_title"
    NORMALIZE_SIZE = "normalize_size"
    ASSIGN_CATEGORY = "assign_category"
    MERGE_RECORDS = "merge_records"
    CORRECT_PRICE = "correct_price"
    FLAG_FOR_REVIEW = "flag_for_review"
    APPROVE_PROPOSAL = "approve_proposal"
    REJECT_PROPOSAL = "reject_proposal"
    ESCALATE_PROPOSAL = "escalate_proposal"
    FINALIZE_BATCH = "finalize_batch"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    APPLIED = "applied"


class RecordStatus(str, Enum):
    RAW = "raw"
    CLEANED = "cleaned"
    MERGED = "merged"
    FLAGGED = "flagged"
    INVALID = "invalid"


class Category(str, Enum):
    BEVERAGES = "beverages"
    DAIRY = "dairy"
    SNACKS = "snacks"
    PRODUCE = "produce"
    STAPLES = "staples"
    PERSONAL_CARE = "personal_care"
    HOUSEHOLD = "household"
    FROZEN = "frozen"
    UNKNOWN = "unknown"


class Unit(str, Enum):
    G = "g"
    KG = "kg"
    ML = "ml"
    L = "l"
    PCS = "pcs"


class InventoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    record_id: str
    store_id: str
    raw_title: str
    normalized_title: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[Category] = None
    subcategory: Optional[str] = None
    quantity_value: Optional[float] = None
    quantity_unit: Optional[Unit] = None
    pack_count: Optional[int] = None
    price: Optional[float] = None
    currency: str = "INR"
    barcode: Optional[str] = None
    source: Optional[str] = None
    status: RecordStatus = RecordStatus.RAW
    notes: List[str] = Field(default_factory=list)

    @field_validator("quantity_value")
    @classmethod
    def validate_quantity_value(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("quantity_value must be > 0")
        return value

    @field_validator("pack_count")
    @classmethod
    def validate_pack_count(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError("pack_count must be >= 1")
        return value

    @field_validator("price")
    @classmethod
    def validate_price(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("price must be >= 0")
        return value


class ExpectedRecordOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    record_id: str
    normalized_title: Optional[str] = None
    category: Optional[Category] = None
    quantity_value: Optional[float] = None
    quantity_unit: Optional[Unit] = None
    pack_count: Optional[int] = None
    price: Optional[float] = None
    status: Optional[RecordStatus] = None
    merged_into: Optional[str] = None
    should_flag: bool = False


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str
    title: str
    difficulty: Difficulty
    objective: str
    records: List[InventoryRecord] = Field(default_factory=list)
    policy_snippets: List[str] = Field(default_factory=list)
    hidden_signals: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    expected_outcomes: List[ExpectedRecordOutcome] = Field(default_factory=list)
    max_steps: int = Field(default=16, ge=8, le=32)


class MultiAgentAction(OpenEnvAction):
    agent_role: AgentRole
    action_type: ActionType
    proposal_id: Optional[str] = None
    record_id: Optional[str] = None
    secondary_record_id: Optional[str] = None
    field_name: Optional[str] = None
    value: Optional[ActionValue] = None
    reason: Optional[str] = None


class AgentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    proposal_id: str
    proposer: AgentRole
    action_type: ActionType
    record_id: Optional[str] = None
    secondary_record_id: Optional[str] = None
    field_name: Optional[str] = None
    value: Optional[ActionValue] = None
    reason: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ProposalStatus = ProposalStatus.PENDING


class RewardComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    contribution: float


class MultiAgentReward(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    delta: float
    total_score: float = Field(ge=0.0, le=1.0)
    progress_score: float = Field(ge=0.0, le=1.0)
    penalty: float = 0.0
    components: List[RewardComponent] = Field(default_factory=list)
    explanation: str


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step: int = Field(ge=1)
    agent_role: AgentRole
    action_type: ActionType
    proposal_id: Optional[str] = None
    record_id: Optional[str] = None
    secondary_record_id: Optional[str] = None
    field_name: Optional[str] = None
    value: Optional[ActionValue] = None
    reward: float = 0.0
    error: Optional[str] = None


class MultiAgentObservation(OpenEnvObservation):
    task_id: str
    difficulty: Difficulty
    active_agent: AgentRole
    objective: str
    records: List[InventoryRecord] = Field(default_factory=list)
    policy_snippets: List[str] = Field(default_factory=list)
    pending_proposals: List[AgentProposal] = Field(default_factory=list)
    action_history: List[ActionRecord] = Field(default_factory=list)
    remaining_steps: int = Field(ge=0)
    last_action_error: Optional[str] = None
    reward_details: Optional[MultiAgentReward] = None
    allowed_actions: List[str] = Field(
        default_factory=lambda: [action.value for action in ActionType]
    )


class MultiAgentState(OpenEnvState):
    task_id: Optional[str] = None
    difficulty: Optional[Difficulty] = None
    max_steps: int = Field(default=0, ge=0)
    done: bool = False
    current_agent: AgentRole = AgentRole.CURATION
    records: List[InventoryRecord] = Field(default_factory=list)
    pending_proposals: List[AgentProposal] = Field(default_factory=list)
    approved_proposals: List[str] = Field(default_factory=list)
    rejected_proposals: List[str] = Field(default_factory=list)
    escalated_records: List[str] = Field(default_factory=list)
    merged_pairs: List[List[str]] = Field(default_factory=list)
    flagged_records: List[str] = Field(default_factory=list)
    action_history: List[ActionRecord] = Field(default_factory=list)
    last_action_error: Optional[str] = None
    last_reward: Optional[MultiAgentReward] = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    progress_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @computed_field
    @property
    def remaining_steps(self) -> int:
        return max(self.max_steps - self.step_count, 0)
