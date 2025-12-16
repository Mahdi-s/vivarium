from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from aam.scheduler import SortMode


class ExperimentRunSection(BaseModel):
    steps: int = Field(10, ge=0)
    agents: int = Field(2, ge=1)
    seed: int = 42
    run_id: Optional[str] = None
    deterministic_timestamps: bool = True
    runs_dir: str = "./runs"


class ExperimentSchedulerSection(BaseModel):
    per_agent_timeout_s: float = Field(60.0, gt=0)
    max_concurrency: int = Field(50, gt=0)
    sort_mode: SortMode = "agent_id"


PolicyKind = Literal["random", "cognitive", "transformerlens"]


class ExperimentPolicySection(BaseModel):
    kind: PolicyKind = "cognitive"

    # Cognitive policy / gateways
    model: str = "gpt-3.5-turbo"
    mock_llm: bool = False
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    message_history: int = Field(20, ge=0)

    # Rate limiting
    rate_limit_rpm: Optional[int] = Field(None, description="Requests per minute limit")
    rate_limit_tpm: Optional[int] = Field(None, description="Tokens per minute limit")
    rate_limit_max_concurrent: int = Field(10, description="Max concurrent requests")
    rate_limit_enabled: bool = Field(True, description="Enable rate limiting by default")

    # TransformerLens policy
    model_id: Optional[str] = None


class ExperimentCaptureSection(BaseModel):
    # Mirrors Phase 3 flags at a high level (kept optional).
    layers: str = "0"
    components: str = "resid_post"
    trigger_actions: str = "post_message"
    token_position: int = -1
    dtype: Literal["float16", "float32"] = "float16"


class ExperimentConfig(BaseModel):
    run: ExperimentRunSection = Field(default_factory=ExperimentRunSection)
    scheduler: ExperimentSchedulerSection = Field(default_factory=ExperimentSchedulerSection)
    policy: ExperimentPolicySection = Field(default_factory=ExperimentPolicySection)
    capture: Optional[ExperimentCaptureSection] = None


def load_experiment_config(path: str) -> ExperimentConfig:
    """
    Load and validate an experiment config from JSON.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)


