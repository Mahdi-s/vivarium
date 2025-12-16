from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


JsonDict = Dict[str, Any]


class ActionRequest(BaseModel):
    run_id: str = Field(..., description="Unique identifier for the experiment run")
    time_step: int = Field(..., ge=0, description="Logical clock time of the simulation")
    agent_id: str = Field(..., description="The internal simulation identity of the agent")
    action_name: str = Field(..., description="Stable identifier for the tool/action")
    arguments: JsonDict = Field(default_factory=dict, description="Typed arguments validated by schema")
    reasoning: Optional[str] = Field(None, description="Optional reasoning text captured from the policy/LLM")
    metadata: JsonDict = Field(default_factory=dict, description="Model ID, latency, token usage, etc.")

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json")


class ActionResult(BaseModel):
    success: bool
    data: Optional[JsonDict] = Field(None, description="Structured return payload")
    error: Optional[str] = Field(None, description="Error message if validation/execution failed")
    trace_id: str = Field(..., description="UUID of the generated trace event")

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json")


class TraceEvent(BaseModel):
    trace_id: str
    run_id: str
    time_step: int = Field(..., ge=0)
    timestamp: float = Field(..., description="Wall-clock timestamp (seconds since epoch)")
    agent_id: str
    action_type: str
    info: JsonDict = Field(default_factory=dict, description="Action payload")
    outcome: JsonDict = Field(default_factory=dict, description="Action outcome payload")
    environment_state_hash: Optional[str] = Field(None, description="Optional integrity hash of environment state")

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json")


class RunMetadata(BaseModel):
    run_id: str
    seed: int
    created_at: float
    config: JsonDict = Field(default_factory=dict)

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json")


Observation = JsonDict


