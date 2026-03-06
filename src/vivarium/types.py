from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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


class ScientificReport(BaseModel):
    """
    Scientific report validating experimental rigor.
    
    This "Run Certificate" is generated at the end of each experiment run
    to verify scientific validity before human interpretation.
    
    See Implementation Plan Section C for specification details.
    """
    
    run_id: str = Field(..., description="Unique identifier for the experiment run")
    git_hash: Optional[str] = Field(None, description="Git commit hash for reproducibility")
    backend: str = Field(
        default="pytorch",
        description="Inference backend used: 'pytorch' or 'llamacpp'"
    )
    duration_seconds: float = Field(..., description="Total runtime in seconds")
    
    # Validity flags
    integrity_verified: bool = Field(
        ...,
        description="True if Merkle hash chain is intact (no data corruption)"
    )
    dual_stack_risk: bool = Field(
        default=False,
        description="True if generation and probing used different model weights (validity threat)"
    )
    
    # Core metrics
    metrics: JsonDict = Field(
        default_factory=dict,
        description="Dictionary of computed metrics: truth_vector_alignment, sycophancy_rate, cot_consistency, response_entropy"
    )
    
    # Anomaly detection
    anomalies: List[str] = Field(
        default_factory=list,
        description="List of detected anomalies or validation failures"
    )
    
    # Convergence analysis
    convergence_data: Optional[JsonDict] = Field(
        None,
        description="Step-by-step consensus rate and 'turn' analysis"
    )
    
    # Output quality assessment
    output_quality: Optional[JsonDict] = Field(
        None,
        description="Diversity metrics, failure categorization (JSONDecodeError, Refusal, ToolError)"
    )
    
    def save(self, path: str) -> None:
        """
        Save report to JSON file.
        
        Args:
            path: Output file path
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: str) -> "ScientificReport":
        """
        Load report from JSON file.
        
        Args:
            path: Input file path
            
        Returns:
            ScientificReport instance
        """
        p = Path(path)
        return cls.model_validate_json(p.read_text())
    
    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json")
    
    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        lines = [
            f"Scientific Report: {self.run_id}",
            f"  Backend: {self.backend}",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  Integrity: {'✓ Verified' if self.integrity_verified else '✗ FAILED'}",
            f"  Dual-Stack Risk: {'⚠ YES' if self.dual_stack_risk else '✓ No'}",
        ]
        
        if self.metrics:
            lines.append("  Metrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.4f}")
                else:
                    lines.append(f"    {key}: {value}")
        
        if self.anomalies:
            lines.append(f"  Anomalies ({len(self.anomalies)}):")
            for anomaly in self.anomalies[:5]:  # Show first 5
                lines.append(f"    - {anomaly}")
            if len(self.anomalies) > 5:
                lines.append(f"    ... and {len(self.anomalies) - 5} more")
        
        return "\n".join(lines)


Observation = JsonDict
