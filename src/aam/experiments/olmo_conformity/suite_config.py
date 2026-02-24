from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

JsonDict = Dict[str, Any]


class SuiteDatasetSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    version: str = "v0"
    path: str
    notes: Optional[str] = None


class SuiteConditionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    params: JsonDict = Field(default_factory=dict)
    notes: Optional[str] = None


class SuiteModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    variant: str
    model_id: str
    notes: Optional[str] = None


class SuiteRunSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int = 42
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_items_per_dataset: Optional[int] = None
    notes: Optional[str] = None


class SuiteConfig(BaseModel):
    """
    Typed/validated schema for experiments/olmo_conformity/configs/suite_*.json.
    """

    model_config = ConfigDict(extra="allow")

    paths_config: Optional[str] = None
    suite_name: str
    suite_version: str = "v0"
    description: Optional[str] = None
    datasets: List[SuiteDatasetSpec] = Field(default_factory=list)
    conditions: List[SuiteConditionSpec] = Field(default_factory=list)
    models: List[SuiteModelSpec] = Field(default_factory=list)
    run: SuiteRunSpec = Field(default_factory=SuiteRunSpec)

