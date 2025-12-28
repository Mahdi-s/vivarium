from __future__ import annotations

import os
import hashlib
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CaptureConfig:
    """
    Phase 3 (TransformerLens) integration point.

    This is intentionally light-weight: Phase 2 can run without these deps.
    
    Attributes:
        layers: List of layer indices to capture activations from
        components: List of component names (e.g. ["resid_post", "attn_out"])
        trigger_actions: List of action names that trigger activation capture
        token_position: Token position to slice (-1 = last token)
        layer_sample_rate: Probability of capturing each layer (0.0 to 1.0, default 1.0)
        max_layers_per_step: Maximum number of layers to capture per step (None = no limit)
        sampling_seed: Random seed for reproducible sparse sampling
    """

    layers: List[int]
    components: List[str]  # e.g. ["resid_post", "attn_out"]
    trigger_actions: List[str]
    token_position: int = -1  # -1 = last token
    
    # Sparse capture options (Task 013)
    layer_sample_rate: float = 1.0  # 1.0 = capture all layers
    max_layers_per_step: Optional[int] = None  # None = no limit
    sampling_seed: Optional[int] = None  # For reproducible sampling
    
    def __post_init__(self) -> None:
        # Validate sample rate
        if not 0.0 <= self.layer_sample_rate <= 1.0:
            raise ValueError(f"layer_sample_rate must be between 0.0 and 1.0, got {self.layer_sample_rate}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptureConfig":
        """
        Create CaptureConfig from a dictionary (e.g., loaded from JSON config).
        
        This allows users to configure sparse capture via experiment config files.
        """
        return cls(
            layers=data.get("layers", []),
            components=data.get("components", []),
            trigger_actions=data.get("trigger_actions", []),
            token_position=data.get("token_position", -1),
            layer_sample_rate=data.get("layer_sample_rate", 1.0),
            max_layers_per_step=data.get("max_layers_per_step"),
            sampling_seed=data.get("sampling_seed"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layers": list(self.layers),
            "components": list(self.components),
            "trigger_actions": list(self.trigger_actions),
            "token_position": self.token_position,
            "layer_sample_rate": self.layer_sample_rate,
            "max_layers_per_step": self.max_layers_per_step,
            "sampling_seed": self.sampling_seed,
        }


@dataclass
class ActivationRecordRef:
    run_id: str
    time_step: int
    agent_id: str
    model_id: str
    layer_index: int
    component: str
    token_position: int
    shard_file_path: str
    tensor_key: str
    shape: Tuple[int, ...]
    dtype: str


class CaptureContext:
    """
    Phase 3 capture context: buffers activations during a single model inference,
    then commits or discards them based on the decided action (sampling policy),
    and finally flushes per-step safetensors shards aligned with trace steps.

    Features:
    - Sparse capture: Control storage explosion with layer_sample_rate and max_layers_per_step
    - CoT indexing: Mark Chain-of-Thought regions for separate analysis
    - Merkle logging: Cryptographic provenance tracking

    Notes:
    - Requires extras: `pip install -e .[interpretability]`
    - Intended for local models run via TransformerLens (not remote LiteLLM providers).
    """

    def __init__(
        self, *, output_dir: str, config: CaptureConfig, dtype: str = "float16", trace_db: Optional[Any] = None
    ):
        self.output_dir = output_dir
        self.config = config
        self.dtype = dtype
        self._trace_db = trace_db
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            import torch  # type: ignore
            from safetensors.torch import save_file  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Interpretability deps not installed. Install extras: `pip install -e .[interpretability]`"
            ) from e

        self._torch = torch
        self._save_file = save_file

        # Pending tensors from the most recent inference (keyed by hook.name).
        self._pending: Dict[str, Any] = {}
        # Committed tensors for each step (keyed by tensor_key within the shard).
        self._committed_by_step: Dict[int, Dict[str, Any]] = {}
        # Optional per-step metadata to embed into safetensors shards.
        self._metadata_by_step: Dict[int, Dict[str, str]] = {}
        # Best-effort: remember run_id/model_id for provenance usage during flush.
        self._last_run_id: Optional[str] = None
        self._last_model_id: Optional[str] = None
        self._merkle_logger: Optional[Any] = None
        
        # Sparse capture state
        self._sampling_rng = random.Random(config.sampling_seed)
        self._step_layer_count: Dict[int, int] = {}  # Track layers captured per step
        
        if self._trace_db is not None:
            try:
                from aam.provenance import MerkleLogger
                self._merkle_logger = MerkleLogger()
            except Exception:
                self._merkle_logger = None

    def begin_inference(self) -> None:
        self._pending = {}

    def build_fwd_hooks(self) -> List[tuple[str, Callable[[Any, Any], Any]]]:
        """
        Returns TransformerLens fwd_hooks list: [(hook_name, hook_fn), ...]
        
        Respects sparse capture settings (layer_sample_rate, max_layers_per_step).
        """

        hook_names = self._expand_hook_names()

        def hook_fn(activations: Any, hook: Any) -> Any:
            self.record_activation(
                hook_name=str(getattr(hook, "name", "unknown_hook")),
                activations=activations,
            )
            return activations

        return [(name, hook_fn) for name in hook_names]

    def record_activation(self, *, hook_name: str, activations: Any) -> None:
        """
        Record a single activation vector into the pending buffer.

        This is used by TransformerLens hooks and can also be used by
        plain PyTorch forward hooks for HF models (to emulate TL-style hook names).
        
        Respects sparse capture settings.
        """
        # Apply sparse capture: probabilistic layer sampling
        if self.config.layer_sample_rate < 1.0:
            if self._sampling_rng.random() > self.config.layer_sample_rate:
                return  # Skip this activation
        
        # activations are usually one of:
        # - [batch, pos, d_model] (residual stream)
        # - [batch, pos, heads, head_dim] (head-separated)
        # - [batch, heads, pos, pos] (attention pattern)  # best-effort
        pos = int(self.config.token_position)
        try:
            if getattr(activations, "ndim", None) == 4:
                # Prefer token-position slicing on axis=1 when present.
                # Many HF projection hooks yield [B, S, ...]; attention patterns may be [B, H, S, S].
                # If this is [B, H, S, S], slicing [:, pos, ...] is wrong; fall back in that case.
                try:
                    vec = activations[:, pos, :, :].detach()
                except Exception:
                    vec = activations.detach()
            else:
                vec = activations[:, pos, :].detach()
        except Exception:
            # If activation shape isn't token-positioned, fall back to raw tensor.
            try:
                vec = activations.detach()
            except Exception:
                return
        vec = vec.to("cpu")

        # Store a single vector per hook (batch index 0) to keep files small.
        try:
            if vec.ndim >= 2:
                vec = vec[0]
        except Exception:
            pass

        if self.dtype == "float16":
            vec = vec.to(self._torch.float16)
        elif self.dtype == "float32":
            vec = vec.to(self._torch.float32)

        self._pending[str(hook_name)] = vec.contiguous()

    def set_step_metadata(self, *, time_step: int, metadata: Dict[str, Any]) -> None:
        """
        Set per-step metadata to embed into the safetensors shard header when flushing.

        Notes:
        - safetensors metadata values must be strings
        - this is best-effort; metadata may be dropped if the installed safetensors
          version doesn't support it
        """
        md: Dict[str, str] = {}
        for k, v in (metadata or {}).items():
            if v is None:
                continue
            md[str(k)] = str(v)
        self._metadata_by_step[int(time_step)] = md

    def mark_cot_region(
        self,
        *,
        time_step: int,
        agent_id: str,
        start_token_idx: int,
        end_token_idx: int,
    ) -> None:
        """
        Mark a Chain-of-Thought (CoT) region for separate indexing.
        
        This is used to track <think>...</think> token boundaries for
        deceptive rationalization analysis in Think models.
        
        Args:
            time_step: Simulation time step
            agent_id: Agent identifier
            start_token_idx: Token index where CoT begins (e.g., <think> token)
            end_token_idx: Token index where CoT ends (e.g., </think> token)
        """
        if self._trace_db is None:
            return
        
        # Store in metadata for inclusion in safetensors header
        md = self._metadata_by_step.setdefault(int(time_step), {})
        md[f"cot_start_{agent_id}"] = str(start_token_idx)
        md[f"cot_end_{agent_id}"] = str(end_token_idx)
        
        # Also store in a more structured format for analysis
        cot_key = f"cot_regions_{time_step}"
        if not hasattr(self, "_cot_regions"):
            self._cot_regions: Dict[str, List[Dict[str, Any]]] = {}
        
        self._cot_regions.setdefault(cot_key, []).append({
            "agent_id": agent_id,
            "start_token_idx": start_token_idx,
            "end_token_idx": end_token_idx,
        })

    def get_cot_regions(self, time_step: int) -> List[Dict[str, Any]]:
        """
        Get marked CoT regions for a given time step.
        
        Returns:
            List of dicts with agent_id, start_token_idx, end_token_idx
        """
        if not hasattr(self, "_cot_regions"):
            return []
        return self._cot_regions.get(f"cot_regions_{time_step}", [])

    def on_action_decided(
        self,
        *,
        run_id: str,
        time_step: int,
        agent_id: str,
        model_id: str,
        action_name: str,
    ) -> None:
        """
        Commit or discard pending activations based on trigger_actions.
        Called after the policy parses an action_name.
        
        Respects max_layers_per_step sparse capture setting.
        """
        self._last_run_id = str(run_id)
        self._last_model_id = str(model_id)
        if not self._pending:
            return
        if self.config.trigger_actions and action_name not in set(self.config.trigger_actions):
            # Sampling policy: discard
            self._pending = {}
            return

        step_buf = self._committed_by_step.setdefault(int(time_step), {})
        
        # Apply max_layers_per_step limit
        layers_committed = self._step_layer_count.get(int(time_step), 0)
        
        for hook_name, tensor in self._pending.items():
            # Check max_layers_per_step
            if self.config.max_layers_per_step is not None:
                if layers_committed >= self.config.max_layers_per_step:
                    break  # Stop committing for this step
            
            key = f"{agent_id}.{hook_name}"
            step_buf[key] = tensor
            layers_committed += 1

            # Index activation metadata if trace_db is available
            if self._trace_db is not None:
                # Parse layer index and component from hook name
                layer_index = 0
                component = hook_name
                if "blocks." in hook_name:
                    parts = hook_name.split(".")
                    if len(parts) >= 2:
                        try:
                            layer_index = int(parts[1])
                            component = ".".join(parts[2:]) if len(parts) > 2 else hook_name
                        except ValueError:
                            pass

                # Get tensor shape and dtype
                shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
                dtype_str = str(tensor.dtype) if hasattr(tensor, "dtype") else self.dtype

                # Create activation record reference
                from aam.interpretability import ActivationRecordRef

                record = ActivationRecordRef(
                    run_id=run_id,
                    time_step=time_step,
                    agent_id=agent_id,
                    model_id=model_id,
                    layer_index=layer_index,
                    component=component,
                    token_position=self.config.token_position,
                    shard_file_path="",  # Will be set in flush_step
                    tensor_key=key,
                    shape=shape,
                    dtype=dtype_str,
                )
                # Store record temporarily; will update with shard path in flush_step
                if not hasattr(self, "_pending_records"):
                    self._pending_records: Dict[int, List[ActivationRecordRef]] = {}
                self._pending_records.setdefault(int(time_step), []).append(record)

        self._step_layer_count[int(time_step)] = layers_committed
        self._pending = {}

    def flush_step(self, *, time_step: int) -> Optional[str]:
        """
        Write `activations/step_{time_step:06d}.safetensors` if there are committed tensors.
        Returns the shard path if written.
        """
        buf = self._committed_by_step.pop(int(time_step), None) or {}
        if not buf:
            return None
        # IMPORTANT:
        # Multiple experiment phases (e.g. trials, truth probe capture, social probe capture)
        # may reuse the same time_step indices. Never overwrite an existing shard file,
        # otherwise we silently corrupt earlier activations and break downstream analysis.
        shard_path = os.path.join(self.output_dir, f"step_{int(time_step):06d}.safetensors")
        if os.path.exists(shard_path):
            # Create a unique sibling shard to preserve prior data.
            suffix = uuid.uuid4().hex[:8]
            shard_path = os.path.join(self.output_dir, f"step_{int(time_step):06d}__{suffix}.safetensors")

        def _hash_tensor_dict(tensors: Dict[str, Any]) -> str:
            # Deterministic hash: SHA256 of concatenated (key + bytes) in sorted key order.
            h = hashlib.sha256()
            for k in sorted(tensors.keys()):
                h.update(k.encode("utf-8"))
                t = tensors[k]
                try:
                    tb = t.detach().contiguous().cpu().numpy().tobytes()
                except Exception:
                    tb = repr(t).encode("utf-8")
                h.update(tb)
            return h.hexdigest()

        # Best-effort metadata embedding (requires safetensors>=0.4).
        metadata = dict(self._metadata_by_step.pop(int(time_step), None) or {})
        # Provide minimal defaults if we can.
        if self._last_run_id is not None:
            metadata.setdefault("run_id", str(self._last_run_id))
        metadata.setdefault("step_id", str(int(time_step)))
        if self._last_model_id is not None:
            metadata.setdefault("model_id", str(self._last_model_id))
        
        # Include sparse capture config in metadata for reproducibility
        metadata.setdefault("layer_sample_rate", str(self.config.layer_sample_rate))
        if self.config.max_layers_per_step is not None:
            metadata.setdefault("max_layers_per_step", str(self.config.max_layers_per_step))

        # File-level activation hash (may include multiple agents).
        activation_hash_all = _hash_tensor_dict(buf)
        metadata.setdefault("provenance_hash", activation_hash_all)

        # Best-effort Merkle logging per (time_step, agent_id) using agent-scoped activation hashes.
        # We do this BEFORE writing so the shard metadata can include the root.
        if self._trace_db is not None and self._merkle_logger is not None:
            try:
                # Partition tensors by agent_id prefix (agent_id.<hook_name>)
                by_agent: Dict[str, Dict[str, Any]] = {}
                for tensor_key, tensor in buf.items():
                    agent = str(tensor_key).split(".", 1)[0] if "." in str(tensor_key) else "unknown"
                    by_agent.setdefault(agent, {})[str(tensor_key)] = tensor

                root_at_step: Optional[str] = None
                for agent_id, agent_buf in sorted(by_agent.items(), key=lambda x: x[0]):
                    activation_hash = _hash_tensor_dict(agent_buf)

                    # Try to look up a conformity prompt hash for this (time_step, agent_id).
                    prompt_hash = ""
                    try:
                        row = self._trace_db.conn.execute(
                            """
                            SELECT p.rendered_prompt_hash
                            FROM conformity_trial_steps s
                            JOIN conformity_prompts p ON p.trial_id = s.trial_id
                            WHERE s.time_step = ? AND s.agent_id = ?
                            ORDER BY p.created_at ASC
                            LIMIT 1;
                            """,
                            (int(time_step), str(agent_id)),
                        ).fetchone()
                        if row is not None:
                            prompt_hash = str(row["rendered_prompt_hash"] or "")
                    except Exception:
                        prompt_hash = ""

                    leaf, root = self._merkle_logger.add_step(
                        step_id=str(int(time_step)),
                        agent_id=str(agent_id),
                        prompt_hash=prompt_hash,
                        activation_hash=activation_hash,
                    )
                    root_at_step = root
                    try:
                        self._trace_db.insert_merkle_log(
                            run_id=str(self._last_run_id or ""),
                            time_step=int(time_step),
                            agent_id=str(agent_id),
                            prompt_hash=prompt_hash,
                            activation_hash=activation_hash,
                            leaf_hash=leaf,
                            merkle_root=root,
                        )
                    except Exception:
                        pass

                if root_at_step is not None:
                    metadata.setdefault("merkle_root_at_step", str(root_at_step))
            except Exception:
                pass

        try:
            self._save_file(buf, shard_path, metadata=metadata)
        except TypeError:
            # Older safetensors: ignore metadata.
            self._save_file(buf, shard_path)

        # Index activation metadata if trace_db is available
        if self._trace_db is not None and hasattr(self, "_pending_records"):
            records = self._pending_records.pop(int(time_step), [])
            for record in records:
                # Update shard file path
                from aam.interpretability import ActivationRecordRef

                updated_record = ActivationRecordRef(
                    run_id=record.run_id,
                    time_step=record.time_step,
                    agent_id=record.agent_id,
                    model_id=record.model_id,
                    layer_index=record.layer_index,
                    component=record.component,
                    token_position=record.token_position,
                    shard_file_path=shard_path,
                    tensor_key=record.tensor_key,
                    shape=record.shape,
                    dtype=record.dtype,
                )
                self._trace_db.insert_activation_metadata(updated_record)
        
        # Clean up step layer count
        self._step_layer_count.pop(int(time_step), None)

        return shard_path

    @staticmethod
    def list_available_hooks(model: Any) -> List[str]:
        """
        Return available hook names for a HookedTransformer (best-effort).
        """
        hook_dict = getattr(model, "hook_dict", None)
        if isinstance(hook_dict, dict):
            return sorted(str(k) for k in hook_dict.keys())
        return []

    @staticmethod
    def get_model_layers(model: Any) -> Dict[str, Any]:
        """
        Get model layer information for dynamic layer selection.
        
        Returns a dictionary with:
        - num_layers: Total number of layers
        - layer_names: List of available layer hook names
        - components: Available components per layer
        """
        hook_dict = getattr(model, "hook_dict", None)
        if not isinstance(hook_dict, dict):
            return {"num_layers": 0, "layer_names": [], "components": {}}

        # Extract layer information
        layers = set()
        components_by_layer: Dict[int, List[str]] = {}

        for hook_name in hook_dict.keys():
            # Parse hook names like "blocks.10.attn.hook_z"
            if "blocks." in str(hook_name):
                parts = str(hook_name).split(".")
                if len(parts) >= 2 and parts[0] == "blocks":
                    try:
                        layer_idx = int(parts[1])
                        layers.add(layer_idx)
                        if layer_idx not in components_by_layer:
                            components_by_layer[layer_idx] = []
                        component = ".".join(parts[2:]) if len(parts) > 2 else str(hook_name)
                        if component not in components_by_layer[layer_idx]:
                            components_by_layer[layer_idx].append(component)
                    except ValueError:
                        pass

        num_layers = max(layers) + 1 if layers else 0
        layer_names = sorted([f"blocks.{i}" for i in sorted(layers)])

        return {
            "num_layers": num_layers,
            "layer_names": layer_names,
            "components": {k: sorted(v) for k, v in components_by_layer.items()},
        }

    def _expand_hook_names(self) -> List[str]:
        """
        Expand (layers, components) into concrete TransformerLens hook names.

        - If a component contains '.', it is treated as a suffix under `blocks.{layer}.`
          (e.g. 'attn.hook_z' -> 'blocks.10.attn.hook_z') unless it already starts with 'blocks.'.
        - Otherwise, it is treated as a hookpoint name (e.g. 'resid_post' -> 'hook_resid_post').
        """
        components = list(self.config.components or [])
        layers = list(self.config.layers or [])

        # If layers aren't specified, accept components as fully qualified hook names.
        if not layers:
            return [c for c in components if c]

        base_map = {
            "resid_pre": "hook_resid_pre",
            "resid_mid": "hook_resid_mid",
            "resid_post": "hook_resid_post",
        }

        out: List[str] = []
        for layer in layers:
            for comp in components:
                if not comp:
                    continue
                if comp.startswith("blocks."):
                    out.append(comp)
                    continue
                if "." in comp:
                    out.append(f"blocks.{layer}.{comp}")
                    continue
                hook = base_map.get(comp, comp if comp.startswith("hook_") else f"hook_{comp}")
                out.append(f"blocks.{layer}.{hook}")
        # De-dupe while preserving order
        seen = set()
        deduped: List[str] = []
        for h in out:
            if h in seen:
                continue
            seen.add(h)
            deduped.append(h)
        return deduped
