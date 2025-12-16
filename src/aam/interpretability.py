from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CaptureConfig:
    """
    Phase 3 (TransformerLens) integration point.

    This is intentionally light-weight: Phase 2 can run without these deps.
    """

    layers: List[int]
    components: List[str]  # e.g. ["resid_post", "attn_out"]
    trigger_actions: List[str]
    token_position: int = -1  # -1 = last token


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

    def begin_inference(self) -> None:
        self._pending = {}

    def build_fwd_hooks(self) -> List[tuple[str, Callable[[Any, Any], Any]]]:
        """
        Returns TransformerLens fwd_hooks list: [(hook_name, hook_fn), ...]
        """

        hook_names = self._expand_hook_names()

        def hook_fn(activations: Any, hook: Any) -> Any:
            # activations usually shape [batch, pos, d_model]
            pos = int(self.config.token_position)
            try:
                vec = activations[:, pos, :].detach()
            except Exception:
                # If activation shape isn't token-positioned, fall back to raw tensor.
                vec = activations.detach()
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

            self._pending[str(getattr(hook, "name", "unknown_hook"))] = vec.contiguous()
            return activations

        return [(name, hook_fn) for name in hook_names]

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
        """
        _ = (run_id, model_id)  # reserved for future metadata indexing
        if not self._pending:
            return
        if self.config.trigger_actions and action_name not in set(self.config.trigger_actions):
            # Sampling policy: discard
            self._pending = {}
            return

        step_buf = self._committed_by_step.setdefault(int(time_step), {})
        for hook_name, tensor in self._pending.items():
            key = f"{agent_id}.{hook_name}"
            step_buf[key] = tensor

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

        self._pending = {}

    def flush_step(self, *, time_step: int) -> Optional[str]:
        """
        Write `activations/step_{time_step:06d}.safetensors` if there are committed tensors.
        Returns the shard path if written.
        """
        buf = self._committed_by_step.pop(int(time_step), None) or {}
        if not buf:
            return None
        shard_path = os.path.join(self.output_dir, f"step_{int(time_step):06d}.safetensors")
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


