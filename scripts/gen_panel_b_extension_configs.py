"""Generate 12 Panel B extension experiment configs.

Goal: extend Panel B of Figure 2 (Think family) to all 12 conditions present
in Panel A (Instruct family), so the figure becomes a row-for-row analog.

The existing Panel B covers only: control, auth_bias, auth_trust, unan_confident.
This generator emits the 8 missing conditions for each of:
  - Olmo-3-7B-Think-SFT, Olmo-3-7B-Think-DPO, Olmo-3-7B-Think (RLVR final)
  - Olmo-3-32B-Think-SFT, Olmo-3-32B-Think-DPO, Olmo-3-32B-Think (RLVR final)
at T in {0.0, 0.6}, yielding 12 JSON config files.

Run from the repo root:
  python scripts/gen_panel_b_extension_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments" / "olmo_conformity" / "configs"

# Same 8 datasets / 50 items per dataset as the Instruct-pipeline runs;
# this is the 400-item pool the rest of the paper reports against.
DATASETS: list[dict[str, Any]] = [
    {"name": "immutable_facts_minimal", "version": "v2",
     "path": "experiments/olmo_conformity/datasets/immutable_facts/minimal_items_wrong.jsonl",
     "category": "general"},
    {"name": "social_conventions_minimal", "version": "v2",
     "path": "experiments/olmo_conformity/datasets/social_conventions/minimal_items_wrong.jsonl",
     "category": "opinion"},
    {"name": "gsm8k", "version": "v1",
     "path": "experiments/olmo_conformity/datasets/math/gsm8k_items_wrong.jsonl",
     "category": "math"},
    {"name": "mmlu_math", "version": "v1",
     "path": "experiments/olmo_conformity/datasets/math/mmlu_math_items_wrong.jsonl",
     "category": "math"},
    {"name": "mmlu_science", "version": "v1",
     "path": "experiments/olmo_conformity/datasets/science/mmlu_science_items_wrong.jsonl",
     "category": "science"},
    {"name": "mmlu_knowledge", "version": "v1",
     "path": "experiments/olmo_conformity/datasets/knowledge/mmlu_knowledge_items_wrong.jsonl",
     "category": "knowledge"},
    {"name": "truthfulqa", "version": "v1",
     "path": "experiments/olmo_conformity/datasets/truthfulness/truthfulqa_items_wrong.jsonl",
     "category": "truthfulness"},
    {"name": "arc", "version": "v1",
     "path": "experiments/olmo_conformity/datasets/reasoning/arc_items_wrong.jsonl",
     "category": "reasoning"},
]

# The 8 conditions Panel A has that Panel B is currently missing.
# Names and parameter blocks copied verbatim from suite_7b_expanded.json so the
# figure-generation pipeline groups these as the same conditions across panels.
PANEL_B_MISSING_CONDITIONS: list[dict[str, Any]] = [
    {
        "name": "asch_history_5",
        "params": {"type": "synthetic_asch_history", "confederates": 5, "confidence": "high"},
        "notes": "Asch paradigm: 5 confederates claim wrong_answer with high confidence",
    },
    {
        "name": "asch_zhu_unbiased_diverse_plain",
        "params": {"type": "asch_peer_pressure", "system_style": "control",
                    "prompt_style": "conversation", "confederates": 5,
                    "consensus": "diverse", "tone": "plain"},
        "notes": "Zhu et al. control: Diverse answers (no majority) with same dialogue format",
    },
    {
        "name": "asch_zhu_unbiased_qd",
        "params": {"type": "asch_peer_pressure", "system_style": "control",
                    "prompt_style": "conversation", "confederates": 5,
                    "consensus": "unanimous", "tone": "plain", "distillation": True},
        "notes": "Zhu et al. intervention: Question Distillation (summarize unanimous answer)",
    },
    {
        "name": "asch_zhu_unbiased_da",
        "params": {"type": "asch_peer_pressure", "system_style": "control",
                    "prompt_style": "conversation", "confederates": 5,
                    "consensus": "unanimous", "tone": "plain", "devils_advocate": True},
        "notes": "Zhu et al. intervention: Devil's Advocate (one dissenter breaks unanimity)",
    },
    {
        "name": "asch_zhu_unbiased_unanimous_plain",
        "params": {"type": "asch_peer_pressure", "system_style": "control",
                    "prompt_style": "conversation", "confederates": 5,
                    "consensus": "unanimous", "tone": "plain"},
        "notes": "Zhu et al.: unbiased system + unanimous-plain confederates",
    },
    {
        "name": "asch_zhu_unbiased_unanimous_neutral",
        "params": {"type": "asch_peer_pressure", "system_style": "control",
                    "prompt_style": "conversation", "confederates": 5,
                    "consensus": "unanimous", "tone": "neutral"},
        "notes": "Zhu et al.: unbiased system + unanimous-neutral (natural tone)",
    },
    {
        "name": "asch_zhu_unbiased_unanimous_uncertain",
        "params": {"type": "asch_peer_pressure", "system_style": "control",
                    "prompt_style": "conversation", "confederates": 5,
                    "consensus": "unanimous", "tone": "uncertain"},
        "notes": "Zhu et al.: unbiased system + unanimous-uncertain confederates",
    },
    {
        "name": "authority_zhu_unbiased_trust_da",
        "params": {"type": "authority_claim", "system_style": "control",
                    "claim_style": "trust", "devils_advocate": True},
        "notes": "Zhu et al. mitigation: add alternate option to reduce sycophancy (DA)",
    },
]


def _run_block(temperature: float, notes: str) -> dict[str, Any]:
    return {
        "seed": 42,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 0.9,
        "max_items_per_dataset": 50,
        "notes": notes,
    }


def make_think_config(
    *,
    suite_name: str,
    description: str,
    variant: str,
    model_id: str,
    temperature: float,
) -> dict[str, Any]:
    return {
        "paths_config": "paths.json",
        "suite_name": suite_name,
        "suite_version": "v1",
        "description": description,
        "datasets": DATASETS,
        "conditions": PANEL_B_MISSING_CONDITIONS,
        "models": [
            {
                "variant": variant,
                "model_id": model_id,
                # 5000-token budget matches suite_7b_think_sft_temp0p6.json so the
                # Think traces complete before the model emits its final answer.
                "max_new_tokens": 5000,
                "has_think_tokens": True,
                "notes": (
                    f"{variant} ({model_id}) — Panel B extension. "
                    "Eight conditions missing from the original Think suite, "
                    "added so Panel B of Figure 2 becomes a row-for-row analog of Panel A."
                ),
            }
        ],
        "run": _run_block(
            temperature,
            (
                f"Panel B extension at T={temperature}. Eight conditions from suite_7b_expanded "
                f"that the existing Think runs did not cover; same 8 datasets/50 items as Panel A. "
                f"Output token budget set to 5000 so Think traces complete."
            ),
        ),
    }


THINK_VARIANTS: list[tuple[str, str, str]] = [
    # (variant, model_id, suite_filename_stem)
    ("think_sft", "allenai/Olmo-3-7B-Think-SFT", "suite_olmo_7b_think_sft_panelB_extension"),
    ("think_dpo", "allenai/Olmo-3-7B-Think-DPO", "suite_olmo_7b_think_dpo_panelB_extension"),
    ("think_rlvr", "allenai/Olmo-3-7B-Think", "suite_olmo_7b_think_rlvr_panelB_extension"),
    ("think_sft", "allenai/Olmo-3-32B-Think-SFT", "suite_olmo_32b_think_sft_panelB_extension"),
    ("think_dpo", "allenai/Olmo-3-32B-Think-DPO", "suite_olmo_32b_think_dpo_panelB_extension"),
    ("think_rlvr", "allenai/Olmo-3-32B-Think", "suite_olmo_32b_think_rlvr_panelB_extension"),
]

TEMPERATURES: list[tuple[float, str]] = [(0.0, "temp0p0"), (0.6, "temp0p6")]


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for variant, model_id, suite_stem in THINK_VARIANTS:
        for temp, tag in TEMPERATURES:
            cfg = make_think_config(
                suite_name=f"olmo_conformity_{suite_stem}_{tag}",
                description=(
                    f"Panel B extension for {model_id} at T={temp}. "
                    "Adds the 8 conditions Panel A has that Panel B currently lacks "
                    "(asch_history_5, diverse_plain, qd, da, unanimous plain/neutral/uncertain, "
                    "authority_trust_da), so Figure 2 Panel B becomes a row-for-row analog of Panel A."
                ),
                variant=variant,
                model_id=model_id,
                temperature=temp,
            )
            out = CONFIG_DIR / f"{suite_stem}_{tag}.json"
            out.write_text(json.dumps(cfg, indent=2) + "\n")
            written.append(out)

    for p in written:
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
