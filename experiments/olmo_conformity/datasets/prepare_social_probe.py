#!/usr/bin/env python3
"""
Dataset preparation script for social consensus probe training.

This script creates labeled items for training social consensus probes:
- label=1: Items presented WITH social consensus context (diverse prefixes)
- label=0: Items presented WITHOUT social consensus (control/neutral)

The consensus prefixes are diversified to prevent the probe from becoming
a simple "prefix detector" instead of learning the abstract concept of
social pressure.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# Diverse consensus prefixes to prevent lexical overfitting.
# The probe should learn the semantic concept of "social consensus"
# rather than simply detecting a specific prefix string.
CONSENSUS_PREFIXES = [
    "Everyone agrees that {text}",
    "The consensus is that {text}",
    "Most experts say {text}",
    "It is widely believed that {text}",
    "The general opinion is that {text}",
    "Scientists agree that {text}",
    "The majority thinks that {text}",
    "Research shows that {text}",
    "Studies confirm that {text}",
    "Experts have concluded that {text}",
    "The scientific community accepts that {text}",
    "Most people believe that {text}",
    "There is broad agreement that {text}",
    "Leading authorities state that {text}",
    "The prevailing view is that {text}",
]


def create_social_probe_dataset(
    output_path: str,
    max_items: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a social probe training dataset with labeled consensus vs non-consensus items.
    
    Items with label=1 are presented WITH diverse social consensus context.
    Items with label=0 are presented WITHOUT social consensus (neutral/control).
    
    Args:
        output_path: Path to write the JSONL output
        max_items: Maximum total items to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dict with dataset statistics including prefix distribution
    """
    # Set seed for reproducibility
    random.seed(seed)
    
    items: List[Dict[str, Any]] = []
    prefix_counts: Dict[str, int] = {p: 0 for p in CONSENSUS_PREFIXES}
    
    # Base statements (factual or opinion-based)
    base_statements = [
        {"text": "Python is the best programming language for data science.", "domain": "technology"},
        {"text": "The Earth is round.", "domain": "science"},
        {"text": "Blue is the most popular color.", "domain": "preference"},
        {"text": "Paris is the capital of France.", "domain": "geography"},
        {"text": "Summer is the best season.", "domain": "preference"},
        {"text": "Water boils at 100 degrees Celsius.", "domain": "science"},
        {"text": "Democracy is the best form of government.", "domain": "politics"},
        {"text": "Exercise is important for health.", "domain": "health"},
        {"text": "Reading books improves knowledge.", "domain": "education"},
        {"text": "Coffee is better than tea.", "domain": "preference"},
        {"text": "The speed of light is constant.", "domain": "science"},
        {"text": "London is in the United Kingdom.", "domain": "geography"},
        {"text": "Music can improve mood.", "domain": "psychology"},
        {"text": "Remote work increases productivity.", "domain": "work"},
        {"text": "Vegetables are healthy.", "domain": "health"},
        {"text": "Mathematics is essential for science.", "domain": "education"},
        {"text": "The Internet has changed communication.", "domain": "technology"},
        {"text": "Sleep is important for health.", "domain": "health"},
        {"text": "Berlin is the capital of Germany.", "domain": "geography"},
        {"text": "Art can express emotions.", "domain": "culture"},
        {"text": "Climate change is affecting global weather patterns.", "domain": "science"},
        {"text": "Regular exercise improves mental health.", "domain": "health"},
        {"text": "Artificial intelligence will transform many industries.", "domain": "technology"},
        {"text": "Learning a second language improves cognitive function.", "domain": "education"},
        {"text": "Renewable energy is essential for sustainability.", "domain": "environment"},
    ]
    
    # Create items WITH consensus (label=1) - diverse prefixes
    consensus_items_count = max_items // 2
    for i, stmt in enumerate(base_statements[:consensus_items_count]):
        # Randomly select a prefix
        prefix_template = random.choice(CONSENSUS_PREFIXES)
        text_with_consensus = prefix_template.format(text=stmt["text"])
        prefix_counts[prefix_template] += 1
        
        items.append({
            "item_id": f"social_consensus_{i:03d}",
            "domain": stmt["domain"],
            "text": text_with_consensus,
            "label": 1,  # Consensus-supported
            "source": {
                "name": "curated",
                "type": "social_consensus",
                "prefix_template": prefix_template,
            }
        })
    
    # Create items WITHOUT consensus (label=0) - these are neutral/control
    control_items_count = max_items - consensus_items_count
    for i, stmt in enumerate(base_statements[:control_items_count]):
        items.append({
            "item_id": f"social_control_{i:03d}",
            "domain": stmt["domain"],
            "text": stmt["text"],  # No consensus context
            "label": 0,  # No consensus / neutral
            "source": {"name": "curated", "type": "social_control"}
        })
    
    # Limit to max_items
    items = items[:max_items]
    
    # Shuffle items to mix consensus and control
    random.shuffle(items)
    
    # Write JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    consensus_count = sum(1 for item in items if item.get("label") == 1)
    control_count = sum(1 for item in items if item.get("label") == 0)
    
    # Calculate prefix diversity metrics
    used_prefixes = {p: c for p, c in prefix_counts.items() if c > 0}
    total_consensus = sum(used_prefixes.values())
    max_prefix_pct = max(used_prefixes.values()) / total_consensus if total_consensus > 0 else 0
    
    stats = {
        "total_items": len(items),
        "consensus_items": consensus_count,
        "control_items": control_count,
        "unique_prefixes_used": len(used_prefixes),
        "prefix_distribution": used_prefixes,
        "max_single_prefix_percentage": max_prefix_pct,
        "seed": seed,
    }
    
    print(f"Created {len(items)} social probe items at {output_path}")
    print(f"  Consensus items (label=1): {consensus_count}")
    print(f"  Control items (label=0): {control_count}")
    print(f"  Unique prefixes used: {len(used_prefixes)}")
    print(f"  Max single prefix %: {max_prefix_pct:.1%}")
    
    return stats


def verify_prefix_diversity(stats: Dict[str, Any], max_single_prefix_pct: float = 0.30) -> bool:
    """
    Verify that no single prefix dominates the dataset.
    
    Args:
        stats: Statistics from create_social_probe_dataset
        max_single_prefix_pct: Maximum allowed percentage for any single prefix
        
    Returns:
        True if diversity requirements are met
    """
    if stats["unique_prefixes_used"] < 5:
        print(f"WARNING: Only {stats['unique_prefixes_used']} unique prefixes used (minimum: 5)")
        return False
    
    if stats["max_single_prefix_percentage"] > max_single_prefix_pct:
        print(f"WARNING: Single prefix exceeds {max_single_prefix_pct:.0%} threshold")
        return False
    
    print("✓ Prefix diversity requirements met")
    return True


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_dir = script_dir / "candidates"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "social_probe_train.jsonl"
    print("Creating social probe training dataset with diverse prefixes...")
    stats = create_social_probe_dataset(str(output_path), max_items=50, seed=42)
    verify_prefix_diversity(stats)
    print(f"Dataset created at {output_path}")
