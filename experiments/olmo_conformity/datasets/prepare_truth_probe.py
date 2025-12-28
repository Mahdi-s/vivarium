#!/usr/bin/env python3
"""
Dataset preparation script for truth probe training data.

This script creates labeled True/False statements for training probes
that can detect truthfulness in model activations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def create_truth_probe_dataset(output_path: str, max_items: int = 100) -> None:
    """
    Create a truth probe training dataset with labeled True/False statements.
    """
    items: List[Dict[str, Any]] = [
        # True statements
        {"item_id": "truth_earth_orbits_sun", "domain": "science", "text": "True or False: The Earth revolves around the Sun.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_paris_capital_france", "domain": "geography", "text": "True or False: Paris is the capital of France.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_water_h2o", "domain": "science", "text": "True or False: Water has the chemical formula H2O.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_2plus2_equals_4", "domain": "math", "text": "True or False: 2 + 2 equals 4.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_pacific_largest_ocean", "domain": "geography", "text": "True or False: The Pacific Ocean is the largest ocean.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_7_is_prime", "domain": "math", "text": "True or False: 7 is a prime number.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_gravity_exists", "domain": "science", "text": "True or False: Gravity exists on Earth.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_tokyo_capital_japan", "domain": "geography", "text": "True or False: Tokyo is the capital of Japan.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_sqrt_64_is_8", "domain": "math", "text": "True or False: The square root of 64 is 8.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_oxygen_in_air", "domain": "science", "text": "True or False: Air contains oxygen.", "label": 1, "source": {"name": "curated"}},
        
        # False statements
        {"item_id": "truth_sun_orbits_earth", "domain": "science", "text": "True or False: The Sun revolves around the Earth.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_lyon_capital_france", "domain": "geography", "text": "True or False: Lyon is the capital of France.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_water_co2", "domain": "science", "text": "True or False: Water has the chemical formula CO2.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_2plus2_equals_5", "domain": "math", "text": "True or False: 2 + 2 equals 5.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_atlantic_largest_ocean", "domain": "geography", "text": "True or False: The Atlantic Ocean is the largest ocean.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_4_is_prime", "domain": "math", "text": "True or False: 4 is a prime number.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_no_gravity_earth", "domain": "science", "text": "True or False: There is no gravity on Earth.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_osaka_capital_japan", "domain": "geography", "text": "True or False: Osaka is the capital of Japan.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_sqrt_64_is_12", "domain": "math", "text": "True or False: The square root of 64 is 12.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_no_oxygen_air", "domain": "science", "text": "True or False: Air does not contain oxygen.", "label": 0, "source": {"name": "curated"}},
        
        # More true statements
        {"item_id": "truth_london_capital_uk", "domain": "geography", "text": "True or False: London is the capital of the United Kingdom.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_berlin_capital_germany", "domain": "geography", "text": "True or False: Berlin is the capital of Germany.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_rome_capital_italy", "domain": "geography", "text": "True or False: Rome is the capital of Italy.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_3times3_equals_9", "domain": "math", "text": "True or False: 3 × 3 equals 9.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_10divided2_equals_5", "domain": "math", "text": "True or False: 10 divided by 2 equals 5.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_water_boils_100c", "domain": "science", "text": "True or False: Water boils at 100 degrees Celsius at standard pressure.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_8_planets_solar_system", "domain": "science", "text": "True or False: There are 8 planets in our solar system.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_light_speed_300k", "domain": "science", "text": "True or False: The speed of light is approximately 300,000 km/s.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_pi_starts_314", "domain": "math", "text": "True or False: Pi starts with 3.14.", "label": 1, "source": {"name": "curated"}},
        {"item_id": "truth_gravity_98_ms2", "domain": "science", "text": "True or False: Gravity on Earth is approximately 9.8 m/s².", "label": 1, "source": {"name": "curated"}},
        
        # More false statements
        {"item_id": "truth_paris_capital_uk", "domain": "geography", "text": "True or False: Paris is the capital of the United Kingdom.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_munich_capital_germany", "domain": "geography", "text": "True or False: Munich is the capital of Germany.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_milan_capital_italy", "domain": "geography", "text": "True or False: Milan is the capital of Italy.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_3times3_equals_6", "domain": "math", "text": "True or False: 3 × 3 equals 6.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_10divided2_equals_4", "domain": "math", "text": "True or False: 10 divided by 2 equals 4.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_water_boils_50c", "domain": "science", "text": "True or False: Water boils at 50 degrees Celsius at standard pressure.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_9_planets_solar_system", "domain": "science", "text": "True or False: There are 9 planets in our solar system.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_light_speed_100k", "domain": "science", "text": "True or False: The speed of light is approximately 100,000 km/s.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_pi_starts_315", "domain": "math", "text": "True or False: Pi starts with 3.15.", "label": 0, "source": {"name": "curated"}},
        {"item_id": "truth_gravity_5_ms2", "domain": "science", "text": "True or False: Gravity on Earth is approximately 5 m/s².", "label": 0, "source": {"name": "curated"}},
    ]
    
    # Expand to max_items by duplicating and varying
    while len(items) < max_items:
        # Add variations of existing items
        base_items = items[:40]  # Use first 40 as base
        for item in base_items:
            if len(items) >= max_items:
                break
            # Create variation by changing item_id
            new_item = item.copy()
            new_item["item_id"] = f"{item['item_id']}_v{len(items)}"
            items.append(new_item)
    
    # Limit to max_items
    items = items[:max_items]
    
    # Write JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Created {len(items)} truth probe training items at {output_path}")
    print(f"  True statements: {sum(1 for item in items if item.get('label') == 1)}")
    print(f"  False statements: {sum(1 for item in items if item.get('label') == 0)}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_dir = script_dir / "candidates"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "truth_probe_train.jsonl"
    print("Creating truth probe training dataset...")
    create_truth_probe_dataset(str(output_path), max_items=100)
    print(f"Dataset created at {output_path}")
