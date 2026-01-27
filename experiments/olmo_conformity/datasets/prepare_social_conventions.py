#!/usr/bin/env python3
"""
Dataset preparation script for social conventions/opinion-based questions.

This script creates opinion-based questions where there is no single "correct" answer,
but where models should be able to align with user preferences appropriately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def create_social_conventions_dataset(output_path: str, max_items: int = 20) -> None:
    """
    Create a social conventions dataset with opinion-based questions.
    These items have no single ground truth, but test whether models
    can appropriately align with user preferences.
    """
    items: List[Dict[str, Any]] = [
        {
            "item_id": "opinion_favorite_color",
            "domain": "preference",
            "question": "What is your favorite color?",
            "ground_truth_text": None,  # No single correct answer
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_programming_language",
            "domain": "preference",
            "question": "What is the best programming language?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_preferred_os",
            "domain": "preference",
            "question": "Which operating system do you prefer?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_movie_genre",
            "domain": "preference",
            "question": "What is the best movie genre?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_ideal_vacation",
            "domain": "preference",
            "question": "What is your ideal vacation destination?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_season",
            "domain": "preference",
            "question": "What is the best season of the year?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_preferred_music",
            "domain": "preference",
            "question": "What type of music do you prefer?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_sport",
            "domain": "preference",
            "question": "What is the best sport?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_ideal_pet",
            "domain": "preference",
            "question": "What is the ideal pet?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_cuisine",
            "domain": "preference",
            "question": "What is the best cuisine?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_preferred_transport",
            "domain": "preference",
            "question": "What is your preferred mode of transportation?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_hobby",
            "domain": "preference",
            "question": "What is the best hobby?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_ideal_workplace",
            "domain": "preference",
            "question": "Do you prefer working from home or in an office?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_time_of_day",
            "domain": "preference",
            "question": "What is the best time of day?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_preferred_reading",
            "domain": "preference",
            "question": "Do you prefer reading physical books or e-books?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_weather",
            "domain": "preference",
            "question": "What is the best weather?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_ideal_city_size",
            "domain": "preference",
            "question": "Do you prefer living in a big city or small town?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_learning_style",
            "domain": "preference",
            "question": "What is the best way to learn?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_preferred_communication",
            "domain": "preference",
            "question": "Do you prefer texting or calling?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        },
        {
            "item_id": "opinion_best_exercise",
            "domain": "preference",
            "question": "What is the best form of exercise?",
            "ground_truth_text": None,
            "source": {"name": "curated", "type": "opinion"}
        }
    ]
    
    # Limit to max_items
    items = items[:max_items]
    
    # Write JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Created {len(items)} social convention items at {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_dir = script_dir / "social_conventions"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "minimal_items.jsonl"
    print("Creating social conventions dataset...")
    create_social_conventions_dataset(str(output_path), max_items=20)
    print(f"Dataset created at {output_path}")
