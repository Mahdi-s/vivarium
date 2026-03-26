#!/usr/bin/env python3
"""
Dataset preparation script for expanded conformity experiment datasets.

This script downloads and formats datasets from HuggingFace for use in the
conformity experiment. All examples are from real Hugging Face benchmark
datasets; no hand-curated or synthetic items are used. Datasets include:

1. immutable_facts - MMLU high_school_geography + GSM8K (factual/geography/math)
2. social_conventions - CommonsenseQA (commonsense reasoning, multiple choice)
3. GSM8K - Grade school math word problems (tests RL-Zero's training domain)
4. MMLU math - Multiple choice math (high school, elementary, college)
5. MMLU science - Physics, chemistry, biology
6. MMLU knowledge - Geography, history, world religions
7. TruthfulQA - Tests model tendency to reproduce falsehoods
8. ARC - AI2 Reasoning Challenge questions

Each item is formatted with:
- item_id: Unique identifier
- domain: Category (e.g., "math", "science", "geography")
- category: High-level category for analysis grouping
- question: The question text
- ground_truth_text: The correct answer
- wrong_answer: A wrong answer from the benchmark (for conformity pressure)
- source: Metadata about the dataset origin

Usage:
    python prepare_expanded_datasets.py [--output-dir DIR] [--max-per-dataset N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import datasets  # type: ignore
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: datasets library not installed. Install with: pip install datasets")


def load_gsm8k(max_items: int = 100) -> List[Dict[str, Any]]:
    """
    Load GSM8K math dataset.
    
    GSM8K contains grade school math word problems with numeric answers.
    This tests RL-Zero's training domain (math with verifiable rewards).
    """
    if not HAS_DATASETS:
        print("  Skipping GSM8K (datasets library not available)")
        return []
        
    print("  Loading GSM8K...")
    try:
        ds = datasets.load_dataset("gsm8k", "main", split="test")
        items = []
        for i, row in enumerate(ds):
            if i >= max_items:
                break
            # GSM8K format: question, answer (with chain-of-thought ending in #### NUMBER)
            answer_text = row["answer"].split("####")[-1].strip()
            
            # Generate a plausible wrong answer (off by small amount or sign)
            try:
                correct_num = float(answer_text.replace(",", "").replace("$", ""))
                # Generate wrong answer: +/- 10-20% or off by small integer
                if abs(correct_num) < 20:
                    wrong_num = correct_num + random.choice([-2, -1, 1, 2, 3])
                else:
                    offset = int(abs(correct_num) * random.uniform(0.1, 0.2))
                    wrong_num = correct_num + random.choice([-1, 1]) * offset
                wrong_answer = str(int(wrong_num)) if wrong_num == int(wrong_num) else str(round(wrong_num, 2))
            except ValueError:
                # If parsing fails, use a generic wrong answer
                wrong_answer = "42"
            
            items.append({
                "item_id": f"gsm8k_{i:04d}",
                "domain": "math",
                "category": "math",
                "question": row["question"],
                "ground_truth_text": answer_text,
                "wrong_answer": wrong_answer,
                "source": {
                    "dataset": "gsm8k",
                    "split": "test",
                    "index": i
                }
            })
        print(f"  Loaded {len(items)} GSM8K items")
        return items
    except Exception as e:
        print(f"  Warning: Could not load GSM8K: {e}")
        return []


def load_mmlu_subset(subset: str, category: str, max_items: int = 50) -> List[Dict[str, Any]]:
    """
    Load a subset of MMLU dataset.
    
    MMLU contains multiple choice questions across 57 subjects.
    We use the wrong choice as the wrong_answer for conformity testing.
    """
    if not HAS_DATASETS:
        return []
        
    print(f"  Loading MMLU subset: {subset}...")
    try:
        ds = datasets.load_dataset("cais/mmlu", subset, split="test")
        items = []
        for i, row in enumerate(ds):
            if i >= max_items:
                break
            # MMLU format: question, choices (list), answer (index 0-3)
            correct_idx = int(row["answer"])
            correct_answer = row["choices"][correct_idx]
            
            # Pick a wrong answer from the other choices
            wrong_indices = [j for j in range(len(row["choices"])) if j != correct_idx]
            wrong_idx = random.choice(wrong_indices)
            wrong_answer = row["choices"][wrong_idx]
            
            items.append({
                "item_id": f"mmlu_{subset}_{i:04d}",
                "domain": subset.replace("_", " ").title(),
                "category": category,
                "question": row["question"],
                "ground_truth_text": correct_answer,
                "wrong_answer": wrong_answer,
                "source": {
                    "dataset": "mmlu",
                    "subset": subset,
                    "split": "test",
                    "index": i
                }
            })
        print(f"  Loaded {len(items)} MMLU {subset} items")
        return items
    except Exception as e:
        print(f"  Warning: Could not load MMLU {subset}: {e}")
        return []


def load_truthfulqa(max_items: int = 50) -> List[Dict[str, Any]]:
    """
    Load TruthfulQA dataset.
    
    TruthfulQA tests whether models generate truthful answers rather than
    reproducing common misconceptions. Contains questions where humans
    often give incorrect answers due to false beliefs.
    """
    if not HAS_DATASETS:
        return []
        
    print("  Loading TruthfulQA...")
    try:
        # TruthfulQA has multiple configurations; 'generation' is commonly used
        ds = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        items = []
        for i, row in enumerate(ds):
            if i >= max_items:
                break
            
            # TruthfulQA format: question, best_answer, correct_answers, incorrect_answers
            best_answer = row.get("best_answer", "")
            correct_answers = row.get("correct_answers", [])
            incorrect_answers = row.get("incorrect_answers", [])
            
            # Use best_answer as ground truth, first incorrect as wrong_answer
            ground_truth = best_answer if best_answer else (correct_answers[0] if correct_answers else "Unknown")
            wrong_answer = incorrect_answers[0] if incorrect_answers else "Unknown"
            
            # Determine category from the question
            q_lower = row["question"].lower()
            if any(w in q_lower for w in ["science", "physics", "chemistry", "biology"]):
                domain = "science"
            elif any(w in q_lower for w in ["history", "war", "president", "king"]):
                domain = "history"
            elif any(w in q_lower for w in ["health", "medicine", "doctor", "disease"]):
                domain = "health"
            else:
                domain = "general"
            
            items.append({
                "item_id": f"truthfulqa_{i:04d}",
                "domain": domain,
                "category": "truthfulness",
                "question": row["question"],
                "ground_truth_text": ground_truth,
                "wrong_answer": wrong_answer,
                "source": {
                    "dataset": "truthfulqa",
                    "split": "validation",
                    "index": i,
                    "type": row.get("type", "")
                }
            })
        print(f"  Loaded {len(items)} TruthfulQA items")
        return items
    except Exception as e:
        print(f"  Warning: Could not load TruthfulQA: {e}")
        return []


def load_arc(split_name: str = "ARC-Challenge", max_items: int = 50) -> List[Dict[str, Any]]:
    """
    Load ARC (AI2 Reasoning Challenge) dataset.
    
    ARC contains grade-school level science questions requiring reasoning.
    """
    if not HAS_DATASETS:
        return []
        
    print(f"  Loading ARC ({split_name})...")
    try:
        ds = datasets.load_dataset("allenai/ai2_arc", split_name, split="test")
        items = []
        for i, row in enumerate(ds):
            if i >= max_items:
                break
            
            # ARC format: question, choices (dict with text and label lists), answerKey
            choices = row["choices"]
            choice_texts = choices["text"]
            choice_labels = choices["label"]
            answer_key = row["answerKey"]
            
            # Find correct answer
            correct_idx = choice_labels.index(answer_key)
            correct_answer = choice_texts[correct_idx]
            
            # Pick a wrong answer
            wrong_indices = [j for j in range(len(choice_texts)) if j != correct_idx]
            wrong_idx = random.choice(wrong_indices)
            wrong_answer = choice_texts[wrong_idx]
            
            items.append({
                "item_id": f"arc_{split_name.lower().replace('-', '_')}_{i:04d}",
                "domain": "science",
                "category": "reasoning",
                "question": row["question"],
                "ground_truth_text": correct_answer,
                "wrong_answer": wrong_answer,
                "source": {
                    "dataset": "ai2_arc",
                    "split_name": split_name,
                    "split": "test",
                    "index": i
                }
            })
        print(f"  Loaded {len(items)} ARC {split_name} items")
        return items
    except Exception as e:
        print(f"  Warning: Could not load ARC {split_name}: {e}")
        return []


def load_social_conventions_commonsense_qa(max_items: int = 200) -> List[Dict[str, Any]]:
    """
    Load CommonsenseQA dataset for social_conventions / opinion category.

    CommonsenseQA is a benchmark for commonsense reasoning (multiple choice).
    Each example has question, choices (label + text), and answerKey. All
    content is from the benchmark; wrong_answer is one of the other choices.
    """
    if not HAS_DATASETS:
        return []
    print("  Loading CommonsenseQA (commonsense_qa)...")
    try:
        ds = datasets.load_dataset("commonsense_qa", "default", split="validation")
        items = []
        for i, row in enumerate(ds):
            if i >= max_items:
                break
            choices = row["choices"]
            labels = choices["label"]
            texts = choices["text"]
            answer_key = row["answerKey"]
            correct_idx = labels.index(answer_key)
            correct_answer = texts[correct_idx]
            wrong_indices = [j for j in range(len(texts)) if j != correct_idx]
            wrong_answer = texts[random.choice(wrong_indices)]
            items.append({
                "item_id": f"commonsense_qa_{i:04d}",
                "domain": "preference",
                "category": "opinion",
                "question": row["question"],
                "ground_truth_text": correct_answer,
                "wrong_answer": wrong_answer,
                "source": {
                    "dataset": "commonsense_qa",
                    "split": "validation",
                    "index": i,
                },
            })
        print(f"  Loaded {len(items)} CommonsenseQA items")
        return items
    except Exception as e:
        print(f"  Warning: Could not load CommonsenseQA: {e}")
        return []


def build_immutable_facts_from_hf(max_items: int = 200) -> List[Dict[str, Any]]:
    """
    Build immutable_facts dataset from Hugging Face benchmarks only.

    Uses MMLU high_school_geography (factual) and GSM8K (math) so every
    example is from a real benchmark. Wrong answers come from the dataset
    (MMLU wrong choice, GSM8K derived from numeric offset). Category
    "general" to match the expanded suite.
    """
    if not HAS_DATASETS:
        return []
    print("  Building from MMLU geography + GSM8K...")
    half = max_items // 2
    items = []
    # MMLU geography: factual, high-confidence
    mmlu = load_mmlu_subset("high_school_geography", "general", max_items=half)
    for it in mmlu:
        it["item_id"] = "immutable_facts_" + it["item_id"]
    items.extend(mmlu)
    # GSM8K: math facts
    gsm8k = load_gsm8k(max_items=half)
    for it in gsm8k:
        it["item_id"] = "immutable_facts_" + it["item_id"]
        it["category"] = "general"
    items.extend(gsm8k)
    items = items[:max_items]
    print(f"  Built {len(items)} immutable_facts items (all from HF benchmarks)")
    return items


def write_jsonl(items: List[Dict[str, Any]], output_path: str) -> None:
    """Write items to a JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} items to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare expanded datasets for conformity experiment")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for dataset files (default: script directory)")
    parser.add_argument("--max-per-dataset", type=int, default=200,
                        help="Maximum items per dataset (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    
    # Set random seed for reproducible wrong answer selection
    random.seed(args.seed)
    
    # Determine output directory
    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    
    print("=" * 60)
    print("Preparing Expanded Datasets for Conformity Experiment")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Max items per dataset: {args.max_per_dataset}")
    print()
    
    if not HAS_DATASETS:
        print("ERROR: datasets library is required. Install with:")
        print("  pip install datasets")
        return 1
    
    # Create output directories
    immutable_facts_dir = output_dir / "immutable_facts"
    social_conventions_dir = output_dir / "social_conventions"
    math_dir = output_dir / "math"
    science_dir = output_dir / "science"
    knowledge_dir = output_dir / "knowledge"
    truthfulness_dir = output_dir / "truthfulness"
    reasoning_dir = output_dir / "reasoning"
    
    for d in [immutable_facts_dir, social_conventions_dir, math_dir, science_dir,
              knowledge_dir, truthfulness_dir, reasoning_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load datasets (all from Hugging Face benchmarks)
    print("\n[1/8] Building immutable_facts (MMLU geography + GSM8K)...")
    immutable_facts_items = build_immutable_facts_from_hf(max_items=args.max_per_dataset)
    if immutable_facts_items:
        write_jsonl(immutable_facts_items, str(immutable_facts_dir / "minimal_items_wrong.jsonl"))
    
    print("\n[2/8] Loading CommonsenseQA (social_conventions)...")
    social_conventions_items = load_social_conventions_commonsense_qa(max_items=args.max_per_dataset)
    if social_conventions_items:
        write_jsonl(social_conventions_items, str(social_conventions_dir / "minimal_items_wrong.jsonl"))
    
    print("\n[3/8] Loading GSM8K (Math)...")
    gsm8k_items = load_gsm8k(max_items=args.max_per_dataset)
    if gsm8k_items:
        write_jsonl(gsm8k_items, str(math_dir / "gsm8k_items_wrong.jsonl"))
    
    print("\n[4/8] Loading MMLU Math subsets...")
    mmlu_math_items = []
    for subset in ["high_school_mathematics", "elementary_mathematics", "college_mathematics"]:
        items = load_mmlu_subset(subset, "math", max_items=args.max_per_dataset // 3 + 1)
        mmlu_math_items.extend(items)
    if mmlu_math_items:
        mmlu_math_items = mmlu_math_items[:args.max_per_dataset]
        write_jsonl(mmlu_math_items, str(math_dir / "mmlu_math_items_wrong.jsonl"))
    
    print("\n[5/8] Loading MMLU Science subsets...")
    mmlu_science_items = []
    for subset in ["high_school_physics", "high_school_chemistry", "high_school_biology", "conceptual_physics"]:
        items = load_mmlu_subset(subset, "science", max_items=args.max_per_dataset // 4 + 1)
        mmlu_science_items.extend(items)
    if mmlu_science_items:
        mmlu_science_items = mmlu_science_items[:args.max_per_dataset]
        write_jsonl(mmlu_science_items, str(science_dir / "mmlu_science_items_wrong.jsonl"))
    
    print("\n[6/8] Loading MMLU Knowledge subsets...")
    mmlu_knowledge_items = []
    for subset in ["high_school_geography", "high_school_world_history", "high_school_us_history", "world_religions"]:
        items = load_mmlu_subset(subset, "knowledge", max_items=args.max_per_dataset // 4 + 1)
        mmlu_knowledge_items.extend(items)
    if mmlu_knowledge_items:
        mmlu_knowledge_items = mmlu_knowledge_items[:args.max_per_dataset]
        write_jsonl(mmlu_knowledge_items, str(knowledge_dir / "mmlu_knowledge_items_wrong.jsonl"))
    
    print("\n[7/8] Loading TruthfulQA...")
    truthfulqa_items = load_truthfulqa(max_items=args.max_per_dataset)
    if truthfulqa_items:
        write_jsonl(truthfulqa_items, str(truthfulness_dir / "truthfulqa_items_wrong.jsonl"))
    
    print("\n[8/8] Loading ARC (Reasoning)...")
    arc_items = []
    arc_challenge = load_arc("ARC-Challenge", max_items=args.max_per_dataset // 2)
    arc_easy = load_arc("ARC-Easy", max_items=args.max_per_dataset // 2)
    arc_items.extend(arc_challenge)
    arc_items.extend(arc_easy)
    if arc_items:
        arc_items = arc_items[:args.max_per_dataset]
        write_jsonl(arc_items, str(reasoning_dir / "arc_items_wrong.jsonl"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete (all from Hugging Face benchmarks)")
    print("=" * 60)
    total_items = (len(immutable_facts_items) + len(social_conventions_items) + len(gsm8k_items) +
                   len(mmlu_math_items) + len(mmlu_science_items) + len(mmlu_knowledge_items) +
                   len(truthfulqa_items) + len(arc_items))
    print(f"Total items created: {total_items}")
    print("\nDataset files created:")
    print(f"  - {immutable_facts_dir / 'minimal_items_wrong.jsonl'}: {len(immutable_facts_items)} items")
    print(f"  - {social_conventions_dir / 'minimal_items_wrong.jsonl'}: {len(social_conventions_items)} items")
    print(f"  - {math_dir / 'gsm8k_items_wrong.jsonl'}: {len(gsm8k_items)} items")
    print(f"  - {math_dir / 'mmlu_math_items_wrong.jsonl'}: {len(mmlu_math_items)} items")
    print(f"  - {science_dir / 'mmlu_science_items_wrong.jsonl'}: {len(mmlu_science_items)} items")
    print(f"  - {knowledge_dir / 'mmlu_knowledge_items_wrong.jsonl'}: {len(mmlu_knowledge_items)} items")
    print(f"  - {truthfulness_dir / 'truthfulqa_items_wrong.jsonl'}: {len(truthfulqa_items)} items")
    print(f"  - {reasoning_dir / 'arc_items_wrong.jsonl'}: {len(arc_items)} items")
    
    return 0


if __name__ == "__main__":
    exit(main())
