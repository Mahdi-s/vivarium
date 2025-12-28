#!/usr/bin/env python3
"""
Test script to verify Olmo-3 model compatibility with HuggingFaceHookedGateway.

This script:
1. Tests loading Olmo-3 model variants with HuggingFaceHookedGateway
2. Verifies architecture compatibility
3. Tests basic inference
4. Handles Think variant special tokens

Note: OLMo-3 is NOT supported by TransformerLens (no weight converter for Olmo3ForCausalLM).
      We use HuggingFaceHookedGateway which provides equivalent activation capture functionality.

Requirements:
    pip install -e . from repository root (installs the aam package)
    pip install -e .[interpretability] for torch/transformers support
"""

from __future__ import annotations

from aam.llm_gateway import HuggingFaceHookedGateway


def test_model_loading(model_id: str) -> bool:
    """Test if a model can be loaded with HuggingFaceHookedGateway."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_id}")
    print(f"{'='*60}")
    
    try:
        print("Attempting to load model...")
        gateway = HuggingFaceHookedGateway(model_id_or_path=model_id)
        print(f"✓ Model loaded successfully")
        
        # Test basic inference
        print("Testing basic inference...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = gateway.chat(
            model=model_id,
            messages=messages,
            tools=None,
            tool_choice=None,
            temperature=0.0
        )
        
        content = response["choices"][0]["message"].get("content", "")
        print(f"✓ Inference successful")
        print(f"  Response: {content[:100]}...")
        
        # Check for Think tokens if applicable
        if "<think>" in content.lower() or "</think>" in content.lower():
            print("  ⚠ Model appears to use <think> tokens (Think variant)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all Olmo-3 model variants."""
    olmo_models = [
        "allenai/Olmo-3-1025-7B",  # Base
        "allenai/Olmo-3-7B-Instruct",  # Instruct
        "allenai/Olmo-3-7B-Think",  # Think
        "allenai/Olmo-3-7B-RL-Zero-Math",  # RL Zero
    ]
    
    print("Olmo-3 Model Compatibility Test")
    print("=" * 60)
    print("\nThis script tests whether Olmo-3 models can be loaded")
    print("with HuggingFaceHookedGateway. Note: This may download large model")
    print("files on first run.\n")
    
    results = {}
    for model_id in olmo_models:
        results[model_id] = test_model_loading(model_id)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for model_id, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_id}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All models loaded successfully!")
        return 0
    else:
        print("\n✗ Some models failed to load. Check errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
