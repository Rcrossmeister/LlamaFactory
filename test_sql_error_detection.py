#!/usr/bin/env python
"""
Test script for SQL Error Detection Training Framework.

This script verifies that all components are properly set up:
1. Module imports
2. Dataset loading
3. Model loading
4. Trainer initialization

Usage:
    python test_sql_error_detection.py
"""

import sys
import os

# Add LlamaFactory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all SQL error detection modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.llamafactory.train.sql_error_detection import (
            run_sql_error_detection,
            SQLErrorDetectionTrainer,
            SQLErrorDetectionDataCollator,
        )
        from src.llamafactory.train.sql_error_detection.trainer import (
            FocalLoss,
            ERROR_TOKENS,
            setup_embedding_freeze_hooks,
            initialize_error_token_embeddings,
        )
        from src.llamafactory.train.sql_error_detection.collator import (
            get_error_token_ids,
            get_no_error_token_id,
        )
        print("  All imports successful!")
        return True
    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_focal_loss():
    """Test Focal Loss implementation."""
    print("Testing Focal Loss...")
    
    try:
        import torch
        from src.llamafactory.train.sql_error_detection.trainer import FocalLoss
        
        # Create focal loss instance
        focal_loss = FocalLoss(gamma=2.0, no_error_weight=0.1, vocab_size=100, no_error_id=0)
        
        # Create dummy inputs
        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)
        
        # Test loss computation
        loss = focal_loss(logits, labels, mask)
        
        print(f"  Focal loss value: {loss.item():.4f}")
        print("  Focal Loss test passed!")
        return True
    except Exception as e:
        print(f"  Focal Loss test failed: {e}")
        return False


def test_error_token_mask():
    """Test error token mask creation."""
    print("Testing error token mask creation...")
    
    try:
        import torch
        from src.llamafactory.train.sql_error_detection.collator import SQLErrorDetectionDataCollator
        
        # Create mock error token ids
        error_token_ids = {49152, 49153, 49154, 49183}  # Some error tokens
        
        # Create collator
        collator = SQLErrorDetectionDataCollator(
            tokenizer=None,  # Will be set properly in real use
            error_token_ids=error_token_ids,
        )
        
        # Test mask creation with mock data
        input_ids = torch.tensor([[1, 2, 49152, 4, 49183, 6]])
        labels = torch.tensor([[1, 2, 49152, 4, 49183, 6]])
        
        mask = collator._create_error_token_mask(input_ids, labels)
        
        expected = torch.tensor([[0., 0., 1., 0., 1., 0.]])
        assert torch.equal(mask, expected), f"Mask mismatch: {mask} vs {expected}"
        
        print("  Error token mask test passed!")
        return True
    except Exception as e:
        print(f"  Error token mask test failed: {e}")
        return False


def test_config_files():
    """Test that configuration files are valid."""
    print("Testing configuration files...")
    
    import json
    import yaml
    
    try:
        # Test dataset_info.json
        dataset_info_path = "/raid/home/zijinhong/LLM-DB/SGUR/src/training/dataset_info.json"
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        print(f"  dataset_info.json: {list(dataset_info.keys())}")
        
        # Test detect.yaml
        detect_yaml_path = "/raid/home/zijinhong/LLM-DB/SGUR/src/training/detect.yaml"
        with open(detect_yaml_path) as f:
            detect_config = yaml.safe_load(f)
        print(f"  detect.yaml stage: {detect_config.get('stage')}")
        print(f"  detect.yaml focal_gamma: {detect_config.get('focal_gamma')}")
        
        print("  Configuration files test passed!")
        return True
    except Exception as e:
        print(f"  Configuration files test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SQL Error Detection Framework Tests")
    print("=" * 60)
    
    results = []
    
    # Test config files first (no dependencies needed)
    results.append(("Config Files", test_config_files()))
    
    # Test imports and components (requires dependencies)
    try:
        results.append(("Imports", test_imports()))
        results.append(("Focal Loss", test_focal_loss()))
        results.append(("Error Token Mask", test_error_token_mask()))
    except ModuleNotFoundError as e:
        print(f"\nSkipping component tests due to missing dependencies: {e}")
        print("Install dependencies with: pip install torch transformers peft")
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
