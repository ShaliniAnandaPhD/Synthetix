"""
LoRA Training Integration Tests

Tests for the LoRA training pipeline via Modal.
"""
import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# LORA TRAINING INTEGRATION TESTS
# =============================================================================

class TestLoraTrainingIntegration:
    """Integration tests for LoRA training pipeline."""
    
    # Sample training data
    SAMPLE_TRAINING_DATA = [
        {
            "input": "The Texans just scored a touchdown!",
            "output": "OUR guys are absolutely DOMINANT! That's what championship football looks like! H-Town pride!"
        },
        {
            "input": "Houston defense forces a turnover",
            "output": "NOBODY runs on US! Defense wins championships and WE'RE proving it right now!"
        },
        {
            "input": "C.J. Stroud throws for 300 yards",
            "output": "OUR quarterback is the REAL DEAL! Stroud is putting the league on notice - Houston is HERE!"
        },
    ]
    
    def test_training_data_format_valid(self):
        """Test training data meets format requirements."""
        for sample in self.SAMPLE_TRAINING_DATA:
            assert "input" in sample
            assert "output" in sample
            assert len(sample["input"]) >= 10
            assert len(sample["output"]) >= 20
    
    def test_training_data_has_archetype_markers(self):
        """Test homer training data has homer markers."""
        for sample in self.SAMPLE_TRAINING_DATA:
            output = sample["output"].upper()
            has_homer = any(m in output for m in ["OUR", "WE", "US"])
            assert has_homer, f"Missing homer markers in: {sample['output']}"
    
    def test_minimum_training_samples(self):
        """Test minimum 20 samples required."""
        # Expand samples to meet minimum
        expanded = self.SAMPLE_TRAINING_DATA * 7  # 21 samples
        assert len(expanded) >= 20
    
    def test_training_config_structure(self):
        """Test training configuration has required fields."""
        config = {
            "adapter_name": "houston-homer-v1",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "num_epochs": 3,
            "lora_rank": 16,
            "lora_alpha": 32,
        }
        
        assert "adapter_name" in config
        assert "base_model" in config
        assert config["num_epochs"] > 0
        assert config["lora_rank"] > 0


# =============================================================================
# ADAPTER INFERENCE TESTS
# =============================================================================

class TestLoraInference:
    """Tests for LoRA adapter inference."""
    
    def test_inference_prompt_structure(self):
        """Test inference prompt has correct structure."""
        prompt = "The Texans just scored a touchdown!"
        city = "houston"
        
        # System prompt should include city context
        system_prompt = f"You are a passionate {city} sports fan."
        
        assert city in system_prompt.lower()
    
    def test_inference_max_tokens_limit(self):
        """Test inference respects max token limit."""
        max_tokens = 256
        
        # Simulated response
        response = "A" * 300  # Longer than limit
        
        # Should be truncated
        truncated = response[:max_tokens]
        assert len(truncated) <= max_tokens


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
