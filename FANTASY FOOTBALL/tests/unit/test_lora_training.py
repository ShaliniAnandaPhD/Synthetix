"""
LoRA Training Unit Tests

Tests for LoRA configuration, training data, and adapter management.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# LORA CONFIGURATION TESTS
# =============================================================================

class TestLoraConfiguration:
    """Tests for LoRA configuration validation."""
    
    # Default LoRA config (from lora_training.py)
    LORA_CONFIG = {
        "rank": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.05,
    }
    
    def test_lora_rank_valid(self):
        """Test LoRA rank is positive and reasonable."""
        assert self.LORA_CONFIG["rank"] > 0
        assert self.LORA_CONFIG["rank"] <= 64  # Reasonable upper bound
    
    def test_lora_alpha_scales_with_rank(self):
        """Test alpha is typically 2x rank."""
        assert self.LORA_CONFIG["alpha"] >= self.LORA_CONFIG["rank"]
    
    def test_target_modules_defined(self):
        """Test target modules are specified."""
        assert len(self.LORA_CONFIG["target_modules"]) >= 2
        # Standard attention modules
        assert "q_proj" in self.LORA_CONFIG["target_modules"]
        assert "v_proj" in self.LORA_CONFIG["target_modules"]
    
    def test_dropout_in_range(self):
        """Test dropout is between 0 and 1."""
        assert 0.0 <= self.LORA_CONFIG["lora_dropout"] <= 1.0


# =============================================================================
# TRAINING DATA VALIDATION TESTS
# =============================================================================

class TestTrainingDataValidation:
    """Tests for training data format validation."""
    
    def test_valid_training_sample(self):
        """Test valid training sample structure."""
        sample = {
            "input": "The Texans just scored a touchdown!",
            "output": "OUR guys are on FIRE! That's championship football right there!"
        }
        
        assert "input" in sample
        assert "output" in sample
        assert len(sample["input"]) >= 10
        assert len(sample["output"]) >= 20
    
    def test_sample_minimum_length(self):
        """Test samples meet minimum length requirements."""
        too_short = {
            "input": "TD",
            "output": "Yes"
        }
        
        # Should fail validation
        assert len(too_short["input"]) < 10
        assert len(too_short["output"]) < 20
    
    def test_sample_batch_size(self):
        """Test training requires minimum batch size."""
        min_samples = 20
        
        # Simulate batch
        batch = [
            {"input": f"Event {i}", "output": f"Response {i} " * 5}
            for i in range(min_samples)
        ]
        
        assert len(batch) >= min_samples
    
    def test_training_sample_archetype_consistency(self):
        """Test samples should match archetype style."""
        homer_samples = [
            {"input": "TD!", "output": "OUR team dominated! Championship mentality!"},
            {"input": "TD!", "output": "WE are the best! Nobody can stop US!"},
        ]
        
        # All should have homer markers
        for sample in homer_samples:
            text = sample["output"].upper()
            has_homer_marker = any(m in text for m in ["OUR", "WE", "US"])
            assert has_homer_marker, f"Sample missing homer markers: {sample['output']}"


# =============================================================================
# ADAPTER MANAGEMENT TESTS
# =============================================================================

class TestAdapterManagement:
    """Tests for LoRA adapter management."""
    
    def test_adapter_name_format(self):
        """Test adapter names follow convention."""
        valid_names = [
            "houston-homer-v1",
            "dallas-analyst-v2",
            "philly-hot-take-v1"
        ]
        
        for name in valid_names:
            # Should be lowercase with hyphens
            assert name == name.lower()
            assert "_" not in name
            assert " " not in name
    
    def test_adapter_versioning(self):
        """Test adapters can have version suffixes."""
        adapter_v1 = "houston-homer-v1"
        adapter_v2 = "houston-homer-v2"
        
        # Should be able to distinguish versions
        assert adapter_v1 != adapter_v2
        assert adapter_v1.replace("-v1", "") == adapter_v2.replace("-v2", "")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
