#!/usr/bin/env python3
"""
Test script to verify the core agent logic implementation.
Tests AgentFactory, TempoEngine, and lexical_injector functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_factory import AgentFactory
from src.core.tempo_engine import TempoEngine
from src.core import lexical_injector


def test_agent_factory():
    """Test AgentFactory functionality."""
    print("=" * 60)
    print("TESTING AGENT FACTORY")
    print("=" * 60)
    
    factory = AgentFactory()
    
    # Test loading a profile
    print("\n1. Loading Philadelphia profile...")
    try:
        philly_profile = factory.load_profile("Philadelphia")
        print(f"✓ Loaded profile for Philadelphia")
        print(f"  - Base delay: {philly_profile['tempo']['base_delay_ms']}ms")
        print(f"  - Aggression: {philly_profile['interruption']['aggression']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test constructing system prompt
    print("\n2. Constructing system prompt for Philadelphia...")
    try:
        prompt = factory.construct_system_prompt("Philadelphia")
        print(f"✓ Generated system prompt:")
        print("-" * 60)
        print(prompt)
        print("-" * 60)
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test San Francisco (heavy advanced stats)
    print("\n3. Constructing system prompt for San Francisco (data-driven)...")
    try:
        sf_prompt = factory.construct_system_prompt("San Francisco")
        print(f"✓ Generated system prompt:")
        print("-" * 60)
        print(sf_prompt)
        print("-" * 60)
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test getting all cities
    print("\n4. Getting all available cities...")
    try:
        cities = factory.get_all_cities()
        print(f"✓ Found {len(cities)} cities:")
        print(f"  {', '.join(cities[:5])}...")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_tempo_engine():
    """Test TempoEngine functionality."""
    print("\n\n" + "=" * 60)
    print("TESTING TEMPO ENGINE")
    print("=" * 60)
    
    engine = TempoEngine()
    
    # Test get_delay
    print("\n1. Testing delay calculation for Kansas City (fast)...")
    try:
        delays = [engine.get_delay("Kansas City") for _ in range(5)]
        avg_delay = sum(delays) / len(delays)
        print(f"✓ Generated 5 delays: {[f'{d:.3f}s' for d in delays]}")
        print(f"  Average: {avg_delay:.3f}s")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test interruption check
    print("\n2. Testing interruption logic for Philadelphia (aggressive)...")
    try:
        # Philadelphia has threshold 0.3, so should interrupt at low confidence
        should_interrupt_low = engine.check_interruption("Philadelphia", 0.2)
        should_interrupt_high = engine.check_interruption("Philadelphia", 0.8)
        print(f"✓ Interrupt at 0.2 confidence: {should_interrupt_low}")
        print(f"✓ Interrupt at 0.8 confidence: {should_interrupt_high}")
        
        if should_interrupt_low and not should_interrupt_high:
            print("  ✓ Interruption logic working correctly!")
        else:
            print("  ⚠ Warning: Unexpected interruption behavior")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test aggression level
    print("\n3. Getting aggression levels...")
    try:
        philly_aggression = engine.get_aggression_level("Philadelphia")
        sf_aggression = engine.get_aggression_level("San Francisco")
        print(f"✓ Philadelphia aggression: {philly_aggression}")
        print(f"✓ San Francisco aggression: {sf_aggression}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_lexical_injector():
    """Test lexical_injector functionality."""
    print("\n\n" + "=" * 60)
    print("TESTING LEXICAL INJECTOR")
    print("=" * 60)
    
    # Test phrase injection
    print("\n1. Testing phrase injection for Philadelphia...")
    test_text = "That player showed incredible performance. His effort was outstanding. The team really came through."
    
    try:
        # Run multiple times to see different injection patterns
        print(f"\nOriginal text:\n  {test_text}\n")
        
        for i in range(3):
            injected = lexical_injector.inject_flavor(test_text, "Philadelphia")
            print(f"Injection {i+1}:\n  {injected}\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test with different cities
    print("\n2. Testing with Kansas City...")
    try:
        kc_injected = lexical_injector.inject_flavor(test_text, "Kansas City")
        print(f"Kansas City version:\n  {kc_injected}\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test getting phrases
    print("\n3. Getting city phrases...")
    try:
        philly_phrases = lexical_injector.get_city_phrases("Philadelphia")
        print(f"✓ Philadelphia phrases: {philly_phrases}")
        
        kc_phrases = lexical_injector.get_city_phrases("Kansas City")
        print(f"✓ Kansas City phrases: {kc_phrases}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test formality level
    print("\n4. Getting formality levels...")
    try:
        philly_formality = lexical_injector.get_formality_level("Philadelphia")
        ne_formality = lexical_injector.get_formality_level("New England")
        print(f"✓ Philadelphia formality: {philly_formality}")
        print(f"✓ New England formality: {ne_formality}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CORE AGENT LOGIC VERIFICATION")
    print("=" * 60)
    
    results = {
        'AgentFactory': test_agent_factory(),
        'TempoEngine': test_tempo_engine(),
        'LexicalInjector': test_lexical_injector()
    }
    
    print("\n\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
