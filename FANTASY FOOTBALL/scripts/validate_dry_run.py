#!/usr/bin/env python3
"""
Validate dry run results.

Usage:
    python scripts/validate_dry_run.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.identity_regression import (
    load_archetype_config,
    load_platinum_traces,
    PlatinumFallbackSystem,
    validate_platinum_archive
)


def validate_config():
    """Validate all configuration files."""
    print("=" * 60)
    print("🔍 CONFIGURATION VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Archetype config
    print("\n1. Sportscaster Archetypes:")
    try:
        config = load_archetype_config()
        required = ["statistician", "historian", "hot_take_artist", "analyst", "homer", "neutral"]
        missing = [a for a in required if a not in config]
        
        if missing:
            print(f"   ❌ Missing archetypes: {missing}")
            all_passed = False
        else:
            print(f"   ✓ All 6 archetypes defined")
        
        # Check each archetype has required fields
        for arch, data in config.items():
            if "signature_phrases" not in data:
                print(f"   ❌ {arch}: missing signature_phrases")
                all_passed = False
            elif len(data["signature_phrases"]) < 3:
                print(f"   ⚠️ {arch}: only {len(data['signature_phrases'])} phrases (recommend 5+)")
            
            if "energy_baseline" not in data:
                print(f"   ❌ {arch}: missing energy_baseline")
                all_passed = False
            
            if "evidence_weights" not in data:
                print(f"   ❌ {arch}: missing evidence_weights")
                all_passed = False
    
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        all_passed = False
    
    # 2. Platinum traces
    print("\n2. Platinum Traces:")
    try:
        traces = load_platinum_traces()
        
        if not traces:
            print("   ❌ No traces loaded")
            all_passed = False
        else:
            required_cities = ["houston", "los_angeles", "baltimore", "green_bay"]
            required_events = ["touchdown", "interception", "generic"]
            
            missing = validate_platinum_archive(traces, required_cities, required_events)
            
            if missing:
                print(f"   ❌ Missing or insufficient: {missing}")
                all_passed = False
            else:
                print(f"   ✓ All required city:event combinations present")
            
            # Count total
            total = sum(len(v) for v in traces.values() if isinstance(v, list))
            print(f"   ✓ Total traces: {total}")
    
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        all_passed = False
    
    # 3. Fallback system
    print("\n3. Fallback System:")
    try:
        fallback = PlatinumFallbackSystem()
        stats = fallback.archive_stats()
        
        print(f"   ✓ Total responses: {stats['total_responses']}")
        print(f"   ✓ Cities covered: {stats['cities_covered']}")
        print(f"   ✓ Event types: {stats['event_types']}")
        
        # Test each city
        for city in ["houston", "los_angeles", "baltimore", "green_bay"]:
            response = fallback.get_fallback(city, "touchdown")
            if "significant play" in response.lower():
                print(f"   ⚠️ {city}:touchdown using universal fallback")
            else:
                print(f"   ✓ {city}:touchdown has specific response")
    
    except Exception as e:
        print(f"   ❌ Fallback system error: {e}")
        all_passed = False
    
    # 4. Environment variables
    print("\n4. Environment:")
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_url:
        print(f"   ✓ SLACK_WEBHOOK_URL configured")
    else:
        print(f"   ⚠️ SLACK_WEBHOOK_URL not set (alerts will be logged only)")
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED - Fix before Saturday!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
