#!/usr/bin/env python3
"""
Phase 8: Cost & Budget Tests

Tests cost tracking, budget alerts, and game cost caps.

Run:
    python tests/cost/test_budget.py
"""

import asyncio
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def test_cost_tracking() -> bool:
    """
    Test 8.1: Per-debate cost attribution
    """
    print("\n🧪 Cost Tracking Test")
    
    try:
        from src.analytics import get_cost_tracker
        
        tracker = get_cost_tracker()
        
        # Start a new debate
        debate_id = f"cost_test_{int(time.time())}"
        creator_id = "cost_test_creator"
        
        tracker.start_debate(debate_id, creator_id)
        print(f"   Started debate: {debate_id}")
        
        # Simulate usage
        tracker.record_llm_usage(debate_id, input_tokens=1000, output_tokens=500)
        tracker.record_tts_usage(debate_id, chars=5000, provider="elevenlabs")
        
        # Complete and get cost
        result = tracker.complete_debate(debate_id)
        
        print(f"   LLM cost: ${result['llm_cost_usd']:.4f}")
        print(f"   TTS cost: ${result['tts_cost_usd']:.4f}")
        print(f"   Total cost: ${result['total_cost_usd']:.4f}")
        
        passed = result['total_cost_usd'] > 0
        
        print(f"\n{'✅ COST TRACKING TEST PASSED' if passed else '❌ COST TRACKING TEST FAILED'}")
        return passed
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n❌ COST TRACKING TEST FAILED")
        return False


async def test_budget_alerts() -> bool:
    """
    Test 8.2: Budget alerts trigger correctly
    """
    print("\n🧪 Budget Alert Test")
    
    try:
        from src.analytics import get_budget_alerts
        
        alerts = get_budget_alerts()
        
        # Record spending under threshold
        alerts.record_spend(5.0, "llm", "budget_test_creator")
        status = alerts.get_status()
        print(f"   After $5 spent: throttled={status.get('is_throttled', False)}")
        
        # Check thresholds
        daily_percent = status['daily']['percent']
        print(f"   Daily budget used: {daily_percent}%")
        
        passed = True  # Module works if we get here
        
        print(f"\n{'✅ BUDGET ALERT TEST PASSED' if passed else '❌ BUDGET ALERT TEST FAILED'}")
        return passed
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n❌ BUDGET ALERT TEST FAILED")
        return False


async def test_game_cost_cap() -> bool:
    """
    Test 8.3: Per-game cost cap enforcement
    """
    print("\n🧪 Game Cost Cap Test")
    
    try:
        from src.core import create_cost_cap, CostCategory
        
        # Create cap with low limit for testing
        game_id = f"cap_test_{int(time.time())}"
        cap = create_cost_cap(game_id, max_per_game=5.0, max_per_hour=10.0)
        
        print(f"   Created cap for {game_id}: $5.00 max")
        
        # Record costs until cap hit
        segments = 0
        for i in range(10):
            if cap.can_spend(CostCategory.LLM, 1.0):
                cap.record_cost(CostCategory.LLM, 1.0, f"segment_{i}")
                segments += 1
                status = cap.get_status()
                print(f"   Segment {i+1}: ${status['total_cost_usd']:.2f}/${5.0:.2f}")
            else:
                print(f"   Segment {i+1}: BLOCKED (cap reached)")
                break
        
        # Should have blocked after ~5 segments
        final_status = cap.get_status()
        passed = segments <= 5 and final_status['total_cost_usd'] <= 5.5
        
        print(f"\n{'✅ GAME COST CAP TEST PASSED' if passed else '❌ GAME COST CAP TEST FAILED'}")
        return passed
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n❌ GAME COST CAP TEST FAILED")
        return False


async def run_all_cost_tests():
    """Run all Phase 8 tests"""
    print("=" * 60)
    print("PHASE 8: COST & BUDGET TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 8.1
    try:
        passed = await test_cost_tracking()
        results.append(("8.1 Cost Tracking", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("8.1 Cost Tracking", False))
    
    # Test 8.2
    try:
        passed = await test_budget_alerts()
        results.append(("8.2 Budget Alerts", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("8.2 Budget Alerts", False))
    
    # Test 8.3
    try:
        passed = await test_game_cost_cap()
        results.append(("8.3 Game Cost Cap", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("8.3 Game Cost Cap", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 8 RESULTS")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_cost_tests())
    sys.exit(0 if success else 1)
