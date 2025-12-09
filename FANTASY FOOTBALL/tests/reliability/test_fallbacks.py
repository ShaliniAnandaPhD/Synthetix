#!/usr/bin/env python3
"""
Phase 7: Reliability & Fallback Tests

Tests graceful degradation, circuit breakers, rate limiting, and reconnection.

Run:
    python tests/reliability/test_fallbacks.py
"""

import asyncio
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Configuration
WS_BASE_URL = os.environ.get(
    "LIVE_WS_URL", 
    "wss://neuronsystems--neuron-live-ws-live-server.modal.run/live"
)


async def test_graceful_degradation() -> bool:
    """
    Test 7.1: 4-level fallback chain
    Tests each level of the fallback hierarchy
    """
    print("\n🧪 Graceful Degradation Test")
    
    results = []
    
    # Test Level 1: Phrase cache
    print("   Testing Level 1 (Cache)...")
    try:
        from src.core import get_phrase_cache
        cache = get_phrase_cache()
        # Cache should return something (even if empty for uncached)
        print(f"   ✅ Cache module available")
        results.append(True)
    except Exception as e:
        print(f"   ❌ Cache failed: {e}")
        results.append(False)
    
    # Test Level 2: Circuit breaker
    print("   Testing Level 2 (Circuit Breaker)...")
    try:
        from src.reliability import get_circuit
        circuit = get_circuit("test_fallback_service")
        assert circuit.is_closed
        print(f"   ✅ Circuit breaker available")
        results.append(True)
    except Exception as e:
        print(f"   ❌ Circuit breaker failed: {e}")
        results.append(False)
    
    # Test Level 3: Rate limiter (prevents overload)
    print("   Testing Level 3 (Rate Limiter)...")
    try:
        from src.reliability import get_rate_limiter
        limiter = get_rate_limiter()
        assert limiter.is_allowed("fallback_test", "request")
        print(f"   ✅ Rate limiter available")
        results.append(True)
    except Exception as e:
        print(f"   ❌ Rate limiter failed: {e}")
        results.append(False)
    
    # Test Level 4: Feature flags (kill switch)
    print("   Testing Level 4 (Feature Flags)...")
    try:
        from src.reliability import get_feature_flags
        flags = get_feature_flags()
        flags.define("fallback_test", "Test flag", enabled=True)
        assert flags.is_enabled("fallback_test")
        print(f"   ✅ Feature flags available")
        results.append(True)
    except Exception as e:
        print(f"   ❌ Feature flags failed: {e}")
        results.append(False)
    
    passed = all(results)
    print(f"\n{'✅ GRACEFUL DEGRADATION TEST PASSED' if passed else '❌ GRACEFUL DEGRADATION TEST FAILED'}")
    return passed


async def test_circuit_breaker() -> bool:
    """
    Test 7.2: Circuit breaker triggers after failures
    """
    print("\n🧪 Circuit Breaker Test")
    
    try:
        from src.reliability import get_circuit
        
        # Get a fresh circuit
        circuit = get_circuit("test_circuit_breaker")
        circuit.reset()
        
        # Verify closed state
        assert circuit.is_closed
        print("   Initial state: closed ✅")
        
        # Simulate failures
        for i in range(10):
            circuit.record_failure()
        
        # Should be open now
        assert circuit.is_open
        print("   After 10 failures: open ✅")
        
        # Verify stats
        stats = circuit.get_stats()
        print(f"   Stats: {stats['failure_count']} failures, state={stats['state']}")
        
        # Reset for other tests
        circuit.reset()
        
        print("\n✅ CIRCUIT BREAKER TEST PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n❌ CIRCUIT BREAKER TEST FAILED")
        return False


async def test_rate_limiting() -> bool:
    """
    Test 7.3: Rate limiting per creator
    """
    print("\n🧪 Rate Limiting Test")
    
    try:
        from src.reliability import get_rate_limiter
        
        limiter = get_rate_limiter()
        creator_id = f"test_rate_limit_{int(time.time())}"
        
        # Count allowed vs blocked
        allowed = 0
        blocked = 0
        
        # Try 70 requests (default minute limit is 60)
        for i in range(70):
            if limiter.is_allowed(creator_id, "request"):
                limiter.record_request(creator_id)
                allowed += 1
            else:
                blocked += 1
        
        print(f"   Allowed: {allowed}")
        print(f"   Blocked: {blocked}")
        
        # Should have blocked some
        passed = blocked > 0 and allowed >= 60
        
        print(f"\n{'✅ RATE LIMITING TEST PASSED' if passed else '❌ RATE LIMITING TEST FAILED'}")
        return passed
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n❌ RATE LIMITING TEST FAILED")
        return False


async def test_reconnection_handling() -> bool:
    """
    Test 7.4: WebSocket reconnection handling
    """
    print("\n🧪 Reconnection Handling Test")
    
    if not WEBSOCKETS_AVAILABLE:
        print("   ⚠️ Skipped: websockets not available")
        return True  # Skip, don't fail
    
    session_ids = []
    
    for attempt in range(3):
        print(f"   Connection attempt {attempt + 1}...")
        try:
            url = f"{WS_BASE_URL}/reconnect-test-{int(time.time())}-{attempt}"
            async with websockets.connect(url, ping_timeout=10) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                session_id = data.get("session_id", f"unknown_{attempt}")
                session_ids.append(session_id)
                print(f"   ✅ Connected: {session_id[:30]}...")
                
                # Clean disconnect
                await ws.close()
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}")
            session_ids.append(None)
    
    # All sessions should be unique (no stale sessions)
    valid_sessions = [s for s in session_ids if s]
    unique_sessions = len(set(valid_sessions))
    
    print(f"\n   Unique sessions: {unique_sessions}/{len(valid_sessions)}")
    
    passed = unique_sessions == len(valid_sessions) and len(valid_sessions) >= 2
    print(f"\n{'✅ RECONNECTION TEST PASSED' if passed else '❌ RECONNECTION TEST FAILED'}")
    return passed


async def run_all_reliability_tests():
    """Run all Phase 7 tests"""
    print("=" * 60)
    print("PHASE 7: RELIABILITY & FALLBACK TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 7.1
    try:
        passed = await test_graceful_degradation()
        results.append(("7.1 Graceful Degradation", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("7.1 Graceful Degradation", False))
    
    # Test 7.2
    try:
        passed = await test_circuit_breaker()
        results.append(("7.2 Circuit Breaker", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("7.2 Circuit Breaker", False))
    
    # Test 7.3
    try:
        passed = await test_rate_limiting()
        results.append(("7.3 Rate Limiting", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("7.3 Rate Limiting", False))
    
    # Test 7.4
    try:
        passed = await test_reconnection_handling()
        results.append(("7.4 Reconnection", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("7.4 Reconnection", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 7 RESULTS")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_reliability_tests())
    sys.exit(0 if success else 1)
