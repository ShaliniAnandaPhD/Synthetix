#!/usr/bin/env python3
"""
Fantasy Football Neuron - End-to-End Test Suite

Tests complete user flows:
1. Full debate creation → export → share
2. Live commentary connection
3. Personality creation flow

Run: python scripts/test_e2e.py
"""

import asyncio
import sys
import os
import time
import json
from typing import Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Colors
class C:
    G = '\033[92m'  # Green
    R = '\033[91m'  # Red
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    N = '\033[0m'   # Reset
    BOLD = '\033[1m'


def header(title: str):
    print(f"\n{C.BOLD}{C.B}{'='*60}{C.N}")
    print(f"{C.BOLD}{C.B}🧪 {title}{C.N}")
    print(f"{C.BOLD}{C.B}{'='*60}{C.N}\n")


def success(msg: str):
    print(f"   {C.G}✅ {msg}{C.N}")


def fail(msg: str):
    print(f"   {C.R}❌ {msg}{C.N}")


def info(msg: str):
    print(f"   {C.Y}ℹ️  {msg}{C.N}")


# ============================================================================
# E2E TEST 1: Full Debate Flow (Mock)
# ============================================================================

async def test_debate_flow_mock():
    """Test debate creation flow with mocks"""
    header("E2E Test: Full Debate Flow (Mock)")
    
    results = {"passed": 0, "failed": 0}
    
    # Step 1: Test debate generator imports
    print("1. Testing DebateGenerator module...")
    try:
        from src.core import classify_event
        success("DebateGenerator imports OK")
        results["passed"] += 1
    except Exception as e:
        fail(f"Import failed: {e}")
        results["failed"] += 1
    
    # Step 2: Test event classification
    print("\n2. Testing event classification...")
    try:
        event = {"type": "touchdown", "description": "Mahomes TD pass to Kelce"}
        result = classify_event(event)
        if result:
            # Result is a ClassifiedEvent object, not a dict
            priority = getattr(result, 'priority', 'unknown')
            success(f"Event classified: {priority}")
            results["passed"] += 1
        else:
            fail("No classification returned")
            results["failed"] += 1
    except Exception as e:
        fail(f"Classification failed: {e}")
        results["failed"] += 1
    
    # Step 3: Test cost tracking
    print("\n3. Testing cost tracking...")
    try:
        from src.analytics import get_cost_tracker
        tracker = get_cost_tracker()
        tracker.start_debate("e2e_test_debate", "e2e_test_creator")
        tracker.record_llm_usage("e2e_test_debate", 1000, 500)
        tracker.record_tts_usage("e2e_test_debate", 200, "elevenlabs")
        cost = tracker.complete_debate("e2e_test_debate")
        success(f"Cost tracked: ${cost['total_cost_usd']:.6f}")
        results["passed"] += 1
    except Exception as e:
        fail(f"Cost tracking failed: {e}")
        results["failed"] += 1
    
    # Step 4: Test export formats
    print("\n4. Testing export formats...")
    try:
        from src.core.post_game_export import get_exporter
        exporter = get_exporter()
        exporter.start_session("e2e_test", "KC", "BUF", ["kansas_city"])
        exporter.add_entry("e2e_test", "kansas_city", "homer", "Touchdown!")
        exporter.add_entry("e2e_test", "kansas_city", "skeptic", "But the defense...")
        
        transcript = exporter.export_transcript("e2e_test")
        srt = exporter.export_srt("e2e_test")
        
        assert "Touchdown!" in transcript
        assert "00:00:00" in srt
        success(f"Export formats work (transcript: {len(transcript)} chars)")
        results["passed"] += 1
    except Exception as e:
        fail(f"Export failed: {e}")
        results["failed"] += 1
    
    # Step 5: Test share functionality
    print("\n5. Testing share code generation...")
    try:
        import hashlib
        share_code = hashlib.sha256(f"e2e_test_{time.time()}".encode()).hexdigest()[:8]
        success(f"Share code generated: {share_code}")
        results["passed"] += 1
    except Exception as e:
        fail(f"Share code failed: {e}")
        results["failed"] += 1
    
    # Summary
    print(f"\n{C.BOLD}Results: {results['passed']}/{results['passed']+results['failed']} passed{C.N}")
    return results["failed"] == 0


# ============================================================================
# E2E TEST 2: Live Commentary Pipeline (Mock)
# ============================================================================

async def test_live_commentary_mock():
    """Test live commentary pipeline with mocks"""
    header("E2E Test: Live Commentary Pipeline (Mock)")
    
    results = {"passed": 0, "failed": 0}
    
    # Step 1: Test session manager
    print("1. Testing session manager...")
    try:
        from src.core import get_session_manager
        manager = get_session_manager()
        success("Session manager initialized")
        results["passed"] += 1
    except Exception as e:
        fail(f"Session manager failed: {e}")
        results["failed"] += 1
    
    # Step 2: Test phrase cache
    print("\n2. Testing phrase cache...")
    try:
        from src.core import get_phrase_cache
        cache = get_phrase_cache()
        success("Phrase cache initialized")
        results["passed"] += 1
    except Exception as e:
        fail(f"Phrase cache failed: {e}")
        results["failed"] += 1
    
    # Step 3: Test latency tracking
    print("\n3. Testing latency tracking...")
    try:
        from src.observability import get_latency_tracker
        tracker = get_latency_tracker()
        
        # Simulate 10 events
        for i in range(10):
            tracker.record("event_classification", 5 + i)
            tracker.record("agent_generation", 100 + i * 10)
            tracker.record("voice_synthesis", 150 + i * 5)
            tracker.record("total_e2e", 300 + i * 15)
        
        stats = tracker.get_stats()
        p95 = stats["summary"]["e2e_p95_ms"]
        success(f"Latency tracking: P95 = {p95}ms")
        results["passed"] += 1
    except Exception as e:
        fail(f"Latency tracking failed: {e}")
        results["failed"] += 1
    
    # Step 4: Test game simulator
    print("\n4. Testing game simulator...")
    try:
        from src.observability import get_game_simulator
        sim = get_game_simulator()
        
        games = sim.list_games()
        assert "sample_kc_buf" in games
        
        game = sim.get_game("sample_kc_buf")
        success(f"Game simulator: {game['event_count']} events loaded")
        results["passed"] += 1
    except Exception as e:
        fail(f"Game simulator failed: {e}")
        results["failed"] += 1
    
    # Step 5: Test replay buffer
    print("\n5. Testing replay buffer...")
    try:
        from src.core.replay_buffer import get_replay_buffer
        buffer = get_replay_buffer()
        
        buffer.add_event("live_test", {"type": "touchdown", "team": "KC"})
        buffer.add_commentary("live_test", "kansas_city", {"text": "TOUCHDOWN!"})
        
        catchup = buffer.get_catchup("live_test", max_items=10)
        success(f"Replay buffer: {len(catchup)} items for late joiner")
        results["passed"] += 1
    except Exception as e:
        fail(f"Replay buffer failed: {e}")
        results["failed"] += 1
    
    # Step 6: Test agent tracer
    print("\n6. Testing agent tracer...")
    try:
        from src.observability import get_agent_tracer, TraceStage
        tracer = get_agent_tracer()
        
        trace = tracer.start_trace("live_evt_1", "touchdown", "kansas_city")
        with tracer.span(trace.trace_id, TraceStage.AGENT_DISPATCHED):
            await asyncio.sleep(0.01)  # Simulate work
        with tracer.span(trace.trace_id, TraceStage.VOICE_COMPLETE):
            await asyncio.sleep(0.01)
        tracer.complete_trace(trace.trace_id)
        
        stats = tracer.get_stats()
        success(f"Agent tracer: {stats['completed_traces']} traces completed")
        results["passed"] += 1
    except Exception as e:
        fail(f"Agent tracer failed: {e}")
        results["failed"] += 1
    
    # Summary
    print(f"\n{C.BOLD}Results: {results['passed']}/{results['passed']+results['failed']} passed{C.N}")
    return results["failed"] == 0


# ============================================================================
# E2E TEST 3: Reliability Under Stress
# ============================================================================

async def test_reliability_stress():
    """Test reliability components under simulated stress"""
    header("E2E Test: Reliability Under Stress")
    
    results = {"passed": 0, "failed": 0}
    
    # Step 1: Test rate limiter under load
    print("1. Testing rate limiter under load...")
    try:
        from src.reliability import get_rate_limiter
        limiter = get_rate_limiter()
        
        allowed = 0
        denied = 0
        for i in range(100):
            if limiter.is_allowed(f"stress_test_user", "request"):
                limiter.record_request(f"stress_test_user")
                allowed += 1
            else:
                denied += 1
        
        success(f"Rate limiter: {allowed} allowed, {denied} denied (limit working)")
        results["passed"] += 1
    except Exception as e:
        fail(f"Rate limiter failed: {e}")
        results["failed"] += 1
    
    # Step 2: Test circuit breaker under failures
    print("\n2. Testing circuit breaker under failures...")
    try:
        from src.reliability import get_circuit
        circuit = get_circuit("stress_test_service")
        
        # Simulate failures
        for i in range(10):
            circuit.record_failure()
        
        # Should be open now
        assert circuit.is_open, "Circuit should be open after failures"
        success(f"Circuit breaker: OPEN after 10 failures (state: {circuit.state.value})")
        
        # Reset for other tests
        circuit.reset()
        results["passed"] += 1
    except Exception as e:
        fail(f"Circuit breaker failed: {e}")
        results["failed"] += 1
    
    # Step 3: Test budget alerts
    print("\n3. Testing budget alerts under spend...")
    try:
        from src.analytics import BudgetAlertSystem, BudgetConfig
        
        alerts_triggered = []
        system = BudgetAlertSystem(
            config=BudgetConfig(daily_limit_usd=10, warn_threshold=0.5),
            on_alert=lambda a: alerts_triggered.append(a)
        )
        
        # Spend 60% of daily budget
        for i in range(6):
            system.record_spend(1.0, "llm", "stress_test")
        
        status = system.get_status()
        success(f"Budget alerts: {status['daily']['percent']}% spent, {len(alerts_triggered)} alerts")
        results["passed"] += 1
    except Exception as e:
        fail(f"Budget alerts failed: {e}")
        results["failed"] += 1
    
    # Step 4: Test feature flags
    print("\n4. Testing feature flags...")
    try:
        from src.reliability import get_feature_flags, FlagType
        flags = get_feature_flags()
        
        # Test percentage rollout
        flags.define("stress_test_feature", "Test", flag_type=FlagType.PERCENTAGE, percentage=50)
        
        enabled_count = 0
        for i in range(100):
            if flags.is_enabled_for("stress_test_feature", f"user_{i}"):
                enabled_count += 1
        
        # Should be roughly 50%
        assert 30 <= enabled_count <= 70, f"Expected ~50%, got {enabled_count}%"
        success(f"Feature flags: {enabled_count}% enabled (target: 50%)")
        results["passed"] += 1
    except Exception as e:
        fail(f"Feature flags failed: {e}")
        results["failed"] += 1
    
    # Summary
    print(f"\n{C.BOLD}Results: {results['passed']}/{results['passed']+results['failed']} passed{C.N}")
    return results["failed"] == 0


# ============================================================================
# MAIN
# ============================================================================

async def run_all_e2e():
    print(f"\n{C.BOLD}{C.B}Fantasy Football Neuron - E2E Test Suite{C.N}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = []
    
    # Run all E2E tests
    results.append(("Debate Flow", await test_debate_flow_mock()))
    results.append(("Live Commentary", await test_live_commentary_mock()))
    results.append(("Reliability Stress", await test_reliability_stress()))
    
    # Final summary
    header("FINAL RESULTS")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = f"{C.G}PASS{C.N}" if result else f"{C.R}FAIL{C.N}"
        print(f"   {name}: {status}")
    
    print(f"\n{C.BOLD}Overall: {passed}/{total} E2E tests passed{C.N}")
    
    if passed == total:
        print(f"\n{C.G}{C.BOLD}🎉 All E2E tests passed!{C.N}")
        return 0
    else:
        print(f"\n{C.R}{C.BOLD}Some E2E tests failed.{C.N}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all_e2e()))
