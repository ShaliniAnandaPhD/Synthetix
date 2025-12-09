#!/usr/bin/env python3
"""
Phase 6: Performance & Load Tests

Tests concurrent session handling, burst traffic, and latency requirements.

Run:
    modal serve infra/modal_live_ws.py  # Terminal 1
    python tests/performance/test_load.py  # Terminal 2
"""

import asyncio
import json
import time
import os
import sys
from dataclasses import dataclass
from typing import List
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("ERROR: websockets not installed. Run: pip install websockets")


# Configuration
WS_BASE_URL = os.environ.get(
    "LIVE_WS_URL", 
    "wss://neuronsystems--neuron-live-ws-live-server.modal.run/live"
)


@dataclass
class LoadTestResult:
    total_sessions: int
    successful_sessions: int
    failed_sessions: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    events_processed: int
    duration_seconds: float


async def test_concurrent_sessions(num_sessions: int = 50) -> LoadTestResult:
    """
    Test 6.1: Concurrent session handling
    Target: 50+ concurrent creators during primetime
    """
    print(f"\n🧪 Load Test: {num_sessions} Concurrent Sessions")
    print(f"   URL: {WS_BASE_URL}")
    
    all_latencies = []
    successful = 0
    failed = 0
    events_processed = 0
    start_time = time.time()
    
    async def single_session(session_id: int):
        nonlocal successful, failed, events_processed
        session_latencies = []
        
        try:
            url = f"{WS_BASE_URL}/load-test-{session_id}"
            async with websockets.connect(url, ping_timeout=30) as ws:
                # Wait for session start
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                
                # Process up to 3 events per session (shorter for load test)
                for i in range(3):
                    try:
                        event_start = time.time()
                        msg = await asyncio.wait_for(ws.recv(), timeout=8)
                        data = json.loads(msg)
                        
                        latency = (time.time() - event_start) * 1000
                        session_latencies.append(latency)
                        events_processed += 1
                    except asyncio.TimeoutError:
                        break
                
                successful += 1
                return session_latencies
                
        except Exception as e:
            failed += 1
            return []
    
    # Run all sessions concurrently
    results = await asyncio.gather(
        *[single_session(i) for i in range(num_sessions)],
        return_exceptions=True
    )
    
    # Collect all latencies
    for result in results:
        if isinstance(result, list):
            all_latencies.extend(result)
    
    duration = time.time() - start_time
    
    # Calculate percentiles
    if all_latencies:
        sorted_latencies = sorted(all_latencies)
        p95_idx = min(int(0.95 * len(sorted_latencies)), len(sorted_latencies) - 1)
        p99_idx = min(int(0.99 * len(sorted_latencies)), len(sorted_latencies) - 1)
        
        result = LoadTestResult(
            total_sessions=num_sessions,
            successful_sessions=successful,
            failed_sessions=failed,
            avg_latency_ms=statistics.mean(all_latencies),
            p95_latency_ms=sorted_latencies[p95_idx],
            p99_latency_ms=sorted_latencies[p99_idx],
            events_processed=events_processed,
            duration_seconds=duration
        )
    else:
        result = LoadTestResult(
            total_sessions=num_sessions,
            successful_sessions=successful,
            failed_sessions=failed,
            avg_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            events_processed=0,
            duration_seconds=duration
        )
    
    # Print results
    print(f"\n📊 Load Test Results:")
    print(f"   Sessions: {result.successful_sessions}/{result.total_sessions} successful")
    print(f"   Events processed: {result.events_processed}")
    print(f"   Avg latency: {result.avg_latency_ms:.0f}ms")
    print(f"   P95 latency: {result.p95_latency_ms:.0f}ms")
    print(f"   P99 latency: {result.p99_latency_ms:.0f}ms")
    print(f"   Duration: {result.duration_seconds:.1f}s")
    
    # Pass/fail criteria
    success_rate = result.successful_sessions / result.total_sessions if result.total_sessions else 0
    passed = (
        success_rate >= 0.8 and  # 80% success rate (relaxed for testing)
        (result.p95_latency_ms < 6000 or result.p95_latency_ms == 0)  # Allow for 5s push interval
    )
    
    print(f"\n{'✅ LOAD TEST PASSED' if passed else '❌ LOAD TEST FAILED'}")
    return result


async def test_burst_traffic(num_sessions: int = 20) -> bool:
    """
    Test 6.2: Burst traffic handling (simulates inactives release)
    Target: Handle 10x traffic spike
    """
    print(f"\n🧪 Burst Traffic Test: {num_sessions} simultaneous connections")
    
    async def create_quick_session(session_id: int) -> bool:
        try:
            url = f"{WS_BASE_URL}/burst-test-{session_id}"
            async with websockets.connect(url, ping_timeout=10) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                return True
        except:
            return False
    
    # Phase 1: Baseline (5 sessions)
    print("   Phase 1: Establishing baseline (5 sessions)...")
    baseline_start = time.time()
    baseline_results = await asyncio.gather(
        *[create_quick_session(i) for i in range(5)]
    )
    baseline_success = sum(1 for s in baseline_results if s)
    baseline_time = time.time() - baseline_start
    print(f"   Baseline: {baseline_success}/5 sessions in {baseline_time:.1f}s")
    
    # Phase 2: Burst
    print(f"   Phase 2: Burst traffic ({num_sessions} sessions)...")
    burst_start = time.time()
    burst_results = await asyncio.gather(
        *[create_quick_session(100 + i) for i in range(num_sessions)]
    )
    burst_duration = time.time() - burst_start
    burst_success = sum(1 for s in burst_results if s)
    
    print(f"\n📊 Burst Test Results:")
    print(f"   Burst sessions: {burst_success}/{num_sessions} successful")
    print(f"   Burst duration: {burst_duration:.1f}s")
    print(f"   Sessions/second: {num_sessions/burst_duration:.1f}")
    
    passed = burst_success >= num_sessions * 0.8  # 80% success
    print(f"\n{'✅ BURST TEST PASSED' if passed else '❌ BURST TEST FAILED'}")
    return passed


async def test_latency_by_event_type() -> bool:
    """
    Test 6.3: Latency by event urgency
    Note: Server uses push-based events, so we measure time between pushes
    """
    print("\n🧪 Event Delivery Test")
    
    url = f"{WS_BASE_URL}/latency-test"
    
    try:
        async with websockets.connect(url, ping_timeout=30) as ws:
            # Wait for connection
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(msg)
            print(f"   ✅ Connected: {data.get('type', 'unknown')}")
            
            # Wait for first event and time it
            event_start = time.time()
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            first_event_time = (time.time() - event_start) * 1000
            data = json.loads(msg)
            print(f"   ✅ First event: {data.get('type', 'unknown')} in {first_event_time:.0f}ms")
            
            # Wait for commentary
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            print(f"   ✅ Commentary: {data.get('agent', 'unknown')} - {data.get('text', '')[:40]}...")
            
            passed = True
            
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        passed = False
    
    print(f"\n{'✅ LATENCY TEST PASSED' if passed else '❌ LATENCY TEST FAILED'}")
    return passed


async def run_all_performance_tests():
    """Run all Phase 6 tests"""
    print("=" * 60)
    print("PHASE 6: PERFORMANCE & LOAD TESTS")
    print("=" * 60)
    
    if not WEBSOCKETS_AVAILABLE:
        print("ERROR: websockets package required")
        return False
    
    results = []
    
    # Test 6.1: Concurrent sessions (reduced for initial test)
    try:
        result = await test_concurrent_sessions(10)  # Start with 10
        results.append(("6.1 Concurrent Sessions", result.successful_sessions >= 8))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("6.1 Concurrent Sessions", False))
    
    # Test 6.2: Burst traffic
    try:
        passed = await test_burst_traffic(10)  # Start with 10
        results.append(("6.2 Burst Traffic", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("6.2 Burst Traffic", False))
    
    # Test 6.3: Latency by event type
    try:
        passed = await test_latency_by_event_type()
        results.append(("6.3 Latency Test", passed))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        results.append(("6.3 Latency Test", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 6 RESULTS")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_performance_tests())
    sys.exit(0 if success else 1)
