#!/usr/bin/env python3
"""
Fantasy Football Neuron - Comprehensive Test Script

Runs through all backend modules to verify they load and function correctly.
Run with: python scripts/test_all_modules.py
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime
from typing import Callable, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")

def fail(msg: str, error: str = ""):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")
    if error:
        print(f"  {Colors.RED}{error}{Colors.RESET}")

def section(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


results = {"passed": 0, "failed": 0, "errors": []}


def test(name: str):
    """Decorator for test functions"""
    def decorator(func: Callable):
        def wrapper():
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func())
                else:
                    func()
                success(name)
                results["passed"] += 1
            except Exception as e:
                fail(name, str(e))
                results["failed"] += 1
                results["errors"].append((name, str(e)))
        return wrapper
    return decorator


# ============================================================================
# ANALYTICS MODULE TESTS
# ============================================================================

@test("Analytics: CreatorDashboard")
def test_creator_dashboard():
    from src.analytics import get_creator_dashboard
    dashboard = get_creator_dashboard()
    dashboard.record_debate_created("test_creator", "template1", ["kansas_city"])
    dashboard.record_cost("test_creator", 0.15, "llm")
    metrics = dashboard.get_creator_metrics("test_creator")
    assert metrics is not None
    assert metrics["usage"]["debates_created"] >= 1

@test("Analytics: DebateCostTracker")
def test_cost_tracker():
    from src.analytics import get_cost_tracker
    tracker = get_cost_tracker()
    tracker.start_debate("test_debate", "test_creator")
    tracker.record_llm_usage("test_debate", 1000, 500)
    tracker.record_tts_usage("test_debate", 200, "elevenlabs")
    result = tracker.complete_debate("test_debate")
    assert result is not None
    assert result["total_cost_usd"] > 0

@test("Analytics: BudgetAlerts")
def test_budget_alerts():
    from src.analytics import get_budget_alerts
    alerts = get_budget_alerts()
    alerts.record_spend(1.0, "llm", "test_creator")
    status = alerts.get_status()
    assert "daily" in status
    assert status["daily"]["spent_usd"] >= 1.0

@test("Analytics: TemplateAnalytics")
def test_template_analytics():
    from src.analytics import get_template_analytics
    analytics = get_template_analytics()
    analytics.record_selection("head_to_head", "Head to Head", "debate")
    analytics.record_completion("head_to_head", 120, 6)
    stats = analytics.get_template_stats("head_to_head")
    assert stats is not None
    assert stats["usage"]["times_selected"] >= 1


# ============================================================================
# RELIABILITY MODULE TESTS
# ============================================================================

@test("Reliability: CircuitBreaker")
def test_circuit_breaker():
    from src.reliability import get_circuit
    circuit = get_circuit("test_service")
    assert circuit.is_closed
    stats = circuit.get_stats()
    assert stats["state"] == "closed"

@test("Reliability: RateLimiter")
def test_rate_limiter():
    from src.reliability import get_rate_limiter
    limiter = get_rate_limiter()
    assert limiter.is_allowed("test_user", "request")
    limiter.record_request("test_user")
    limits = limiter.get_limits("test_user")
    assert limits["minute"]["used"] >= 1

@test("Reliability: GameRateLimiter")
def test_game_rate_limiter():
    from src.reliability import get_game_rate_limiter
    limiter = get_game_rate_limiter()
    assert limiter.is_event_allowed("test_game")
    limiter.record_event("test_game")
    stats = limiter.get_game_stats("test_game")
    assert stats["events_last_minute"] >= 1

@test("Reliability: FeatureFlags")
def test_feature_flags():
    from src.reliability import get_feature_flags
    flags = get_feature_flags()
    flags.define("test_flag", "Test feature", enabled=True)
    assert flags.is_enabled("test_flag")
    flags.disable("test_flag")
    assert not flags.is_enabled("test_flag")


# Async tests (no decorator - called directly)
async def test_health_checker_async():
    from src.reliability import get_health_checker
    checker = get_health_checker()
    status = await checker.get_status()
    assert "status" in status
    assert "services" in status

async def test_retry_async():
    from src.reliability import RetryExecutor
    executor = RetryExecutor(max_attempts=2)
    call_count = 0
    
    async def succeed():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = await executor.execute(succeed)
    assert result == "success"
    assert call_count == 1


# ============================================================================
# OBSERVABILITY MODULE TESTS
# ============================================================================

@test("Observability: AgentTracer")
def test_agent_tracer():
    from src.observability import get_agent_tracer, TraceStage
    tracer = get_agent_tracer()
    trace = tracer.start_trace("evt_test", "touchdown", "kansas_city")
    with tracer.span(trace.trace_id, TraceStage.AGENT_DISPATCHED):
        pass  # Simulate work
    tracer.complete_trace(trace.trace_id)
    stats = tracer.get_stats()
    assert stats["completed_traces"] >= 1

@test("Observability: DriftDetector")
def test_drift_detector():
    from src.observability import get_drift_detector
    detector = get_drift_detector()
    detector.set_baseline("test_region", tempo=1.0, interruption_rate=0.3)
    detector.record_sample("test_region", tempo=1.05, sentiment=0.5)
    status = detector.get_status("test_region")
    assert "test_region" in status

@test("Observability: LatencyTracker")
def test_latency_tracker():
    from src.observability import get_latency_tracker
    tracker = get_latency_tracker()
    for i in range(10):
        tracker.record("total_e2e", 100 + i * 10)
    stats = tracker.get_stats()
    assert stats["summary"]["e2e_p50_ms"] > 0

@test("Observability: GameSimulator")
def test_game_simulator():
    from src.observability import get_game_simulator
    simulator = get_game_simulator()
    games = simulator.list_games()
    assert "sample_kc_buf" in games
    game = simulator.get_game("sample_kc_buf")
    assert game["event_count"] > 0


# ============================================================================
# CORE MODULE TESTS
# ============================================================================

@test("Core: EventClassifier")
def test_event_classifier():
    from src.core import classify_event
    event = {"type": "touchdown", "description": "Mahomes TD pass"}
    result = classify_event(event)
    assert result is not None

@test("Core: LivePhraseCache")
def test_phrase_cache():
    from src.core import get_phrase_cache
    cache = get_phrase_cache()
    assert cache is not None

@test("Core: LiveSessionManager")
def test_session_manager():
    from src.core import get_session_manager
    manager = get_session_manager()
    assert manager is not None

@test("Core: LiveAudioQueue")
def test_audio_queue():
    from src.core import LiveAudioQueue, AudioQueueItem, AudioPriority
    queue = LiveAudioQueue()
    item = AudioQueueItem(
        item_id="test1",
        audio_data="ZmFrZV9hdWRpbw==",  # base64 encoded
        text="Test commentary",
        region="kansas_city",
        agent_type="homer",
        priority=AudioPriority.HIGH,
        duration_ms=1000
    )
    # Just test the item creation works
    assert item.item_id == "test1"
    assert item.priority == AudioPriority.HIGH

@test("Core: LiveLatencyDashboard")
def test_latency_dashboard():
    from src.core import get_dashboard, MetricType
    dashboard = get_dashboard()
    dashboard.record_latency(MetricType.E2E_LATENCY, 150)
    dashboard.record_cache_hit()
    stats = dashboard.get_stats()
    assert stats.events_processed >= 0

@test("Core: LiveCostCap")
def test_cost_cap():
    from src.core import create_cost_cap, CostCategory
    # Create with higher limits to avoid triggering cap
    cap = create_cost_cap("test_game_2", max_per_game=100.0, max_per_hour=50.0)
    assert cap.can_spend(CostCategory.LLM, 1.0)
    cap.record_cost(CostCategory.LLM, 1.0, "test")
    status = cap.get_status()
    assert status["total_cost_usd"] >= 1.0

@test("Core: MultiCreatorSync")
def test_multi_creator_sync():
    from src.core.multi_creator_sync import get_multi_creator_sync
    sync = get_multi_creator_sync()
    assert sync is not None
    games = sync.get_all_games()
    assert isinstance(games, list)

@test("Core: ReplayBuffer")
def test_replay_buffer():
    from src.core.replay_buffer import get_replay_buffer
    buffer = get_replay_buffer()
    buffer.add_event("test_game", {"type": "play", "desc": "test"})
    buffer.add_commentary("test_game", "kansas_city", {"text": "Nice play!"})
    catchup = buffer.get_catchup("test_game", max_items=5)
    assert len(catchup) >= 1

@test("Core: ManualOverride")
def test_manual_override():
    from src.core.manual_override import get_override_controller
    controller = get_override_controller()
    controller.mute_agent("session1", "homer")
    assert not controller.should_play("session1", "kansas_city", "homer")
    controller.unmute_agent("session1", "homer")
    assert controller.should_play("session1", "kansas_city", "homer")

@test("Core: PostGameExport")
def test_post_game_export():
    from src.core.post_game_export import get_exporter
    exporter = get_exporter()
    exporter.start_session("test_game", "KC", "BUF", ["kansas_city"])
    exporter.add_entry("test_game", "kansas_city", "homer", "Touchdown!")
    transcript = exporter.export_transcript("test_game")
    assert "Touchdown!" in transcript


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    print(f"\n{Colors.BOLD}Fantasy Football Neuron - Test Suite{Colors.RESET}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Analytics
    section("Analytics Module")
    test_creator_dashboard()
    test_cost_tracker()
    test_budget_alerts()
    test_template_analytics()
    
    # Reliability
    section("Reliability Module")
    test_circuit_breaker()
    test_rate_limiter()
    test_game_rate_limiter()
    
    # Async tests - run directly
    try:
        asyncio.run(test_health_checker_async())
        success("Reliability: HealthChecker")
        results["passed"] += 1
    except Exception as e:
        fail("Reliability: HealthChecker", str(e))
        results["failed"] += 1
    
    try:
        asyncio.run(test_retry_async())
        success("Reliability: RetryExecutor")
        results["passed"] += 1
    except Exception as e:
        fail("Reliability: RetryExecutor", str(e))
        results["failed"] += 1
    
    test_feature_flags()
    
    # Observability
    section("Observability Module")
    test_agent_tracer()
    test_drift_detector()
    test_latency_tracker()
    test_game_simulator()
    
    # Core
    section("Core Module")
    test_event_classifier()
    test_phrase_cache()
    test_session_manager()
    test_audio_queue()
    test_latency_dashboard()
    test_cost_cap()
    test_multi_creator_sync()
    test_replay_buffer()
    test_manual_override()
    test_post_game_export()
    
    # Summary
    section("Test Results")
    total = results["passed"] + results["failed"]
    
    print(f"Total:  {total}")
    print(f"{Colors.GREEN}Passed: {results['passed']}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {results['failed']}{Colors.RESET}")
    
    if results["errors"]:
        print(f"\n{Colors.RED}Failed Tests:{Colors.RESET}")
        for name, error in results["errors"]:
            print(f"  • {name}: {error}")
    
    print()
    
    if results["failed"] == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}All tests passed! 🎉{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}Some tests failed.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
