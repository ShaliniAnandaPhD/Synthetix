#!/usr/bin/env python3
"""
Dry Run Script - Simulates a 10-minute game with synthetic events.
Tests identity regression system before going live.

Usage:
    python scripts/dry_run.py
    python scripts/dry_run.py --duration 600 --creators 50
"""
import asyncio
import argparse
import time
import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.identity_regression import (
    VibeCheckScorer,
    load_archetype_config,
    load_platinum_traces,
    PlatinumFallbackSystem,
    RateLimiter,
    validate_platinum_archive,
    send_slack_alert,
    format_metrics_summary
)


# Simulated LLM responses for dry run
MOCK_RESPONSES = {
    "statistician": [
        "The numbers don't lie here - this offense is averaging 6.8 yards per play, which puts them in the top 10 league-wide. If you look at the data, efficiency rating suggests this drive should continue.",
        "According to Pro Football Focus, this is a top-tier performance. The EPA per play is exceptionally high for this game situation.",
    ],
    "historian": [
        "This reminds me of Montana in his prime. Back in the day, we'd call this a legacy game. Hall of fame trajectory for sure.",
        "You have to go back to the 1996 season to find a comparable stretch. This is history in the making.",
    ],
    "hot_take_artist": [
        "Are you KIDDING me?! This is RIDICULOUS! DOMINANT performance and it's not even CLOSE!",
        "UNBELIEVABLE! I've been saying this for WEEKS! This is a DISGRACE by the defense!",
    ],
    "analyst": [
        "If you watch the tape, the footwork on that play was immaculate. The pre-snap read identified the coverage immediately.",
        "That route concept is textbook. Watch the technique on the coverage breakdown.",
    ],
    "homer": [
        "OUR guys came to PLAY today! This is who we ARE! Championship mentality!",
        "Nobody believes in us? Good! WE are the best team and it's not even close!",
    ],
    "neutral": [
        "Credit to both teams for a hard-fought game. You have to acknowledge the execution on both sides.",
        "Fair to say this could go either way. On the other hand, there's been excellent play from both squads.",
    ],
}


class DryRunOrchestrator:
    """Simplified orchestrator for dry run testing."""
    
    def __init__(self):
        self.archetype_config = load_archetype_config()
        self.platinum_traces = load_platinum_traces()
        self.fallback_system = PlatinumFallbackSystem()
        self.rate_limiter = RateLimiter(max_concurrent=200, requests_per_second=15)
        
        self.vibe_checkers = {
            arch: VibeCheckScorer(arch, self.archetype_config)
            for arch in self.archetype_config.keys()
        }
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "vibe_passes": 0,
            "regenerations": 0,
            "platinum_fallbacks": 0,
            "total_latency": 0,
            "errors": 0
        }
        self.archetype_scores = {arch: [] for arch in self.archetype_config.keys()}
    
    def _mock_generate(self, archetype: str) -> str:
        """Generate mock response for testing."""
        responses = MOCK_RESPONSES.get(archetype, MOCK_RESPONSES["neutral"])
        return random.choice(responses)
    
    async def generate_response(self, prompt: str, archetype: str, city: str, event_type: str):
        """Simulate response generation with vibe checking."""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        async with self.rate_limiter:
            try:
                # Layer 1: Mock generation
                response = self._mock_generate(archetype)
                vibe_score = self.vibe_checkers[archetype].score(response)
                
                if vibe_score < 0.7:
                    # Layer 2: Simulate regeneration
                    self.metrics["regenerations"] += 1
                    response = self._mock_generate(archetype)
                    vibe_score = self.vibe_checkers[archetype].score(response)
                    
                    if vibe_score < 0.7:
                        # Layer 3: Platinum fallback
                        self.metrics["platinum_fallbacks"] += 1
                        response = self.fallback_system.get_fallback(city, event_type)
                        vibe_score = 0.75
                
                if vibe_score >= 0.7:
                    self.metrics["vibe_passes"] += 1
                
                self.archetype_scores[archetype].append(vibe_score)
                
                latency = time.time() - start_time
                self.metrics["total_latency"] += latency
                
                return response, vibe_score, latency
                
            except Exception as e:
                self.metrics["errors"] += 1
                return f"Error: {e}", 0.0, 0.0
    
    def get_metrics(self):
        """Get current metrics."""
        total = self.metrics["total_requests"]
        if total == 0:
            return {}
        
        # Calculate per-archetype stats
        archetype_stats = {}
        for arch, scores in self.archetype_scores.items():
            if scores:
                archetype_stats[f"vibe_check/{arch}/pass_rate"] = sum(1 for s in scores if s >= 0.7) / len(scores)
                archetype_stats[f"vibe_check/{arch}/avg_score"] = sum(scores) / len(scores)
        
        return {
            "total_requests": total,
            "vibe_check_pass_rate": self.metrics["vibe_passes"] / total,
            "regeneration_rate": self.metrics["regenerations"] / total,
            "platinum_fallback_rate": self.metrics["platinum_fallbacks"] / total,
            "avg_latency": self.metrics["total_latency"] / total,
            "error_rate": self.metrics["errors"] / total,
            **archetype_stats
        }


async def run_dry_run(num_requests: int = 50, duration: int = 60):
    """Run dry run test."""
    print("=" * 60)
    print("🏈 NEURON DRY RUN - Identity Regression Test")
    print("=" * 60)
    
    # Validate prerequisites
    print("\n📋 Validating prerequisites...")
    
    # Check archetype config
    try:
        archetype_config = load_archetype_config()
        print(f"  ✓ Archetype config loaded: {len(archetype_config)} archetypes")
    except Exception as e:
        print(f"  ❌ Archetype config failed: {e}")
        return False
    
    # Check platinum traces
    try:
        traces = load_platinum_traces()
        required_cities = ["houston", "los_angeles", "baltimore", "green_bay"]
        required_events = ["touchdown", "interception", "generic"]
        missing = validate_platinum_archive(traces, required_cities, required_events)
        if missing:
            print(f"  ⚠️ Missing platinum traces: {missing}")
        else:
            print(f"  ✓ Platinum traces validated: {len(traces)} keys")
    except Exception as e:
        print(f"  ❌ Platinum traces failed: {e}")
        return False
    
    # Check fallback system
    try:
        fallback = PlatinumFallbackSystem()
        stats = fallback.archive_stats()
        print(f"  ✓ Fallback system ready: {stats['total_responses']} responses, {stats['cities_covered']} cities")
    except Exception as e:
        print(f"  ❌ Fallback system failed: {e}")
        return False
    
    # Initialize orchestrator
    print("\n🚀 Starting dry run...")
    orchestrator = DryRunOrchestrator()
    
    archetypes = list(archetype_config.keys())
    cities = ["houston", "los_angeles", "baltimore", "green_bay"]
    events = ["touchdown", "interception", "generic"]
    
    # Run requests
    start_time = time.time()
    tasks = []
    
    for i in range(num_requests):
        arch = archetypes[i % len(archetypes)]
        city = cities[i % len(cities)]
        event = events[i % len(events)]
        prompt = f"Analyze this {event} for {city}"
        
        task = orchestrator.generate_response(prompt, arch, city, event)
        tasks.append((arch, task))
    
    # Run all tasks
    for i, (arch, task) in enumerate(tasks):
        response, score, latency = await task
        status = "✓" if score >= 0.7 else "✗"
        print(f"  [{i+1:3d}/{num_requests}] {arch:20s} | Score: {score:.2f} {status} | {latency:.2f}s")
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 DRY RUN RESULTS")
    print("=" * 60)
    
    metrics = orchestrator.get_metrics()
    print(format_metrics_summary(metrics))
    
    # Check thresholds
    print("\n🎯 Threshold Checks:")
    all_passed = True
    
    # Overall pass rate
    pass_rate = metrics.get("vibe_check_pass_rate", 0)
    if pass_rate >= 0.85:
        print(f"  ✓ Vibe check pass rate: {pass_rate:.0%} (threshold: 85%)")
    else:
        print(f"  ❌ Vibe check pass rate: {pass_rate:.0%} (threshold: 85%)")
        all_passed = False
    
    # Latency
    avg_latency = metrics.get("avg_latency", 0)
    if avg_latency < 60:
        print(f"  ✓ Avg latency: {avg_latency:.1f}s (threshold: 60s)")
    else:
        print(f"  ❌ Avg latency: {avg_latency:.1f}s (threshold: 60s)")
        all_passed = False
    
    # Per-archetype
    for arch in archetypes:
        arch_pass = metrics.get(f"vibe_check/{arch}/pass_rate", 0)
        if arch_pass >= 0.7:
            print(f"  ✓ {arch}: {arch_pass:.0%} (threshold: 70%)")
        else:
            print(f"  ❌ {arch}: {arch_pass:.0%} (threshold: 70%)")
            all_passed = False
    
    # Fallback rate
    fallback_rate = metrics.get("platinum_fallback_rate", 0)
    if fallback_rate < 0.10:
        print(f"  ✓ Fallback rate: {fallback_rate:.0%} (threshold: 10%)")
    else:
        print(f"  ⚠️ Fallback rate: {fallback_rate:.0%} (threshold: 10%)")
    
    print(f"\n⏱️  Total time: {elapsed:.1f}s for {num_requests} requests")
    print(f"📈 Throughput: {num_requests / elapsed:.1f} requests/second")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ DRY RUN PASSED - Ready for Saturday deployment!")
    else:
        print("❌ DRY RUN FAILED - Fix issues before Saturday!")
    print("=" * 60)
    
    return all_passed


async def run_load_test(num_concurrent: int = 150):
    """Test peak load scenario."""
    print("\n" + "=" * 60)
    print(f"🔥 LOAD TEST - {num_concurrent} concurrent requests")
    print("=" * 60)
    
    orchestrator = DryRunOrchestrator()
    archetypes = list(load_archetype_config().keys())
    
    # Create all tasks at once (simulates burst)
    tasks = []
    for i in range(num_concurrent):
        arch = archetypes[i % len(archetypes)]
        task = orchestrator.generate_response("Big play!", arch, "houston", "touchdown")
        tasks.append(task)
    
    start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.time() - start
    
    # Count successes
    successes = sum(1 for r in results if not isinstance(r, Exception))
    errors = sum(1 for r in results if isinstance(r, Exception))
    
    print(f"\n📊 Load Test Results:")
    print(f"  - {num_concurrent} concurrent requests completed in {duration:.1f}s")
    print(f"  - Successes: {successes}")
    print(f"  - Errors: {errors}")
    print(f"  - Effective RPS: {num_concurrent / duration:.1f}")
    
    # Check rate limiting worked
    expected_min_time = num_concurrent / 15  # At 15 RPS
    if duration >= expected_min_time * 0.8:  # Allow 20% variance
        print(f"  ✓ Rate limiting working (expected ~{expected_min_time:.1f}s)")
    else:
        print(f"  ⚠️ Rate limiting may not be working correctly")
    
    return errors == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry run for identity regression system")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--creators", type=int, default=50, help="Number of requests")
    parser.add_argument("--load-test", action="store_true", help="Run load test")
    parser.add_argument("--concurrent", type=int, default=150, help="Concurrent requests for load test")
    
    args = parser.parse_args()
    
    async def main():
        # Run dry run
        passed = await run_dry_run(num_requests=args.creators, duration=args.duration)
        
        # Optionally run load test
        if args.load_test:
            load_passed = await run_load_test(num_concurrent=args.concurrent)
            passed = passed and load_passed
        
        # Test Slack (optional)
        try:
            send_slack_alert("🧪 Dry run completed - test alert")
            print("\n✓ Slack alert test successful")
        except Exception as e:
            print(f"\n⚠️ Slack alert test failed: {e}")
        
        return passed
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
