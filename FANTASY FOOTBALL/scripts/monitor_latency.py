#!/usr/bin/env python3
"""
Latency monitoring script for P0 Cultural Cognition system.
Run: python3 scripts/monitor_latency.py
"""

import requests
import time
import statistics
from datetime import datetime
from typing import Dict, List, Optional

# Vercel production URL
VERCEL_API = "https://cultural-cognition-s0dlc8qne-neuron-systems.vercel.app/api/cultural"

# Test these cities for latency
TEST_CITIES = [
    "Philadelphia",
    "Kansas City", 
    "Buffalo",
    "Dallas",
    "New England",
    "San Francisco",
    "Pittsburgh",
    "Seattle"
]


def test_city_latency(city: str, runs: int = 5) -> Optional[Dict[str, float]]:
    """
    Test latency for a specific city.
    
    Args:
        city: City name to test
        runs: Number of test runs
    
    Returns:
        Dictionary with avg, p95, min, max latencies in ms
    """
    latencies = []
    
    for i in range(runs):
        try:
            start = time.time()
            response = requests.post(
                VERCEL_API,
                json={
                    "city": city,
                    "user_input": "Test query for latency monitoring",
                    "game_context": {}
                },
                timeout=15
            )
            end = time.time()
            
            if response.status_code == 200:
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                print(f"  Run {i+1}: {latency_ms:.0f}ms")
            else:
                print(f"  Run {i+1}: ERROR {response.status_code}")
        
        except requests.exceptions.Timeout:
            print(f"  Run {i+1}: TIMEOUT (>15s)")
        except Exception as e:
            print(f"  Run {i+1}: ERROR {str(e)}")
    
    if latencies:
        return {
            "avg": statistics.mean(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            "min": min(latencies),
            "max": max(latencies)
        }
    
    return None


def main():
    """Run latency monitoring across test cities."""
    print("=" * 70)
    print(f"P0 Latency Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nTarget: <300ms P95 latency")
    print(f"Testing {len(TEST_CITIES)} cities with 5 runs each\n")
    
    results = {}
    
    for city in TEST_CITIES:
        print(f"Testing {city}...")
        results[city] = test_city_latency(city, runs=5)
        time.sleep(2)  # Rate limiting between cities
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for city, metrics in results.items():
        if metrics:
            status = "✅" if metrics['p95'] < 300 else "⚠️"
            print(f"{status} {city:20} Avg: {metrics['avg']:>6.0f}ms | P95: {metrics['p95']:>6.0f}ms | Min: {metrics['min']:>6.0f}ms | Max: {metrics['max']:>6.0f}ms")
        else:
            print(f"❌ {city:20} FAILED - No successful requests")
    
    # Overall statistics
    all_avgs = [m['avg'] for r in results.values() if r for m in [r]]
    all_p95s = [m['p95'] for r in results.values() if r for m in [r]]
    
    if all_p95s:
        overall_p95 = max(all_p95s)  # Worst case P95
        overall_avg = statistics.mean(all_avgs)
        
        print("\n" + "=" * 70)
        print("OVERALL METRICS")
        print("=" * 70)
        print(f"Average Latency: {overall_avg:.0f}ms")
        print(f"Worst P95:       {overall_p95:.0f}ms")
        print(f"Target:          <300ms")
        
        if overall_p95 < 300:
            print("\n✅ SUCCESS: All cities within latency target")
        else:
            print(f"\n⚠️  WARNING: P95 latency {overall_p95:.0f}ms exceeds 300ms target")
            print(f"   Optimization needed - check Modal logs and Redis cache")
    else:
        print("\n❌ FAILED: No successful requests across all cities")
        print("   Check Vercel API endpoint and Modal deployment")


if __name__ == "__main__":
    main()
