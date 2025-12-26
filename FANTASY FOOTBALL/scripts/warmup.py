#!/usr/bin/env python3
"""
Warmup script - Pre-warms cache for specified cities.

Usage:
    python scripts/warmup.py --cities houston,los_angeles,baltimore,green_bay
"""
import argparse
import asyncio
import aiohttp
import sys

MODAL_BASE_URL = "https://neuronsystems--neuron-orchestrator"


async def warmup_city(session: aiohttp.ClientSession, city: str):
    """Warmup cache for a single city."""
    print(f"  Warming up {city}...")
    
    try:
        # Generate a test response
        payload = {
            "city1": city,
            "city2": "neutral",
            "topic": f"Warmup test for {city}",
            "rounds": 1
        }
        
        async with session.post(
            f"{MODAL_BASE_URL}-run-debate.modal.run",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status == 200:
                print(f"  ✓ {city} warmed up successfully")
                return True
            else:
                print(f"  ❌ {city} warmup failed: {resp.status}")
                return False
    
    except Exception as e:
        print(f"  ❌ {city} warmup error: {e}")
        return False


async def warmup_cache(cities: list):
    """Warmup cache for all specified cities."""
    print("=" * 60)
    print("🔥 CACHE WARMUP")
    print("=" * 60)
    print(f"\nCities to warm up: {', '.join(cities)}")
    
    async with aiohttp.ClientSession() as session:
        # Warmup each city
        results = await asyncio.gather(*[
            warmup_city(session, city) for city in cities
        ])
    
    # Summary
    successes = sum(results)
    print(f"\n✓ {successes}/{len(cities)} cities warmed up successfully")
    
    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warmup cache for cities")
    parser.add_argument(
        "--cities",
        type=str,
        default="houston,los_angeles,baltimore,green_bay",
        help="Comma-separated list of cities"
    )
    
    args = parser.parse_args()
    cities = [c.strip() for c in args.cities.split(",")]
    
    success = asyncio.run(warmup_cache(cities))
    sys.exit(0 if success else 1)
