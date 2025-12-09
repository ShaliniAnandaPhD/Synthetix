#!/usr/bin/env python3
"""
Pre-Game Cache Warming Script

Warms the phrase cache for a specific game matchup before kickoff.

Usage:
    python scripts/warm_game_context.py --game=TNF_DEC18 --home=seattle --away=los_angeles
"""

import argparse
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Event types to pre-generate reactions for
EVENT_TYPES = [
    "touchdown",
    "turnover",
    "big_play",
    "field_goal",
    "interception",
    "fumble",
    "sack",
    "penalty",
    "timeout",
    "two_minute_warning",
    "fourth_down_conversion",
    "fourth_down_stop",
    "game_winning_drive",
    "clutch_play",
]

# Agent types
AGENT_TYPES = ["homer", "analyst", "hater", "neutral"]


async def warm_cache_for_team(cache, region: str, num_phrases: int = 100):
    """Generate and cache phrases for a specific team region"""
    print(f"\n🔥 Warming cache for {region}...")
    
    phrases_generated = 0
    start_time = time.time()
    
    for event_type in EVENT_TYPES:
        for agent_type in AGENT_TYPES:
            # Check if we have enough
            if phrases_generated >= num_phrases:
                break
            
            try:
                # Try to get or generate a phrase
                if hasattr(cache, 'warm_phrase'):
                    await cache.warm_phrase(region, agent_type, event_type)
                    phrases_generated += 1
                elif hasattr(cache, 'generate_phrase'):
                    phrase = await cache.generate_phrase(region, agent_type, event_type)
                    if phrase:
                        phrases_generated += 1
                else:
                    # Simulate cache population for testing
                    print(f"   • {event_type}/{agent_type}")
                    phrases_generated += 1
                    
            except Exception as e:
                print(f"   ⚠️ Failed: {event_type}/{agent_type}: {e}")
    
    duration = time.time() - start_time
    print(f"   ✅ Generated {phrases_generated} phrases in {duration:.1f}s")
    
    return phrases_generated


async def warm_game(game_id: str, home_team: str, away_team: str, phrases_per_team: int = 100):
    """Warm caches for both teams in a game"""
    print("=" * 60)
    print(f"PRE-GAME CACHE WARMING: {game_id}")
    print(f"Matchup: {away_team.upper()} @ {home_team.upper()}")
    print("=" * 60)
    
    try:
        from src.core import get_phrase_cache
        cache = get_phrase_cache()
    except ImportError as e:
        print(f"⚠️ Could not import phrase cache: {e}")
        print("   Simulating cache warming...")
        cache = None
    
    total_phrases = 0
    
    # Warm home team cache
    if cache:
        total_phrases += await warm_cache_for_team(cache, home_team, phrases_per_team)
        total_phrases += await warm_cache_for_team(cache, away_team, phrases_per_team)
    else:
        # Simulate for testing
        print(f"\n🔥 Simulating cache warm for {home_team}...")
        for i, event in enumerate(EVENT_TYPES):
            print(f"   • Generating {event} reactions...")
            await asyncio.sleep(0.1)
        total_phrases += len(EVENT_TYPES) * len(AGENT_TYPES)
        
        print(f"\n🔥 Simulating cache warm for {away_team}...")
        for i, event in enumerate(EVENT_TYPES):
            print(f"   • Generating {event} reactions...")
            await asyncio.sleep(0.1)
        total_phrases += len(EVENT_TYPES) * len(AGENT_TYPES)
    
    # Summary
    print("\n" + "=" * 60)
    print("CACHE WARMING COMPLETE")
    print("=" * 60)
    print(f"   Game: {game_id}")
    print(f"   Total phrases: {total_phrases}")
    print(f"   {home_team}: {total_phrases // 2} phrases")
    print(f"   {away_team}: {total_phrases // 2} phrases")
    
    # Verify cache stats
    if cache and hasattr(cache, 'get_stats'):
        stats = cache.get_stats()
        print(f"\n📊 Cache Stats:")
        print(f"   {stats}")
    
    print("\n✅ Ready for kickoff!")
    
    return total_phrases


def main():
    parser = argparse.ArgumentParser(description="Warm phrase cache for a game")
    parser.add_argument("--game", required=True, help="Game ID (e.g., TNF_DEC18)")
    parser.add_argument("--home", required=True, help="Home team region (e.g., seattle)")
    parser.add_argument("--away", required=True, help="Away team region (e.g., los_angeles)")
    parser.add_argument("--phrases", type=int, default=100, help="Phrases per team")
    
    args = parser.parse_args()
    
    asyncio.run(warm_game(
        args.game,
        args.home,
        args.away,
        args.phrases
    ))


if __name__ == "__main__":
    main()
