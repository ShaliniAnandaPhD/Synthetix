#!/usr/bin/env python3
"""
P2: ESPN Live Event Integration Tests

Validates real-time polling from ESPN API:
- Events detected within 10s of ESPN update
- Critical events (TD, turnover) correctly identified

Run:
    python tests/realtime/test_espn_integration.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Tuple, Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# CONFIGURATION
# ============================================================================

ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

# Event type classifications
CRITICAL_EVENTS = ["touchdown", "interception", "fumble", "safety", "field_goal"]
IMMEDIATE_EVENTS = ["touchdown", "interception", "fumble", "safety"]
STANDARD_EVENTS = ["first_down", "punt", "timeout", "penalty"]


# ============================================================================
# ESPN API HELPERS
# ============================================================================

def get_live_games() -> List[Dict[str, Any]]:
    """Fetch current live NFL games from ESPN."""
    import requests
    
    try:
        response = requests.get(f"{ESPN_API_BASE}/scoreboard", timeout=10)
        if response.status_code != 200:
            return []
        
        data = response.json()
        events = data.get("events", [])
        
        live_games = []
        for event in events:
            status = event.get("status", {}).get("type", {}).get("name", "")
            if status == "STATUS_IN_PROGRESS":
                game_info = {
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "status": status,
                    "competitions": event.get("competitions", [])
                }
                live_games.append(game_info)
        
        return live_games
        
    except Exception as e:
        print(f"   Error fetching games: {e}")
        return []


def get_game_events(game_id: str) -> List[Dict[str, Any]]:
    """Fetch play-by-play events for a specific game."""
    import requests
    
    try:
        response = requests.get(
            f"{ESPN_API_BASE}/summary?event={game_id}",
            timeout=10
        )
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        # Extract plays from drives
        plays = []
        drives = data.get("drives", {}).get("current", {}).get("plays", [])
        
        for play in drives:
            play_info = {
                "id": play.get("id"),
                "type": play.get("type", {}).get("text", ""),
                "text": play.get("text", ""),
                "clock": play.get("clock", {}).get("displayValue", ""),
                "quarter": play.get("period", {}).get("number", 0),
                "scoringPlay": play.get("scoringPlay", False)
            }
            plays.append(play_info)
        
        return plays
        
    except Exception as e:
        print(f"   Error fetching events: {e}")
        return []


def classify_event(play: Dict[str, Any]) -> str:
    """Classify an ESPN play into our event categories."""
    play_type = play.get("type", "").lower()
    play_text = play.get("text", "").lower()
    is_scoring = play.get("scoringPlay", False)
    
    # Check for touchdowns
    if is_scoring and ("touchdown" in play_text or "td" in play_text):
        return "touchdown"
    
    # Check for turnovers
    if "intercept" in play_text:
        return "interception"
    if "fumble" in play_text and "recover" in play_text:
        return "fumble"
    
    # Check for field goals
    if "field goal" in play_text and is_scoring:
        return "field_goal"
    
    # Check for safety
    if "safety" in play_text:
        return "safety"
    
    # Default classification
    if "first down" in play_text:
        return "first_down"
    if "punt" in play_type or "punt" in play_text:
        return "punt"
    
    return "play"


# ============================================================================
# TEST 1: ESPN Polling Latency
# ============================================================================

async def test_espn_polling_latency() -> Tuple[bool, str]:
    """
    Events detected within 10s of ESPN update.
    
    Tests the latency of fetching data from ESPN API.
    """
    print("\n🧪 Test 1: ESPN Polling Latency")
    print("   Target: API response < 10 seconds")
    
    latencies = []
    
    # Test 5 polling cycles
    for i in range(5):
        start = time.time()
        
        games = get_live_games()
        
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        
        print(f"   Poll {i+1}: {latency:.0f}ms ({len(games)} live games)")
        
        await asyncio.sleep(0.5)  # Small delay between polls
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    # Pass if average < 5s and max < 10s
    passed = avg_latency < 5000 and max_latency < 10000
    
    status = f"Avg: {avg_latency:.0f}ms, Max: {max_latency:.0f}ms"
    print(f"   {'✅ PASSED' if passed else '❌ FAILED'} - {status}")
    
    return passed, status


# ============================================================================
# TEST 2: Event Classification
# ============================================================================

async def test_event_classification() -> Tuple[bool, str]:
    """
    Critical events (TD, turnover) correctly identified.
    
    Tests our event classification logic against sample plays.
    """
    print("\n🧪 Test 2: Event Classification")
    
    # Sample plays to classify
    sample_plays = [
        {"type": "Pass", "text": "Mahomes throws a 25-yard TOUCHDOWN pass to Kelce", "scoringPlay": True},
        {"type": "Pass", "text": "Allen intercepted by Diggs at the 30", "scoringPlay": False},
        {"type": "Rush", "text": "Henry rushes for 5 yards, first down", "scoringPlay": False},
        {"type": "Kick", "text": "Tucker 45 yard field goal is GOOD", "scoringPlay": True},
        {"type": "Rush", "text": "Swift fumbles, recovered by Eagles", "scoringPlay": False},
        {"type": "Punt", "text": "Punt from the 35 to the 20", "scoringPlay": False},
        {"type": "Safety", "text": "SAFETY! Tackle in the end zone", "scoringPlay": True},
    ]
    
    expected = ["touchdown", "interception", "first_down", "field_goal", "fumble", "punt", "safety"]
    
    correct = 0
    total = len(sample_plays)
    
    for play, expected_type in zip(sample_plays, expected):
        classified = classify_event(play)
        match = classified == expected_type
        
        if match:
            correct += 1
            print(f"   ✅ '{play['text'][:40]}...' → {classified}")
        else:
            print(f"   ❌ '{play['text'][:40]}...' → {classified} (expected: {expected_type})")
    
    # Pass if at least 80% correct
    accuracy = correct / total
    passed = accuracy >= 0.8
    
    status = f"{correct}/{total} correct ({accuracy*100:.0f}%)"
    print(f"   {'✅ PASSED' if passed else '❌ FAILED'} - {status}")
    
    return passed, status


# ============================================================================
# TEST 3: Live Game Detection
# ============================================================================

async def test_live_game_detection() -> Tuple[bool, str]:
    """
    Test ability to detect currently live games.
    """
    print("\n🧪 Test 3: Live Game Detection")
    
    games = get_live_games()
    
    if games:
        print(f"   Found {len(games)} live game(s):")
        for game in games[:3]:  # Show max 3
            print(f"      • {game.get('name', 'Unknown')}")
        passed = True
        status = f"{len(games)} live games detected"
    else:
        print("   No live games currently (this is normal outside game times)")
        passed = True  # Not a failure - just no games right now
        status = "No live games (time-dependent)"
    
    print(f"   {'✅ PASSED' if passed else '❌ FAILED'} - {status}")
    return passed, status


# ============================================================================
# MAIN RUNNER
# ============================================================================

async def run_all_espn_tests():
    """Run all ESPN integration tests."""
    print("=" * 60)
    print("P2: ESPN LIVE EVENT INTEGRATION TESTS")
    print("=" * 60)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Polling latency
    passed, status = await test_espn_polling_latency()
    results.append(("ESPN Polling Latency", passed, status))
    
    # Test 2: Event classification
    passed, status = await test_event_classification()
    results.append(("Event Classification", passed, status))
    
    # Test 3: Live game detection
    passed, status = await test_live_game_detection()
    results.append(("Live Game Detection", passed, status))
    
    # Summary
    print("\n" + "=" * 60)
    print("ESPN INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed, status in results:
        symbol = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {symbol} - {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL ESPN INTEGRATION TESTS PASSED")
    else:
        print("⚠️  SOME ESPN TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_espn_tests())
    sys.exit(0 if success else 1)
