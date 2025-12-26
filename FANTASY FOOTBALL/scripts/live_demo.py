#!/usr/bin/env python3
"""
Live Demo Script for Neuron Real-Time Creator Studio

This script monitors live NFL games and demonstrates the full
event-to-content pipeline in real-time.

Usage:
    python scripts/live_demo.py

Features:
- Auto-detects live NFL games from ESPN
- Monitors for scoring plays
- Generates debate content + audio for each event
- Measures and displays timing metrics
"""

import asyncio
import base64
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
MODAL_BASE_URL = "https://neuronsystems--neuron-orchestrator"
ENDPOINTS = {
    "run_debate": f"{MODAL_BASE_URL}-run-debate.modal.run",
    "generate_tts": f"{MODAL_BASE_URL}-generate-tts.modal.run",
}

POLL_INTERVAL = 30  # seconds between checks


# ============================================================================
# ESPN HELPERS
# ============================================================================

def get_live_games() -> List[Dict[str, Any]]:
    """Fetch current live NFL games."""
    try:
        response = requests.get(f"{ESPN_API_BASE}/scoreboard", timeout=10)
        data = response.json()
        
        live_games = []
        for event in data.get("events", []):
            status = event.get("status", {}).get("type", {}).get("name", "")
            if status == "STATUS_IN_PROGRESS":
                competition = event.get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])
                
                home = next((c for c in competitors if c.get("homeAway") == "home"), {})
                away = next((c for c in competitors if c.get("homeAway") == "away"), {})
                
                live_games.append({
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "home_team": home.get("team", {}).get("displayName", "Home"),
                    "away_team": away.get("team", {}).get("displayName", "Away"),
                    "home_score": int(home.get("score", 0)),
                    "away_score": int(away.get("score", 0)),
                    "status_detail": event.get("status", {}).get("type", {}).get("detail", "")
                })
        
        return live_games
        
    except Exception as e:
        print(f"⚠️ Error fetching games: {e}")
        return []


def get_game_summary(game_id: str) -> Dict[str, Any]:
    """Get detailed game info including recent plays."""
    try:
        response = requests.get(f"{ESPN_API_BASE}/summary?event={game_id}", timeout=10)
        return response.json()
    except:
        return {}


# ============================================================================
# CONTENT GENERATION
# ============================================================================

async def generate_content_for_event(
    home_team: str,
    away_team: str,
    event_description: str,
    score: str
) -> Dict[str, Any]:
    """Generate debate + audio for an event."""
    start = time.time()
    
    topic = f"{event_description}. Score is {score}."
    
    # Extract city names (handle multi-word cities like "Kansas City")
    def get_city_name(team_name: str) -> str:
        parts = team_name.split()
        if len(parts) <= 2:
            return parts[0]  # "Bills" -> "Buffalo" won't work, use first word
        # For "Kansas City Chiefs" -> "Kansas City"
        # For "New England Patriots" -> "New England" 
        return " ".join(parts[:-1])  # Remove last word (team mascot)
    
    city1 = get_city_name(home_team)
    city2 = get_city_name(away_team)
    
    # Step 1: Generate debate
    try:
        response = requests.post(
            ENDPOINTS["run_debate"],
            json={
                "city1": city1,
                "city2": city2,
                "topic": topic,
                "rounds": 1,
                "style": "homer"
            },
            timeout=60
        )
        debate_time = time.time() - start
        
        if response.status_code != 200:
            return {"error": f"Debate failed: HTTP {response.status_code}"}
        
        data = response.json()
        if data.get("status") != "success":
            return {"error": "Debate generation failed"}
        
        transcript = data.get("debate", {}).get("transcript", [])
        
    except Exception as e:
        return {"error": f"Debate error: {str(e)[:50]}"}
    
    # Step 2: Generate TTS for first turn
    tts_start = time.time()
    try:
        first_turn = transcript[0] if transcript else {}
        text = first_turn.get("response", "")[:400]
        speaker = first_turn.get("city", home_team)
        
        tts_response = requests.post(
            ENDPOINTS["generate_tts"],
            json={
                "text": text,
                "speaker_id": speaker,
                "provider": "google"
            },
            timeout=30
        )
        tts_time = time.time() - tts_start
        
        audio_b64 = ""
        if tts_response.status_code == 200:
            tts_data = tts_response.json()
            audio_b64 = tts_data.get("audio") or tts_data.get("audio_base64", "")
        
    except Exception as e:
        tts_time = time.time() - tts_start
        audio_b64 = ""
    
    total_time = time.time() - start
    
    return {
        "success": True,
        "debate_time": debate_time,
        "tts_time": tts_time,
        "total_time": total_time,
        "transcript": transcript,
        "has_audio": len(audio_b64) > 100
    }


# ============================================================================
# MAIN DEMO LOOP
# ============================================================================

async def run_demo():
    """Main demo loop - monitors games and generates content."""
    print("=" * 60)
    print("🏈 NEURON LIVE DEMO")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Monitoring for live NFL games...\n")
    
    last_scores = {}  # Track scores to detect changes
    
    while True:
        games = get_live_games()
        
        if not games:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No live games. Checking again in {POLL_INTERVAL}s...")
            await asyncio.sleep(POLL_INTERVAL)
            continue
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(games)} live game(s)")
        
        for game in games:
            game_id = game["id"]
            current_score = f"{game['home_score']}-{game['away_score']}"
            
            # Check if score changed (potential scoring event)
            if game_id in last_scores and last_scores[game_id] != current_score:
                print(f"\n🚨 SCORING EVENT DETECTED!")
                print(f"   {game['name']}")
                print(f"   Score: {last_scores[game_id]} → {current_score}")
                
                # Generate content
                print(f"   Generating content...")
                result = await generate_content_for_event(
                    game["home_team"],
                    game["away_team"],
                    f"Score update in {game['name']}",
                    current_score
                )
                
                if result.get("success"):
                    print(f"   ✅ Content generated in {result['total_time']:.1f}s")
                    print(f"      • Debate: {result['debate_time']:.1f}s")
                    print(f"      • Audio: {'✅' if result['has_audio'] else '❌'}")
                else:
                    print(f"   ❌ {result.get('error', 'Unknown error')}")
            
            last_scores[game_id] = current_score
            
            # Display current status
            print(f"   📺 {game['name']}: {current_score} ({game['status_detail']})")
        
        await asyncio.sleep(POLL_INTERVAL)


def run_single_demo():
    """Run a single demo cycle without live monitoring."""
    print("=" * 60)
    print("🏈 NEURON SINGLE DEMO")
    print("=" * 60)
    
    # Check for live games first
    games = get_live_games()
    
    if games:
        game = games[0]
        print(f"\n📺 Using live game: {game['name']}")
        home = game["home_team"]
        away = game["away_team"]
        event = f"Exciting play in the {game['name']}"
        score = f"{game['home_score']}-{game['away_score']}"
    else:
        print("\nNo live games - using sample event")
        home = "Kansas City Chiefs"
        away = "Buffalo Bills"
        event = "Mahomes throws a 30-yard touchdown to Kelce!"
        score = "21-14"
    
    print(f"   {home} vs {away}")
    print(f"   Event: {event}")
    print(f"   Score: {score}")
    
    print("\n⏳ Generating content...")
    result = asyncio.run(generate_content_for_event(home, away, event, score))
    
    if result.get("success"):
        print(f"\n✅ SUCCESS!")
        print(f"   Total time: {result['total_time']:.1f}s")
        print(f"   Debate: {result['debate_time']:.1f}s ({len(result.get('transcript', []))} turns)")
        print(f"   Audio: {'✅ Generated' if result['has_audio'] else '❌ Failed'}")
        
        # Show first response preview
        transcript = result.get('transcript', [])
        if transcript:
            preview = transcript[0].get('response', '')[:100]
            print(f"\n📝 Preview: \"{preview}...\"")
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown')}")
    
    print("=" * 60)


if __name__ == "__main__":
    if "--monitor" in sys.argv:
        asyncio.run(run_demo())
    else:
        run_single_demo()
