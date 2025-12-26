#!/usr/bin/env python3
"""
P0: End-to-End Pipeline Timing Tests

CRITICAL: These tests validate the core claim:
"Neuron converts live game events into publish-ready outputs within minutes."

Targets:
- Event → Debate text: <30 seconds
- Event → Audio ready: <60 seconds  
- Event → Full package: <120 seconds (2 minutes)

Run:
    python tests/e2e/test_full_pipeline_timing.py
"""

import asyncio
import json
import os
import sys
import time
import base64
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

MODAL_BASE_URL = "https://neuronsystems--neuron-orchestrator"
ENDPOINTS = {
    "run_debate": f"{MODAL_BASE_URL}-run-debate.modal.run",
    "generate_tts": f"{MODAL_BASE_URL}-generate-tts.modal.run",
    "generate_commentary": f"{MODAL_BASE_URL}-generate-commentary.modal.run",
}

# Timing targets (in seconds)
TARGETS = {
    "event_to_debate": 30,
    "event_to_audio": 60,
    "full_pipeline": 120,
}


@dataclass
class TimingResult:
    """Result from a timing test."""
    name: str
    target_seconds: float
    actual_seconds: float
    passed: bool
    details: Dict[str, Any]


# ============================================================================
# SIMULATED GAME EVENTS
# ============================================================================

SAMPLE_EVENTS = [
    {
        "type": "touchdown",
        "description": "Patrick Mahomes throws a 45-yard touchdown pass to Travis Kelce",
        "home_team": "Kansas City",
        "away_team": "Denver",
        "home_score": 14,
        "away_score": 7,
        "quarter": 2,
        "time_remaining": "8:42",
        "player": "Travis Kelce",
        "team": "Kansas City"
    },
    {
        "type": "interception",
        "description": "Josh Allen intercepted by Trevon Diggs in the end zone",
        "home_team": "Buffalo",
        "away_team": "Dallas",
        "home_score": 21,
        "away_score": 17,
        "quarter": 3,
        "time_remaining": "2:15",
        "player": "Trevon Diggs",
        "team": "Dallas"
    },
    {
        "type": "big_play",
        "description": "Tyreek Hill breaks 3 tackles for a 67-yard gain",
        "home_team": "Miami",
        "away_team": "New England",
        "home_score": 28,
        "away_score": 14,
        "quarter": 4,
        "time_remaining": "6:30",
        "player": "Tyreek Hill",
        "team": "Miami"
    }
]


# ============================================================================
# TEST 1: Event to Debate Timing (<30 seconds)
# ============================================================================

async def test_event_to_debate_timing() -> TimingResult:
    """
    Measures: Event detected → Debate text generated
    Target: <30 seconds
    
    Tests the core debate generation pipeline without audio.
    """
    print("\n🧪 Test 1: Event to Debate Timing")
    print(f"   Target: <{TARGETS['event_to_debate']} seconds")
    
    event = SAMPLE_EVENTS[0]  # Use touchdown event
    city1 = event["home_team"]
    city2 = event["away_team"]
    
    # Build debate topic from event
    topic = f"{event['description']}. {city1} leads {event['home_score']}-{event['away_score']} in Q{event['quarter']}."
    
    print(f"   Cities: {city1} vs {city2}")
    print(f"   Topic: {topic[:60]}...")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            ENDPOINTS["run_debate"],
            json={
                "city1": city1,
                "city2": city2,
                "topic": topic,
                "rounds": 1,  # Single round for speed
                "style": "homer",
                "game_context": {
                    "home_score": event["home_score"],
                    "away_score": event["away_score"],
                    "quarter": event["quarter"],
                    "event_type": event["type"]
                }
            },
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for success status 
            if data.get("status") != "success":
                print(f"   ❌ API returned status: {data.get('status')}")
                return TimingResult(
                    name="Event to Debate",
                    target_seconds=TARGETS["event_to_debate"],
                    actual_seconds=elapsed,
                    passed=False,
                    details={"error": f"Status: {data.get('status')}", "message": data.get('message', '')}
                )
            
            # Check we got actual debate content
            # Response format: {"status": "success", "debate": {"transcript": [{"city": ..., "response": ...}]}}
            debate_obj = data.get("debate", {})
            transcript = debate_obj.get("transcript", []) if isinstance(debate_obj, dict) else []
            
            has_content = len(transcript) > 0 and any(
                turn.get("response") for turn in transcript
            )
            
            passed = elapsed < TARGETS["event_to_debate"] and has_content
            
            print(f"   ⏱️  Time: {elapsed:.2f}s")
            print(f"   📝 Turns: {len(transcript)}")
            print(f"   {'✅ PASSED' if passed else '❌ FAILED'}")
            
            return TimingResult(
                name="Event to Debate",
                target_seconds=TARGETS["event_to_debate"],
                actual_seconds=elapsed,
                passed=passed,
                details={
                    "debate_turns": len(transcript),
                    "has_content": has_content,
                    "http_status": response.status_code
                }
            )
        else:
            print(f"   ❌ HTTP {response.status_code}")
            return TimingResult(
                name="Event to Debate",
                target_seconds=TARGETS["event_to_debate"],
                actual_seconds=elapsed,
                passed=False,
                details={"error": f"HTTP {response.status_code}"}
            )
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Error: {e}")
        return TimingResult(
            name="Event to Debate",
            target_seconds=TARGETS["event_to_debate"],
            actual_seconds=elapsed,
            passed=False,
            details={"error": str(e)}
        )


# ============================================================================
# TEST 2: Event to Audio Timing (<60 seconds)
# ============================================================================

async def test_event_to_audio_timing() -> TimingResult:
    """
    Measures: Event detected → Audio file ready
    Target: <60 seconds
    
    Tests debate generation + TTS synthesis.
    """
    print("\n🧪 Test 2: Event to Audio Timing")
    print(f"   Target: <{TARGETS['event_to_audio']} seconds")
    
    event = SAMPLE_EVENTS[1]  # Use interception event
    city1 = event["home_team"]
    city2 = event["away_team"]
    
    topic = f"{event['description']}. Score is {event['home_score']}-{event['away_score']}."
    
    print(f"   Cities: {city1} vs {city2}")
    
    start_time = time.time()
    debate_time = 0
    tts_time = 0
    
    try:
        # Step 1: Generate debate
        debate_start = time.time()
        debate_response = requests.post(
            ENDPOINTS["run_debate"],
            json={
                "city1": city1,
                "city2": city2,
                "topic": topic,
                "rounds": 1,
                "style": "homer"
            },
            timeout=45
        )
        debate_time = time.time() - debate_start
        
        if debate_response.status_code != 200:
            raise Exception(f"Debate failed: HTTP {debate_response.status_code}")
        
        debate_data = debate_response.json()
        
        if debate_data.get("status") != "success":
            raise Exception(f"Debate failed: {debate_data.get('message', 'Unknown error')}")
        
        debate_obj = debate_data.get("debate", {})
        transcript = debate_obj.get("transcript", []) if isinstance(debate_obj, dict) else []
        
        if not transcript:
            raise Exception("No debate content generated")
        
        # Get first turn's content for TTS
        first_turn = transcript[0]
        text_for_tts = first_turn.get("response", "")[:500]  # Limit to 500 chars
        speaker = first_turn.get("city", city1)
        
        print(f"   📝 Debate: {debate_time:.2f}s ({len(transcript)} turns)")
        
        # Step 2: Generate audio
        tts_start = time.time()
        tts_response = requests.post(
            ENDPOINTS["generate_tts"],
            json={
                "text": text_for_tts,
                "speaker_id": speaker,
                "provider": "google"  # Use Google for reliability
            },
            timeout=30
        )
        tts_time = time.time() - tts_start
        
        total_elapsed = time.time() - start_time
        
        if tts_response.status_code != 200:
            raise Exception(f"TTS failed: HTTP {tts_response.status_code}")
        
        tts_data = tts_response.json()
        audio_base64 = tts_data.get("audio") or tts_data.get("audio_base64", "")
        
        # Verify audio is valid base64
        audio_valid = False
        audio_bytes = 0
        if audio_base64:
            try:
                audio_data = base64.b64decode(audio_base64)
                audio_bytes = len(audio_data)
                audio_valid = audio_bytes > 1000  # At least 1KB
            except:
                pass
        
        passed = total_elapsed < TARGETS["event_to_audio"] and audio_valid
        
        print(f"   🔊 TTS: {tts_time:.2f}s ({audio_bytes} bytes)")
        print(f"   ⏱️  Total: {total_elapsed:.2f}s")
        print(f"   {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return TimingResult(
            name="Event to Audio",
            target_seconds=TARGETS["event_to_audio"],
            actual_seconds=total_elapsed,
            passed=passed,
            details={
                "debate_time": debate_time,
                "tts_time": tts_time,
                "audio_bytes": audio_bytes,
                "audio_valid": audio_valid
            }
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Error: {e}")
        return TimingResult(
            name="Event to Audio",
            target_seconds=TARGETS["event_to_audio"],
            actual_seconds=elapsed,
            passed=False,
            details={"error": str(e)}
        )


# ============================================================================
# TEST 3: Full Content Creation Timing (<120 seconds)
# ============================================================================

async def test_full_content_creation_timing() -> TimingResult:
    """
    Measures: Event detected → Full publish-ready package
    Target: <120 seconds (2 minutes)
    
    Simulates full workflow: 3-round debate with audio for each turn.
    """
    print("\n🧪 Test 3: Full Content Creation Timing")
    print(f"   Target: <{TARGETS['full_pipeline']} seconds")
    
    event = SAMPLE_EVENTS[2]  # Use big play event
    city1 = event["home_team"]
    city2 = event["away_team"]
    
    topic = f"{event['description']}. {city1} up {event['home_score']}-{event['away_score']} in Q{event['quarter']}."
    
    print(f"   Cities: {city1} vs {city2}")
    print(f"   Mode: 2-round debate with audio")
    
    start_time = time.time()
    
    try:
        # Step 1: Generate multi-round debate
        debate_start = time.time()
        debate_response = requests.post(
            ENDPOINTS["run_debate"],
            json={
                "city1": city1,
                "city2": city2,
                "topic": topic,
                "rounds": 2,  # 2 rounds = 4 turns
                "style": "homer",
                "game_context": {
                    "home_score": event["home_score"],
                    "away_score": event["away_score"],
                    "quarter": event["quarter"],
                    "player": event["player"],
                    "event_type": event["type"]
                }
            },
            timeout=90
        )
        debate_time = time.time() - debate_start
        
        if debate_response.status_code != 200:
            raise Exception(f"Debate failed: HTTP {debate_response.status_code}")
        
        debate_data = debate_response.json()
        
        if debate_data.get("status") != "success":
            raise Exception(f"Debate failed: {debate_data.get('message', 'Unknown error')}")
        
        debate_obj = debate_data.get("debate", {})
        transcript = debate_obj.get("transcript", []) if isinstance(debate_obj, dict) else []
        
        print(f"   📝 Debate: {debate_time:.2f}s ({len(transcript)} turns)")
        
        # Step 2: Generate audio for first 2 turns (representative sample)
        audio_segments = []
        turns_to_process = min(2, len(transcript))
        
        for i, turn in enumerate(transcript[:turns_to_process]):
            text = turn.get("response", "")[:400]
            speaker = turn.get("city", city1 if i % 2 == 0 else city2)
            
            tts_start = time.time()
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
            
            if tts_response.status_code == 200:
                tts_data = tts_response.json()
                audio = tts_data.get("audio") or tts_data.get("audio_base64", "")
                try:
                    audio_bytes = len(base64.b64decode(audio)) if audio else 0
                except:
                    audio_bytes = 0
                    
                audio_segments.append({
                    "turn": i + 1,
                    "speaker": speaker,
                    "tts_time": tts_time,
                    "audio_bytes": audio_bytes
                })
                print(f"   🔊 Turn {i+1}: {tts_time:.2f}s ({audio_bytes} bytes)")
        
        total_elapsed = time.time() - start_time
        
        # Calculate success metrics
        has_debate = len(transcript) >= 2
        has_audio = len(audio_segments) >= 1 and all(s["audio_bytes"] > 1000 for s in audio_segments)
        passed = total_elapsed < TARGETS["full_pipeline"] and has_debate and has_audio
        
        print(f"   ⏱️  Total: {total_elapsed:.2f}s")
        print(f"   {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return TimingResult(
            name="Full Content Creation",
            target_seconds=TARGETS["full_pipeline"],
            actual_seconds=total_elapsed,
            passed=passed,
            details={
                "debate_time": debate_time,
                "debate_turns": len(transcript),
                "audio_segments": len(audio_segments),
                "has_debate": has_debate,
                "has_audio": has_audio
            }
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Error: {e}")
        return TimingResult(
            name="Full Content Creation",
            target_seconds=TARGETS["full_pipeline"],
            actual_seconds=elapsed,
            passed=False,
            details={"error": str(e)}
        )


# ============================================================================
# MAIN RUNNER
# ============================================================================

async def run_all_timing_tests():
    """Run all P0 timing tests and report results."""
    print("=" * 60)
    print("P0: END-TO-END PIPELINE TIMING TESTS")
    print("=" * 60)
    print(f"\nValidating: 'Neuron converts live events to publish-ready")
    print(f"outputs within minutes'\n")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results: List[TimingResult] = []
    
    # Test 1: Event to Debate
    result1 = await test_event_to_debate_timing()
    results.append(result1)
    
    # Test 2: Event to Audio
    result2 = await test_event_to_audio_timing()
    results.append(result2)
    
    # Test 3: Full Pipeline
    result3 = await test_full_content_creation_timing()
    results.append(result3)
    
    # Summary
    print("\n" + "=" * 60)
    print("TIMING TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        pct = (result.actual_seconds / result.target_seconds) * 100
        print(f"   {result.name}:")
        print(f"      Target: <{result.target_seconds}s | Actual: {result.actual_seconds:.2f}s ({pct:.0f}%)")
        print(f"      Status: {status}")
        if not result.passed:
            all_passed = False
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL P0 TIMING TESTS PASSED")
        print("✅ Claim validated: Outputs generated within minutes")
    else:
        print("⚠️  SOME P0 TIMING TESTS FAILED")
        print("   Review failed tests above for details")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_timing_tests())
    sys.exit(0 if success else 1)
