#!/usr/bin/env python3
"""
Performance Optimization: Parallel Pipeline Execution

This script demonstrates and benchmarks parallel vs sequential execution
of the debate + TTS pipeline.

Run:
    python scripts/parallel_pipeline_benchmark.py
"""

import asyncio
import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

MODAL_BASE_URL = "https://neuronsystems--neuron-orchestrator"
ENDPOINTS = {
    "run_debate": f"{MODAL_BASE_URL}-run-debate.modal.run",
    "generate_tts": f"{MODAL_BASE_URL}-generate-tts.modal.run",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_debate(city1: str, city2: str, topic: str, rounds: int = 1) -> Dict[str, Any]:
    """Generate debate synchronously."""
    response = requests.post(
        ENDPOINTS["run_debate"],
        json={
            "city1": city1,
            "city2": city2,
            "topic": topic,
            "rounds": rounds,
            "style": "homer"
        },
        timeout=90
    )
    return response.json()


def generate_tts(text: str, speaker: str) -> bytes:
    """Generate TTS synchronously."""
    response = requests.post(
        ENDPOINTS["generate_tts"],
        json={
            "text": text[:500],  # Limit text length
            "speaker_id": speaker,
            "provider": "google"
        },
        timeout=30
    )
    data = response.json()
    audio_b64 = data.get("audio") or data.get("audio_base64", "")
    return base64.b64decode(audio_b64) if audio_b64 else b""


# ============================================================================
# SEQUENTIAL PIPELINE
# ============================================================================

def run_sequential_pipeline(city1: str, city2: str, topic: str) -> Tuple[float, Dict]:
    """
    Run pipeline sequentially: All debate turns, then all TTS.
    This is the current approach.
    """
    start = time.time()
    
    # Step 1: Generate full debate
    debate_data = generate_debate(city1, city2, topic, rounds=2)
    debate_time = time.time() - start
    
    if debate_data.get("status") != "success":
        return time.time() - start, {"error": "Debate failed"}
    
    transcript = debate_data.get("debate", {}).get("transcript", [])
    
    # Step 2: Generate TTS for each turn sequentially
    tts_start = time.time()
    audio_segments = []
    
    for turn in transcript[:4]:  # Max 4 turns
        text = turn.get("response", "")
        speaker = turn.get("city", city1)
        audio = generate_tts(text, speaker)
        audio_segments.append(len(audio))
    
    tts_time = time.time() - tts_start
    total_time = time.time() - start
    
    return total_time, {
        "debate_time": debate_time,
        "tts_time": tts_time,
        "turns": len(transcript),
        "audio_segments": len(audio_segments)
    }


# ============================================================================
# PARALLEL PIPELINE
# ============================================================================

async def run_parallel_pipeline(city1: str, city2: str, topic: str) -> Tuple[float, Dict]:
    """
    Run pipeline with parallel TTS generation.
    Generates TTS for all turns concurrently.
    """
    start = time.time()
    
    # Step 1: Generate full debate (still sequential - needed for content)
    loop = asyncio.get_event_loop()
    debate_data = await loop.run_in_executor(
        None, 
        lambda: generate_debate(city1, city2, topic, rounds=2)
    )
    debate_time = time.time() - start
    
    if debate_data.get("status") != "success":
        return time.time() - start, {"error": "Debate failed"}
    
    transcript = debate_data.get("debate", {}).get("transcript", [])
    
    # Step 2: Generate TTS for ALL turns in parallel
    tts_start = time.time()
    
    async def tts_async(text: str, speaker: str) -> bytes:
        return await loop.run_in_executor(
            None,
            lambda: generate_tts(text, speaker)
        )
    
    # Create tasks for all turns
    tasks = [
        tts_async(turn.get("response", ""), turn.get("city", city1))
        for turn in transcript[:4]
    ]
    
    # Run all TTS in parallel
    audio_results = await asyncio.gather(*tasks)
    
    tts_time = time.time() - tts_start
    total_time = time.time() - start
    
    return total_time, {
        "debate_time": debate_time,
        "tts_time": tts_time,
        "turns": len(transcript),
        "audio_segments": len(audio_results)
    }


# ============================================================================
# BENCHMARK
# ============================================================================

async def run_benchmark():
    """Compare sequential vs parallel pipeline performance."""
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    city1 = "Kansas City"
    city2 = "Buffalo"
    topic = "React to this incredible touchdown pass! The crowd goes wild!"
    
    print(f"\n📍 Test Case:")
    print(f"   {city1} vs {city2}")
    print(f"   2-round debate + TTS for all turns")
    
    # Run sequential
    print("\n🔄 Running Sequential Pipeline...")
    seq_time, seq_details = run_sequential_pipeline(city1, city2, topic)
    print(f"   Total: {seq_time:.2f}s")
    print(f"   Debate: {seq_details.get('debate_time', 0):.2f}s")
    print(f"   TTS: {seq_details.get('tts_time', 0):.2f}s (sequential)")
    
    # Run parallel
    print("\n⚡ Running Parallel Pipeline...")
    par_time, par_details = await run_parallel_pipeline(city1, city2, topic)
    print(f"   Total: {par_time:.2f}s")
    print(f"   Debate: {par_details.get('debate_time', 0):.2f}s")
    print(f"   TTS: {par_details.get('tts_time', 0):.2f}s (parallel)")
    
    # Calculate improvement
    improvement = ((seq_time - par_time) / seq_time) * 100 if seq_time > 0 else 0
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"   Sequential: {seq_time:.2f}s")
    print(f"   Parallel:   {par_time:.2f}s")
    print(f"   Improvement: {improvement:.1f}% faster")
    
    if improvement > 10:
        print(f"\n🚀 Parallel execution is {improvement:.0f}% faster!")
    else:
        print(f"\n📊 Minimal difference (mostly debate generation time)")
    
    print("=" * 60)
    
    return improvement > 0


if __name__ == "__main__":
    success = asyncio.run(run_benchmark())
    sys.exit(0 if success else 1)
