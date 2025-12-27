#!/usr/bin/env python3
"""
Working Multi-City Debate Generator
Generates debates with different voices for each city
"""

import requests
import base64
import subprocess
import time

RUN_DEBATE_URL = "https://neuronsystems--neuron-orchestrator-run-debate.modal.run"
GENERATE_TTS_URL = "https://neuronsystems--neuron-orchestrator-generate-tts.modal.run"

def run_multi_city_debate(use_premium=False):
    provider = "elevenlabs" if use_premium else "google"
    tier_name = "Premium (ElevenLabs)" if use_premium else "Standard (Google TTS)"
    
    print(f"🎭 Generating Multi-City Debate - {tier_name}\n")
    print("=" * 80)
    print("📍 Matchup: Philadelphia vs Dallas")
    print(f"🎙️  Audio: {tier_name} with unique voices per city")
    print("🎬 Topic: Jalen Hurts vs Dak Prescott")
    print("=" * 80)
    print()
    
    # Step 1: Generate debate transcript (sitcom style!)
    print("⏳ Step 1/2: Generating debate transcript with sitcom personalities...")
    
    payload = {
        "topic": "Who is the better quarterback: Jalen Hurts or Dak Prescott?",
        "city1": "Philadelphia",
        "city2": "Dallas",
        "num_rounds": 2,  # 2 rounds = 4 turns
        "style": "sitcom"
    }
    
    try:
        response = requests.post(RUN_DEBATE_URL, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") != "success":
            print(f"❌ Debate generation failed: {result}")
            return None
        
        transcript = result["debate"]["transcript"]
        print(f"✅ Generated {len(transcript)} debate turns\n")
        
        # Step 2: Generate TTS for each turn
        print(f"⏳ Step 2/2: Converting to audio with {tier_name}...\n")
        
        audio_files = []
        
        for i, turn in enumerate(transcript):
            speaker = turn["speaker"]
            text = turn["response"]
            round_num = turn["round"]
            
            print(f"🎙️  Round {round_num} - {speaker}")
            print(f"   Text preview: {text[:80]}...")
            
            # Call generate_tts endpoint
            tts_payload = {
                "text": text,
                "speaker_id": speaker,
                "provider": provider,  # "google" or "elevenlabs"
                "force_provider": provider
            }
            
            tts_response = requests.post(GENERATE_TTS_URL, json=tts_payload, timeout=45)
            
            if tts_response.ok:
                tts_data = tts_response.json()
                
                # Check correct response path
                if "audio_base64" in tts_data:
                    audio_base64 = tts_data["audio_base64"]
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    audio_file = f"tests/debate_r{round_num}_{speaker.replace(' ', '_')}_{provider}.mp3"
                    with open(audio_file, "wb") as f:
                        f.write(audio_bytes)
                    
                    audio_files.append(audio_file)
                    provider_used = tts_data.get("provider", provider)
                    print(f"   ✅ Audio saved ({len(audio_bytes):,} bytes, {provider_used})")
                else:
                    print(f"   ❌ No audio in response: {tts_data}")
            else:
                print(f"   ❌ TTS failed: {tts_response.status_code}")
            
            print()
        
        # Step 3: Play the debate
        if audio_files:
            print("=" * 80)
            print(f"🎧 PLAYING DEBATE ({len(audio_files)} segments)")
            print("=" * 80)
            print()
            
            for i, audio_file in enumerate(audio_files):
                speaker = "Philadelphia" if "Philadelphia" in audio_file else "Dallas"
                round_num = int(audio_file.split("_r")[1][0])
                
                print(f"▶️  [{i+1}/{len(audio_files)}] Round {round_num} - {speaker}")
                subprocess.run(["afplay", audio_file])
                time.sleep(0.5)  # Brief pause between speakers
            
            print()
            print("=" * 80)
            print("✅ DEBATE COMPLETE!")
            print("=" * 80)
            print(f"\nAudio Quality: {tier_name}")
            print(f"Files saved:")
            for f in audio_files:
                print(f"  - {f}")
            
            return audio_files
        else:
            print("⚠️  No audio files were generated")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-CITY DEBATE AUDIO DEMO")
    print("=" * 80)
    print()
    
    # Test 1: Google TTS (Standard - FREE)
    print("🔊 TEST 1: GOOGLE TTS (Different voices per city)\n")
    google_files = run_multi_city_debate(use_premium=False)
    
    if google_files:
        print("\n" + "=" * 80)
        print()
        
        # Test 2: ElevenLabs (Premium)
        print("🔊 TEST 2: ELEVENLABS (Ultra-realistic AI voices)\n")
        elevenlabs_files = run_multi_city_debate(use_premium=True)
        
        if elevenlabs_files:
            print("\n" + "=" * 80)
            print("📊 COMPARISON COMPLETE")
            print("=" * 80)
            print("\nYou just heard the SAME debate with:")
            print("  1. Google TTS (standard quality, unique voice per city)")
            print("  2. ElevenLabs (premium quality, ultra-realistic)")
            print("\nBoth use different voices for Philadelphia and Dallas!")
    
    print("\n✅ All tasks complete!")
