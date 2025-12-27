#!/usr/bin/env python3
"""
Generate Multi-City Sitcom Debate with Different Voices
This version generates TTS ourselves for each segment
"""

import requests
import base64
import json
import subprocess

RUN_DEBATE_URL = "https://neuronsystems--neuron-orchestrator-run-debate.modal.run"
GENERATE_TTS_URL = "https://neuronsystems--neuron-orchestrator-generate-tts.modal.run"

def generate_multi_city_debate():
    print("🎭 Generating Multi-City Sitcom Debate with Different Voices...\n")
    print("📍 Matchup: Philadelphia vs Dallas")
    print("🎙️ Each city gets its own unique voice!\n")
    
    # Step 1: Generate the debate transcript
    payload = {
        "topic": "Who has the better quarterback: Jalen Hurts or Dak Prescott?",
        "city1": "Philadelphia",
        "city2": "Dallas",
        "num_rounds": 2,  # 2 rounds = 4 turns total
        "style": "sitcom"
    }
    
    print(f"Topic: {payload['topic']}")
    print(f"Rounds: {payload['num_rounds']}\n")
    print("⏳ Generating debate transcript...\n")
    
    try:
        response = requests.post(
            RUN_DEBATE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=90
        )
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "success" and "debate" in result:
            debate = result["debate"]
            transcript = debate.get("transcript", [])
            
            print(f"✅ Generated {len(transcript)} debate turns!\n")
            
            audio_files = []
            
            # Step 2: Generate TTS for each turn
            for i, turn in enumerate(transcript):
                speaker = turn.get("speaker")
                text = turn.get("response", "")
                round_num = turn.get("round", 0)
                
                print("=" * 80)
                print(f"ROUND {round_num} - {speaker}")
                print("=" * 80)
                print(text[:300] + "..." if len(text) > 300 else text)
                print()
                
                # Generate TTS for this turn
                print(f"🎙️  Generating audio with {speaker}'s voice...")
                
                tts_payload = {
                    "text": text,
                    "speaker_id": speaker,
                    "provider": "google"  # Use Google TTS
                }
                
                tts_response = requests.post(
                    GENERATE_TTS_URL,
                    json=tts_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if tts_response.ok:
                    tts_result = tts_response.json()
                    if "audio" in tts_result and "data" in tts_result["audio"]:
                        audio_base64 = tts_result["audio"]["data"]
                        audio_bytes = base64.b64decode(audio_base64)
                        
                        audio_file = f"tests/debate_r{round_num}_{speaker.replace(' ', '_')}.mp3"
                        with open(audio_file, "wb") as f:
                            f.write(audio_bytes)
                        
                        audio_files.append(audio_file)
                        print(f"   ✅ Saved: {audio_file} ({len(audio_bytes):,} bytes)\n")
                    else:
                        print(f"   ⚠️  No audio in response")
                else:
                    print(f"   ❌ TTS failed: {tts_response.status_code}")
            
            # Step 3: Play all audio files in sequence
            if audio_files:
                print("=" * 80)
                print("🎧 PLAYING FULL DEBATE")
                print("=" * 80)
                print(f"Playing {len(audio_files)} segments with different voices...\n")
                
                for i, audio_file in enumerate(audio_files):
                    print(f"▶️  [{i+1}/{len(audio_files)}] Playing: {audio_file}")
                    subprocess.run(["afplay", audio_file])
                    print()
                
                print("=" * 80)
                print("✅ DEBATE COMPLETE!")
                print("=" * 80)
                print(f"\nAudio files saved:")
                for f in audio_files:
                    print(f"  - {f}")
                
                return audio_files
            else:
                print("⚠️  No audio was generated")
                return None
        else:
            print(f"❌ Unexpected response: {json.dumps(result, indent=2)[:500]}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text[:500]}")
        return None

if __name__ == "__main__":
    generate_multi_city_debate()
