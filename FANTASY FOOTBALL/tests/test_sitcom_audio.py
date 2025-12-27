#!/usr/bin/env python3
"""
Test Sitcom Mode with Audio Output
"""

import requests
import base64
import json

GENERATE_COMMENTARY_URL = "https://neuronsystems--neuron-orchestrator-generate-commentary.modal.run"

def test_sitcom_audio():
    print("🎭 Generating Sitcom-Style Debate with Audio...\n")
    
    payload = {
        "city": "Philadelphia",
        "user_input": "The Cowboys are definitely winning the Super Bowl this year!",
        "style": "sitcom",
        "tier": "standard"  # Uses Google TTS
    }
    
    print(f"Request: Philadelphia reacting to Cowboys Super Bowl prediction")
    print(f"Style: Sitcom Mode\n")
    
    try:
        response = requests.post(
            GENERATE_COMMENTARY_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "data" in result and "response" in result["data"]:
            text = result["data"]["response"]
            
            print("=" * 80)
            print("SITCOM RESPONSE TEXT")
            print("=" * 80)
            print(text)
            print("\n")
            
            # Check if audio was generated
            if "audio" in result and "data" in result["audio"]:
                audio_base64 = result["audio"]["data"]
                audio_bytes = base64.b64decode(audio_base64)
                
                # Save audio file
                output_file = "tests/sitcom_debate_audio.mp3"
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                
                print("=" * 80)
                print("AUDIO GENERATED")
                print("=" * 80)
                print(f"✅ Audio saved to: {output_file}")
                print(f"   Size: {len(audio_bytes):,} bytes")
                print(f"   Format: {result['audio']['metadata'].get('audio_format', 'mp3')}")
                print(f"   Sample rate: {result['audio']['metadata'].get('sample_rate', 'unknown')} Hz")
                print(f"\n🎧 Play it with: afplay {output_file}")
                
                return output_file
            else:
                print("⚠️  No audio in response (audio generation may have failed)")
                return None
                
        else:
            print(f"❌ Unexpected response format: {json.dumps(result, indent=2)}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return None

if __name__ == "__main__":
    audio_file = test_sitcom_audio()
    
    if audio_file:
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Sitcom audio ready to play.")
        print("=" * 80)
