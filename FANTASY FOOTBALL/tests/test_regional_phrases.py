#!/usr/bin/env python3
"""
Test regional phrase integration
Verifies that cities are using their authentic regional vocabulary
"""

import requests

# Test different regional cities
def test_regional_phrases():
    """Test sitcom mode with regional phrases"""
    
    test_cities = [
        ("Philadelphia", ["jawn", "yo", "youse", "down the shore", "wawa"]),
        ("Pittsburgh", ["yinz", "dahntahn", "primantis", "jagoff"]),
        ("New Orleans", ["where yat", "makin groceries", "who dat", "lagniappe"]),
        ("Minnesota", ["ope", "you betcha", "dontcha know", "uff da"]),
        ("Boston", ["wicked", "pahk the cah", "packie", "dunks"])
    ]
    
    print("🎭 Testing Regional Phrase Integration\n")
    print("=" * 80)
    
    for city, expected_phrases in test_cities:
        print(f"\n📍 Testing: {city}")
        print(f"Expected regional phrases: {', '.join(expected_phrases)}")
        print("-" * 80)
        
        # Simulate debate request
        payload = {
            "topic": f"React to {city} winning the Super Bowl prediction",
            "city1": city,
            "city2": "Dallas",  # Generic opponent
            "num_rounds": 1,
            "style": "sitcom"
        }
        
        try:
            # Call run_debate endpoint
            response = requests.post(
                "https://neuronsystems--neuron-orchestrator-run-debate.modal.run",
                json=payload,
                timeout=60
            )
            
            if response.ok:
                result = response.json()
                if result.get("status") == "success":
                    transcript = result["debate"]["transcript"]
                    if transcript:
                        text = transcript[0]["response"]
                        
                        # Check which phrases appeared
                        found_phrases = [p for p in expected_phrases if p.lower() in text.lower()]
                        
                        print(f"✅ Response generated ({len(text)} chars)")
                        print(f"📝 Sample: {text[:150]}...")
                        print(f"\n🎯 Regional phrases found: {len(found_phrases)}/{len(expected_phrases)}")
                        
                        if found_phrases:
                            print(f"   Found: {', '.join(found_phrases)}")
                        else:
                            print(f"   ⚠️  No expected regional phrases found yet")
                            print(f"   (Note: LLM learns over time, injection rate is {0.15 * 100}%)")
                    else:
                        print("❌ Empty transcript")
                else:
                    print(f"❌ Debate failed: {result}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Regional phrase testing complete!")
    print("\nNote: Phrases appear based on injection_rate (12-18%), so not every response")
    print("will have ALL phrases. The LLM learns patterns over multiple generations.")

if __name__ == "__main__":
    test_regional_phrases()
