
import requests
import json
import sys

# Endpoint URL from deployment output
ENDPOINT_URL = "https://neuronsystems--neuron-orchestrator-generate-commentary.modal.run"

def verify_sitcom_mode():
    print(f"Testing Sitcom Mode at: {ENDPOINT_URL}")
    
    payload = {
        "city": "Philadelphia",
        "user_input": "What do you think about the Cowboys winning the Super Bowl?",
        "style": "sitcom",
        "tier": "standard" # standard tier uses Google TTS which is faster for testing
    }
    
    try:
        response = requests.post(ENDPOINT_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        if "data" in data and "response" in data["data"]:
            text_response = data["data"]["response"]
            print("\n✅ Verification Successful! Response received.")
            print("\n--- SITCOM RESPONSE PREVIEW ---")
            print(text_response)
            print("-------------------------------")
            
            # Simple check for audio cues
            cues = ["[Sighs]", "[Laughs]", "[Clears throat]", "...", "!!!"]
            found_cues = [cue for cue in cues if cue in text_response or text_response.count("!") > 3]
            
            if found_cues:
                print(f"✅ Audio/Sitcom cues detected: {found_cues}")
            else:
                print("⚠️ No explicit audio cues found (might just be the random generation), but response was received.")
                
        else:
            print("❌ Invalid response format:")
            print(json.dumps(data, indent=2))
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        sys.exit(1)

if __name__ == "__main__":
    verify_sitcom_mode()
