import requests
import json
import time

# Modal endpoint URL - replace with your actual deployed URL
# You can find this in the Modal dashboard or deployment logs
MODAL_URL = "https://shalini--neuron-orchestrator-run-debate.modal.run"

def test_debate():
    print(f"Testing debate endpoint: {MODAL_URL}")
    
    payload = {
        "city1": "Philadelphia",
        "city2": "Dallas",
        "topic": "Who is the real king of the NFC East?",
        "rounds": 2
    }
    
    print(f"Starting debate: {payload['city1']} vs {payload['city2']}")
    print(f"Topic: {payload['topic']}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        response = requests.post(MODAL_URL, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                debate = data.get("debate", {})
                transcript = debate.get("transcript", [])
                
                print(f"\nDebate completed in {end_time - start_time:.2f} seconds")
                print(f"Total turns: {len(transcript)}")
                print("-" * 50)
                
                for turn in transcript:
                    speaker = turn.get("speaker")
                    text = turn.get("response")
                    print(f"\n[{speaker}]: {text}")
            else:
                print(f"Error in response: {data}")
        else:
            print(f"Request failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_debate()
