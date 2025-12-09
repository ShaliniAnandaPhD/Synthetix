import modal
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.modal_orchestrator import app, run_debate

@app.local_entrypoint()
def main():
    print("Invoking run_debate.remote() for debugging...")
    
    payload = {
        "city1": "Philadelphia",
        "city2": "Dallas",
        "topic": "Who is the real king of the NFC East?",
        "rounds": 1
    }
    
    try:
        # Call the function remotely
        # Note: run_debate is a web endpoint, but since it's also a modal function, 
        # we might be able to call it if we access the underlying function.
        # However, @modal.web_endpoint wraps the function.
        # Let's try calling it. If it fails because it's a web endpoint, 
        # we might need to define a separate function for testing or use a different approach.
        
        # Actually, let's try to call the CulturalAgent directly to see if it works.
        from infra.modal_orchestrator import CulturalAgent
        
        print("Testing CulturalAgent.generate_response directly...")
        agent = CulturalAgent()
        response = agent.generate_response.remote(
            city_name="Philadelphia",
            user_input="Test message",
            conversation_history=[]
        )
        print("CulturalAgent response:", response)
        
        print("\nNow testing run_debate...")
        result = run_debate.remote(payload)
        print("run_debate result:", result)
        
    except Exception as e:
        print("\nERROR OCCURRED:")
        print(e)
