"""
Minimal test script to debug Modal CulturalAgent initialization.
"""
import modal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app from orchestrator
from infra.modal_orchestrator import app, CulturalAgent

@app.local_entrypoint()
def main():
    print("=" * 60)
    print("TESTING CULTURLA AGENT INITIALIZATION")
    print("=" * 60)
    
    try:
        # Try to call a simple method on the CulturalAgent
        agent = CulturalAgent()
        
        print("\nAttempting to call get_city_profile.remote()...")
        result = agent.get_city_profile.remote("Philadelphia")
        print(f"\nSUCCESS! Result: {result}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Run with: modal run scripts/test_modal_init.py")
