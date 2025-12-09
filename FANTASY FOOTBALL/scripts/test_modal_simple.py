"""
Simple test to just invoke the Modal agent and see the error.
"""
import modal
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.modal_orchestrator import app, CulturalAgent

@app.local_entrypoint()
def main():
    print("\n" + "="*60)
    print("MODAL INIT TEST")
    print("="*60 + "\n")
    
    agent = CulturalAgent()
    
    print("Calling get_city_profile.remote('Philadelphia')...")
    try:
        result = agent.get_city_profile.remote("Philadelphia")
        print(f"\n✅ SUCCESS! Got profile with keys: {list(result.keys())[:5]}...")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
