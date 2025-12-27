
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.agent_factory import AgentFactory

def test_sitcom_prompt():
    factory = AgentFactory()
    
    # Test for a few cities to see different archetypes
    cities = ["Philadelphia", "New England", "Kansas City"]
    
    for city in cities:
        print(f"\n--- TESTING SITCOM MODE FOR {city.upper()} ---\n")
        try:
            # Construct prompt in sitcom mode
            prompt = factory.construct_system_prompt(
                city_name=city,
                mode="debate",
                style="sitcom",
                debate_context={
                    "opponent_name": "Dallas",
                    "previous_response": "We have five rings!",
                    "turn_number": 2,
                    "conflict_mode": "aggressive"
                }
            )
            
            # Check for sitcom elements
            if "AUDIO DRAMA MODE: SITCOM STYLE" in prompt:
                print("✅ Sitcom header present")
            else:
                print("❌ Sitcom header MISSING")

            if "YOUR CHARACTER ROLE:" in prompt:
                print("✅ Character role assigned")
                # Extract role
                import re
                match = re.search(r"YOUR CHARACTER ROLE: (.*)", prompt)
                if match:
                    print(f"Role: {match.group(1)}")
            else:
                print("❌ Character role MISSING")
                
            if "PLAY YOUR ROLE:" in prompt:
                 print("✅ Role tactics present")
            else:
                 print("❌ Role tactics MISSING")
                 
            # print preview
            print("\nPreview of instructions (last 500 chars):")
            print(prompt[-500:])

        except Exception as e:
            print(f"❌ Error generating prompt for {city}: {e}")

if __name__ == "__main__":
    test_sitcom_prompt()
