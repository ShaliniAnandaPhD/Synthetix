#!/usr/bin/env python3
"""
Test the Style Capture & Agent Creation Flow

This script simulates what a user would do:
1. Upload content samples (text)
2. Call analyze_style_samples endpoint
3. Get back a personality profile
4. Test generating commentary with that profile
"""

import requests
import json

# Modal endpoint for style analysis
STYLE_ANALYSIS_URL = "https://neuronsystems--neuron-orchestrator-analyze-style-samples.modal.run"

# Sample content from a fictional sportscaster "Big Tony"
SAMPLE_CONTENT_1 = """
Listen, I'm not here to make friends. I'm here to tell you the TRUTH about what I saw on that field. 
And let me tell you something - that quarterback? He's soft. I said it. SOFT. You can't win championships 
with a guy who folds like origami when the pressure's on. I've been watching this game for 30 years, 
and I can tell you right now, this kid ain't it. The stats? Sure, they look pretty. EPA this, CPOE that. 
But when it's fourth quarter, two minutes left, down by six? That's when you see what a man is made of. 
And this guy? He's made of tissue paper. End of story.
"""

SAMPLE_CONTENT_2 = """
Here's the thing nobody wants to talk about - DEFENSE wins championships. Not fancy quarterbacks. 
Not these analytics nerds with their laptops. DEFENSE. Smash-mouth, in-your-face, make-em-quit football. 
You think Tom Brady wins seven rings without a defense? Please. Get outta here with that nonsense. 
The problem with football today is everyone's too busy looking at their phones to watch the actual game. 
They see a completion percentage and think that means something. You know what means something? 
Can your offensive line move people? Can your D-line eat double teams? That's real football.
"""

SAMPLE_CONTENT_3 = """
And another thing - these coaches today don't have the guts they used to. Back in my day, you'd see 
Bill Parcells grab a guy by the facemask and tell him like it is. Now? Everyone's worried about 
feelings. "Player development." "Mental health." Listen, you know what develops a player? Getting hit 
in the mouth and learning to hit back harder. That's development. This league has gone soft, 
and it starts at the top. You want to know why the game was better in the 90s? Because men were MEN. 
None of this analytics garbage. Just pure, physical, dominating football.
"""

def test_style_capture():
    """Test the complete style capture flow"""
    
    print("=" * 80)
    print("TESTING STYLE CAPTURE & AGENT CREATION")
    print("=" * 80)
    
    # Step 1: Prepare samples
    samples = [
        {
            "id": "sample1",
            "type": "text",
            "source": "Pasted text 1",
            "content": SAMPLE_CONTENT_1,
            "status": "ready",
            "metadata": {
                "wordCount": len(SAMPLE_CONTENT_1.split()),
                "title": "Big Tony Take #1"
            }
        },
        {
            "id": "sample2",
            "type": "text",
            "source": "Pasted text 2",
            "content": SAMPLE_CONTENT_2,
            "status": "ready",
            "metadata": {
                "wordCount": len(SAMPLE_CONTENT_2.split()),
                "title": "Big Tony Take #2"
            }
        },
        {
            "id": "sample3",
            "type": "text",
            "source": "Pasted text 3",
            "content": SAMPLE_CONTENT_3,
            "status": "ready",
            "metadata": {
                "wordCount": len(SAMPLE_CONTENT_3.split()),
                "title": "Big Tony Take #3"
            }
        }
    ]
    
    total_words = sum(s["metadata"]["wordCount"] for s in samples)
    print(f"\n📝 Prepared {len(samples)} samples")
    print(f"   Total words: {total_words}")
    print(f"   Average per sample: {total_words // len(samples)}")
    
    # Step 2: Call analyze_style_samples
    print("\n🤖 Calling analyze_style_samples endpoint...")
    print(f"   Endpoint: {STYLE_ANALYSIS_URL}")
    
    payload = {
        "samples": samples,
        "personality_name": "Big Tony"
    }
    
    try:
        response = requests.post(
            STYLE_ANALYSIS_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # Style analysis can take a while
        )
        
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ Style analysis complete!")
        
        # Step 3: Display results
        if "analysis" in result:
            print("\n" + "=" * 80)
            print("STYLE ANALYSIS RESULTS")
            print("=" * 80)
            analysis = result["analysis"]
            
            if "vocabulary" in analysis:
                print("\n📖 Vocabulary:")
                vocab = analysis["vocabulary"]
                print(f"   Complexity: {vocab.get('complexity_level', 'N/A')}")
                print(f"   Jargon density: {vocab.get('sports_jargon_density', 0):.2f}")
                if "unique_phrases" in vocab:
                    print(f"   Signature phrases: {', '.join(vocab['unique_phrases'][:3])}")
            
            if "emotional_profile" in analysis:
                print("\n😠 Emotional Profile:")
                emo = analysis["emotional_profile"]
                print(f"   Energy level: {emo.get('baseline_energy', 0):.2f}")
                print(f"   Hot take tendency: {emo.get('hot_take_tendency', 0):.2f}")
                print(f"   Humor: {emo.get('humor_frequency', 0):.2f}")
            
            if "argumentation" in analysis:
                print("\n🎯 Argumentation Style:")
                arg = analysis["argumentation"]
                print(f"   Style: {arg.get('style', 'N/A')}")
                print(f"   Acknowledges counter-points: {arg.get('acknowledges_counterpoints', False)}")
            
            if "signature_elements" in analysis:
                print("\n🎤 Signature Elements:")
                sig = analysis["signature_elements"]
                if "catchphrases" in sig:
                    print(f"   Catchphrases: {', '.join(sig['catchphrases'][:3])}")
        
        # Step 4: Display personality config
        if "personality_config" in result:
            print("\n" + "=" * 80)
            print("GENERATED PERSONALITY CONFIG")
            print("=" * 80)
            config = result["personality_config"]
            
            print(f"\nCity Profile Name: {config.get('city_name', 'N/A')}")
            
            if "system_prompt_personality" in config:
                print(f"\nSystem Prompt Preview:")
                prompt = config["system_prompt_personality"]
                print(f"   {prompt[:200]}...")
            
            if "lexical_style" in config:
                lex = config["lexical_style"]
                print(f"\nLexical Style:")
                print(f"   Injection rate: {lex.get('injection_rate', 0):.2f}")
                if "phrases" in lex:
                    print(f"   Custom phrases: {', '.join(lex['phrases'][:5])}")
        
        # Step 5: Summary
        if "sample_stats" in result:
            stats = result["sample_stats"]
            print("\n" + "=" * 80)
            print("SAMPLE STATISTICS")
            print("=" * 80)
            print(f"   Total samples: {stats.get('total_samples', 0)}")
            print(f"   Total words: {stats.get('total_words', 0):,}")
            print(f"   Confidence score: {stats.get('confidence_score', 0):.2f}/1.0")
        
        # Save full result to file for inspection
        output_file = "tests/style_analysis_result.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full result saved to: {output_file}")
        
        print("\n" + "=" * 80)
        print("✅ STYLE CAPTURE TEST SUCCESSFUL!")
        print("=" * 80)
        
        return result
        
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out. Style analysis can take 60-120 seconds.")
        print("   Try running again or check Modal logs.")
        return None
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response: {e.response.text}")
        return None

if __name__ == "__main__":
    test_style_capture()
