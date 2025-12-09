# New run_debate endpoint code to be inserted before the main section

@app.function()
@modal.web_endpoint(method="POST")
def run_debate(request_data: dict):
    """
    Run a multi-turn debate between two city agents.
    
    Request body:
    {
        "city1": "Philadelphia",
        "city2": "Dallas",
        "topic": "Who has the better quarterback?",
        "rounds": 3
    }
    
    Returns:
    {
        "status": "success",
        "debate": {
            "city1": "Philadelphia",
            "city2": "Dallas",
            "topic": "Who has the better quarterback?",
            "rounds": 3,
            "transcript": [
                {
                    "round": 1,
                    "speaker": "Philadelphia",
                    "response": "...",
                    "timestamp": 1234567890
                },
                {
                    "round": 1,
                    "speaker": "Dallas",
                    "response": "...",
                    "timestamp": 1234567891
                },
                ...
            ]
        }
    }
    """
    import time
    
    city1 = request_data.get("city1")
    city2 = request_data.get("city2")
    topic = request_data.get("topic")
    rounds = request_data.get("rounds", 3)
    
    # Validation
    if not city1 or not city2:
        return {"error": "Missing 'city1' or 'city2' field"}
    if not topic:
        return {"error": "Missing 'topic' field"}
    if rounds < 1 or rounds > 10:
        return {"error": "Rounds must be between 1 and 10"}
    
    # Initialize agent
    agent = CulturalAgent()
    
    # Debate transcript
    transcript = []
    conversation_history = []
    
    # Run debate rounds
    for round_num in range(1, rounds + 1):
        # City 1's turn
        city1_result = agent.generate_response.remote(
            city_name=city1,
            user_input=topic,
            conversation_history=conversation_history,
            game_context={"opponent": city2, "debate_round": round_num}
        )
        
        city1_response = city1_result.get("response", "")
        
        # Add to transcript
        transcript.append({
            "round": round_num,
            "speaker": city1,
            "response": city1_response,
            "timestamp": int(time.time())
        })
        
        # Add to conversation history
        conversation_history.append({
            "role": city1,
            "content": city1_response
        })
        
        # City 2's turn
        city2_result = agent.generate_response.remote(
            city_name=city2,
            user_input=topic,
            conversation_history=conversation_history,
            game_context={"opponent": city1, "debate_round": round_num}
        )
        
        city2_response = city2_result.get("response", "")
        
        # Add to transcript
        transcript.append({
            "round": round_num,
            "speaker": city2,
            "response": city2_response,
            "timestamp": int(time.time())
        })
        
        # Add to conversation history
        conversation_history.append({
            "role": city2,
            "content": city2_response
        })
    
    # Return full debate
    return {
        "status": "success",
        "debate": {
            "city1": city1,
            "city2": city2,
            "topic": topic,
            "rounds": rounds,
            "transcript": transcript,
            "total_turns": len(transcript)
        }
    }
