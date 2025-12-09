"""
Vercel Serverless Function: Cultural Cognition API (P0 Enhanced)

Direct access to Modal's CulturalAgent with P0 memory & cognitive dimensions.
Frontend can call this to get memory-enhanced responses in real-time.

Architecture: React → /api/cultural → Modal generate_commentary → Response
"""

from flask import Flask, request, jsonify
import requests
import os
from typing import Dict, Any

app = Flask(__name__)


# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

def get_modal_cultural_url() -> str:
    """
    Get Modal's cultural cognition endpoint URL.
    
    Set in Vercel environment variables:
    - MODAL_CULTURAL_URL: https://your-modal-url.modal.run/generate_commentary
    
    Returns:
        Modal cultural cognition URL
    """
    url = os.environ.get("MODAL_CULTURAL_URL")
    
    if not url:
        # Fallback to constructing from base URL
        base_url = os.environ.get("MODAL_KAFKA_WEBHOOK_URL", "").replace("/publish_game_event", "")
        if base_url:
            url = f"{base_url}/generate_commentary"
        else:
            raise ValueError("MODAL_CULTURAL_URL or MODAL_KAFKA_WEBHOOK_URL not set")
    
    return url


# ============================================================================
# CULTURAL COGNITION API ENDPOINT (P0 ENHANCED)
# ============================================================================

@app.route('/api/cultural', methods=['POST', 'OPTIONS'])
def generate_cultural_response():
    """
    Generate culturally-aware response with P0 memory enhancements.
    
    This endpoint provides access to the enhanced CulturalAgent:
    - Tiered memory (episodic, semantic, procedural)
    - Cognitive dimensions → emergent biases
    - Context-aware memory retrieval
    
    Request (JSON):
    {
        "city": "Philadelphia",
        "user_input": "I think Dallas is going all the way this year",
        "conversation_history": [
            {"role": "user", "content": "Who's better, Dak or Hurts?"},
            {"role": "assistant", "content": "Hurts all day."}
        ],
        "game_context": {
            "opponent": "Cowboys",
            "score": "Eagles 21 - Cowboys 14"
        }
    }
    
    Response (JSON):
    {
        "status": "success",
        "data": {
            "response": "Dallas? Going all the way? [Memory: Super Bowl LII] We beat TOM BRADY...",
            "city": "Philadelphia",
            "delay_ms": 140,
            "confidence": 0.85,
            "cached": true,
            "latency_ms": 287,
            "memory_invoked": ["Super Bowl LII", "Cowboys archetype: perennial_chokers"]
        }
    }
    """
    # ----------------------------------------------------------------
    # CORS PREFLIGHT
    # ----------------------------------------------------------------
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,GET')
        return response
    
    # ----------------------------------------------------------------
    # VALIDATE REQUEST
    # ----------------------------------------------------------------
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Validate required fields
        city = data.get('city')
        user_input = data.get('user_input')
        
        if not city:
            return jsonify({
                "status": "error",
                "message": "Missing required field: 'city'"
            }), 400
        
        if not user_input:
            return jsonify({
                "status": "error",
                "message": "Missing required field: 'user_input'"
            }), 400
        
        # Extract optional fields
        conversation_history = data.get('conversation_history', [])
        game_context = data.get('game_context', {})
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid JSON: {str(e)}"
        }), 400
    
    # ----------------------------------------------------------------
    # CALL MODAL CULTURAL COGNITION ENDPOINT
    # ----------------------------------------------------------------
    try:
        modal_url = get_modal_cultural_url()
        
        # Prepare payload for Modal
        payload = {
            "city": city,
            "user_input": user_input,
            "conversation_history": conversation_history,
            "game_context": game_context
        }
        
        # Call Modal endpoint
        response = requests.post(
            modal_url,
            json=payload,
            timeout=15  # 15 second timeout (inference can be slow)
        )
        
        response.raise_for_status()
        result = response.json()
        
        # ----------------------------------------------------------------
        # RETURN SUCCESS RESPONSE
        # ----------------------------------------------------------------
        response_data = jsonify({
            "status": "success",
            "data": result.get('data', result)
        })
        
        # Enable CORS
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        
        return response_data, 200
    
    except requests.exceptions.Timeout:
        response_data = jsonify({
            "status": "error",
            "message": "Modal inference timeout (>15s). Vertex AI may be slow."
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
    
    except requests.exceptions.RequestException as e:
        response_data = jsonify({
            "status": "error",
            "message": f"Failed to call Modal: {str(e)}"
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 502
    
    except Exception as e:
        response_data = jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# MULTI-CITY DEBATE ENDPOINT
# ============================================================================

@app.route('/api/cultural/debate', methods=['POST', 'OPTIONS'])
def generate_multi_city_debate():
    """
    Generate responses from multiple cities for panel discussions.
    
    Request (JSON):
    {
        "cities": ["Philadelphia", "Dallas", "New York"],
        "user_input": "Who's winning the NFC East?",
        "game_context": {}
    }
    
    Response (JSON):
    {
        "status": "success",
        "responses": {
            "Philadelphia": { "response": "...", "delay_ms": 140 },
            "Dallas": { "response": "...", "delay_ms": 200 },
            "New York": { "response": "...", "delay_ms": 180 }
        }
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.get_json()
        cities = data.get('cities', [])
        user_input = data.get('user_input')
        game_context = data.get('game_context', {})
        
        if not cities or not user_input:
            return jsonify({
                "status": "error",
                "message": "Missing 'cities' or 'user_input'"
            }), 400
        
        # Call Modal multi-city endpoint
        modal_base = os.environ.get("MODAL_KAFKA_WEBHOOK_URL", "").replace("/publish_game_event", "")
        modal_url = f"{modal_base}/generate_multi_city_commentary"
        
        response = requests.post(
            modal_url,
            json={
                "cities": cities,
                "user_input": user_input,
                "game_context": game_context
            },
            timeout=20  # Multiple cities take longer
        )
        
        response.raise_for_status()
        result = response.json()
        
        response_data = jsonify({
            "status": "success",
            "responses": result.get('responses', {})
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        
        return response_data, 200
        
    except Exception as e:
        response_data = jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# MAIN (FOR LOCAL TESTING)
# ============================================================================

if __name__ == '__main__':
    """
    Local development server.
    
    Usage:
        python api/cultural.py
    
    Test:
        curl -X POST http://localhost:5001/api/cultural \
             -H "Content-Type: application/json" \
             -d '{
               "city": "Philadelphia",
               "user_input": "Cowboys are winning it all",
               "game_context": {"opponent": "Cowboys"}
             }'
    """
    if not os.environ.get("MODAL_KAFKA_WEBHOOK_URL"):
        os.environ["MODAL_KAFKA_WEBHOOK_URL"] = "https://placeholder.modal.run"
        print("[WARNING] Using placeholder webhook URL")
    
    print("=" * 70)
    print("CULTURAL COGNITION API - LOCAL DEVELOPMENT")
    print("=" * 70)
    print("\nEndpoints:")
    print("  POST /api/cultural        - Single city response (P0 enhanced)")
    print("  POST /api/cultural/debate - Multi-city panel")
    print("\nListening on http://localhost:5001")
    print("=" * 70)
    
    app.run(debug=True, port=5001)
