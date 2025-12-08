"""
Vercel Serverless Function: Event API

This module provides the HTTP bridge between the frontend and backend:
- Frontend (React) → Vercel API → Modal Kafka Producer → Confluent Kafka
- Acts as a secure gateway with environment variable protection
- Lightweight and optimized for Vercel's serverless environment

Architecture: React App → /api/events → Modal Webhook → Kafka → Agents
"""

from flask import Flask, request, jsonify
import requests
import os
from typing import Dict, Any

app = Flask(__name__)


# ============================================================================
# MODAL WEBHOOK CONFIGURATION
# ============================================================================

def get_modal_webhook_url() -> str:
    """
    Get the Modal webhook URL from environment variables.
    
    Set in Vercel environment variables:
    - MODAL_KAFKA_WEBHOOK_URL: https://your-modal-url.modal.run/publish_game_event
    
    Returns:
        Modal webhook URL
    """
    webhook_url = os.environ.get("MODAL_KAFKA_WEBHOOK_URL")
    
    if not webhook_url:
        raise ValueError(
            "MODAL_KAFKA_WEBHOOK_URL not set in environment. "
            "Please configure this in Vercel dashboard."
        )
    
    return webhook_url


# ============================================================================
# EVENT API ENDPOINT
# ============================================================================

@app.route('/api/events', methods=['POST', 'OPTIONS'])
def publish_event():
    """
    Publish game event to Kafka via Modal webhook.
    
    This endpoint:
    1. Receives event from React frontend
    2. Validates the payload
    3. Forwards to Modal Kafka producer
    4. Returns response to frontend
    
    Request (JSON):
    {
        "city": "Philadelphia",
        "event_type": "touchdown",
        "user_input": "React to that touchdown!",
        "game_context": {
            "score": "Eagles 21 - Cowboys 14",
            "quarter": 3
        }
    }
    
    Response (JSON):
    {
        "status": "success",
        "message": "Event published successfully",
        "timestamp": 1733189234.567
    }
    """
    # ----------------------------------------------------------------
    # CORS PREFLIGHT
    # ----------------------------------------------------------------
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # ----------------------------------------------------------------
    # VALIDATE REQUEST
    # ----------------------------------------------------------------
    try:
        event_data = request.get_json()
        
        if not event_data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Validate required fields
        if 'city' not in event_data:
            return jsonify({
                "status": "error",
                "message": "Missing required field: 'city'"
            }), 400
        
        # Auto-populate user_input if missing
        if 'user_input' not in event_data:
            event_type = event_data.get('event_type', 'event')
            event_data['user_input'] = f"Simulate {event_type}"
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid JSON: {str(e)}"
        }), 400
    
    # ----------------------------------------------------------------
    # FORWARD TO MODAL KAFKA PRODUCER
    # ----------------------------------------------------------------
    try:
        modal_webhook_url = get_modal_webhook_url()
        
        # Call Modal webhook
        response = requests.post(
            modal_webhook_url,
            json=event_data,
            timeout=10  # 10 second timeout
        )
        
        response.raise_for_status()  # Raise exception for 4xx/5xx
        
        result = response.json()
        
        # ----------------------------------------------------------------
        # RETURN SUCCESS RESPONSE
        # ----------------------------------------------------------------
        response_data = jsonify({
            "status": "success",
            "message": "Event published to Kafka",
            "data": result
        })
        
        # Enable CORS
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        
        return response_data, 200
    
    except requests.exceptions.Timeout:
        response_data = jsonify({
            "status": "error",
            "message": "Modal webhook timeout (>10s)"
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
    
    except requests.exceptions.RequestException as e:
        response_data = jsonify({
            "status": "error",
            "message": f"Failed to forward to Modal: {str(e)}"
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
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        Simple status response
    """
    response_data = jsonify({
        "status": "healthy",
        "service": "neuron-events-api",
        "version": "1.0"
    })
    response_data.headers.add('Access-Control-Allow-Origin', '*')
    return response_data


# ============================================================================
# DEBATE API ENDPOINTS
# ============================================================================

def get_modal_orchestrator_url() -> str:
    """Get the Modal orchestrator URL for debate endpoints."""
    return os.environ.get(
        "MODAL_ORCHESTRATOR_URL",
        "https://shaliniananada--neuron-orchestrator.modal.run"
    )


@app.route('/api/debate/run', methods=['POST', 'OPTIONS'])
def run_debate():
    """
    Run a multi-agent debate with optional SSE streaming.
    
    Proxies to Modal `run_debate_stream` endpoint.
    
    Request body:
    {
        "topic": str,
        "panel": list[str],           # personality IDs
        "config": {
            "rounds": int,
            "tone": {
                "analytical": float,
                "depth": float,
                "energy": float
            },
            "closed_loop": bool
        }
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        
        if not data:
            response_data = jsonify({"error": "No request data provided"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Validate required fields
        topic = data.get("topic", "")
        panel = data.get("panel", [])
        
        if not topic:
            response_data = jsonify({"error": "Topic is required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        if len(panel) < 2:
            response_data = jsonify({"error": "At least 2 agents required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Forward to Modal
        modal_url = get_modal_orchestrator_url()
        
        modal_response = requests.post(
            f"{modal_url}/run_debate_stream",
            json=data,
            timeout=120,  # 2 minutes for long debates
            stream=True  # Enable streaming
        )
        
        if modal_response.status_code != 200:
            error_text = modal_response.text[:500]  # Truncate long errors
            response_data = jsonify({
                "error": f"Modal returned {modal_response.status_code}",
                "details": error_text
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 502
        
        # Check if Modal returned SSE stream or JSON
        content_type = modal_response.headers.get('content-type', '')
        
        if 'text/event-stream' in content_type:
            # Stream SSE events directly to client
            from flask import Response
            
            def generate_sse():
                for line in modal_response.iter_lines():
                    if line:
                        yield line.decode('utf-8') + '\n'
            
            return Response(
                generate_sse(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        else:
            # Return JSON response
            result = modal_response.json()
            response_data = jsonify(result)
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "Debate generation timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except requests.exceptions.RequestException as e:
        print(f"[DEBATE] Modal request failed: {e}")
        response_data = jsonify({"error": f"Modal unreachable: {str(e)}"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 502
        
    except Exception as e:
        print(f"[DEBATE] Unexpected error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


@app.route('/api/debate/regenerate-segment', methods=['POST', 'OPTIONS'])
def regenerate_segment():
    """
    Regenerate a single debate segment.
    
    Proxies to Modal `regenerate_segment` endpoint.
    Used by EditBay when creators want a different take.
    
    Request body:
    {
        "debate_id": str,
        "segment_index": int,
        "speaker_id": str,
        "topic": str,
        "conversation_history": [{ "speaker": str, "text": str }, ...],
        "responding_to": { "speaker": str, "text": str } | null,
        "tone": { "analytical": float, "depth": float, "energy": float }
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        
        if not data:
            response_data = jsonify({"error": "No request data provided"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Validate required fields
        speaker_id = data.get("speaker_id", "")
        topic = data.get("topic", "")
        
        if not speaker_id:
            response_data = jsonify({"error": "speaker_id is required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Forward to Modal
        modal_url = get_modal_orchestrator_url()
        
        modal_response = requests.post(
            f"{modal_url}/regenerate_segment",
            json=data,
            timeout=60
        )
        
        if modal_response.status_code != 200:
            error_text = modal_response.text[:500]
            response_data = jsonify({
                "error": f"Modal returned {modal_response.status_code}",
                "details": error_text
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 502
        
        result = modal_response.json()
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "Segment regeneration timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except requests.exceptions.RequestException as e:
        print(f"[REGENERATE] Modal request failed: {e}")
        response_data = jsonify({"error": f"Modal unreachable: {str(e)}"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 502
        
    except Exception as e:
        print(f"[REGENERATE] Unexpected error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# AUDIO TRANSCRIPTION ENDPOINT
# ============================================================================

@app.route('/api/content/transcribe-audio', methods=['POST', 'OPTIONS'])
def transcribe_audio_route():
    """
    Forward audio transcription request to Modal Whisper endpoint.
    
    Accepts FormData with 'file' and 'sampleId'.
    For now, we expect the frontend to upload to Supabase first and send the URL.
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        # Get the audio URL from request
        data = request.get_json() if request.is_json else {}
        audio_url = data.get('audioUrl')
        sample_id = data.get('sampleId', 'unknown')
        
        if not audio_url:
            response_data = jsonify({
                "error": "No audio URL provided"
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Get Modal webhook URL from environment
        modal_url = os.environ.get(
            "MODAL_TRANSCRIBE_WEBHOOK_URL",
            "https://your-modal-app--transcribe-audio.modal.run"
        )
        
        # Forward to Modal
        response = requests.post(
            modal_url,
            json={
                "audio_url": audio_url,
                "sample_id": sample_id
            },
            timeout=600  # Long timeout for transcription
        )
        
        result = response.json()
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, response.status_code
        
    except requests.Timeout:
        response_data = jsonify({
            "error": "Transcription timed out. Try a shorter audio clip."
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
    except Exception as e:
        response_data = jsonify({
            "error": f"Transcription failed: {str(e)}"
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# TWEET EXTRACTION ENDPOINT
# ============================================================================

@app.route('/api/content/extract-tweets', methods=['POST', 'OPTIONS'])
def extract_tweets_route():
    """
    Forward tweet extraction request to Modal endpoint.
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip().lstrip('@')
        count = data.get('count', 50)
        
        if not username:
            response_data = jsonify({
                "error": "No username provided"
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Get Modal webhook URL from environment
        modal_url = os.environ.get(
            "MODAL_TWEETS_WEBHOOK_URL",
            "https://your-modal-app--extract-tweets.modal.run"
        )
        
        # Forward to Modal
        response = requests.post(
            modal_url,
            json={
                "username": username,
                "count": count,
                "include_replies": data.get('includeReplies', False)
            },
            timeout=120
        )
        
        result = response.json()
        
        # Check for errors from Modal
        if "error" in result:
            response_data = jsonify(result)
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.Timeout:
        response_data = jsonify({
            "error": "Tweet extraction timed out. Try again."
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
    except Exception as e:
        response_data = jsonify({
            "error": f"Tweet extraction failed: {str(e)}"
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# TTS GENERATION ENDPOINT
# ============================================================================

@app.route('/api/tts/generate', methods=['POST', 'OPTIONS'])
def generate_tts():
    """
    Generate text-to-speech audio via Modal.
    
    Used by MasteringDesk for debate audio generation.
    
    Request body:
    {
        "text": str,
        "speaker_id": str,
        "voice_id": str (optional),
        "intensity": str (optional)
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        
        if not data or not data.get("text"):
            response_data = jsonify({"error": "Text is required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        modal_url = get_modal_orchestrator_url()
        
        modal_response = requests.post(
            f"{modal_url}/generate_tts",
            json=data,
            timeout=60
        )
        
        if modal_response.status_code != 200:
            response_data = jsonify({
                "error": f"TTS generation failed: {modal_response.status_code}",
                "details": modal_response.text[:500]
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 502
        
        result = modal_response.json()
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "TTS generation timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except Exception as e:
        print(f"[TTS] Error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# YOUTUBE EXTRACTION ENDPOINT
# ============================================================================

@app.route('/api/content/extract-youtube', methods=['POST', 'OPTIONS'])
def extract_youtube():
    """
    Extract transcript from YouTube video.
    
    Used by ContentSampleUpload for style capture.
    
    Request body:
    {
        "url": str (YouTube URL)
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        url = data.get("url", "")
        
        if not url:
            response_data = jsonify({"error": "YouTube URL is required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        modal_url = get_modal_orchestrator_url()
        
        modal_response = requests.post(
            f"{modal_url}/extract_youtube",
            json={"url": url},
            timeout=60
        )
        
        if modal_response.status_code != 200:
            response_data = jsonify({
                "error": f"YouTube extraction failed: {modal_response.status_code}",
                "details": modal_response.text[:500]
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 502
        
        result = modal_response.json()
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "YouTube extraction timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except Exception as e:
        print(f"[YOUTUBE] Error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# ARTICLE EXTRACTION ENDPOINT
# ============================================================================

@app.route('/api/content/extract-article', methods=['POST', 'OPTIONS'])
def extract_article():
    """
    Extract text content from article URL.
    
    Used by ContentSampleUpload for style capture.
    
    Request body:
    {
        "url": str (Article URL)
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        url = data.get("url", "")
        
        if not url:
            response_data = jsonify({"error": "Article URL is required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        modal_url = get_modal_orchestrator_url()
        
        modal_response = requests.post(
            f"{modal_url}/extract_article",
            json={"url": url},
            timeout=30
        )
        
        if modal_response.status_code != 200:
            response_data = jsonify({
                "error": f"Article extraction failed: {modal_response.status_code}",
                "details": modal_response.text[:500]
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 502
        
        result = modal_response.json()
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "Article extraction timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except Exception as e:
        print(f"[ARTICLE] Error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# STYLE ANALYSIS ENDPOINT
# ============================================================================

@app.route('/api/style/analyze', methods=['POST', 'OPTIONS'])
def analyze_style():
    """
    Analyze writing/speaking style from content samples.
    
    Used by PersonalityWizard for AI-powered style capture.
    
    Request body:
    {
        "samples": [{ "type": str, "content": str, "source": str }, ...],
        "personality_name": str
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        samples = data.get("samples", [])
        
        if not samples:
            response_data = jsonify({"error": "At least one content sample is required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        modal_url = get_modal_orchestrator_url()
        
        modal_response = requests.post(
            f"{modal_url}/analyze_style_samples",
            json=data,
            timeout=90  # Analysis can take a while
        )
        
        if modal_response.status_code != 200:
            response_data = jsonify({
                "error": f"Style analysis failed: {modal_response.status_code}",
                "details": modal_response.text[:500]
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 502
        
        result = modal_response.json()
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "Style analysis timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except Exception as e:
        print(f"[STYLE] Error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# ESPN FANTASY FOOTBALL DATA ENDPOINT
# ============================================================================

def get_espn_credentials():
    """
    Get ESPN Fantasy Football credentials from environment variables.
    
    Required environment variables:
    - ESPN_S2: ESPN session cookie (espn_s2)
    - ESPN_SWID: ESPN user ID cookie (swid)
    - ESPN_LEAGUE_ID: Your ESPN Fantasy Football league ID
    """
    return {
        "espn_s2": os.environ.get("ESPN_S2", ""),
        "swid": os.environ.get("ESPN_SWID", ""),
        "league_id": os.environ.get("ESPN_LEAGUE_ID", "")
    }


@app.route('/api/espn/players', methods=['GET', 'OPTIONS'])
def get_espn_players():
    """
    Fetch live player data from ESPN Fantasy Football API.
    
    Returns player projections, stats, and injury status.
    
    Query parameters:
    - week: NFL week number (optional, defaults to current)
    - scoring_period: ESPN scoring period ID (optional)
    
    Response:
    {
        "players": [...],
        "week": int,
        "season": int,
        "source": "espn"
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
    
    try:
        creds = get_espn_credentials()
        
        if not creds["league_id"]:
            response_data = jsonify({
                "error": "ESPN League ID not configured",
                "hint": "Set ESPN_LEAGUE_ID environment variable in Vercel"
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Get current season and week
        from datetime import datetime
        current_date = datetime.now()
        season = current_date.year
        
        # Calculate approximate NFL week (rough estimate)
        week = request.args.get('week', None)
        if not week:
            # NFL season typically starts first week of September
            season_start = datetime(season, 9, 5)
            if current_date < season_start:
                week = 1
            else:
                week = min(18, max(1, (current_date - season_start).days // 7 + 1))
        else:
            week = int(week)
        
        # ESPN Fantasy API endpoints
        base_url = f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
        
        # Build cookies for authentication (private leagues)
        cookies = {}
        if creds["espn_s2"] and creds["swid"]:
            cookies = {
                "espn_s2": creds["espn_s2"],
                "SWID": creds["swid"]
            }
        
        # Fetch players with projections
        # Using the free agent view to get all players
        players_url = f"{base_url}/segments/0/leagues/{creds['league_id']}"
        
        params = {
            "view": "kona_player_info",
            "scoringPeriodId": week
        }
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        print(f"[ESPN] Fetching players for week {week}, season {season}")
        
        espn_response = requests.get(
            players_url,
            params=params,
            cookies=cookies if cookies else None,
            headers=headers,
            timeout=30
        )
        
        if espn_response.status_code != 200:
            print(f"[ESPN] API returned {espn_response.status_code}")
            # Fall back to public player data
            fallback_url = f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/players"
            fallback_params = {"scoringPeriodId": week, "view": "players_wl"}
            
            fallback_response = requests.get(
                fallback_url,
                params=fallback_params,
                headers=headers,
                timeout=30
            )
            
            if fallback_response.status_code != 200:
                response_data = jsonify({
                    "error": f"ESPN API request failed: {espn_response.status_code}",
                    "details": "Check your ESPN credentials or league settings"
                })
                response_data.headers.add('Access-Control-Allow-Origin', '*')
                return response_data, 502
            
            raw_data = fallback_response.json()
        else:
            raw_data = espn_response.json()
        
        # Parse player data from ESPN response
        players = []
        
        # ESPN positions mapping
        position_map = {
            1: "QB", 2: "RB", 3: "WR", 4: "TE", 
            5: "K", 16: "DST", 17: "DST"
        }
        
        # Injury status mapping
        injury_map = {
            "ACTIVE": "healthy",
            "QUESTIONABLE": "questionable", 
            "DOUBTFUL": "doubtful",
            "OUT": "out",
            "INJURED_RESERVE": "IR",
            "SUSPENSION": "suspended"
        }
        
        # Process players from response
        player_list = raw_data.get("players", [])
        if not player_list and isinstance(raw_data, list):
            player_list = raw_data
        
        for entry in player_list[:200]:  # Limit to top 200 players
            try:
                player = entry.get("player", entry)
                if not player:
                    continue
                
                player_id = str(player.get("id", ""))
                full_name = player.get("fullName", "Unknown")
                
                # Get position
                default_pos_id = player.get("defaultPositionId", 0)
                position = position_map.get(default_pos_id, "FLEX")
                
                # Get team
                team_abbrev = player.get("proTeamId", 0)
                team_map = {
                    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE",
                    6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
                    11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA",
                    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
                    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF",
                    26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
                    33: "BAL", 34: "HOU"
                }
                team = team_map.get(team_abbrev, "FA")
                
                # Get injury status
                injury_status_raw = player.get("injuryStatus", "ACTIVE")
                injury_status = injury_map.get(injury_status_raw, "healthy")
                
                # Get stats/projections
                stats = player.get("stats", [])
                projection = 0.0
                
                for stat in stats:
                    # statSourceId 1 = projected, 0 = actual
                    if stat.get("statSourceId") == 1 and stat.get("scoringPeriodId") == week:
                        projection = stat.get("appliedTotal", 0.0)
                        break
                
                # Ownership data
                ownership = player.get("ownership", {})
                ownership_pct = ownership.get("percentOwned", 50.0)
                
                players.append({
                    "id": player_id,
                    "name": full_name,
                    "position": position,
                    "team": team,
                    "salary": int(6000 + projection * 100),  # Estimate DFS salary
                    "projection": round(projection, 1),
                    "ownership_projection": round(ownership_pct, 1),
                    "injury_status": injury_status,
                    "opponent": "",  # Would need schedule data
                    "game_time": "",  # Would need schedule data
                    "is_active": injury_status not in ["out", "IR"]
                })
                
            except Exception as parse_error:
                print(f"[ESPN] Error parsing player: {parse_error}")
                continue
        
        # Sort by projection
        players.sort(key=lambda x: x["projection"], reverse=True)
        
        response_data = jsonify({
            "players": players,
            "week": week,
            "season": season,
            "source": "espn",
            "count": len(players)
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "ESPN API request timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except Exception as e:
        print(f"[ESPN] Error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# CULTURAL COMMENTARY ENDPOINT
# ============================================================================

@app.route('/api/cultural', methods=['POST', 'OPTIONS'])
def cultural_commentary():
    """
    Generate cultural commentary for a city/event.
    
    Used by CulturalCommentary component.
    
    Request body:
    {
        "city": str,
        "event_type": str,
        "context": dict (optional)
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        city = data.get("city", "")
        event_type = data.get("event_type", "")
        
        if not city:
            response_data = jsonify({"error": "City is required"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        modal_url = get_modal_orchestrator_url()
        
        modal_response = requests.post(
            f"{modal_url}/generate_cultural_commentary",
            json=data,
            timeout=30
        )
        
        if modal_response.status_code != 200:
            response_data = jsonify({
                "error": f"Cultural commentary failed: {modal_response.status_code}",
                "details": modal_response.text[:500]
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 502
        
        result = modal_response.json()
        response_data = jsonify(result)
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except requests.exceptions.Timeout:
        response_data = jsonify({"error": "Cultural commentary timed out"})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 504
        
    except Exception as e:
        print(f"[CULTURAL] Error: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# ANALYTICS AGGREGATION ENDPOINT
# ============================================================================

@app.route('/api/analytics/aggregate', methods=['POST', 'OPTIONS'])
def aggregate_analytics():
    """
    Aggregate analytics events into creator_stats.
    Should be called by a cron job every few hours.
    
    This endpoint:
    1. Gets all users with recent activity
    2. Calculates aggregated stats for each user
    3. Updates creator_stats table
    
    Note: In production, this would use proper Supabase service role key.
    Consider moving this to a Supabase Edge Function or Modal scheduled function.
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        # Verify admin/cron authorization
        auth_header = request.headers.get('Authorization', '')
        cron_secret = os.environ.get('CRON_SECRET', '')
        
        if not auth_header or not cron_secret:
            # For development, allow without auth
            pass
        elif auth_header != f'Bearer {cron_secret}':
            response_data = jsonify({
                "error": "Unauthorized"
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 401
        
        # Get Supabase service URL and key
        supabase_url = os.environ.get('SUPABASE_URL', '')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
        
        if not supabase_url or not supabase_key:
            response_data = jsonify({
                "error": "Supabase configuration missing",
                "message": "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in environment"
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 500
        
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        
        # Get distinct users with recent activity (last 24 hours)
        users_response = requests.get(
            f"{supabase_url}/rest/v1/analytics_events?select=user_id&created_at=gte.{get_24h_ago()}",
            headers=headers,
            timeout=30
        )
        
        if users_response.status_code != 200:
            response_data = jsonify({
                "error": "Failed to fetch users",
                "details": users_response.text
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 500
        
        user_ids = list(set([row['user_id'] for row in users_response.json() if row.get('user_id')]))
        users_updated = 0
        
        for user_id in user_ids:
            try:
                # Fetch all events for this user
                events_response = requests.get(
                    f"{supabase_url}/rest/v1/analytics_events?user_id=eq.{user_id}&select=event_type,event_data",
                    headers=headers,
                    timeout=30
                )
                
                if events_response.status_code != 200:
                    continue
                
                events = events_response.json()
                
                # Calculate aggregates
                debates_started = sum(1 for e in events if e['event_type'] == 'debate_started')
                debates_completed = sum(1 for e in events if e['event_type'] == 'debate_completed')
                total_exports = sum(1 for e in events if 'exported' in e['event_type'])
                total_regenerations = sum(1 for e in events if e['event_type'] == 'segment_regenerated')
                
                completion_rate = debates_completed / debates_started if debates_started > 0 else 0
                
                # Calculate avg segments per debate
                completed_events = [e for e in events if e['event_type'] == 'debate_completed']
                total_segments = sum(
                    e.get('event_data', {}).get('segment_count', 0) 
                    for e in completed_events
                )
                avg_segments = total_segments / len(completed_events) if completed_events else 0
                
                # Update creator_stats using upsert
                stats_data = {
                    "user_id": user_id,
                    "total_debates": debates_started,
                    "total_exports": total_exports,
                    "total_regenerations": total_regenerations,
                    "completion_rate": completion_rate,
                    "avg_segments_per_debate": avg_segments,
                    "stats_updated_at": get_current_timestamp()
                }
                
                upsert_response = requests.post(
                    f"{supabase_url}/rest/v1/creator_stats",
                    headers={**headers, "Prefer": "resolution=merge-duplicates"},
                    json=stats_data,
                    timeout=30
                )
                
                if upsert_response.status_code in [200, 201]:
                    users_updated += 1
                    
            except Exception as e:
                print(f"[ANALYTICS] Error processing user {user_id}: {e}")
                continue
        
        response_data = jsonify({
            "status": "ok",
            "users_updated": users_updated,
            "total_users": len(user_ids)
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except Exception as e:
        response_data = jsonify({
            "error": f"Aggregation failed: {str(e)}"
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


# ============================================================================
# FINE-TUNING API ENDPOINTS
# ============================================================================

def get_lora_endpoint_url() -> str:
    """Get the Modal LoRA training endpoint URL."""
    return os.environ.get(
        "MODAL_LORA_ENDPOINT_URL", 
        "https://shaliniananada--sportscaster-lora.modal.run"
    )


@app.route('/api/fine-tuning/status', methods=['POST', 'OPTIONS'])
def fine_tuning_status():
    """
    Get fine-tuning status and list trained adapters.
    
    Request:
        { "action": "list_models" }
    
    Response:
        { "tuned_models": [...], "status": "ok" }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        lora_url = get_lora_endpoint_url()
        
        # Call Modal to list adapters
        modal_response = requests.post(
            f"{lora_url}/list_adapters_api",
            json={},
            timeout=30
        )
        
        if modal_response.status_code == 200:
            data = modal_response.json()
            adapters = data.get("adapters", [])
            
            # Format as tuned models
            tuned_models = [
                {
                    "name": adapter,
                    "resource_name": f"lora/{adapter}",
                    "created": None
                }
                for adapter in adapters
            ]
            
            response_data = jsonify({
                "status": "ok",
                "tuned_models": tuned_models
            })
        else:
            # Modal not reachable - return empty list
            response_data = jsonify({
                "status": "ok",
                "tuned_models": [],
                "note": "LoRA endpoint not configured or unreachable"
            })
        
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except Exception as e:
        print(f"[FINE-TUNING] Status check failed: {e}")
        response_data = jsonify({
            "status": "ok",
            "tuned_models": [],
            "error": str(e)
        })
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data


@app.route('/api/fine-tuning/lora', methods=['POST', 'OPTIONS'])
def fine_tuning_lora():
    """
    Start LoRA fine-tuning job on Modal (A100 GPU).
    
    Request:
        {
            "personality_id": str,
            "personality_name": str,
            "training_data": str (JSONL format)
        }
    
    Response:
        { "status": "ok", "job_name": str }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        personality_id = data.get("personality_id")
        personality_name = data.get("personality_name", "custom")
        training_data_jsonl = data.get("training_data", "")
        
        if not training_data_jsonl:
            response_data = jsonify({"error": "No training data provided"})
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Parse JSONL to list
        import json
        training_examples = []
        for line in training_data_jsonl.strip().split('\n'):
            if line.strip():
                try:
                    training_examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if len(training_examples) < 50:
            response_data = jsonify({
                "error": f"Insufficient training data: {len(training_examples)} examples (minimum: 50)"
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 400
        
        # Create adapter name
        safe_name = personality_name.lower().replace(" ", "_").replace("-", "_")[:20]
        adapter_name = f"{safe_name}_{personality_id[:8]}"
        
        lora_url = get_lora_endpoint_url()
        
        # Submit to Modal LoRA training
        modal_response = requests.post(
            f"{lora_url}/train_adapter_api",
            json={
                "training_data": training_examples,
                "adapter_name": adapter_name,
                "num_epochs": 3
            },
            timeout=120  # Training can take a while to start
        )
        
        if modal_response.status_code == 200:
            result = modal_response.json()
            response_data = jsonify({
                "status": "ok",
                "job_name": adapter_name,
                "message": f"LoRA training started for {personality_name}",
                "examples_count": len(training_examples),
                "modal_response": result
            })
        else:
            response_data = jsonify({
                "error": f"Modal training failed: {modal_response.text}"
            })
            response_data.headers.add('Access-Control-Allow-Origin', '*')
            return response_data, 500
        
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except Exception as e:
        print(f"[FINE-TUNING] LoRA training failed: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


@app.route('/api/fine-tuning/vertex', methods=['POST', 'OPTIONS'])
def fine_tuning_vertex():
    """
    Start Vertex AI fine-tuning job (Gemini 1.5 Pro).
    
    Note: This is a placeholder - Vertex AI fine-tuning requires
    the training data to be uploaded to GCS first.
    
    Request:
        {
            "personality_id": str,
            "personality_name": str,
            "training_data": str (JSONL format)
        }
    """
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        personality_name = data.get("personality_name", "custom")
        
        # Vertex AI fine-tuning requires more setup:
        # 1. Upload training data to GCS
        # 2. Call Vertex AI Tuning API
        # 3. Monitor long-running operation
        
        # For now, return a placeholder response
        response_data = jsonify({
            "status": "pending",
            "job_name": f"vertex_{personality_name.lower().replace(' ', '_')}",
            "message": "Vertex AI fine-tuning is not yet configured. Use LoRA training instead.",
            "note": "To enable Vertex AI tuning, configure GCS bucket and Vertex AI API access."
        })
        
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
        
    except Exception as e:
        print(f"[FINE-TUNING] Vertex AI failed: {e}")
        response_data = jsonify({"error": str(e)})
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data, 500


def get_24h_ago():
    """Get ISO timestamp for 24 hours ago."""
    from datetime import datetime, timedelta
    return (datetime.utcnow() - timedelta(hours=24)).isoformat() + 'Z'


def get_current_timestamp():
    """Get current ISO timestamp."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + 'Z'


# ============================================================================
# MAIN (FOR LOCAL TESTING)
# ============================================================================

if __name__ == '__main__':
    """
    Local development server.
    
    Usage:
        python api/events.py
    
    Then test with:
        curl -X POST http://localhost:5000/api/events \
             -H "Content-Type: application/json" \
             -d '{"city": "Philadelphia", "event_type": "test"}'
    """
    # For local testing, set a dummy webhook URL
    if not os.environ.get("MODAL_KAFKA_WEBHOOK_URL"):
        os.environ["MODAL_KAFKA_WEBHOOK_URL"] = "https://placeholder.modal.run/publish_game_event"
        print("[WARNING] Using placeholder webhook URL for local testing")
    
    print("=" * 70)
    print("NEURON EVENTS API - LOCAL DEVELOPMENT")
    print("=" * 70)
    print("\nEndpoints:")
    print("  POST /api/events  - Publish game event")
    print(" GET /api/health   - Health check")
    print("\nListening on http://localhost:5000")
    print("=" * 70)
    
    app.run(debug=True, port=5000)
