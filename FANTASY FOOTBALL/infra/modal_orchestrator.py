"""
Modal Orchestrator for Event-Driven Multi-Agent System.

This module implements the Neuron Hybrid Stack architecture:
- Modal for sub-100ms container spawning
- Redis for hot state management (sub-5ms reads)
- Vertex AI for inference (Gemini Flash/Pro + Claude)
- Google Cloud TTS for voice synthesis

Architecture: See docs/architecture.md for full system design.
"""

import json
import os
import sys
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
import modal

# Add project root to Python path for imports
sys.path.insert(0, '/root')

# Import existing core components
# NOTE: These are imported inside Modal functions after sys.path is set
# from src.core.agent_factory import AgentFactory
# from src.core.tempo_engine import TempoEngine
# from src.llm.gemini_client import GeminiAgent  # Gemini 1.5 Pro for deep reasoning


# ============================================================================
# COST CONSTANTS & CONFIGURATION
# ============================================================================
COST_PER_TOKEN_INPUT = 0.00000015   # Gemini Flash
COST_PER_TOKEN_OUTPUT = 0.0000006   # Gemini Flash
COST_PER_CHAR_ELEVENLABS = 0.0003   # Premium Voice
COST_PER_CHAR_GOOGLE = 0.000016     # Standard Voice
DAILY_SPEND_LIMIT = 50.0            # $50.00 Hard Limit


# ============================================================================
# QUICK WIN 1: COST LOGGING UTILITY
# ============================================================================

def log_cost(service: str, amount: float, details: str = "", redis_client=None):
    """
    Simple cost logging for TNF monitoring.
    
    Usage:
        log_cost("gemini", 0.002, "debate generation, 1500 tokens")
        log_cost("elevenlabs", 0.01, "tts, 500 chars")
    """
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    log_line = f"{timestamp} | {service} | ${amount:.6f} | {details}"
    print(f"💰 {log_line}")
    
    # Store in Redis for post-game analysis
    if redis_client:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            redis_client.lpush(f"cost_log:{today}", log_line)
            redis_client.expire(f"cost_log:{today}", 604800)  # 7 days TTL
        except Exception as e:
            print(f"[COST LOG] Redis storage failed: {e}")


# ============================================================================
# QUICK WIN 2: EMERGENCY KILL SWITCH
# ============================================================================

def check_kill_switch(redis_client=None) -> bool:
    """
    Check if emergency stop is activated.
    
    Returns True if the system should stop processing.
    
    Activation methods:
        1. Environment variable: EMERGENCY_STOP=true
        2. Redis key: neuron:kill_switch = "true"
    
    Usage:
        if check_kill_switch(redis_client):
            return {"status": "stopped", "reason": "Emergency stop activated"}
    """
    # Method 1: Environment variable
    env_stop = os.environ.get("EMERGENCY_STOP", "false").lower()
    if env_stop == "true":
        print("🛑 EMERGENCY STOP activated via environment variable")
        return True
    
    # Method 2: Redis key (allows runtime control)
    if redis_client:
        try:
            redis_stop = redis_client.get("neuron:kill_switch")
            if redis_stop and redis_stop.lower() == "true":
                print("🛑 EMERGENCY STOP activated via Redis key")
                return True
        except Exception as e:
            print(f"[KILL SWITCH] Redis check failed: {e}")
    
    return False


# ============================================================================
# QUICK WIN 3: WEAVE TRACE LOGGING
# ============================================================================

_weave_initialized = False

def init_weave(project: str = "shalini-nfl-neuron-systems/ffn-debate-testing"):
    """Initialize Weave for tracing if not already done."""
    global _weave_initialized
    if _weave_initialized:
        return True
    
    try:
        import weave
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            print("[WEAVE] No WANDB_API_KEY found, skipping initialization")
            return False
        
        weave.init(project)
        _weave_initialized = True
        print(f"[WEAVE] Initialized project={project}")
        return True
    except Exception as e:
        print(f"[WEAVE] Initialization failed: {e}")
        return False


# Keep init_wandb as alias for backward compatibility
def init_wandb(project: str = "ffn-debate-testing", entity: str = "shalini-nfl-neuron-systems", run_name: str = None):
    """Initialize Weave (legacy alias)."""
    return init_weave(f"{entity}/{project}")


def log_trace(
    endpoint: str,
    latency_ms: float,
    status: str = "success",
    city: str = None,
    player: str = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    extra: dict = None
):
    """
    Log a trace to Weave for real-time monitoring.
    
    Usage:
        log_trace("run_debate", 250.5, city="seattle", player="Geno Smith")
    """
    try:
        import weave
        if not _weave_initialized:
            print(f"[WEAVE] Skipping log_trace - not initialized")
            return
        
        # Create trace data
        trace_data = {
            "endpoint": endpoint,
            "latency_ms": latency_ms,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        
        if city:
            trace_data["city"] = city
        if player:
            trace_data["player"] = player
        if tokens_in:
            trace_data["tokens_in"] = tokens_in
        if tokens_out:
            trace_data["tokens_out"] = tokens_out
        if cost_usd:
            trace_data["cost_usd"] = cost_usd
        if extra:
            trace_data.update(extra)
        
        # Call traced function based on endpoint
        if endpoint == "run_debate":
            traced_fn = get_traced_run_debate()
            if traced_fn and city:
                cities = city.split(" vs ") if " vs " in city else [city, ""]
                traced_fn(
                    city1=cities[0] if cities else "",
                    city2=cities[1] if len(cities) > 1 else "",
                    topic=extra.get("topic", "") if extra else "",
                    rounds=extra.get("rounds", 3) if extra else 3
                )
        elif endpoint == "generate_tts":
            traced_fn = get_traced_generate_tts()
            if traced_fn:
                traced_fn(
                    text="",  # Don't log full text
                    speaker_id=extra.get("speaker_id", "") if extra else "",
                    city=city or "",
                    audio_bytes=extra.get("audio_bytes", 0) if extra else 0
                )
        
        print(f"[WEAVE] Logged trace: {endpoint} - {latency_ms:.0f}ms")
        
    except Exception as e:
        print(f"[WEAVE] Log trace failed: {e}")


# ============================================================================
# WEAVE TRACED CORE FUNCTIONS
# These functions are decorated with @weave.op and called by Modal endpoints
# ============================================================================

def get_traced_run_debate():
    """Get a Weave-traced version of run_debate_core."""
    try:
        import weave
        
        @weave.op()
        def run_debate_core(city1: str, city2: str, topic: str, rounds: int = 3) -> dict:
            """
            Core debate logic - traced by Weave.
            """
            return {
                "city1": city1,
                "city2": city2,
                "topic": topic,
                "rounds": rounds,
                "traced": True
            }
        
        return run_debate_core
    except Exception as e:
        print(f"[WEAVE] Failed to create traced function: {e}")
        return None


def get_traced_generate_tts():
    """Get a Weave-traced version of generate_tts_core."""
    try:
        import weave
        
        @weave.op()
        def generate_tts_core(text: str, speaker_id: str, city: str, audio_bytes: int) -> dict:
            """
            Core TTS logic - traced by Weave.
            """
            return {
                "text_length": len(text),
                "speaker_id": speaker_id,
                "city": city,
                "audio_bytes": audio_bytes,
                "traced": True
            }
        
        return generate_tts_core
    except Exception as e:
        print(f"[WEAVE] Failed to create traced function: {e}")
        return None


# ============================================================================
# CITY VOICE MAPPING (HYBRID: Google TTS + ElevenLabs)
# ============================================================================

# ElevenLabs Voice IDs (Premium Tier)
ELEVENLABS_VOICE_MAP = {
    # Aggressive, high-energy cities - deeper, assertive voices
    "Philadelphia": "pNInz6obpgDQGcFmaJgB",  # Adam - deep, authoritative
    "Baltimore": "VR6AewLTigWG4xSOukaG",  # Arnold - strong, confident
    "New York (Jets)": "EXAVITQu4vr4xnSDxMaL",  # Bella - energetic
    "Las Vegas": "pNInz6obpgDQGcFmaJgB",  # Adam
    "Miami": "21m00Tcm4TlvDq8ikWAM",  # Rachel - vibrant
    
    # Analytical, measured cities - clear, professional voices
    "San Francisco": "AZnzlk1XvdvUeBnXmlld",  # Domi - clear, professional
    "New England": "pqHfZKP75CvOlQylNhV4",  # Bill - authoritative
    "Dallas": "EXAVITQu4vr4xnSDxMaL",  # Bella
    
    # Fast-paced, execution-focused - energetic voices
    "Kansas City": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "Seattle": "AZnzlk1XvdvUeBnXmlld",  # Domi
    "Cincinnati": "EXAVITQu4vr4xnSDxMaL",  # Bella
    
    # Traditional, blue-collar cities - mature, authoritative voices
    "Green Bay": "VR6AewLTigWG4xSOukaG",  # Arnold
    "Pittsburgh": "pNInz6obpgDQGcFmaJgB",  # Adam
    "Chicago": "pqHfZKP75CvOlQylNhV4",  # Bill
    
    # Resilient, long-suffering cities - steady, grounded voices
    "Buffalo": "VR6AewLTigWG4xSOukaG",  # Arnold
    "Cleveland": "pNInz6obpgDQGcFmaJgB",  # Adam
    "Detroit": "VR6AewLTigWG4xSOukaG",  # Arnold
    
    # Confident, swagger cities - bold voices
    "Tampa Bay": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "Los Angeles (Rams)": "AZnzlk1XvdvUeBnXmlld",  # Domi
    "Los Angeles (Chargers)": "EXAVITQu4vr4xnSDxMaL",  # Bella
    
    # Midwest steady cities - calm, measured voices
    "Minnesota": "pqHfZKP75CvOlQylNhV4",  # Bill
    "Indianapolis": "AZnzlk1XvdvUeBnXmlld",  # Domi
    "Tennessee": "VR6AewLTigWG4xSOukaG",  # Arnold
    
    # Southern cities - warm, distinctive voices
    "New Orleans": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "Atlanta": "EXAVITQu4vr4xnSDxMaL",  # Bella
    "Carolina": "pqHfZKP75CvOlQylNhV4",  # Bill
    "Houston": "pNInz6obpgDQGcFmaJgB",  # Adam
    
    # Emerging/rebuilding cities - fresh, optimistic voices
    "Jacksonville": "EXAVITQu4vr4xnSDxMaL",  # Bella
    "Arizona": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    
    # Legacy-burdened cities - reflective voices
    "New York (Giants)": "VR6AewLTigWG4xSOukaG",  # Arnold
    "Washington": "pqHfZKP75CvOlQylNhV4",  # Bill
    
    # Mountain/elevation cities - strong, confident voices
    "Denver": "VR6AewLTigWG4xSOukaG",  # Arnold
}

# Google Cloud TTS Voice IDs (Standard Tier)
GOOGLE_VOICE_MAP = {
    "Washington": "en-US-Studio-O",
    
    # Mountain/elevation cities - strong, confident voices
    "Denver": "en-US-Neural2-D",
}


# ============================================================================
# TTS PROVIDERS
# ============================================================================

class GoogleTTSProvider:
    """
    Wrapper for Google Cloud Text-to-Speech API (Standard Tier).
    """
    def __init__(self):
        from google.cloud import texttospeech
        try:
            self.client = texttospeech.TextToSpeechClient()
            print("[INIT] GoogleTTSProvider initialized")
        except Exception as e:
            print(f"[INIT ERROR] GoogleTTSProvider failed: {e}")
            self.client = None

    def generate_audio(self, text: str, voice_name: str) -> bytes:
        """
        Generate MP3 audio using Google Cloud TTS.
        """
        if not self.client:
            raise RuntimeError("Google TTS client not initialized")
            
        from google.cloud import texttospeech
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build voice params
        voice_params = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name
        )
        
        # Audio configuration
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            sample_rate_hertz=24000,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        # Perform synthesis
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )
        
        return response.audio_content


# ============================================================================
# MODAL APP & IMAGE DEFINITION
# ============================================================================

app = modal.App("neuron-orchestrator")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",                    # Required for web endpoints
        "google-cloud-aiplatform",    # Vertex AI SDK (Gemini)
        "google-generativeai",        # Google AI Studio SDK (Gemini with API key)
        "anthropic[vertex]",          # Anthropic SDK for Claude on Vertex AI
        "redis",                      # Redis client for hot state
        "google-cloud-texttospeech",  # Google Cloud TTS (Standard)
        "elevenlabs>=1.0.0",          # ElevenLabs TTS (Premium) - v1.0+ required for new API
        "websockets",                 # WebSocket for streaming TTS
        "sse-starlette",              # Server-Sent Events for streaming
        # Style Capture v2 dependencies
        "youtube_transcript_api",     # YouTube transcript extraction
        "newspaper3k",                # Article content extraction
        "lxml",                       # HTML parsing for newspaper3k
        "openai",                     # OpenAI Whisper API for transcription
        # Observability
        "wandb",                      # Weights & Biases for tracking
        "weave",                      # Weave for tracing
    )
    .add_local_dir(
        local_path="../src",
        remote_path="/root/src"
    )
    .add_local_dir(
        local_path="../config",
        remote_path="/root/config"
    )
)

# ============================================================================
# CULTURAL AGENT CLASS (MODAL-MANAGED)
# ============================================================================

@app.cls(
    image=image,
    secrets=[
        modal.Secret.from_name("redis-credentials"),     # REDIS_URL
        modal.Secret.from_name("googlecloud-secret"),    # GCP credentials (Google TTS)
        modal.Secret.from_name("gcp-vertex-ai"),         # Vertex AI credentials (Gemini/Claude)
        modal.Secret.from_name("elevenlabs-secret"),     # ElevenLabs API Keys (Premium Tier)
        modal.Secret.from_name("gemini-api-key"),        # Google AI Studio API Key (fallback)
        modal.Secret.from_name("wandb-secret"),          # Weights & Biases API Key
    ],
    min_containers=20,         # OPTIMIZATION: Increased from 10 → 20 for faster debate starts
    max_containers=150,        # OPTIMIZATION: Increased from 100 → 150 to handle spikes
    timeout=30,                # 30s timeout per request
)
class CulturalAgent:
    """
    Modal-managed agent class for culturally-aware sports commentary.
    
    Each instance handles inference for a specific city profile with:
    - Redis hot state for context window
    - Vertex AI for model inference
    - Tempo and lexical engines for cultural reasoning
    """
    
    @modal.enter()
    def setup(self):
        """
        Initialize agent resources on container startup.
        
        This runs once per container lifecycle, establishing:
        - Redis connection (hot state)
        - Vertex AI client (inference)
        - Google Cloud TTS client (standard voice synthesis)
        - ElevenLabs client (premium voice synthesis)
        - Agent factory (city profiles)
        - Tempo engine (timing logic)
        """
        import traceback
        
        # ----------------------------------------------------------------
        # 1. REDIS CONNECTION
        # ----------------------------------------------------------------
        try:
            import redis
            redis_url = os.environ.get("REDIS_URL")
            if not redis_url:
                print("[INIT ERROR] REDIS_URL not found in environment")
                raise ValueError("REDIS_URL not found in environment")
            
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            print("[INIT] Redis connection established")
        except Exception as e:
            print(f"[INIT ERROR] Redis failed: {e}")
            traceback.print_exc()
            raise
        
        # ----------------------------------------------------------------
        # 2. GCP CREDENTIALS SETUP (Write JSON to temp file for SDK)
        # ----------------------------------------------------------------
        try:
            gcp_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if gcp_creds_json:
                # Write credentials to a temp file for SDKs that expect file path
                import tempfile
                creds_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.json',
                    delete=False
                )
                creds_file.write(gcp_creds_json)
                creds_file.close()
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file.name
                print(f"[INIT] GCP credentials written to temp file: {creds_file.name}")
            else:
                print("[INIT] No GOOGLE_APPLICATION_CREDENTIALS_JSON found, using default credentials")
        except Exception as e:
            print(f"[INIT ERROR] GCP credentials setup failed: {e}")
            traceback.print_exc()
            # Continue - might still work with default credentials
        
        # ----------------------------------------------------------------
        # 3. VERTEX AI CLIENT (Gemini/Claude via Vertex AI SDK)
        # ----------------------------------------------------------------
        try:
            # Import the new VertexAIClient
            from src.llm.vertex_client import VertexAIClient
            
            # Get project ID from new gcp-vertex-ai secret
            project_id = os.environ.get("GCP_PROJECT_ID")
            
            # Check which model to use (can be configured via env)
            # Using Gemini 2.0 Flash via Vertex AI
            model_name = os.environ.get("VERTEX_MODEL", "gemini-2.0-flash-001")
            
            if project_id:
                self.vertex_client = VertexAIClient(
                    project_id=project_id,
                    location="us-central1",
                    model_name=model_name
                )
                print(f"[INIT] VertexAI initialized: project={project_id}, model={model_name}")
                print(f"[INIT] USING CLAUDE: {'YES' if 'claude' in model_name else 'NO'}")
            else:
                # Fallback to old GeminiAgent with API key
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    self.vertex_client = GeminiAgent(
                        model_name="gemini-2.0-flash"
                    )
                    print("[INIT] Fallback to GeminiAgent (API Key)")
                else:
                    raise ValueError("No Vertex AI or Gemini credentials found")
        except Exception as e:
            print(f"[INIT ERROR] Vertex AI failed: {e}")
            traceback.print_exc()
            raise
        
        # ----------------------------------------------------------------
        # 3. GOOGLE TTS PROVIDER (Standard Tier)
        # ----------------------------------------------------------------
        try:
            self.google_tts_provider = GoogleTTSProvider()
            print("[INIT] Google TTS initialized")
        except Exception as e:
            print(f"[INIT ERROR] Google TTS failed: {e}")
            traceback.print_exc()
            # Continue without TTS - not fatal
            self.google_tts_provider = None
        
        # ----------------------------------------------------------------
        # 4. ELEVENLABS KEYS (Premium Tier)
        # ----------------------------------------------------------------
        try:
            self.elevenlabs_keys = []
            key1 = os.environ.get("ELEVENLABS_API_KEY")
            if key1:
                self.elevenlabs_keys.append(key1)
            
            # Try secondary keys (rotation)
            for i in range(2, 6):
                key = os.environ.get(f"ELEVENLABS_API_KEY_{i}")
                if key:
                    self.elevenlabs_keys.append(key)
            
            if self.elevenlabs_keys:
                print(f"[INIT] ElevenLabs: {len(self.elevenlabs_keys)} keys loaded")
            else:
                print("[INIT] No ElevenLabs API keys found - premium tier disabled")
        except Exception as e:
            print(f"[INIT ERROR] ElevenLabs failed: {e}")
            self.elevenlabs_keys = []
        
        # ----------------------------------------------------------------
        # 5. CORE ENGINES (AgentFactory, TempoEngine)
        # ----------------------------------------------------------------
        try:
            from src.core.agent_factory import AgentFactory
            from src.core.tempo_engine import TempoEngine
            
            self.agent_factory = AgentFactory(config_path="/root/config/city_profiles.json")
            self.tempo_engine = TempoEngine(config_path="/root/config/city_profiles.json")
            print("[INIT] AgentFactory and TempoEngine initialized")
        except Exception as e:
            print(f"[INIT ERROR] Core engines failed: {e}")
            traceback.print_exc()
            raise
        
        # ----------------------------------------------------------------
        # 6. MEMORY MODULE
        # ----------------------------------------------------------------
        try:
            from src.core.memory_module import MemoryModule
            self.memory_module = MemoryModule(config_path="/root/config/city_profiles.json")
            print("[INIT] MemoryModule initialized")
        except Exception as e:
            print(f"[INIT ERROR] MemoryModule failed: {e}")
            traceback.print_exc()
            # Continue without memory - not fatal
            self.memory_module = None
        
        # ----------------------------------------------------------------
        # 7. WEIGHTS & BIASES TRACING
        # ----------------------------------------------------------------
        try:
            init_wandb(project="neuron-live", run_name=f"modal-{datetime.now().strftime('%m%d-%H%M')}")
        except Exception as e:
            print(f"[INIT WARNING] W&B init failed (non-fatal): {e}")
        
        print("[INIT] CulturalAgent ready (Hybrid Voice: Google TTS + ElevenLabs)")
    
    @modal.method()
    def generate_response(
        self,
        city_name: str,
        user_input: str,
        conversation_history: Optional[list] = None,
        game_context: Optional[Dict[str, Any]] = None,
        tier: str = "standard",
        style: str = "standard",
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Generate a culturally-aware response for a specific city.
        
        Args:
            city_name: City profile to use (e.g., 'Philadelphia', 'Dallas')
            user_input: User's message/question
            conversation_history: Optional list of previous messages
            game_context: Optional game state (score, time, etc.)
        
        Returns:
            Dictionary containing:
                - response: Generated text response
                - city: City name used
                - delay_ms: Recommended delay before next response
                - confidence: Response confidence (0.0-1.0)
                - cached: Whether context was cached in Redis
        
        Raises:
            KeyError: If city_name not found in profiles
            RuntimeError: If inference fails
        """
        import time
        from datetime import datetime
        start_time = time.time()
        
        # ----------------------------------------------------------------
        # 0a. KILL SWITCH CHECK: Emergency stop
        # ----------------------------------------------------------------
        if check_kill_switch(self.redis_client):
            return {
                "status": "stopped",
                "error": "Emergency stop activated",
                "city": city_name
            }
        
        # ----------------------------------------------------------------
        # 0b. COST GUARD: Circuit Breaker
        # ----------------------------------------------------------------
        today = datetime.now().strftime("%Y-%m-%d")
        daily_spend_key = f"spend:daily:{today}"
        
        try:
            current_spend = float(self.redis_client.get(daily_spend_key) or 0.0)
            if current_spend >= DAILY_SPEND_LIMIT:
                print(f"[COST GUARD] Daily limit reached: ${current_spend:.4f} / ${DAILY_SPEND_LIMIT}")
                raise RuntimeError("Daily cost limit exceeded. Please try again tomorrow.")
        except ValueError:
            pass  # Ignore redis parsing errors
        except RuntimeError as e:
            raise e
        except Exception as e:
            print(f"[COST GUARD ERROR] Failed to check limit: {e}")
        
        # ----------------------------------------------------------------
        # 1. REDIS HOT STATE: Check for cached context
        # ----------------------------------------------------------------
        cache_key = f"context:{city_name}:latest"
        cached_context = None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                cached_context = json.loads(cached_data)
                print(f"[CACHE HIT] Retrieved context for {city_name} (sub-5ms)")
        except Exception as e:
            print(f"[CACHE MISS] Redis read failed: {e}")
        
        # ----------------------------------------------------------------
        # 2. AGENT FACTORY: Load city profile and construct system prompt
        # ----------------------------------------------------------------
        try:
            city_profile = self.agent_factory.load_profile(city_name)
            
            # Detect debate mode from game_context
            is_debate_mode = bool(
                game_context and 
                (game_context.get('debate_round') or game_context.get('responding_to'))
            )
            
            if is_debate_mode:
                # Build debate context for enhanced prompts
                debate_context = {
                    'opponent_name': game_context.get('responding_to', 'opponent'),
                    'previous_response': '',  # Will be in conversation_history
                    'turn_number': game_context.get('turn_in_round', 1) + 
                                   ((game_context.get('debate_round', 1) - 1) * 2),
                    'conflict_mode': 'aggressive' if game_context.get('tone_energy', 0.5) > 0.6 else 'balanced'
                }
                
                # Get previous response from conversation history if available
                if conversation_history and len(conversation_history) > 0:
                    last_msg = conversation_history[-1]
                    debate_context['previous_response'] = last_msg.get('content', '')[:500]
                    debate_context['opponent_name'] = last_msg.get('role', 'opponent')
                
                system_prompt = self.agent_factory.construct_system_prompt(
                    city_name, 
                    mode="debate",
                    style=style,
                    debate_context=debate_context
                )
            else:
                system_prompt = self.agent_factory.construct_system_prompt(
                    city_name,
                    style=style
                )
                
        except KeyError as e:
            return {
                "error": f"City '{city_name}' not found in profiles",
                "available_cities": self.agent_factory.get_all_cities()
            }
        
        # ----------------------------------------------------------------
        # 3. TEMPO ENGINE: Calculate response delay
        # ----------------------------------------------------------------
        delay_seconds = self.tempo_engine.get_delay(city_name)
        delay_ms = int(delay_seconds * 1000)
        
        # ----------------------------------------------------------------
        # 4. CONTEXT ASSEMBLY: Combine game state + conversation history + memory
        # ----------------------------------------------------------------
        context_parts = []
        
        # Add game context if provided
        if game_context:
            context_parts.append("CURRENT GAME STATE:")
            for key, value in game_context.items():
                context_parts.append(f"- {key}: {value}")
        
        # Add conversation history if provided
        if conversation_history and len(conversation_history) > 0:
            context_parts.append("\nCONVERSATION HISTORY:")
            for msg in conversation_history[-5:]:  # Last 5 turns
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_parts.append(f"[{role}]: {content}")
        
        # P0 ENHANCEMENT: Add memory-enhanced context
        memory_situation = {
            'opponent_team': game_context.get('opponent') if game_context else None,
            'keywords': []
        }
        
        # Extract keywords from user input
        if user_input:
            # Simple keyword extraction (can be enhanced with NLP)
            keywords = [word.strip('.,!?').lower() for word in user_input.split() 
                       if len(word) > 3]
            memory_situation['keywords'] = keywords
        
        # Get memory-enhanced context
        memory_context = self.memory_module.construct_memory_context(
            city_name, 
            memory_situation
        )
        
        if memory_context:
            context_parts.append(f"\n{memory_context}")
        
        # Combine into full context
        full_context = "\n".join(context_parts) if context_parts else "No prior context."
        
        # ----------------------------------------------------------------
        # 5. VERTEX AI INFERENCE: Generate response
        # ----------------------------------------------------------------
        try:
            # Check if user_input contains pre-formatted debate instructions
            is_preformatted_debate = (
                "TOPIC/INPUT:" in user_input or 
                "Respond directly to" in user_input or
                "Open the debate on:" in user_input
            )
            
            # Build the prompt
            if is_preformatted_debate:
                # Debate workshop mode: user_input already contains full instructions
                # Just add context and pass through
                prompt = f"{full_context}\n\n{user_input}"
            elif conversation_history and len(conversation_history) > 0:
                # Legacy debate mode: React to the last speaker
                last_message = conversation_history[-1]
                last_speaker = last_message.get("role", "opponent")
                last_content = last_message.get("content", "")
                
                prompt = f"{full_context}\n\n"
                prompt += f"DEBATE TOPIC: {user_input}\n\n"
                prompt += f"OPPONENT'S LAST ARGUMENT:\n{last_content}\n\n"
                prompt += f"React specifically to what they just said. Counter their points directly. "
                prompt += f"Use your city's debate arsenal and historical knowledge. Attack back if they attacked you.\n\n"
                prompt += f"Your response:"
            else:
                # Normal mode: Respond to user input
                prompt = f"{full_context}\n\nUSER: {user_input}\n\nRespond as a {city_name} sports commentator:"
            
            # Add randomization seed to prevent response caching
            import random
            import time
            random_seed = f"\n\n[Response ID: {int(time.time() * 1000)}_{random.randint(1000, 9999)}]"
            prompt += random_seed
            
            # Call Vertex AI (Gemini Flash for speed)
            response_text = self.vertex_client.send_message(
                system_instruction=system_prompt,
                user_message=prompt,
                max_tokens=512,
                temperature=0.9  # High temperature for personality
            )
            
            if not response_text:
                raise RuntimeError("Empty response from Vertex AI")
            
            # ----------------------------------------------------------------
            # 5b. VIBE CHECK: Cultural Circuit Breaker
            # ----------------------------------------------------------------
            try:
                from src.core.vibe_check import compute_vibe_score, build_reinforced_prompt, get_archive_fallback
                
                vibe_score = compute_vibe_score(response_text, city_name)
                regeneration_attempts = 0
                MAX_REGENERATIONS = 1
                VIBE_THRESHOLD = 0.5  # Lower threshold since we adjusted weights
                
                while vibe_score < VIBE_THRESHOLD and regeneration_attempts < MAX_REGENERATIONS:
                    regeneration_attempts += 1
                    print(f"[VIBE CHECK] Score {vibe_score:.2f} < {VIBE_THRESHOLD} — Regenerating ({regeneration_attempts})")
                    
                    # Build reinforced prompt with few-shot examples
                    reinforced_prompt, higher_temp = build_reinforced_prompt(prompt, city_name)
                    
                    # Regenerate with higher temperature
                    response_text = self.vertex_client.send_message(
                        system_instruction=system_prompt,
                        user_message=reinforced_prompt,
                        max_tokens=512,
                        temperature=higher_temp  # 0.95 to escape neutral attractor
                    )
                    
                    if response_text:
                        vibe_score = compute_vibe_score(response_text, city_name)
                        print(f"[VIBE CHECK] Regenerated score: {vibe_score:.2f}")
                
                # Final fail-safe: Archive fallback
                if vibe_score < VIBE_THRESHOLD:
                    print(f"[VIBE CHECK] Regeneration failed — Using archive fallback")
                    response_text = get_archive_fallback(city_name, user_input[:50])
                    
                # Log vibe check result to Redis for monitoring
                try:
                    vibe_key = f"vibe_check:{city_name}:latest"
                    self.redis_client.setex(
                        vibe_key,
                        3600,  # 1 hour TTL
                        json.dumps({
                            "score": vibe_score,
                            "regenerations": regeneration_attempts,
                            "passed": vibe_score >= VIBE_THRESHOLD,
                            "timestamp": time.time()
                        })
                    )
                except Exception as ve:
                    print(f"[VIBE CHECK] Failed to log to Redis: {ve}")
                    
            except ImportError as ie:
                print(f"[VIBE CHECK] Module not available (non-fatal): {ie}")
            except Exception as ve:
                print(f"[VIBE CHECK] Error (non-fatal): {ve}")
            
        except Exception as e:
            print(f"[INFERENCE ERROR] Vertex AI failed: {e}")
            return {
                "error": f"Inference failed: {str(e)}",
                "city": city_name
            }
        
        # ----------------------------------------------------------------
        # 6. REDIS HOT STATE: Update cache with latest context
        # ----------------------------------------------------------------
        try:
            updated_context = {
                "last_user_input": user_input,
                "last_response": response_text,
                "timestamp": time.time(),
                "game_context": game_context
            }
            
            self.redis_client.setex(
                cache_key,
                14400,  # 4 hour TTL (game duration + buffer)
                json.dumps(updated_context)
            )
            print(f"[CACHE UPDATE] Stored context for {city_name}")
        except Exception as e:
            print(f"[CACHE ERROR] Failed to update Redis: {e}")
        
        # ----------------------------------------------------------------
        # 7. METRICS: Calculate latency
        # ----------------------------------------------------------------
        total_latency_ms = int((time.time() - start_time) * 1000)
        
        print(f"[METRICS] {city_name} | Latency: {total_latency_ms}ms | Delay: {delay_ms}ms")
        
        # ----------------------------------------------------------------
        # 8. COST TRACKING: Calculate and record spend with enhanced logging
        # ----------------------------------------------------------------
        try:
            # Estimate tokens (char count / 4 is a rough heuristic)
            input_chars = len(full_context) + len(system_prompt) + len(prompt)
            output_chars = len(response_text)
            
            input_tokens = input_chars / 4
            output_tokens = output_chars / 4
            
            # Calculate LLM Cost
            llm_cost = (input_tokens * COST_PER_TOKEN_INPUT) + (output_tokens * COST_PER_TOKEN_OUTPUT)
            
            # Calculate Voice Cost
            voice_cost = 0.0
            if tier == "premium":
                voice_cost = output_chars * COST_PER_CHAR_ELEVENLABS
            else:
                voice_cost = output_chars * COST_PER_CHAR_GOOGLE
                
            total_cost = llm_cost + voice_cost
            
            # Update Redis (Atomic Increment)
            self.redis_client.incrbyfloat(daily_spend_key, total_cost)
            self.redis_client.incrbyfloat(f"spend:user:{user_id}", total_cost)
            
            # Enhanced logging with log_cost utility
            log_cost("gemini", llm_cost, f"city={city_name}, in={int(input_tokens)}tok, out={int(output_tokens)}tok", self.redis_client)
            if voice_cost > 0:
                log_cost("voice", voice_cost, f"city={city_name}, tier={tier}, chars={output_chars}", self.redis_client)
            
        except Exception as e:
            print(f"[COST TRACKING ERROR] Failed to record spend: {e}")
        
        # ----------------------------------------------------------------
        # 8. RETURN: Response payload
        # ----------------------------------------------------------------
        return {
            "response": response_text,
            "city": city_name,
            "delay_ms": delay_ms,
            "confidence": 0.85,  # TODO: Extract from model metadata
            "cached": cached_context is not None,
            "latency_ms": total_latency_ms,
            "tempo": {
                "base_delay_ms": city_profile.get("tempo", {}).get("base_delay_ms", 150),
                "variance_ms": city_profile.get("tempo", {}).get("variance_ms", 20)
            }
        }
    
    @modal.method()
    def generate_voice_response(
        self,
        text: str,
        city_name: str,
        tier: str = "standard"
    ) -> bytes:
        """
        Generate voice audio from text using hybrid TTS with Redis caching.
        
        Args:
            text: Text to convert to speech
            city_name: City profile (determines voice selection)
            tier: Voice quality tier ('standard' = Google TTS, 'premium' = ElevenLabs)
        
        Returns:
            Raw audio bytes (MP3 format)
        
        Raises:
            ValueError: If city not found in voice mapping or invalid tier
            RuntimeError: If TTS synthesis fails
        """
        from google.cloud import texttospeech
        import hashlib
        import base64
        
        # Validate tier
        if tier not in ["standard", "premium"]:
            tier = "standard"  # Default to standard
            print(f"[TTS WARNING] Invalid tier, defaulting to 'standard'")
        
        # Select voice provider based on tier
        if tier == "premium":
            if not self.elevenlabs_keys:
                print(f"[TTS WARNING] ElevenLabs keys not available, falling back to Google TTS")
                tier = "standard"
            else:
                voice_map = ELEVENLABS_VOICE_MAP
                default_voice = "pNInz6obpgDQGcFmaJgB"  # Adam
        
        if tier == "standard":
            voice_map = GOOGLE_VOICE_MAP
            default_voice = "en-US-Wavenet-D"
        
        # Get voice ID for this city
        voice_id = voice_map.get(city_name)
        if not voice_id:
            voice_id = default_voice
            print(f"[TTS WARNING] City '{city_name}' not in voice map, using default")
        
        # ----------------------------------------------------------------
        # 1. CACHE CHECK: Check Redis for existing audio
        # ----------------------------------------------------------------
        # Generate secure cache key: hash(text + voice_id + tier)
        cache_key_str = f"{text}:{voice_id}:{tier}"
        cache_hash = hashlib.sha256(cache_key_str.encode('utf-8')).hexdigest()
        redis_key = f"voice_cache:{cache_hash}"
        
        try:
            cached_audio_b64 = self.redis_client.get(redis_key)
            if cached_audio_b64:
                print(f"[CACHE HIT] Returning cached audio for {city_name} ({tier}) (key: {redis_key[:8]})")
                return base64.b64decode(cached_audio_b64)
        except Exception as e:
            print(f"[CACHE READ ERROR] Failed to check Redis: {e}")
        
        # 2. GENERATE AUDIO: Try ElevenLabs first, then Google TTS as fallback
        # ----------------------------------------------------------------
        try:
            audio_bytes = None
            
            # ALWAYS try ElevenLabs first if keys are available
            if self.elevenlabs_keys:
                from elevenlabs.client import ElevenLabs
                
                elevenlabs_voice_id = ELEVENLABS_VOICE_MAP.get(city_name, "pNInz6obpgDQGcFmaJgB")
                last_error = None
                
                # Try keys in rotation
                for api_key in self.elevenlabs_keys:
                    try:
                        client = ElevenLabs(api_key=api_key)
                        # Use new SDK method: text_to_speech.convert()
                        audio_generator = client.text_to_speech.convert(
                            text=text,
                            voice_id=elevenlabs_voice_id,
                            model_id="eleven_monolingual_v1",
                            output_format="mp3_44100_128"
                        )
                        # Collect audio bytes from generator
                        audio_bytes = b"".join(audio_generator)
                        tier = "premium"  # Mark as premium since ElevenLabs worked
                        print(f"[TTS SUCCESS - ELEVENLABS] Audio for {city_name} (voice: {elevenlabs_voice_id})")
                        break  # Success!
                    except Exception as e:
                        print(f"[TTS ROTATION] ElevenLabs key failed: {e}")
                        last_error = e
                        continue  # Try next key
            
            # Fall back to Google TTS if ElevenLabs failed or no keys
            if audio_bytes is None:
                print(f"[TTS FALLBACK] Trying Google TTS...")
                if self.google_tts_provider:
                    try:
                        google_voice_id = GOOGLE_VOICE_MAP.get(city_name, "en-US-Wavenet-D")
                        audio_bytes = self.google_tts_provider.generate_audio(text, google_voice_id)
                        tier = "standard"
                        print(f"[TTS SUCCESS - GOOGLE] Audio for {city_name} (voice: {google_voice_id})")
                    except Exception as e:
                        print(f"[TTS ERROR] Google TTS also failed: {e}")
                        raise RuntimeError(f"All TTS providers failed. ElevenLabs: {last_error}, Google: {e}")
                else:
                    raise RuntimeError(f"ElevenLabs failed and Google TTS not available")
            
            # ----------------------------------------------------------------
            # 3. CACHE WRITE: Store in Redis with 90-day TTL
            # ----------------------------------------------------------------
            try:
                # Store as Base64 string since Redis handles strings best
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # 90 days = 7,776,000 seconds
                TTL_SECONDS = 7776000
                
                self.redis_client.setex(
                    redis_key,
                    TTL_SECONDS,
                    audio_b64
                )
                print(f"[CACHE WRITE] Stored audio in Redis (TTL: 90 days)")
            except Exception as e:
                print(f"[CACHE WRITE ERROR] Failed to store in Redis: {e}")
            
            return audio_bytes
            
        except Exception as e:
            raise RuntimeError(f"TTS synthesis failed for {city_name}: {e}")
    
    @modal.method()
    def check_interruption(
        self,
        city_name: str,
        opponent_confidence: float
    ) -> Dict[str, bool]:
        """
        Check if this city should interrupt based on opponent confidence.
        
        Args:
            city_name: City profile to check
            opponent_confidence: Opponent's confidence level (0.0-1.0)
        
        Returns:
            Dictionary with interruption decision and metadata
        """
        should_interrupt = self.tempo_engine.check_interruption(
            city_name, 
            opponent_confidence
        )
        
        aggression = self.tempo_engine.get_aggression_level(city_name)
        backs_down = self.tempo_engine.should_back_down(city_name)
        
        return {
            "should_interrupt": should_interrupt,
            "aggression_level": aggression,
            "will_back_down": backs_down,
            "city": city_name
        }
    
    @modal.method()
    def get_city_profile(self, city_name: str) -> Dict[str, Any]:
        """
        Retrieve the full configuration for a specific city.
        
        Args:
            city_name: Name of the city
        
        Returns:
            Full city profile dictionary
        """
        try:
            return self.agent_factory.load_profile(city_name)
        except KeyError as e:
            return {
                "error": str(e),
                "available_cities": self.agent_factory.get_all_cities()
            }


# ============================================================================
# HTTP WEBHOOK ENDPOINT (EVENT INGESTION)
# ============================================================================

@app.function(
    image=image,
)
@modal.fastapi_endpoint(method="POST")
def generate_commentary(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP webhook endpoint for generating sports commentary.
    
    This is the entry point for live game events and creator requests.
    
    Expected JSON payload:
    {
        "city": "Philadelphia",
        "user_input": "What do you think about that touchdown?",
        "tier": "standard",             // Optional: 'standard' or 'premium' (default: 'standard')
        "conversation_history": [...],  // Optional
        "game_context": {               // Optional
            "score": "Eagles 21 - Cowboys 14",
            "quarter": 3,
            "time_remaining": "8:45"
        }
    }
    
    Returns:
        JSON response with generated commentary and metadata
    
    Example:
        curl -X POST https://your-modal-url.modal.run/generate_commentary \
             -H "Content-Type: application/json" \
             -d '{"city": "Philadelphia", "user_input": "React to that interception!", "tier": "premium"}'
    """
    # ----------------------------------------------------------------
    # 1. VALIDATE REQUEST
    # ----------------------------------------------------------------
    city = request_data.get("city")
    user_input = request_data.get("user_input")
    tier = request_data.get("tier", "standard")  # Default to standard
    style = request_data.get("style", "standard") # Default to standard (or sitcom)
    
    if not city or not user_input:
        return {
            "error": "Missing required fields: 'city' and 'user_input'",
            "status": "invalid_request"
        }
    
    # ----------------------------------------------------------------
    # 2. INSTANTIATE AGENT (Modal spawns container sub-100ms)
    # ----------------------------------------------------------------
    agent = CulturalAgent()
    
    # ----------------------------------------------------------------
    # 3. GENERATE RESPONSE (Vertex AI inference + Redis state)
    # ----------------------------------------------------------------
    conversation_history = request_data.get("conversation_history", [])
    game_context = request_data.get("game_context", {})
    
    user_id = request_data.get("user_id", "anonymous")
    
    result = agent.generate_response.remote(
        city_name=city,
        user_input=user_input,
        conversation_history=conversation_history,
        game_context=game_context,
        tier=tier,
        style=style,
        user_id=user_id
    )
    
    # ----------------------------------------------------------------
    # 4. GENERATE VOICE AUDIO (Hybrid: Google TTS or ElevenLabs)
    # ----------------------------------------------------------------
    import base64
    
    audio_base64 = None
    voice_metadata = {}
    
    # Only generate audio if text response was successful
    if "response" in result and not "error" in result:
        try:
            # CRITICAL: Strip all markdown formatting before TTS
            # LLM sometimes uses *emphasis*, _italics_, [actions], etc.
            # TTS will read these literally as "asterisk word asterisk"
            import re
            original_text = result["response"]
            
            # Remove markdown formatting characters
            sanitized_text = original_text
            sanitized_text = re.sub(r'\*+', '', sanitized_text)  # Remove all asterisks
            sanitized_text = re.sub(r'_+', '', sanitized_text)   # Remove all underscores
            sanitized_text = re.sub(r'\[.*?\]', '', sanitized_text)  # Remove [brackets]
            sanitized_text = re.sub(r'\((?:laughs|sighs|pauses|chuckles)\)', '', sanitized_text, flags=re.IGNORECASE)  # Remove (actions)
            sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()  # Clean up whitespace
            
            if sanitized_text != original_text:
                print(f"[TTS SANITIZE] Removed markdown formatting from text")
            
            # Generate voice audio with tier selection
            audio_bytes = agent.generate_voice_response.remote(
                text=sanitized_text,  # Use sanitized text!
                city_name=city,
                tier=tier
            )
            
            # Encode to Base64 for JSON transport
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Add voice metadata
            voice_map = ELEVENLABS_VOICE_MAP if tier == "premium" else GOOGLE_VOICE_MAP
            voice_metadata = {
                "voice_id": voice_map.get(city, "default"),
                "audio_format": "mp3",
                "sample_rate": 44100 if tier == "premium" else 24000,
                "encoding": "base64",
                "tier": tier
            }
            
            print(f"[VOICE] Generated {tier} audio for {city} ({len(audio_bytes)} bytes)")
            
        except Exception as e:
            print(f"[VOICE ERROR] TTS failed for {city}: {e}")
            # Continue without audio - text response still valid
    
    # ----------------------------------------------------------------
    # 5. RETURN RESPONSE WITH AUDIO
    # ----------------------------------------------------------------
    response_payload = {
        "status": "success",
        "data": result,
        "endpoint": "generate_commentary",
        "version": "1.0"
    }
    
    # Add audio if generated successfully
    if audio_base64:
        response_payload["audio"] = {
            "data": audio_base64,
            "metadata": voice_metadata
        }
    
    return response_payload


# ============================================================================
# STREAMING COMMENTARY ENDPOINT (SSE)
# ============================================================================

@app.function(
    image=image,
)
# @modal.web_endpoint(method="POST")  # DISABLED - Modal 8-endpoint limit (use generate_commentary instead)
def stream_live_commentary(request_data: Dict[str, Any]):
    """
    Server-Sent Events (SSE) endpoint for real-time streaming commentary.
    
    Streams text and audio chunks as they're generated for low-latency playback.
    
    Expected JSON payload:
    {
        "city": "Philadelphia",
        "user_input": "React to that touchdown!",
        "event_type": "touchdown",  // Optional: for priority handling
        "game_context": {...}       // Optional
    }
    
    SSE Events:
        event: text
        data: {"chunk": "First chunk of text...", "index": 0}
        
        event: audio
        data: {"chunk": "<base64 audio>", "index": 0, "format": "mp3"}
        
        event: complete
        data: {"total_chunks": 5, "latency_ms": 1234}
    
    Example:
        curl -N -X POST https://your-modal-url.modal.run/stream_live_commentary \\
             -H "Content-Type: application/json" \\
             -d '{"city": "Philadelphia", "user_input": "TOUCHDOWN!"}'
    """
    from fastapi.responses import StreamingResponse
    import base64
    import time
    
    # ----------------------------------------------------------------
    # 1. VALIDATE INPUT
    # ----------------------------------------------------------------
    city = request_data.get("city")
    user_input = request_data.get("user_input")
    event_type = request_data.get("event_type", "normal")
    game_context = request_data.get("game_context", {})
    
    if not city or not user_input:
        return {"error": "Missing required fields: 'city' and 'user_input'"}
    
    # ----------------------------------------------------------------
    # 2. GENERATE RESPONSE
    # ----------------------------------------------------------------
    start_time = time.time()
    
    try:
        agent = CulturalAgent()
        result = agent.generate_response.remote(
            city_name=city,
            user_input=user_input,
            conversation_history=[],
            game_context=game_context,
            tier="premium"  # Use premium for streaming
        )
        
        response_text = result.get("response", "")
        
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}
    
    # ----------------------------------------------------------------
    # 3. RETURN RESPONSE (Audio via separate endpoint for now)
    # ----------------------------------------------------------------
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "status": "success",
        "city": city,
        "event_type": event_type,
        "text": response_text,
        "latency_ms": latency_ms,
        "streaming": True,
        "audio_endpoint": "generate_tts"  # Use separate TTS endpoint
    }


# ============================================================================
# BATCH PROCESSING ENDPOINT (MULTIPLE CITIES)
# ============================================================================

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_multi_city_commentary(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate commentary from multiple cities simultaneously.
    
    Useful for panel discussions or comparative analysis.
    
    Expected JSON payload:
    {
        "cities": ["Philadelphia", "Dallas", "New York"],
        "user_input": "Who's winning the NFC East?",
        "game_context": {...}
    }
    
    Returns:
        Dictionary mapping city names to their responses
    """
    cities = request_data.get("cities", [])
    user_input = request_data.get("user_input")
    game_context = request_data.get("game_context", {})
    
    if not cities or not user_input:
        return {
            "error": "Missing required fields: 'cities' (list) and 'user_input'",
            "status": "invalid_request"
        }
    
    # Instantiate agent once (reuse across cities)
    agent = CulturalAgent()
    
    # Generate responses for all cities in parallel
    responses = {}
    for city in cities:
        try:
            result = agent.generate_response.remote(
                city_name=city,
                user_input=user_input,
                game_context=game_context
            )
            responses[city] = result
        except Exception as e:
            responses[city] = {
                "error": str(e),
                "status": "failed"
            }
    
    return {
        "status": "success",
        "responses": responses,
        "cities_count": len(cities)
    }


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.function(image=image)
# @modal.web_endpoint(method="GET")  # DISABLED
def list_cities() -> Dict[str, Any]:
    """
    List all available city profiles.
    
    Returns:
        Dictionary with list of available cities
    """
    agent = CulturalAgent()
    cities = agent.get_city_profile.remote("Kansas City")  # Dummy call to init
    
    factory = AgentFactory(config_path="/root/config/city_profiles.json")
    all_cities = factory.get_all_cities()
    
    return {
        "cities": all_cities,
        "count": len(all_cities),
        "status": "success"
    }


@app.function(image=image)
# @modal.web_endpoint(method="POST")  # DISABLED - Modal 8-endpoint limit
def check_interruption_endpoint(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a city should interrupt another city.
    
    Expected JSON payload:
    {
        "city": "Philadelphia",
        "opponent_confidence": 0.65
    }
    """
    city = request_data.get("city")
    opponent_confidence = request_data.get("opponent_confidence", 0.5)
    
    if not city:
        return {"error": "Missing 'city' field"}
    
    agent = CulturalAgent()
    result = agent.check_interruption.remote(city, opponent_confidence)
    
    return {
        "status": "success",
        "data": result
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
@modal.fastapi_endpoint(method="POST")
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
    import redis
    start_time = time.time()
    
    # Initialize W&B for this function
    init_wandb()
    
    city1 = request_data.get("city1")
    city2 = request_data.get("city2")
    topic = request_data.get("topic")
    rounds = request_data.get("rounds", 1)  # OPTIMIZATION: Default to 1 round for <10s response
    
    # Validation
    if not city1 or not city2:
        return {"error": "Missing 'city1' or 'city2' field"}
    if not topic:
        return {"error": "Missing 'topic' field"}
    if rounds < 1 or rounds > 10:
        return {"error": "Rounds must be between 1 and 10"}
    
    # ------------------------------------------------------------------------
    # CACHE CHECK: Look for prewarmed debate using standardized key format
    # Key format: neuron:debate:{normalized_city}:{topic_hash}
    # ------------------------------------------------------------------------
    try:
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Generate standardized cache key (matches prewarm script)
            normalized_city = city1.lower().replace(" ", "_").replace("(", "").replace(")", "")
            topic_hash = hashlib.md5(topic.lower().encode()).hexdigest()[:12]
            cache_key = f"neuron:debate:{normalized_city}:{topic_hash}"
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                import json as json_module
                cache_result = json_module.loads(cached_data)
                latency_ms = int((time.time() - start_time) * 1000)
                print(f"[CACHE HIT] Debate found for {city1} - {topic[:30]}... ({latency_ms}ms)")
                
                # Return cached response with cache indicator
                cached_response = cache_result.get("response", cache_result)
                if isinstance(cached_response, dict) and "debate" in cached_response:
                    cached_response["debate"]["cached"] = True
                    cached_response["latency_ms"] = latency_ms
                    return cached_response
                else:
                    return {
                        "status": "success",
                        "debate": cached_response if isinstance(cached_response, dict) else {"data": cached_response},
                        "cached": True,
                        "latency_ms": latency_ms
                    }
    except Exception as e:
        print(f"[CACHE] Cache check failed (continuing without cache): {e}")
    
    # Initialize agent
    agent = CulturalAgent()
    
    # Debate transcript
    transcript = []
    conversation_history = []
    
    # OPTIMIZATION: Pre-generate TTS tier setting
    tts_tier = request_data.get("tier", "premium")  # Default to premium for debates
    
    # Run debate rounds with PARALLEL TTS GENERATION
    for round_num in range(1, rounds + 1):
        # --------------------------------------------------------------------
        # CITY 1'S TURN
        # --------------------------------------------------------------------
        # Generate text response
        city1_result = agent.generate_response.remote(
            city_name=city1,
            user_input=topic,
            conversation_history=conversation_history,
            game_context={"opponent": city2, "debate_round": round_num}
        )
        
        # Check for errors in generate_response
        if "error" in city1_result:
            return {
                "status": "error",
                "error": f"City1 generate_response failed: {city1_result.get('error')}",
                "debug": city1_result
            }
        
        city1_response = city1_result.get("response", "")
        
        # OPTIMIZATION: Sanitize markdown for TTS immediately
        import re
        city1_tts_text = city1_response
        city1_tts_text = re.sub(r'\*+', '', city1_tts_text)  # Remove asterisks
        city1_tts_text = re.sub(r'_+', '', city1_tts_text)   # Remove underscores
        city1_tts_text = re.sub(r'\[.*?\]', '', city1_tts_text)  # Remove [brackets]
        city1_tts_text = re.sub(r'\((?:laughs|sighs|pauses|chuckles)\)', '', city1_tts_text, flags=re.IGNORECASE)
        city1_tts_text = re.sub(r'\s+', ' ', city1_tts_text).strip()
        
        # OPTIMIZATION: Generate TTS IMMEDIATELY (don't wait for City 2)
        city1_audio_base64 = None
        try:
            import base64
            city1_audio_bytes = agent.generate_voice_response.remote(
                text=city1_tts_text,
                city_name=city1,
                tier=tts_tier
            )
            city1_audio_base64 = base64.b64encode(city1_audio_bytes).decode('utf-8')
        except Exception as e:
            print(f"[TTS WARNING] City1 audio generation failed: {e}")
            # Continue without audio - text is still valid
        
        # Add to transcript WITH AUDIO
        transcript.append({
            "round": round_num,
            "speaker": city1,
            "response": city1_response,
            "timestamp": int(time.time()),
            "audio_base64": city1_audio_base64,  # OPTIMIZATION: Include audio!
            "has_audio": city1_audio_base64 is not None,
            "tier": tts_tier if city1_audio_base64 else None
        })
        
        # Add to conversation history
        conversation_history.append({
            "role": city1,
            "content": city1_response
        })
        
        # --------------------------------------------------------------------
        # CITY 2'S TURN
        # --------------------------------------------------------------------
        # Generate text response
        city2_result = agent.generate_response.remote(
            city_name=city2,
            user_input=topic,
            conversation_history=conversation_history,
            game_context={"opponent": city1, "debate_round": round_num}
        )
        
        city2_response = city2_result.get("response", "")
        
        # OPTIMIZATION: Sanitize markdown for TTS immediately
        city2_tts_text = city2_response
        city2_tts_text = re.sub(r'\*+', '', city2_tts_text)
        city2_tts_text = re.sub(r'_+', '', city2_tts_text)
        city2_tts_text = re.sub(r'\[.*?\]', '', city2_tts_text)
        city2_tts_text = re.sub(r'\((?:laughs|sighs|pauses|chuckles)\)', '', city2_tts_text, flags=re.IGNORECASE)
        city2_tts_text = re.sub(r'\s+', ' ', city2_tts_text).strip()
        
        # OPTIMIZATION: Generate TTS IMMEDIATELY
        city2_audio_base64 = None
        try:
            import base64
            city2_audio_bytes = agent.generate_voice_response.remote(
                text=city2_tts_text,
                city_name=city2,
                tier=tts_tier
            )
            city2_audio_base64 = base64.b64encode(city2_audio_bytes).decode('utf-8')
        except Exception as e:
            print(f"[TTS WARNING] City2 audio generation failed: {e}")
        
        # Add to transcript WITH AUDIO
        transcript.append({
            "round": round_num,
            "speaker": city2,
            "response": city2_response,
            "timestamp": int(time.time()),
            "audio_base64": city2_audio_base64,  # OPTIMIZATION: Include audio!
            "has_audio": city2_audio_base64 is not None,
            "tier": tts_tier if city2_audio_base64 else None
        })
        
        # Add to conversation history
        conversation_history.append({
            "role": city2,
            "content": city2_response
        })
    
    # Log trace to W&B
    latency_ms = (time.time() - start_time) * 1000
    log_trace(
        endpoint="run_debate",
        latency_ms=latency_ms,
        status="success",
        city=f"{city1} vs {city2}",
        extra={"topic": topic, "rounds": rounds, "total_turns": len(transcript)}
    )
    
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


# ============================================================================
# PERSONALITY TO CITY MAPPING (for Creator Studio)
# ============================================================================

# Maps frontend personality IDs to city profile names
PERSONALITY_MAP = {
    # Built-in personalities (from ControlRoom.tsx)
    "marcus": {"display_name": "Marcus Chen", "city": "San Francisco", "style": "analytical"},
    "bigmike": {"display_name": "Big Mike", "city": "Philadelphia", "style": "aggressive"},
    "zareena": {"display_name": "Zareena Volkov", "city": "New England", "style": "strategic"},
    "sam": {"display_name": "Sam Rodriguez", "city": "Dallas", "style": "conservative"},
    "leo": {"display_name": "Leo Kim", "city": "Kansas City", "style": "explosive"},
    "architect": {"display_name": "The Architect", "city": "New York (Giants)", "style": "philosophical"},
    
    # City-based personalities (direct mapping)
    "philadelphia": {"display_name": "Philly Voice", "city": "Philadelphia", "style": "aggressive"},
    "dallas": {"display_name": "Dallas Voice", "city": "Dallas", "style": "confident"},
    "sanfrancisco": {"display_name": "Bay Area Voice", "city": "San Francisco", "style": "analytical"},
    "kansascity": {"display_name": "KC Voice", "city": "Kansas City", "style": "fast-paced"},
    "newengland": {"display_name": "Patriots Voice", "city": "New England", "style": "strategic"},
    "buffalo": {"display_name": "Bills Voice", "city": "Buffalo", "style": "resilient"},
    "baltimore": {"display_name": "Ravens Voice", "city": "Baltimore", "style": "aggressive"},
    "miami": {"display_name": "Dolphins Voice", "city": "Miami", "style": "energetic"},
    "detroit": {"display_name": "Lions Voice", "city": "Detroit", "style": "resilient"},
    "greenbay": {"display_name": "Packers Voice", "city": "Green Bay", "style": "traditional"},
}


# ============================================================================
# STREAMING DEBATE ENDPOINT (SSE)
# ============================================================================

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("redis-credentials"),
        modal.Secret.from_name("googlecloud-secret"),
    ],
    timeout=120,  # 2 minutes for long debates
)
@modal.fastapi_endpoint(method="POST")
def run_debate_stream(request_data: Dict[str, Any]) -> Any:
    """
    Run a multi-agent debate with Server-Sent Events (SSE) streaming.
    
    This endpoint supports the ControlRoom frontend by:
    1. Receiving debate configuration
    2. Running closed-loop debate (agents respond to each other)
    3. Streaming progress events back in real-time
    
    Request body:
    {
        "topic": str,
        "panel": list[str],           # personality IDs
        "config": {
            "rounds": int,            # How many times each agent speaks
            "tone": {
                "analytical": float,  # 0.0-1.0
                "depth": float,       # 0.0-1.0
                "energy": float       # 0.0-1.0
            },
            "closed_loop": bool       # Always true for debates
        }
    }
    
    Streams SSE events:
    - { type: "progress", speaker: str, progress: float }
    - { type: "turn_complete", segment: {...} }
    - { type: "debate_complete", debate_id: str, segments: [...], full_script: str }
    """
    import time
    import uuid
    from datetime import datetime
    
    # Parse request
    topic = request_data.get("topic", "")
    panel = request_data.get("panel", [])
    config = request_data.get("config", {})
    
    rounds = config.get("rounds", 2)  # OPTIMIZATION: Default to 2 rounds for streaming
    tone = config.get("tone", {"analytical": 0.5, "depth": 0.5, "energy": 0.5})
    closed_loop = config.get("closed_loop", True)
    
    # Validation
    if not topic:
        return {"error": "Missing 'topic' field", "type": "error"}
    if len(panel) < 2:
        return {"error": "Need at least 2 personalities for a debate", "type": "error"}
    if rounds < 1 or rounds > 10:
        return {"error": "Rounds must be between 1 and 10", "type": "error"}
    
    # Resolve personalities to cities
    resolved_panel = []
    for personality_id in panel:
        pid_lower = personality_id.lower()
        if pid_lower in PERSONALITY_MAP:
            resolved_panel.append({
                "id": personality_id,
                **PERSONALITY_MAP[pid_lower]
            })
        else:
            # Try to use as city name directly
            resolved_panel.append({
                "id": personality_id,
                "display_name": personality_id,
                "city": personality_id,
                "style": "balanced"
            })
    
    # Generate debate ID
    debate_id = f"debate_{uuid.uuid4().hex[:12]}"
    
    # Initialize agent
    agent = CulturalAgent()
    
    # Debate state
    segments = []
    conversation_history = []
    total_turns = rounds * len(resolved_panel)
    
    # Helper: Detect intensity from response text
    def detect_intensity(text: str) -> str:
        """Analyze response text for combative intensity."""
        text_lower = text.lower()
        
        # Explosive indicators
        explosive_phrases = [
            "absolutely wrong", "ridiculous", "laughable", "pathetic",
            "delusional", "clown show", "wake up", "embarrassing", "trash"
        ]
        if any(phrase in text_lower for phrase in explosive_phrases):
            return "explosive"
        
        # High intensity indicators
        high_phrases = [
            "completely disagree", "wrong about", "that's absurd",
            "nonsense", "think again", "clearly wrong", "dead wrong"
        ]
        if any(phrase in text_lower for phrase in high_phrases):
            return "high"
        
        # Medium intensity (some disagreement)
        medium_phrases = [
            "i disagree", "not quite", "that's not", "however",
            "but actually", "on the other hand", "let's be real"
        ]
        if any(phrase in text_lower for phrase in medium_phrases):
            return "medium"
        
        return "low"
    
    # Helper: Build debate instruction based on turn
    def get_debate_instruction(turn_index: int, previous_speaker: str = None) -> str:
        """Generate debate instruction based on turn position."""
        if turn_index == 0:
            return f"Open the debate on: '{topic}'. Take a strong position. Be assertive and memorable."
        else:
            return (
                f"Respond directly to {previous_speaker}. "
                f"Challenge their points. Defend your position. "
                f"Be specific - quote what they said and dismantle it."
            )
    
    # Build segments (non-streaming version that returns JSON with all segments)
    # Frontend handles the streaming display
    
    for round_num in range(1, rounds + 1):
        for panel_idx, personality in enumerate(resolved_panel):
            turn_index = (round_num - 1) * len(resolved_panel) + panel_idx
            speaker_id = personality["id"]
            speaker_name = personality["display_name"]
            city_name = personality["city"]
            
            # Calculate progress
            progress = ((turn_index + 1) / total_turns) * 100
            
            # Determine what this agent is responding to
            if len(conversation_history) > 0 and closed_loop:
                # Closed-loop: respond to previous speaker
                previous = conversation_history[-1]
                previous_speaker = previous.get("role", "opponent")
                user_input = previous.get("content", topic)
                instruction = get_debate_instruction(turn_index, previous_speaker)
                responding_to = previous_speaker
            else:
                # First turn: respond to topic
                user_input = topic
                instruction = get_debate_instruction(0)
                responding_to = None
            
            # Add instruction to user input
            full_prompt = f"{instruction}\n\nTOPIC/INPUT: {user_input}"
            
            # Apply tone settings
            temperature = 0.7 + (tone.get("energy", 0.5) * 0.4)  # 0.7-1.1
            
            try:
                # Generate response
                result = agent.generate_response.remote(
                    city_name=city_name,
                    user_input=full_prompt,
                    conversation_history=conversation_history[-6:],  # Last 6 turns for context
                    game_context={
                        "debate_round": round_num,
                        "turn_in_round": panel_idx + 1,
                        "responding_to": responding_to,
                        "tone_analytical": tone.get("analytical", 0.5),
                        "tone_depth": tone.get("depth", 0.5),
                        "tone_energy": tone.get("energy", 0.5)
                    },
                    tier="standard"  # Use standard voice for debates (cost-effective)
                )
                
                response_text = result.get("response", "")
                
                if not response_text:
                    response_text = f"*{speaker_name} pauses thoughtfully* Let me gather my thoughts on that..."
                
            except Exception as e:
                print(f"[DEBATE ERROR] Failed to generate for {speaker_name}: {e}")
                # Retry once with simpler context
                try:
                    result = agent.generate_response.remote(
                        city_name=city_name,
                        user_input=topic,
                        conversation_history=[],  # No history
                        game_context={"debate_round": round_num}
                    )
                    response_text = result.get("response", f"*{speaker_name} responds briefly*")
                except Exception as retry_error:
                    print(f"[DEBATE ERROR] Retry failed: {retry_error}")
                    response_text = f"*{speaker_name} acknowledges the point*"
            
            # Detect intensity
            intensity = detect_intensity(response_text)
            
            # Build segment
            segment = {
                "speaker": speaker_name,
                "speaker_id": speaker_id,
                "text": response_text,
                "responding_to": responding_to,
                "timestamp_ms": int(time.time() * 1000),
                "turn_index": turn_index,
                "round": round_num,
                "intensity": intensity
            }
            
            # ----------------------------------------------------------------
            # TTS AUDIO GENERATION: Convert text to speech
            # ----------------------------------------------------------------
            try:
                # Check if we have ElevenLabs keys available
                if hasattr(agent, 'elevenlabs_keys') and agent.elevenlabs_keys:
                    from elevenlabs.client import ElevenLabs
                    import base64
                    
                    # Initialize ElevenLabs client with first available key
                    client = ElevenLabs(api_key=agent.elevenlabs_keys[0])
                    
                    # Get voice ID for speaker (use city name or default)
                    voice_id = ELEVENLABS_VOICE_MAP.get(speaker_name)
                    if not voice_id:
                        # Fallback to default voice
                        voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam voice as default
                        print(f"[TTS] No voice mapping for {speaker_name}, using default")
                    
                    # Generate audio using ElevenLabs Turbo model (faster)
                    audio_generator = client.text_to_speech.convert(
                        voice_id=voice_id,
                        text=response_text,
                        model_id="eleven_turbo_v2"  # Fast, cost-effective model
                    )
                    
                    # Collect all audio chunks
                    audio_bytes = b"".join(audio_generator)
                    
                    # Encode to base64 for JSON transport
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    segment["audio_base64"] = audio_base64
                    
                    # Log cost for monitoring
                    char_count = len(response_text)
                    log_cost(
                        "elevenlabs", 
                        char_count * COST_PER_CHAR_ELEVENLABS,
                        f"TTS for {speaker_name}, {char_count} chars"
                    )
                    
                    print(f"[TTS SUCCESS] Generated audio for {speaker_name}: {len(audio_base64)} base64 chars")
                else:
                    print(f"[TTS SKIP] No ElevenLabs keys available for {speaker_name}")
                    
            except Exception as tts_error:
                # Don't fail the debate if TTS fails - just log and continue
                print(f"[TTS ERROR] Failed for {speaker_name}: {str(tts_error)[:200]}")
                # Frontend will fallback to text-only or call TTS endpoint directly
            
            segments.append(segment)
            
            # Add to conversation history
            conversation_history.append({
                "role": speaker_name,
                "content": response_text
            })
    
    # Build full script
    full_script_parts = []
    for seg in segments:
        responding_str = f" [responding to {seg['responding_to']}]" if seg['responding_to'] else ""
        full_script_parts.append(f"[{seg['speaker']}]{responding_str}\n{seg['text']}")
    
    full_script = "\n\n---\n\n".join(full_script_parts)
    
    # Return complete debate (frontend handles progressive display)
    return {
        "type": "debate_complete",
        "status": "success",
        "debate_id": debate_id,
        "topic": topic,
        "panel": [p["display_name"] for p in resolved_panel],
        "segments": segments,
        "full_script": full_script,
        "total_turns": len(segments),
        "config": {
            "rounds": rounds,
            "tone": tone,
            "closed_loop": closed_loop
        },
        "created_at": datetime.utcnow().isoformat() + "Z"
    }


# ============================================================================
# SEGMENT REGENERATION ENDPOINT
# ============================================================================

def summarize_history(history: list) -> str:
    """
    Create brief summary of conversation so far.
    Keeps agent aware of debate flow without using full context.
    """
    if not history:
        return "This is the opening statement."
    
    if len(history) <= 3:
        # Short history, include all
        return "\n".join([
            f"- {turn.get('speaker', 'Unknown')}: {turn.get('text', '')[:100]}..."
            for turn in history
        ])
    
    # Longer history, summarize
    speakers = set(t.get('speaker', 'Unknown') for t in history)
    summary_parts = [
        f"The debate opened with {history[0].get('speaker', 'Unknown')} arguing about the topic.",
        f"Key voices: {', '.join(speakers)}",
        f"Most recent: {history[-1].get('speaker', 'Unknown')} said: \"{history[-1].get('text', '')[:150]}...\""
    ]
    
    return "\n".join(summary_parts)


def detect_intensity_level(text: str) -> str:
    """Detect combative intensity from response text."""
    text_lower = text.lower()
    
    explosive_phrases = [
        "absolutely wrong", "ridiculous", "laughable", "pathetic",
        "delusional", "clown show", "wake up", "embarrassing", "trash"
    ]
    if any(phrase in text_lower for phrase in explosive_phrases):
        return "explosive"
    
    high_phrases = [
        "completely disagree", "wrong about", "that's absurd",
        "nonsense", "think again", "clearly wrong", "dead wrong"
    ]
    if any(phrase in text_lower for phrase in high_phrases):
        return "high"
    
    medium_phrases = [
        "i disagree", "not quite", "that's not", "however",
        "but actually", "on the other hand", "let's be real"
    ]
    if any(phrase in text_lower for phrase in medium_phrases):
        return "medium"
    
    return "low"


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("redis-credentials"),
        modal.Secret.from_name("googlecloud-secret"),
    ],
    timeout=60,
)
@modal.fastapi_endpoint(method="POST")
def regenerate_segment(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Regenerate a single debate segment with conversation context.
    
    Used by EditBay when creators want a different take on a specific moment.
    Maintains conversation flow while producing varied content.
    
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
    
    Returns:
    {
        "speaker": str,
        "speaker_id": str,
        "text": str,
        "responding_to": str | null,
        "timestamp_ms": int,
        "intensity": str,
        "regenerated": bool
    }
    """
    import time
    
    # Parse request
    speaker_id = request_data.get("speaker_id", "")
    topic = request_data.get("topic", "")
    history = request_data.get("conversation_history", [])
    responding_to = request_data.get("responding_to")
    tone = request_data.get("tone", {"analytical": 0.5, "depth": 0.5, "energy": 0.5})
    segment_index = request_data.get("segment_index", 0)
    
    # Validate speaker_id
    speaker_id_lower = speaker_id.lower()
    if speaker_id_lower not in PERSONALITY_MAP:
        # Try to use as city name directly
        personality = {
            "id": speaker_id,
            "display_name": speaker_id,
            "city": speaker_id,
            "style": "balanced"
        }
    else:
        personality = {
            "id": speaker_id,
            **PERSONALITY_MAP[speaker_id_lower]
        }
    
    city_name = personality["city"]
    speaker_name = personality["display_name"]
    
    # Initialize agent
    agent = CulturalAgent()
    
    # Determine mode and input
    if responding_to is None:
        # Opening statement
        mode = "opening"
        input_text = topic
        opponent_name = None
        debate_instruction = (
            f"Open the debate on: '{topic}'. "
            f"Take a strong position. Be assertive and memorable. "
            f"This is a FRESH TAKE - try a different angle than usual."
        )
    else:
        # Rebuttal mode
        mode = "debate"
        input_text = responding_to.get("text", topic)
        opponent_name = responding_to.get("speaker", "opponent")
        debate_instruction = (
            f"Respond directly to {opponent_name}. "
            f"Challenge their points. This is a FRESH TAKE - "
            f"try a different approach than you normally would."
        )
    
    # Build conversation summary
    history_summary = summarize_history(history)
    
    # Build variation instruction for regeneration
    variation_instruction = """

=== REGENERATION MODE ===
This is a FRESH TAKE. The creator wants something different.

REQUIREMENTS:
- Take a DIFFERENT angle than you might normally
- Use a DIFFERENT signature phrase or opening
- If you'd normally lead with stats, try leading with a challenge
- If you'd normally be aggressive, try calculated precision
- Same position, different approach
- Surprise the listener with an unexpected angle

Make it feel like an alternate timeline of this debate.
"""
    
    # Build full prompt
    full_prompt = f"{debate_instruction}\n\n"
    full_prompt += f"DEBATE SO FAR:\n{history_summary}\n\n"
    
    if responding_to:
        full_prompt += f"YOU ARE RESPONDING TO:\n\"{input_text[:500]}\"\n\n"
    else:
        full_prompt += f"TOPIC: {topic}\n\n"
    
    full_prompt += variation_instruction
    
    # Build game context for debate mode detection
    game_context = {
        "debate_round": (segment_index // 2) + 1,
        "turn_in_round": (segment_index % 2) + 1,
        "responding_to": opponent_name,
        "tone_analytical": tone.get("analytical", 0.5),
        "tone_depth": tone.get("depth", 0.5),
        "tone_energy": tone.get("energy", 0.5)
    }
    
    # Format history for agent
    formatted_history = [
        {"role": h.get("speaker", "Unknown"), "content": h.get("text", "")}
        for h in history[-6:]  # Last 6 turns for context
    ]
    
    try:
        # Generate response with higher temperature for variation
        result = agent.generate_response.remote(
            city_name=city_name,
            user_input=full_prompt,
            conversation_history=formatted_history,
            game_context=game_context,
            tier="standard"
        )
        
        response_text = result.get("response", "")
        
        if not response_text:
            response_text = f"*{speaker_name} takes a moment to reconsider* Let me approach this differently..."
            
    except Exception as e:
        print(f"[REGEN ERROR] Failed to regenerate for {speaker_name}: {e}")
        # Retry with simpler context
        try:
            result = agent.generate_response.remote(
                city_name=city_name,
                user_input=topic,
                conversation_history=[],
                game_context={"debate_round": 1}
            )
            response_text = result.get("response", f"*{speaker_name} offers a fresh perspective*")
        except Exception as retry_error:
            print(f"[REGEN ERROR] Retry failed: {retry_error}")
            return {
                "error": f"Failed to regenerate segment: {str(retry_error)}",
                "speaker": speaker_name,
                "speaker_id": speaker_id
            }
    
    # Detect intensity
    intensity = detect_intensity_level(response_text)
    
    # Build response segment
    new_segment = {
        "speaker": speaker_name,
        "speaker_id": speaker_id,
        "text": response_text,
        "responding_to": opponent_name,
        "timestamp_ms": int(time.time() * 1000),
        "intensity": intensity,
        "regenerated": True
    }
    
    return new_segment


# ============================================================================
# TTS GENERATION ENDPOINT
# ============================================================================

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_name("elevenlabs-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=60,
)
@modal.fastapi_endpoint(method="POST")
def generate_tts(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate TTS audio for a debate segment.
    
    Uses 75% Google Cloud TTS, 25% ElevenLabs with automatic fallback.
    
    Request:
    {
        "text": str,
        "speaker_id": str,
        "intensity": str (optional: low/medium/high/explosive),
        "force_provider": str (optional: "google" or "elevenlabs")
    }
    
    Response:
    {
        "audio_base64": str,
        "duration_ms": int,
        "format": str,
        "provider": str  # "google" or "elevenlabs"
    }
    """
    import base64
    import os
    import time
    import random
    import tempfile
    start_time = time.time()
    
    # Initialize W&B for this function
    init_wandb()
    
    text = request_data.get("text", "")
    speaker_id = request_data.get("speaker_id", "narrator")
    intensity = request_data.get("intensity", "medium")
    force_provider = request_data.get("force_provider", None)
    
    if not text:
        return {"error": "Text is required"}
    
    if len(text) > 5000:
        return {"error": "Text too long. Maximum 5000 characters."}
    
    # =========================================================================
    # VOICE ROUTER: 75% Google, 25% ElevenLabs
    # =========================================================================
    
    # 6 agents PER DEBATE - assigned by agent number (1-6)
    # Zareena is always agent_6 (female), others are male
    GOOGLE_VOICE_POOL = {
        # Agent-based assignment (for 6 agents per debate)
        "agent_1": "en-US-Studio-Q",      # Male - Deep authoritative
        "agent_2": "en-US-Neural2-J",     # Male - Energetic
        "agent_3": "en-US-Neural2-D",     # Male - Smooth
        "agent_4": "en-US-Neural2-A",     # Male - Clear
        "agent_5": "en-US-News-N",        # Male - Broadcaster
        "agent_6": "en-US-Studio-O",      # Female - Zareena
        
        # City fallbacks (legacy support)
        "seattle": "en-US-Neural2-J",
        "dallas": "en-US-Studio-Q",
        "kansas_city": "en-US-Neural2-D",
        "kansas city": "en-US-Neural2-D",
        "philadelphia": "en-US-Neural2-A",
        "buffalo": "en-US-News-N",
        "zareena": "en-US-Studio-O",
        "minneapolis": "en-US-Studio-O",
        
        # Numbered fallback (agent1, agent2, etc)
        "1": "en-US-Studio-Q",
        "2": "en-US-Neural2-J",
        "3": "en-US-Neural2-D",
        "4": "en-US-Neural2-A",
        "5": "en-US-News-N",
        "6": "en-US-Studio-O",
        
        "default": "en-US-Neural2-J",
    }
    
    # Get voice from speaker_id (supports: "agent_1", "1", "seattle", etc)
    speaker_lower = speaker_id.lower().replace(" ", "_")
    google_voice_selected = GOOGLE_VOICE_POOL.get(speaker_lower, GOOGLE_VOICE_POOL["default"])
    
    # Resolve city name from speaker_id
    # Default: Use speaker_id as city name (for cities not in PERSONALITY_MAP)
    city_name = speaker_id  # Use the actual speaker_id, not hardcoded Kansas City!
    
    # Normalize speaker_id for matching (lowercase, remove spaces)
    speaker_normalized = speaker_id.lower().replace(" ", "").replace("_", "")
    
    for key, data in PERSONALITY_MAP.items():
        key_normalized = key.lower().replace(" ", "").replace("_", "")
        city_value = data.get("city", "")
        city_normalized = city_value.lower().replace(" ", "").replace("_", "")
        display_name_normalized = data.get("display_name", "").lower().replace(" ", "").replace("_", "")
        
        if (speaker_normalized == key_normalized or 
            speaker_normalized == city_normalized or
            speaker_normalized in city_normalized or
            city_normalized in speaker_normalized or
            speaker_normalized == display_name_normalized):
            city_name = city_value  # Use the actual city name from data
            break
    
    print(f"[TTS] Resolved city_name='{city_name}' for speaker_id='{speaker_id}'")
    
    # Select provider: 75% Google, 25% ElevenLabs
    if force_provider:
        selected_provider = force_provider
    else:
        selected_provider = "google" if random.random() < 0.75 else "elevenlabs"
    
    print(f"[TTS] Selected provider: {selected_provider} (75/25 split)")
    
    # =========================================================================
    # TRY SELECTED PROVIDER, FALLBACK IF NEEDED
    # =========================================================================
    
    audio_bytes = None
    provider_used = None
    audio_format = "mp3"
    
    # Define generation functions
    def try_google_tts():
        """Generate with Google Cloud TTS."""
        from google.cloud import texttospeech
        
        # Setup credentials
        gcp_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if gcp_creds:
            creds_path = tempfile.mktemp(suffix=".json")
            with open(creds_path, "w") as f:
                f.write(gcp_creds)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        
        client = texttospeech.TextToSpeechClient()
        
        # Get Google voice - use pre-selected voice from speaker_id
        # Falls back to city-based or default
        voice_to_use = google_voice_selected
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_to_use
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            sample_rate_hertz=24000
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        print(f"[TTS] Google success: voice={voice_to_use}, bytes={len(response.audio_content)}")
        return response.audio_content, "google"
    
    def try_elevenlabs_tts():
        """Generate with ElevenLabs."""
        from elevenlabs.client import ElevenLabs
        
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise Exception("ElevenLabs API key not configured")
        
        voice_id = ELEVENLABS_VOICE_MAP.get(city_name, "pNInz6obpgDQGcFmaJgB")
        
        client = ElevenLabs(api_key=api_key)
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_monolingual_v1",
            output_format="mp3_44100_128"
        )
        
        audio = b"".join(audio_generator)
        print(f"[TTS] ElevenLabs success: voice_id={voice_id}, bytes={len(audio)}")
        return audio, "elevenlabs"
    
    # Try selected provider first, then fallback
    providers_to_try = []
    if selected_provider == "google":
        providers_to_try = [("google", try_google_tts), ("elevenlabs", try_elevenlabs_tts)]
    else:
        providers_to_try = [("elevenlabs", try_elevenlabs_tts), ("google", try_google_tts)]
    
    errors = []
    for provider_name, provider_func in providers_to_try:
        try:
            audio_bytes, provider_used = provider_func()
            if audio_bytes:
                break
        except Exception as e:
            errors.append(f"{provider_name}: {str(e)[:100]}")
            print(f"[TTS] {provider_name} failed: {e}")
    
    if not audio_bytes:
        return {"error": f"All TTS providers failed: {'; '.join(errors)}"}
    
    # =========================================================================
    # RESPONSE
    # =========================================================================
    
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    word_count = len(text.split())
    duration_ms = int((word_count / 2.5) * 1000)
    
    latency_ms = (time.time() - start_time) * 1000
    print(f"[TTS SUCCESS] provider={provider_used}, bytes={len(audio_bytes)}, latency={latency_ms:.0f}ms")
    
    # Log trace to W&B
    log_trace(
        endpoint="generate_tts",
        latency_ms=latency_ms,
        status="success",
        city=city_name,
        extra={
            "text_length": len(text),
            "audio_bytes": len(audio_bytes),
            "provider": provider_used,
            "selected_provider": selected_provider,
            "fallback_used": provider_used != selected_provider
        }
    )
    
    return {
        "audio_base64": audio_base64,
        "duration_ms": duration_ms,
        "format": audio_format,
        "provider": provider_used,
        "tier_used": "premium" if provider_used == "elevenlabs" else "google",
        "speaker_id": speaker_id,
        "city": city_name
    }


# ============================================================================
# ASK WITH VOICE - Combined LLM + TTS (Internal function - call via .remote())
# ============================================================================

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_name("elevenlabs-secret"),
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=120,
)
# NOTE: Not a web endpoint to stay within Modal's 8-endpoint limit
# Call via Modal SDK: ask_with_voice.remote(request_data)
def ask_with_voice(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combined endpoint: Question → Gemini → TTS → Audio
    
    Takes a question, generates response with Gemini, converts to speech.
    
    Request:
    {
        "question": str,          # Question to answer
        "agent_id": str,          # "agent_1" through "agent_6" (agent_6 = Zareena/female)
        "context": str,           # Optional game/player context
        "max_words": int,         # Optional max response length (default 100)
        "force_tts_provider": str # Optional: "google" or "elevenlabs"
    }
    
    Response:
    {
        "text": str,              # Generated text response
        "audio_base64": str,      # Base64 encoded audio
        "duration_ms": int,
        "agent_id": str,
        "tts_provider": str
    }
    """
    import base64
    import os
    import time
    import random
    import tempfile
    
    start_time = time.time()
    init_wandb()
    
    question = request_data.get("question", "")
    agent_id = request_data.get("agent_id", "agent_1")
    context = request_data.get("context", "")
    max_words = request_data.get("max_words", 100)
    force_tts_provider = request_data.get("force_tts_provider", None)
    
    if not question:
        return {"error": "Question is required"}
    
    print(f"[ASK] Question: '{question[:80]}...', Agent: {agent_id}")
    
    # =========================================================================
    # STEP 1: Generate response with Gemini
    # =========================================================================
    
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "Gemini API key not configured"}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Build prompt
        system_prompt = f"""You are a fantasy football expert sportscaster (Agent {agent_id}).
Give concise, expert analysis. Be enthusiastic but informative.
Keep your response under {max_words} words."""

        if context:
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {question}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {question}"
        
        response = model.generate_content(full_prompt)
        generated_text = response.text.strip()
        
        print(f"[ASK] Generated {len(generated_text)} chars")
        
    except Exception as e:
        print(f"[ASK ERROR] Gemini failed: {e}")
        return {"error": f"LLM generation failed: {str(e)}"}
    
    # =========================================================================
    # STEP 2: Convert to speech with Google TTS / ElevenLabs
    # =========================================================================
    
    # Voice mapping for 6 agents
    GOOGLE_VOICE_MAP = {
        "agent_1": "en-US-Studio-Q",
        "agent_2": "en-US-Neural2-J",
        "agent_3": "en-US-Neural2-D",
        "agent_4": "en-US-Neural2-A",
        "agent_5": "en-US-News-N",
        "agent_6": "en-US-Studio-O",  # Female - Zareena
    }
    
    ELEVENLABS_VOICE_MAP_LOCAL = {
        "agent_1": "pNInz6obpgDQGcFmaJgB",
        "agent_2": "21m00Tcm4TlvDq8ikWAM",
        "agent_3": "EXAVITQu4vr4xnSDxMaL",
        "agent_4": "VR6AewLTigWG4xSOukaG",
        "agent_5": "yoZ06aMxZJJ28mfd3POQ",
        "agent_6": "21m00Tcm4TlvDq8ikWAM",
    }
    
    # Select provider: 75% Google, 25% ElevenLabs
    if force_tts_provider:
        selected_provider = force_tts_provider
    else:
        selected_provider = "google" if random.random() < 0.75 else "elevenlabs"
    
    audio_bytes = None
    provider_used = None
    
    def try_google():
        from google.cloud import texttospeech
        
        gcp_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if gcp_creds:
            creds_path = tempfile.mktemp(suffix=".json")
            with open(creds_path, "w") as f:
                f.write(gcp_creds)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        
        client = texttospeech.TextToSpeechClient()
        voice_name = GOOGLE_VOICE_MAP.get(agent_id, "en-US-Neural2-J")
        
        synthesis_input = texttospeech.SynthesisInput(text=generated_text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            sample_rate_hertz=24000
        )
        
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content, "google"
    
    def try_elevenlabs():
        from elevenlabs.client import ElevenLabs
        
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise Exception("ElevenLabs API key not configured")
        
        voice_id = ELEVENLABS_VOICE_MAP_LOCAL.get(agent_id, "pNInz6obpgDQGcFmaJgB")
        client = ElevenLabs(api_key=api_key)
        
        audio_gen = client.text_to_speech.convert(
            text=generated_text,
            voice_id=voice_id,
            model_id="eleven_monolingual_v1",
            output_format="mp3_44100_128"
        )
        return b"".join(audio_gen), "elevenlabs"
    
    # Try with fallback
    providers = [(selected_provider, try_google if selected_provider == "google" else try_elevenlabs),
                 ("elevenlabs" if selected_provider == "google" else "google", 
                  try_elevenlabs if selected_provider == "google" else try_google)]
    
    for name, func in providers:
        try:
            audio_bytes, provider_used = func()
            if audio_bytes:
                break
        except Exception as e:
            print(f"[ASK] TTS {name} failed: {e}")
    
    if not audio_bytes:
        return {"error": "TTS generation failed", "text": generated_text}
    
    # =========================================================================
    # RESPONSE
    # =========================================================================
    
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    word_count = len(generated_text.split())
    duration_ms = int((word_count / 2.5) * 1000)
    
    total_latency = (time.time() - start_time) * 1000
    print(f"[ASK SUCCESS] agent={agent_id}, provider={provider_used}, latency={total_latency:.0f}ms")
    
    log_trace(
        endpoint="ask_with_voice",
        latency_ms=total_latency,
        status="success",
        city=agent_id,
        extra={"question_len": len(question), "response_len": len(generated_text), "provider": provider_used}
    )
    
    return {
        "text": generated_text,
        "audio_base64": audio_base64,
        "duration_ms": duration_ms,
        "format": "mp3",
        "agent_id": agent_id,
        "tts_provider": provider_used,
        "latency_ms": int(total_latency)
    }


# ============================================================================
# STYLE ANALYSIS & PERSONALITY GENERATION
# ============================================================================

def convert_analysis_to_personality(analysis: dict, name: str, sample_count: int) -> dict:
    """
    Convert style analysis into personality config compatible with agent system.
    
    Args:
        analysis: Style analysis from AI
        name: Personality name
        sample_count: Number of samples analyzed
    
    Returns:
        Personality config dict
    """
    evidence = analysis.get("evidence_style", {})
    emotional = analysis.get("emotional_profile", {})
    argumentation = analysis.get("argumentation", {})
    tempo = analysis.get("tempo_indicators", {})
    signature = analysis.get("signature_elements", {})
    vocab = analysis.get("vocabulary", {})
    
    # Map tempo to delay
    tempo_map = {"rapid": 120, "moderate": 200, "deliberate": 300}
    base_delay = tempo_map.get(tempo.get("pace", "moderate"), 200)
    
    # Determine archetype from argumentation style
    style_to_archetype = {
        "aggressive": "hot_take",
        "measured": "analyst",
        "diplomatic": "diplomat",
        "provocative": "contrarian"
    }
    archetype = style_to_archetype.get(argumentation.get("style", "measured"), "analyst")
    
    # Determine conflict mode
    hot_take_tendency = emotional.get("hot_take_tendency", 0)
    if hot_take_tendency > 0.6:
        conflict_mode = "aggressive"
    elif argumentation.get("acknowledges_counterpoints", False):
        conflict_mode = "bridge_builder"
    else:
        conflict_mode = "defensive"
    
    return {
        "name": name.lower().replace(" ", "_"),
        "display_name": name,
        "archetype": archetype,
        "source": "style_capture",
        "sample_count": sample_count,
        
        "cognitive_params": {
            "base_delay_ms": base_delay,
            "evidence_weights": {
                "advanced_stats": evidence.get("advanced_stats", 0.25),
                "historical": evidence.get("historical_references", 0.25),
                "film_study": evidence.get("film_study", 0.25),
                "narrative": evidence.get("anecdotes", 0.25)
            }
        },
        
        "voice_profile": {
            "signature_phrases": signature.get("catchphrases", [])[:5],
            "opening_patterns": signature.get("opening_patterns", []),
            "closing_patterns": signature.get("closing_patterns", []),
            "rhetorical_style": "data_driven" if evidence.get("advanced_stats", 0) > 0.5 else "anecdotal",
            "conflict_mode": conflict_mode
        },
        
        "linguistic_markers": {
            "vocabulary_complexity": vocab.get("complexity_level", "moderate"),
            "unique_phrases": vocab.get("unique_phrases", []),
            "power_words": vocab.get("power_words", []),
            "question_frequency": analysis.get("sentence_patterns", {}).get("question_frequency", 0.3)
        },
        
        "emotional_params": {
            "baseline_energy": emotional.get("baseline_energy", 0.5),
            "humor_frequency": emotional.get("humor_frequency", 0.3),
            "sarcasm_level": emotional.get("sarcasm_level", 0.2)
        },
        
        "tts_config": {
            "speed": 1.1 if tempo.get("pace") == "rapid" else (0.9 if tempo.get("pace") == "deliberate" else 1.0),
            "stability": 0.6 if emotional.get("baseline_energy", 0.5) > 0.6 else 0.75
        }
    }


def calculate_confidence(samples: list, analysis: dict) -> float:
    """
    Calculate confidence score based on sample quantity and diversity.
    
    Args:
        samples: List of content samples
        analysis: Style analysis result
    
    Returns:
        Confidence score 0.0-1.0
    """
    sample_count = len(samples)
    source_types = len(set(s.get("type") for s in samples))
    total_words = sum(len(s.get("content", "").split()) for s in samples)
    
    # Base score from quantity
    quantity_score = min(sample_count / 10, 1.0) * 0.4
    
    # Diversity score
    diversity_score = min(source_types / 3, 1.0) * 0.3
    
    # Word count score
    word_score = min(total_words / 5000, 1.0) * 0.3
    
    return round(quantity_score + diversity_score + word_score, 2)



@app.function(
    image=image, 
    timeout=300,
    secrets=[modal.Secret.from_name("gcp-vertex-ai")]
)
@modal.fastapi_endpoint(method="POST")
def analyze_style_samples(request: dict):
    """
    Analyze content samples to extract style markers and generate personality profile.
    
    Extracts:
    - Vocabulary patterns (frequent words, unique phrases)
    - Sentence structure (length, complexity)
    - Evidence preferences (stats vs anecdotes vs film)
    - Emotional tone (intensity, humor, sarcasm)
    - Signature phrases and catchphrases
    - Rhetorical patterns (how they build arguments)
    - Speech tempo indicators
    
    Args:
        request: dict with 'samples' list and 'personality_name' string
    
    Returns:
        dict with 'analysis', 'personality_config', 'confidence_score', 'sample_stats'
    """
    import json
    import re
    from google.cloud import aiplatform
    from vertexai.generative_models import GenerativeModel, Part, Content
    
    print("[STYLE ANALYSIS] Starting style analysis request")
    
    samples = request.get("samples", [])
    personality_name = request.get("personality_name", "Custom")
    
    if not samples:
        return {"error": "No samples provided"}
    
    # Combine all content
    all_content = "\n\n---\n\n".join([
        s.get("content", "") for s in samples if s.get("content")
    ])
    
    total_words = len(all_content.split())
    print(f"[STYLE ANALYSIS] Analyzing {len(samples)} samples, {total_words} total words")
    
    if total_words < 300:
        return {"error": "Insufficient content for analysis. Need at least 300 words."}
    
    # Limit content to avoid token overflow
    content_sample = all_content[:15000]
    
    # Style analysis prompt
    analysis_prompt = f"""Analyze the following content samples from a sports commentator. Extract their unique style markers.

CONTENT SAMPLES:
{content_sample}

ANALYZE AND RETURN ONLY VALID JSON (no markdown, no explanation):
{{
  "vocabulary": {{
    "complexity_level": "simple|moderate|advanced",
    "sports_jargon_density": 0.5,
    "unique_phrases": ["phrase1", "phrase2", "phrase3"],
    "filler_words": ["like", "you know"],
    "power_words": ["dominant", "elite"]
  }},
  
  "sentence_patterns": {{
    "avg_length": "short|medium|long",
    "structure_preference": "simple|compound|complex",
    "question_frequency": 0.3,
    "exclamation_frequency": 0.2
  }},
  
  "evidence_style": {{
    "advanced_stats": 0.5,
    "traditional_stats": 0.5,
    "film_study": 0.3,
    "historical_references": 0.3,
    "anecdotes": 0.4,
    "insider_sources": 0.2
  }},
  
  "emotional_profile": {{
    "baseline_energy": 0.6,
    "humor_frequency": 0.3,
    "sarcasm_level": 0.2,
    "hot_take_tendency": 0.4,
    "empathy_markers": 0.3
  }},
  
  "argumentation": {{
    "style": "aggressive|measured|diplomatic|provocative",
    "acknowledges_counterpoints": true,
    "uses_hypotheticals": true,
    "appeals_to_authority": true,
    "data_visualization_mentions": false
  }},
  
  "signature_elements": {{
    "catchphrases": ["phrase1", "phrase2"],
    "opening_patterns": ["how they typically start"],
    "closing_patterns": ["how they typically end"],
    "transition_phrases": ["moving on", "but here's the thing"]
  }},
  
  "tempo_indicators": {{
    "pace": "rapid|moderate|deliberate",
    "pause_usage": "frequent|occasional|rare",
    "emphasis_style": "volume|repetition|silence"
  }}
}}

Be specific and accurate. These markers will be used to generate an AI personality.
Return ONLY the JSON object, no other text."""

    try:
        # Get GCP credentials from environment (injected by Modal secret)
        import os
        import base64
        import tempfile
        
        credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        project_id = os.environ.get("GCP_PROJECT_ID", "leafy-sanctuary-476515-t2")
        
        if credentials_json:
            # Decode base64 and write credentials to temp file
            creds_data = base64.b64decode(credentials_json)
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as f:
                f.write(creds_data)
                creds_path = f.name
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        
        # Initialize Vertex AI with proper credentials  
        aiplatform.init(project=project_id, location="us-central1")
        model = GenerativeModel("gemini-2.0-flash-001")
        
        # Generate analysis
        response = model.generate_content(
            analysis_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2000
            }
        )
        
        analysis_text = response.text.strip()
        print(f"[STYLE ANALYSIS] Raw response length: {len(analysis_text)}")
        
        # Parse JSON from response
        try:
            # Try direct parse first
            style_analysis = json.loads(analysis_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
            if json_match:
                style_analysis = json.loads(json_match.group())
            else:
                print(f"[STYLE ANALYSIS ERROR] Could not parse JSON: {analysis_text[:500]}")
                return {"error": "Failed to parse style analysis"}
        
        # Convert analysis to personality config
        personality_config = convert_analysis_to_personality(
            analysis=style_analysis,
            name=personality_name,
            sample_count=len(samples)
        )
        
        # Calculate confidence
        confidence = calculate_confidence(samples, style_analysis)
        
        print(f"[STYLE ANALYSIS SUCCESS] Generated profile for '{personality_name}', confidence={confidence}")
        
        return {
            "analysis": style_analysis,
            "personality_config": personality_config,
            "confidence_score": confidence,
            "sample_stats": {
                "count": len(samples),
                "total_words": total_words,
                "sources": list(set(s.get("type") for s in samples))
            }
        }
        
    except Exception as e:
        print(f"[STYLE ANALYSIS ERROR] {str(e)}")
        return {"error": f"Style analysis failed: {str(e)}"}


# ============================================================================
# WHISPER AUDIO TRANSCRIPTION
# ============================================================================

# Define a separate image with Whisper dependencies
whisper_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi",
        "openai-whisper",
        "torch",
        "torchaudio",
        "requests"
    )
)


@app.function(
    image=whisper_image,
    gpu="T4",  # Whisper runs well on T4
    timeout=600,  # 10 min timeout for long audio
)
# @modal.web_endpoint(method="POST")  # DISABLED
def transcribe_audio(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transcribe audio file using OpenAI Whisper.
    
    Accepts audio URL, downloads, transcribes, returns text.
    
    Request body:
    {
        "audio_url": "https://...",
        "sample_id": "optional-tracking-id"
    }
    
    Returns:
    {
        "transcript": "...",
        "duration": 120.5,
        "language": "en",
        "word_count": 1500,
        "sample_id": "..."
    }
    """
    import whisper
    import requests
    import tempfile
    import os as local_os
    
    audio_url = request_data.get("audio_url")
    sample_id = request_data.get("sample_id", "unknown")
    
    if not audio_url:
        return {"error": "No audio URL provided"}
    
    try:
        # Download audio file
        print(f"[WHISPER] Downloading audio from {audio_url}")
        response = requests.get(audio_url, timeout=60)
        response.raise_for_status()
        
        # Determine file extension from content type
        content_type = response.headers.get("content-type", "audio/mpeg")
        ext = ".mp3"
        if "wav" in content_type:
            ext = ".wav"
        elif "m4a" in content_type or "mp4" in content_type:
            ext = ".m4a"
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        # Load Whisper model
        print("[WHISPER] Loading Whisper model (base)...")
        model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
        
        # Transcribe
        print("[WHISPER] Transcribing...")
        result = model.transcribe(
            tmp_path,
            language=None,  # Auto-detect
            task="transcribe",
            verbose=False
        )
        
        # Clean up
        local_os.unlink(tmp_path)
        
        transcript = result.get("text", "").strip()
        language = result.get("language", "en")
        
        # Estimate duration from segments
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0
        
        print(f"[WHISPER] Transcription complete: {len(transcript)} chars, {duration:.1f}s, lang={language}")
        
        return {
            "transcript": transcript,
            "duration": duration,
            "language": language,
            "word_count": len(transcript.split()),
            "sample_id": sample_id
        }
        
    except requests.RequestException as e:
        print(f"[WHISPER ERROR] Download failed: {e}")
        return {"error": f"Failed to download audio: {str(e)}"}
    except Exception as e:
        print(f"[WHISPER ERROR] Transcription failed: {e}")
        return {"error": f"Transcription failed: {str(e)}"}


# ============================================================================
# TWITTER TWEET EXTRACTION
# ============================================================================

# Image with BeautifulSoup for HTML parsing
scraping_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "requests",
        "beautifulsoup4",
        "lxml"
    )
)


@app.function(
    image=scraping_image,
    timeout=120
)
# @modal.web_endpoint(method="POST")  # DISABLED
def extract_tweets(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract tweets from a Twitter/X user for style analysis.
    
    Uses Nitter mirrors (Twitter frontend) as they don't require API keys.
    Falls back across multiple instances if one fails.
    
    Request body:
    {
        "username": "sportscaster123",
        "count": 50,
        "include_replies": false
    }
    
    Returns:
    {
        "tweets": [{"text": "...", "length": 140}, ...],
        "profile": {"username": "...", "name": "...", "bio": "..."},
        "totalExtracted": 50,
        "source": "nitter"
    }
    """
    import requests
    from bs4 import BeautifulSoup
    
    username = request_data.get("username", "").strip().lstrip("@")
    count = min(request_data.get("count", 50), 200)  # Cap at 200
    include_replies = request_data.get("include_replies", False)
    
    if not username:
        return {"error": "No username provided"}
    
    tweets = []
    profile = None
    
    # Try multiple Nitter instances (Twitter frontend mirrors)
    nitter_instances = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.woodland.cafe",
        "https://nitter.net",
        "https://nitter.cz",
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for instance in nitter_instances:
        try:
            # Fetch user page
            url = f"{instance}/{username}"
            print(f"[TWEETS] Trying {url}")
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                
                # Extract profile info
                profile_name = soup.select_one(".profile-card-fullname")
                profile_bio = soup.select_one(".profile-bio")
                
                profile = {
                    "username": username,
                    "name": profile_name.get_text(strip=True) if profile_name else username,
                    "bio": profile_bio.get_text(strip=True) if profile_bio else ""
                }
                
                # Extract tweets
                tweet_elements = soup.select(".timeline-item .tweet-content")
                
                for tweet_el in tweet_elements:
                    if len(tweets) >= count:
                        break
                        
                    tweet_text = tweet_el.get_text(strip=True)
                    
                    # Skip replies if not wanted
                    if not include_replies:
                        parent = tweet_el.find_parent(".timeline-item")
                        if parent and parent.select_one(".replying-to"):
                            continue
                    
                    # Skip very short tweets
                    if len(tweet_text) > 20:
                        tweets.append({
                            "text": tweet_text,
                            "length": len(tweet_text)
                        })
                
                if tweets:
                    print(f"[TWEETS] Extracted {len(tweets)} tweets from {instance}")
                    break  # Success, exit loop
                    
        except requests.RequestException as e:
            print(f"[TWEETS] {instance} failed: {e}")
            continue
        except Exception as e:
            print(f"[TWEETS] Error with {instance}: {e}")
            continue
    
    if not tweets:
        return {
            "error": "Could not fetch tweets. Twitter may be blocking requests. Try again later or paste tweets manually.",
            "suggestion": "Copy some tweets directly and use the 'Paste Text' option instead."
        }
    
    return {
        "tweets": tweets,
        "profile": profile,
        "totalExtracted": len(tweets),
        "source": "nitter"
    }


# ============================================================================
# FINE-TUNING STATUS ENDPOINT
# ============================================================================

# NOTE: Not a web endpoint to stay within Modal's 8-endpoint limit
# Call via generate_commentary or other endpoints that need tuning status
@app.function(image=image, timeout=60, secrets=[modal.Secret.from_name("googlecloud-secret")])
def check_fine_tuning_status(request: dict):
    """
    Check the status of fine-tuning jobs and list available tuned models.
    
    Request:
        {
            "action": "status" | "list_models" | "get_model",
            "job_name": "optional job name for status check",
            "personality_id": "optional personality ID for model lookup"
        }
    
    Returns:
        Status information or list of tuned models
    """
    from google.cloud import aiplatform
    import os
    import json
    import tempfile
    
    action = request.get("action", "list_models")
    
    # Set up GCP credentials
    gcp_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if gcp_creds:
        creds_path = tempfile.mktemp(suffix=".json")
        with open(creds_path, "w") as f:
            f.write(gcp_creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    
    project_id = os.environ.get("GCP_PROJECT_ID", "leafy-sanctuary-476515-t2")
    
    try:
        aiplatform.init(project=project_id, location="us-central1")
        
        if action == "list_models":
            # List all tuned models
            models = aiplatform.Model.list()
            tuned_models = [
                {
                    "name": m.display_name,
                    "resource_name": m.resource_name,
                    "created": str(m.create_time) if hasattr(m, 'create_time') else None
                }
                for m in models
                if "sportscaster" in (m.display_name or "").lower()
            ]
            
            return {
                "status": "success",
                "tuned_models": tuned_models,
                "count": len(tuned_models)
            }
        
        elif action == "get_model":
            # Get specific model for personality
            personality_id = request.get("personality_id", "")
            models = aiplatform.Model.list()
            
            for m in models:
                if personality_id in (m.display_name or ""):
                    return {
                        "status": "success",
                        "model": {
                            "name": m.display_name,
                            "resource_name": m.resource_name,
                            "available": True
                        }
                    }
            
            return {
                "status": "success",
                "model": None,
                "message": f"No tuned model found for personality: {personality_id}"
            }
        
        elif action == "training_stats":
            # Return training data collection stats
            # This would query Supabase for training_examples counts
            return {
                "status": "success",
                "message": "Query training_examples table in Supabase for stats",
                "minimum_required": 100,
                "recommendation": "Collect at least 100 rated examples before fine-tuning"
            }
        
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "valid_actions": ["list_models", "get_model", "training_stats"]
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# STYLE CAPTURE v2: UNIFIED CONTENT EXTRACTION ENDPOINT
# ============================================================================

@app.function(image=image, timeout=300)
# @modal.web_endpoint(method="POST")  # DISABLED - Modal 8-endpoint limit (enable after TNF)
def content_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified content extraction endpoint for Style Capture v2.
    
    Consolidates youtube, article, and audio extraction to stay within
    Modal's 8 endpoint limit.
    
    Request:
        { "action": "youtube" | "article" | "audio", ...params }
    
    Actions:
        - youtube: { "action": "youtube", "url": "..." or "videoId": "..." }
        - article: { "action": "article", "url": "..." }
        - audio: { "action": "audio", "audio_url": "...", "sample_id": "..." }
    """
    action = data.get("action", "").lower()
    
    if action == "youtube":
        return _extract_youtube(data)
    elif action == "article":
        return _extract_article(data)
    elif action == "audio":
        return _transcribe_audio(data)
    else:
        return {
            "error": f"Unknown action: {action}",
            "valid_actions": ["youtube", "article", "audio"]
        }


def _extract_youtube(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract transcript from YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re
        
        url = data.get("url", "")
        video_id = data.get("videoId", "")
        
        if not video_id and url:
            patterns = [
                r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
                r'(?:embed/)([0-9A-Za-z_-]{11})',
                r'(?:youtu\.be/)([0-9A-Za-z_-]{11})'
            ]
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    break
        
        if not video_id:
            return {"error": "Could not extract video ID from URL"}
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([segment['text'] for segment in transcript_list])
        
        duration = 0
        if transcript_list:
            last_segment = transcript_list[-1]
            duration = int(last_segment.get('start', 0) + last_segment.get('duration', 0))
        
        return {
            "transcript": full_transcript,
            "title": f"YouTube Video: {video_id}",
            "duration": duration,
            "videoId": video_id
        }
        
    except Exception as e:
        error_msg = str(e)
        if "TranscriptsDisabled" in error_msg:
            return {"error": "Transcripts are disabled for this video"}
        elif "NoTranscriptFound" in error_msg:
            return {"error": "No transcript available for this video"}
        elif "VideoUnavailable" in error_msg:
            return {"error": "Video is unavailable or private"}
        else:
            return {"error": f"YouTube extraction failed: {error_msg}"}


def _extract_article(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract readable text content from article URL."""
    try:
        from newspaper import Article
        
        url = data.get("url", "")
        if not url:
            return {"error": "URL is required"}
        
        article = Article(url)
        article.download()
        article.parse()
        
        content = article.text
        title = article.title
        
        if not content or len(content) < 50:
            return {"error": "Could not extract meaningful content from this URL"}
        
        return {
            "content": content,
            "title": title or url,
            "url": url,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None
        }
        
    except Exception as e:
        return {"error": f"Article extraction failed: {str(e)}"}


def _transcribe_audio(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe audio from URL using OpenAI Whisper API."""
    try:
        import urllib.request
        import tempfile
        import os
        import openai
        
        audio_url = data.get("audio_url", "")
        sample_id = data.get("sample_id", "unknown")
        
        if not audio_url:
            return {"error": "audio_url is required"}
        
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return {"error": "OpenAI API key not configured for audio transcription"}
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            urllib.request.urlretrieve(audio_url, tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            client = openai.OpenAI(api_key=openai_key)
            
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            return {
                "transcript": transcription.text,
                "duration": int(transcription.duration) if hasattr(transcription, 'duration') else 0,
                "language": transcription.language if hasattr(transcription, 'language') else "en",
                "sample_id": sample_id
            }
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        return {"error": f"Audio transcription failed: {str(e)}"}


# ============================================================================
# LIVE COMMENTARY AGENT FUNCTIONS
# ============================================================================

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
    ],
    cpu=0.5,
    memory=256,
    timeout=2,      # 2s max - fast reactions
    retries=0       # No retries for live - too slow
)
def run_reaction_agent(event: dict, region: str, agent_type: str, params: dict) -> dict:
    """
    Fast reaction agent for immediate events (sub-150ms target).
    
    Used for: touchdowns, turnovers, injuries, big plays.
    Minimal reasoning, maximum speed.
    
    Args:
        event: Classified event dict with type, description, players, etc.
        region: Regional profile (e.g., "dallas", "kansas_city")
        agent_type: Agent archetype ("homer", "analyst", etc.)
        params: Cultural parameters from regional config
    
    Returns:
        dict with text, emotion, confidence, latency_ms
    """
    import time
    start = time.time()
    
    try:
        from vertexai.generative_models import GenerativeModel
        import vertexai
        
        # Fast init - use Gemini Flash for speed
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "neuron-systems"))
        model = GenerativeModel("gemini-1.5-flash")
        
        # Short, punchy prompt for reactions
        event_desc = event.get("description", str(event))[:200]
        event_type = event.get("event_type", "play")
        
        reaction_style = params.get("reaction_style", "energetic and passionate")
        emotional_intensity = params.get("emotional_intensity", "high")
        
        prompt = f"""You are a {region.replace('_', ' ')} sports commentator reacting LIVE to:
{event_desc}

Event type: {event_type}
Your role: {agent_type}
Regional style: {reaction_style}
Emotional level: {emotional_intensity}

Give a 1-2 sentence immediate reaction. Be authentic to your region. Be quick and punchy."""

        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 60,
                "temperature": 0.9
            }
        )
        
        text = response.text.strip()
        
        # Detect emotion from response
        emotion = "excited"
        if any(w in text.lower() for w in ["no", "disaster", "terrible", "hurt"]):
            emotion = "concerned"
        elif any(w in text.lower() for w in ["interesting", "note", "look"]):
            emotion = "analytical"
        
        latency = int((time.time() - start) * 1000)
        
        return {
            "text": text,
            "emotion": emotion,
            "confidence": 0.85,
            "latency_ms": latency
        }
        
    except Exception as e:
        return {
            "text": f"Exciting play there from {region}!",
            "emotion": "excited",
            "confidence": 0.5,
            "latency_ms": int((time.time() - start) * 1000),
            "error": str(e)
        }


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
    ],
    cpu=1.0,
    memory=512,
    timeout=5,      # 5s max - can think more
    retries=0
)
def run_analysis_agent(event: dict, region: str, agent_type: str, params: dict) -> dict:
    """
    Strategic analysis agent for non-urgent events (sub-500ms target).
    
    Used for: penalties, drive ends, timeouts, stats milestones.
    Can reference history, stats, context.
    
    Args:
        event: Classified event dict
        region: Regional profile
        agent_type: Agent archetype ("historian", "contrarian", "stats_expert")
        params: Cultural parameters
    
    Returns:
        dict with text, emotion, confidence, latency_ms
    """
    import time
    start = time.time()
    
    try:
        from vertexai.generative_models import GenerativeModel
        import vertexai
        
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "neuron-systems"))
        model = GenerativeModel("gemini-1.5-flash")
        
        event_desc = event.get("description", str(event))[:300]
        event_type = event.get("event_type", "play")
        players = event.get("players", [])
        teams = event.get("teams", [])
        fantasy_impact = event.get("fantasy_impact", 0.5)
        
        analysis_style = params.get("analysis_style", "balanced and insightful")
        memory_years = params.get("memory_years", 5)
        technical_level = params.get("technical_level", "medium")
        
        # Build context for analysis
        context_parts = []
        if players:
            context_parts.append(f"Players involved: {', '.join(players[:3])}")
        if teams:
            context_parts.append(f"Teams: {', '.join(teams[:2])}")
        if fantasy_impact > 0.6:
            context_parts.append("High fantasy impact")
        
        context = ". ".join(context_parts) if context_parts else ""
        
        prompt = f"""You are a {region.replace('_', ' ')} sports analyst providing analysis on:
{event_desc}

Event type: {event_type}
{context}

Your role: {agent_type}
Regional perspective: {analysis_style}
Historical memory: {memory_years} years
Technical depth: {technical_level}

Provide 2-3 sentences of analysis from your regional perspective.
Reference relevant history or stats if applicable.
For fantasy football context, note any lineup implications."""

        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 120,
                "temperature": 0.7
            }
        )
        
        text = response.text.strip()
        
        # Detect emotion
        emotion = "analytical"
        if any(w in text.lower() for w in ["historically", "years ago", "remember"]):
            emotion = "reflective"
        elif any(w in text.lower() for w in ["disagree", "actually", "but"]):
            emotion = "contrarian"
        elif any(w in text.lower() for w in ["great", "excellent", "impressive"]):
            emotion = "impressed"
        
        latency = int((time.time() - start) * 1000)
        
        return {
            "text": text,
            "emotion": emotion,
            "confidence": 0.8,
            "latency_ms": latency
        }
        
    except Exception as e:
        return {
            "text": f"Interesting development there for {region}.",
            "emotion": "analytical",
            "confidence": 0.5,
            "latency_ms": int((time.time() - start) * 1000),
            "error": str(e)
        }


# ============================================================================
# CONSOLIDATED DASHBOARD API (combines metrics, cache, analytics)
# ============================================================================

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("redis-credentials")],
    timeout=30,
)
@modal.fastapi_endpoint(method="POST")
def dashboard_api(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consolidated dashboard API combining multiple functions.
    
    Request:
    {
        "action": "get_metrics" | "get_cache_status" | "warm_cache" | 
                  "get_analytics" | "update_settings",
        "creator_id": "optional",
        "city": "optional",
        "settings": {...}  // for update_settings
    }
    
    Returns appropriate response based on action.
    """
    import redis
    import os
    import time
    from datetime import datetime
    
    action = request_data.get("action", "get_metrics")
    creator_id = request_data.get("creator_id", "anonymous")
    city = request_data.get("city")
    
    # Connect to Redis
    try:
        redis_url = os.environ.get("REDIS_URL", "")
        r = redis.from_url(redis_url) if redis_url else None
    except Exception as e:
        r = None
    
    # ---- ACTION: get_metrics ----
    if action == "get_metrics":
        # Get real metrics from Redis if available
        metrics = {
            "latency": {"avg": 150, "p50": 120, "p95": 350, "p99": 500},
            "cache_hit_rate": 0.85,
            "provider_split": {"google_tts": 0.75, "elevenlabs": 0.25, "browser_tts": 0.0},
            "requests_today": 0,
            "success_rate": 0.98,
            "cold_starts": 0,
            "errors": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if r:
            try:
                # Get today's stats from Redis
                today = datetime.now().strftime("%Y-%m-%d")
                metrics["requests_today"] = int(r.get(f"neuron:stats:{today}:requests") or 0)
                metrics["errors"] = int(r.get(f"neuron:stats:{today}:errors") or 0)
                
                # Get latency stats
                latencies = r.lrange("neuron:latencies:recent", 0, 99)
                if latencies:
                    lat_values = [float(l) for l in latencies]
                    lat_values.sort()
                    metrics["latency"]["avg"] = sum(lat_values) / len(lat_values)
                    metrics["latency"]["p50"] = lat_values[len(lat_values) // 2]
                    metrics["latency"]["p95"] = lat_values[int(len(lat_values) * 0.95)]
            except:
                pass
        
        return {"status": "success", "metrics": metrics}
    
    # ---- ACTION: get_cache_status ----
    elif action == "get_cache_status":
        cities = ["Philadelphia", "Dallas", "Kansas City", "Buffalo", "Miami", 
                  "New England", "Pittsburgh", "Baltimore", "Denver", "Seattle"]
        cache_status = []
        
        for city_name in cities:
            status = "cold"
            cached_phrases = 0
            last_warmed = None
            
            if r:
                try:
                    cache_key = f"neuron:phrases:{city_name.lower().replace(' ', '_')}"
                    cached_phrases = r.llen(cache_key)
                    if cached_phrases > 0:
                        status = "warm"
                        last_warmed = r.get(f"neuron:cache_warmed:{city_name}") or None
                        if last_warmed:
                            last_warmed = last_warmed.decode() if isinstance(last_warmed, bytes) else str(last_warmed)
                except:
                    pass
            
            cache_status.append({
                "city": city_name,
                "status": status,
                "cached_phrases": cached_phrases,
                "last_warmed": last_warmed
            })
        
        return {"status": "success", "cache_status": cache_status}
    
    # ---- ACTION: warm_cache ----
    elif action == "warm_cache":
        if not city:
            return {"status": "error", "message": "City required for warm_cache"}
        
        phrases_cached = 0
        if r:
            try:
                # Load city profile and cache phrases
                import json
                with open("/root/config/city_profiles.json", "r") as f:
                    profiles = json.load(f)
                
                if city in profiles:
                    profile = profiles[city]
                    phrases = profile.get("lexical_style", {}).get("phrases", [])
                    cache_key = f"neuron:phrases:{city.lower().replace(' ', '_')}"
                    
                    for phrase in phrases:
                        r.rpush(cache_key, phrase)
                    
                    r.set(f"neuron:cache_warmed:{city}", datetime.now().isoformat())
                    phrases_cached = len(phrases)
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        return {"status": "success", "city": city, "phrases_cached": phrases_cached}
    
    # ---- ACTION: get_analytics ----
    elif action == "get_analytics":
        # Return voice analytics for creator
        analytics = {
            "sitcom_exports": 0,
            "studio_exports": 0,
            "completion_by_style": {"sitcom": 0.0, "studio": 0.0},
            "most_used_cities": [],
            "feedback_ratings": {"positive": 0, "negative": 0, "average": 0.0},
            "cost_by_provider": {"google_tts": 0.0, "elevenlabs": 0.0},
            "retention_by_style": {
                "sitcom": {"avg_watch_time": 0, "completion_rate": 0.0},
                "studio": {"avg_watch_time": 0, "completion_rate": 0.0}
            }
        }
        
        if r and creator_id != "anonymous":
            try:
                analytics["sitcom_exports"] = int(r.get(f"neuron:creator:{creator_id}:exports:sitcom") or 0)
                analytics["studio_exports"] = int(r.get(f"neuron:creator:{creator_id}:exports:studio") or 0)
            except:
                pass
        
        return {"status": "success", "analytics": analytics}
    
    # ---- ACTION: update_settings ----
    elif action == "update_settings":
        settings = request_data.get("settings", {})
        
        if r:
            try:
                import json
                settings_key = f"neuron:creator:{creator_id}:settings"
                r.set(settings_key, json.dumps(settings))
                return {"status": "success", "settings": settings}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        return {"status": "success", "settings": settings, "note": "Stored in memory only"}
    
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


@app.local_entrypoint()
def main():
    """
    To run locally:
        modal run infra/modal_orchestrator.py
    
    To deploy to Modal:
        modal deploy infra/modal_orchestrator.py
    
    To monitor logs:
        modal logs neuron-orchestrator
    """
    print("=" * 70)
    print("NEURON ORCHESTRATOR - MODAL DEPLOYMENT")
    print("=" * 70)
    print("\nUsage:")
    print("  modal run infra/modal_orchestrator.py    # Test locally")
    print("  modal deploy infra/modal_orchestrator.py # Deploy to production")
    print("\nEndpoints:")
    print("  POST /generate_commentary          # Single city commentary")
    print("  POST /generate_multi_city_commentary # Multi-city panel")
    print("  POST /check_interruption_endpoint   # Interruption logic")
    print("  POST /run_debate_stream            # Closed-loop debate (SSE)")
    print("  POST /regenerate_segment           # Regenerate single segment")
    print("  POST /generate_tts                 # Generate TTS audio")
    print("  POST /analyze_style_samples        # Style capture & profile gen")
    print("  GET  /list_cities                   # Available cities")
    print("\n" + "=" * 70)

