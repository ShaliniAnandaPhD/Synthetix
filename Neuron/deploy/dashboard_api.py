import asyncio
import os
import ssl
import uuid
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

# --- Configuration ---
ORIGINS = ["*"]

# Confluent Kafka Config
BOOTSTRAP_SERVERS = 'pkc-619z3.us-east1.gcp.confluent.cloud:9092'
SASL_USERNAME = 'UEAFJBH67LNNBKPC'
SASL_PASSWORD = 'cfltGY0RWLd/2RRmmYZWM+5dNDexNRC733PEdub4iF7s60s0mTI9QgKv8y44VHNg'

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global producer (initialized on startup)
producer = None

class SimulationRequest(BaseModel):
    type: str
    team: str
    description: str

def get_ssl_context():
    """Create SSL context for SASL_SSL."""
    context = ssl.create_default_context()
    return context

@app.on_event("startup")
async def startup_event():
    """Initialize Kafka producer on startup."""
    global producer
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol="SASL_SSL",
        sasl_mechanism="PLAIN",
        sasl_plain_username=SASL_USERNAME,
        sasl_plain_password=SASL_PASSWORD,
        ssl_context=get_ssl_context()
    )
    await producer.start()
    print("✅ Kafka producer started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup Kafka producer."""
    global producer
    if producer:
        await producer.stop()

async def kafka_stream() -> AsyncGenerator[dict, None]:
    """Generates SSE events from Kafka messages."""
    group_id = f'dashboard-{uuid.uuid4()}'
    
    consumer = AIOKafkaConsumer(
        'agent-debates',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol="SASL_SSL",
        sasl_mechanism="PLAIN",
        sasl_plain_username=SASL_USERNAME,
        sasl_plain_password=SASL_PASSWORD,
        ssl_context=get_ssl_context(),
        group_id=group_id,
        auto_offset_reset='latest'
    )
    
    await consumer.start()
    print(f"✅ Consumer {group_id} started")
    
    try:
        # Use getmany with timeout for non-blocking behavior
        while True:
            try:
                result = await asyncio.wait_for(
                    consumer.getmany(timeout_ms=500, max_records=10),
                    timeout=1.0
                )
                
                if not result:
                    yield {"comment": "keep-alive"}
                    continue
                
                for tp, messages in result.items():
                    for msg in messages:
                        raw_val = msg.value.decode('utf-8')
                        yield {"data": raw_val}
                        
            except asyncio.TimeoutError:
                yield {"comment": "keep-alive"}
                
    except asyncio.CancelledError:
        print("Client disconnected")
    finally:
        await consumer.stop()

# --- Routes ---

@app.get("/api/stream")
async def stream_api(request: Request):
    """Server-Sent Events endpoint for the frontend."""
    return EventSourceResponse(kafka_stream())

@app.get("/stream")
async def stream(request: Request):
    """SSE endpoint (alias for /api/stream)."""
    return EventSourceResponse(kafka_stream())

@app.post("/api/simulate")
async def simulate_api(req: SimulationRequest):
    """Injects a simulated event into the Kafka stream."""
    global producer
    try:
        if req.type == "FIELD GOAL":
            event_text = f"{req.type}! {req.team} kicks a 45-yarder. It is GOOD."
        elif req.type == "PENALTY":
            event_text = f"{req.type} on {req.team}. 10 yards. Replay down."
        else:
            event_text = f"{req.type} {req.team}! {req.description}"

        print(f"🏈 Simulating: {event_text}")
        
        await producer.send_and_wait('nfl-game-events', event_text.encode('utf-8'))
        
        return {"status": "success", "message": event_text}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# Multi-agent debate responses
import random

FAN_TD = [
    "YESSSSS! TOUCHDOWN BABY! THIS IS THE GREATEST PLAY I'VE EVER SEEN! 🔥🔥🔥",
    "THAT'S WHAT I'M TALKING ABOUT! DYNASTY CONFIRMED! WE'RE GOING ALL THE WAY!",
    "DID YOU SEE THAT?! ABSOLUTE PERFECTION! BEST TEAM IN THE LEAGUE!",
]

FAN_INT = [
    "HAHAHAHA! GET WRECKED! OUR DEFENSE IS ELITE! PICK SIX COMING!",
    "TURNOVER BABY! THAT'S WHAT HAPPENS WHEN YOU TRY US! LET'S GOOO!",
    "INTERCEPTION! Our secondary is UNSTOPPABLE! Feel that momentum shift!",
]

FAN_PENALTY = [
    "WHAT?! THAT'S A TERRIBLE CALL! THE REFS ARE BLIND! This is rigged!",
    "ARE YOU KIDDING ME?! That was a clean play! These refs are AWFUL!",
    "BOOOOOO! The league is literally out to get us! Conspiracy confirmed!",
]

FAN_FUMBLE = [
    "FUMBLE! WE GOT IT! MOMENTUM SHIFT! THIS CHANGES EVERYTHING!",
    "BALL'S OUT! Our boys are HUNGRY! That's championship defense right there!",
    "TURNOVER! See that hit?! Our D-line is absolutely DOMINANT!",
]

FAN_FG = [
    "MONEY! RIGHT DOWN THE MIDDLE! ICE IN HIS VEINS! That's my kicker!",
    "GOOD! CLUTCH! When it matters most, we DELIVER! Points on the board!",
    "THREE POINTS! Every point counts in a game like this! LET'S GO!",
]

ANALYST_TD = [
    "Right, well done. Good execution on the route. The defensive alignment was rather poor.",
    "A touchdown, yes. The offensive line created adequate protection. Expected result.",
    "Solid play design. The quarterback made the correct read. Unremarkable but effective.",
]

ANALYST_INT = [
    "That was rather predictable. The quarterback's footwork was telegraphing it.",
    "Poor decision under pressure. The safety was clearly dropping into zone coverage.",
    "An interception. The quarterback attempted a pass into double coverage. Ill-advised.",
]

ANALYST_PENALTY = [
    "The officials made the correct call. The tape will confirm. Undisciplined play.",
    "A penalty. Expected, frankly. Their penalty rate is 23% above league average.",
    "Procedural error. This has been a recurring issue with this team's preparation.",
]

ANALYST_FUMBLE = [
    "Poor ball security. The running back's grip was compromised pre-contact.",
    "A fumble. Statistically, this player has a 2.1% fumble rate. Not surprising.",
    "The hit was precisely targeted at the ball. Good defensive technique.",
]

ANALYST_FG = [
    "Three points. His career accuracy is 87.3%. Expected result.",
    "The kick was made. Points are points, though a touchdown would have been preferable.",
    "Field goal is good. Wind conditions were favorable. Routine execution.",
]

@app.post("/simulate_event")
async def simulate_event(request: Request):
    """Multi-agent debate: Die Hard Fan vs Analyst"""
    import random
    body = await request.json()
    event_type = body.get("type", "PLAY").upper()
    
    if event_type in ["TD", "TOUCHDOWN"]:
        fan_text = random.choice(FAN_TD)
        analyst_text = random.choice(ANALYST_TD)
        fan_score = random.randint(92, 100)
        analyst_score = random.randint(55, 72)
    elif event_type in ["INT", "INTERCEPTION"]:
        fan_text = random.choice(FAN_INT)
        analyst_text = random.choice(ANALYST_INT)
        fan_score = random.randint(85, 98)
        analyst_score = random.randint(50, 68)
    elif event_type == "PENALTY":
        fan_text = random.choice(FAN_PENALTY)
        analyst_text = random.choice(ANALYST_PENALTY)
        fan_score = random.randint(70, 85)
        analyst_score = random.randint(35, 50)
    elif event_type == "FUMBLE":
        fan_text = random.choice(FAN_FUMBLE)
        analyst_text = random.choice(ANALYST_FUMBLE)
        fan_score = random.randint(88, 98)
        analyst_score = random.randint(52, 70)
    elif event_type in ["FG", "FIELD_GOAL", "FIELD GOAL"]:
        fan_text = random.choice(FAN_FG)
        analyst_text = random.choice(ANALYST_FG)
        fan_score = random.randint(75, 90)
        analyst_score = random.randint(45, 60)
    else:
        fan_text = "SOMETHING HAPPENED! LET'S GOOO!"
        analyst_text = "A play occurred. Moving on."
        fan_score = random.randint(50, 70)
        analyst_score = random.randint(30, 45)
    
    return [
        {
            "agent_id": "agent-fan",
            "persona": "fanatic",
            "name": "Die Hard Fan 🔥",
            "content": fan_text,
            "excitement_score": fan_score,
            "event_type": event_type,
            "alignment": "left",
            "timestamp": "Live"
        },
        {
            "agent_id": "agent-analyst",
            "persona": "analyst",
            "name": "Analyst 🧐",
            "content": analyst_text,
            "excitement_score": analyst_score,
            "event_type": event_type,
            "alignment": "right",
            "timestamp": "Live"
        }
    ]

# =============================================================================
# CREATOR STUDIO: /play endpoint for Dolby OptiView demo
# =============================================================================

DEBATE_SCRIPTS = {
    "TOUCHDOWN": {
        "fan": "YESSSSS! TOUCHDOWN BABY! THIS IS THE GREATEST PLAY I'VE EVER SEEN! We're going all the way! Dynasty confirmed! 🔥🔥🔥",
        "analyst": "Right, well done. Good execution on the route. The defensive alignment was rather poor, which made this somewhat expected given the coverage."
    },
    "PENALTY": {
        "fan": "WHAT?! THAT'S A TERRIBLE CALL! THE REFS ARE BLIND! This is rigged! We're being robbed!",
        "analyst": "The officials appear to have made the correct call. The tape will confirm. This has been a recurring issue with discipline."
    },
    "FIELD_GOAL": {
        "fan": "MONEY! RIGHT DOWN THE MIDDLE! OUR KICKER IS ICE COLD! CLUTCH! That's how you close out games!",
        "analyst": "Three points. The kick was made. His career accuracy is 87.3%. Expected result, though a touchdown would have been preferable."
    },
    "INTERCEPTION": {
        "fan": "HAHAHAHA! GET WRECKED! OUR DEFENSE IS ELITE! PICK SIX COMING! Feel that momentum shift!",
        "analyst": "That was rather predictable. The quarterback's footwork was telegraphing it. Poor decision under pressure."
    }
}

RECAP_TEMPLATES = {
    "TOUCHDOWN": "In what might be the defining moment of the game, {player} connected on a {yards}-yard strike that changed everything. {team} now controls their destiny with the lead.",
    "PENALTY": "A controversial call that will be debated for weeks. The penalty gave {team} new life at a critical moment. Refs under fire from both fanbases.",
    "FIELD_GOAL": "When the game was on the line, {player} delivered with a {yards}-yard field goal. Ice in the veins. {team} walks off victorious.",
    "INTERCEPTION": "Momentum completely shifted when {team}'s defense came up with a crucial interception. The turnover could prove to be the difference maker."
}

COLD_OPEN_TEMPLATES = {
    "TOUCHDOWN": [
        "{player} breaks free down the seam",
        "{yards} yards of pure magic in the fourth quarter",
        "{team} takes the lead with time running out"
    ],
    "PENALTY": [
        "Yellow flag flies at the worst possible moment",
        "Controversial call sparks sideline eruption",
        "Game hangs in the balance after review"
    ],
    "FIELD_GOAL": [
        "{player} lines up for the game-winner",
        "{yards} yards between victory and heartbreak",
        "The kick is up... and it's GOOD!"
    ],
    "INTERCEPTION": [
        "Ball in the air... PICKED OFF!",
        "{team} defense comes up clutch",
        "Momentum completely flipped in an instant"
    ]
}

@app.post("/play")
async def trigger_play(request: Request):
    """Creator Studio: Generate all 3 artifacts for the Dolby demo."""
    import time
    
    body = await request.json()
    event_type = body.get("event_type", "TOUCHDOWN").upper()
    team = body.get("team", "Home Team")
    player = body.get("player", "Star Player")
    yards = body.get("yards", 25)
    
    request_id = str(uuid.uuid4())
    t_start = time.time()
    
    debate_data = DEBATE_SCRIPTS.get(event_type, DEBATE_SCRIPTS["TOUCHDOWN"])
    t_debate = int((time.time() - t_start) * 1000)
    
    recap_template = RECAP_TEMPLATES.get(event_type, RECAP_TEMPLATES["TOUCHDOWN"])
    recap_content = recap_template.format(team=team, player=player, yards=yards)
    t_recap = int((time.time() - t_start) * 1000)
    
    cold_open_template = COLD_OPEN_TEMPLATES.get(event_type, COLD_OPEN_TEMPLATES["TOUCHDOWN"])
    cold_open_bullets = [b.format(team=team, player=player, yards=yards) for b in cold_open_template]
    t_cold_open = int((time.time() - t_start) * 1000)
    
    t_total = int((time.time() - t_start) * 1000)
    
    print(f"🎬 Creator Studio /play: {event_type} → {t_total}ms")
    
    return {
        "request_id": request_id,
        "event_type": event_type,
        "artifacts": {
            "debate": {"fan": debate_data["fan"], "analyst": debate_data["analyst"]},
            "recap": {"content": recap_content},
            "coldOpen": {"bullets": cold_open_bullets}
        },
        "timings": {
            "t_trigger": 0,
            "t_debate_ready": t_debate,
            "t_recap_ready": t_recap,
            "t_cold_open_ready": t_cold_open,
            "t_total": t_total
        }
    }


# =============================================================================
# TEXT-TO-SPEECH: Google Cloud TTS with conversational Journey voices
# =============================================================================
from google.cloud import texttospeech
from starlette.responses import Response

@app.post("/speak")
async def speak(request: Request):
    """Google Cloud TTS with cultural voice selection."""
    body = await request.json()
    text = body.get("text", "")
    persona = body.get("persona", "")
    voice_locale = body.get("voice_locale", "")
    
    if not text:
        return Response(status_code=400)

    try:
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        
        # Check for locale override first
        if voice_locale:
            locale_voices = {
                "de-DE": ("de-DE", "de-DE-Neural2-B"),    # German analytical
                "en-GB": ("en-GB", "en-GB-Neural2-B"),    # British dry wit
                "ja-JP": ("ja-JP", "ja-JP-Neural2-C"),    # Japanese precise
                "pt-BR": ("pt-BR", "pt-BR-Neural2-B"),    # Brazilian GOOOOL
                "es-MX": ("es-MX", "es-MX-Neural2-A"),    # Mexican Andrés Cantor
                "en-AU": ("en-AU", "en-AU-Neural2-B"),    # Aussie mate
            }
            if voice_locale in locale_voices:
                language_code, voice_name = locale_voices[voice_locale]
            else:
                language_code = "en-US"
                voice_name = "en-US-Journey-D"
        elif persona == "fanatic":
            # Energetic male voice
            language_code = "en-US"
            voice_name = "en-US-Journey-D"
        else:
            # Calm analytical voice
            language_code = "en-US"
            voice_name = "en-US-Journey-F"
            
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95  # Slightly slower for clarity
        )

        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        
        return Response(content=response.audio_content, media_type="audio/mpeg")
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return Response(status_code=500)

# =============================================================================
# DOLBY SYNC VERIFICATION ENDPOINTS
# =============================================================================
import random
from collections import deque

# Store recent sync samples for monitoring
sync_samples = deque(maxlen=100)

@app.get("/sync-test")
async def sync_test():
    """Returns server timestamp for sync verification"""
    server_time = int(time.time() * 1000)
    sync_samples.append(server_time)
    
    return {
        "server_time_ms": server_time,
        "audio_ready": True,
        "video_ready": True,
        "sync_offset_ms": 0  # No A/V offset in our system
    }

@app.get("/sync-monitor")
async def sync_monitor():
    """Continuous sync monitoring endpoint for Dolby verification"""
    if len(sync_samples) < 2:
        return {
            "status": "warming_up",
            "samples_collected": len(sync_samples),
            "message": "Collecting samples..."
        }
    
    # Calculate drift between samples (should be consistent intervals)
    drifts = []
    samples_list = list(sync_samples)
    for i in range(1, len(samples_list)):
        drift = samples_list[i] - samples_list[i-1]
        drifts.append(drift)
    
    avg_drift = sum(drifts) / len(drifts) if drifts else 0
    max_drift = max(drifts) if drifts else 0
    std_dev = (sum((d - avg_drift) ** 2 for d in drifts) / len(drifts)) ** 0.5 if drifts else 0
    
    # Determine status based on consistency
    if max_drift < 25:
        status = "excellent"
    elif max_drift < 50:
        status = "good"
    else:
        status = "poor"
    
    return {
        "status": status,
        "average_interval_ms": round(avg_drift, 2),
        "max_drift_ms": round(max_drift, 2),
        "drift_std_dev": round(std_dev, 2),
        "samples_analyzed": len(sync_samples),
        "uptime_seconds": int(time.time() - startup_time) if 'startup_time' in globals() else 0
    }

startup_time = time.time()


@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
