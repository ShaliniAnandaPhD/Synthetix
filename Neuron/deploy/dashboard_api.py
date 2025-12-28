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
async def stream(request: Request):
    """Server-Sent Events endpoint for the frontend."""
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


@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
