import asyncio
import json
import logging
import random
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse
import uvicorn

# Google Cloud TTS
from google.cloud import texttospeech

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardAPI")

app = FastAPI()

# Allow CORS for local React dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SSE Endpoint ---
@app.get("/stream")
async def message_stream(request: Request):
    """
    SSE Endpoint. Pushes real-time events to the frontend.
    """
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            yield {"data": json.dumps({"type": "heartbeat"})}
            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())

# =============================================================================
# SWARM INTELLIGENCE: THE HOMER vs THE SKEPTIC
# =============================================================================

# The Homer (Fanatic) - High energy, emotional, ALL CAPS enthusiasm
HOMER_TD = [
    "YESSSSS! TOUCHDOWN BABY! THIS IS THE GREATEST PLAY I'VE EVER SEEN! 🔥🔥🔥",
    "OH MY GOD! THEY'RE UNSTOPPABLE! DYNASTY! GREATEST TEAM OF ALL TIME!",
    "I'M LITERALLY CRYING! THAT WAS BEAUTIFUL! PURE PERFECTION!",
    "NOBODY CAN STOP US! WE'RE GOING ALL THE WAY! SUPER BOWL HERE WE COME!",
    "DID YOU SEE THAT?! MAHOMES IS A WIZARD! GENERATIONAL TALENT!",
]

HOMER_INT = [
    "HAHAHAHA! GET WRECKED! OUR DEFENSE IS ELITE! BEST IN THE LEAGUE!",
    "I KNEW IT! I KNEW THEY'D CHOKE! THAT'S WHAT HAPPENS WHEN YOU PLAY US!",
    "TURNOVER BABY! MOMENTUM SHIFT! FEEL THAT ENERGY!",
    "PICKED OFF! OUR DB IS A PSYCHIC! HE READ THAT LIKE A BOOK!",
]

HOMER_PENALTY = [
    "WHAT?! THAT'S A TERRIBLE CALL! THE REFS ARE BLIND! RIGGED!",
    "ARE YOU KIDDING ME?! THAT WASN'T A PENALTY! CONSPIRACY!",
    "ROBBED! WE'RE BEING ROBBED! THE NFL HATES US!",
]

HOMER_FUMBLE = [
    "YEAHHHHH STRIP SACK! DEFENSE CAME TO PLAY TODAY! LET'S GOOO!",
    "FUMBLEEEE! CHAOS! I LOVE IT! MOMENTUM IS OURS NOW!",
    "BALL DON'T LIE! THE FOOTBALL GODS ARE WITH US TODAY!",
]

HOMER_FG = [
    "MONEY! RIGHT DOWN THE MIDDLE! OUR KICKER IS ICE COLD! CLUTCH!",
    "THREEEEE POINTS! EVERY POINT COUNTS! WE'RE BUILDING A LEAD!",
    "AUTOMATIC! THAT'S WHY WE PAY HIM! LEG OF GOLD!",
]

# The Skeptic (Analyst) - Calm, analytical, British dry wit
SKEPTIC_TD = [
    "Right, well done. Good execution on the route. Expected result given the coverage.",
    "A touchdown, yes. Though I'd note the defensive alignment was rather poor.",
    "Competent play. The statistics suggested a 67% completion probability there.",
    "Impressive, I suppose. Though let's see if they can maintain this momentum.",
    "Hmm. The data shows they've now scored on 3 of 4 red zone attempts. Acceptable.",
]

SKEPTIC_INT = [
    "That was rather predictable. The quarterback's footwork was telegraphing it.",
    "An interception. Poor decision under pressure. The analytics warned about this.",
    "Yes, well, when you force throws into double coverage, this happens.",
    "Interesting. Their turnover-worthy play rate was concerning coming in.",
]

SKEPTIC_PENALTY = [
    "The officials appear to have made the correct call. The tape will confirm.",
    "A penalty, yes. Undisciplined play. This has been a recurring issue.",
    "Expected, frankly. Their penalty rate is 23% above league average.",
]

SKEPTIC_FUMBLE = [
    "Poor ball security. The running back's grip was compromised pre-contact.",
    "A fumble. Statistically, this player has a 2.1% fumble rate. Not surprising.",
    "The hit was precisely targeted at the ball. Good defensive technique.",
]

SKEPTIC_FG = [
    "Three points. Acceptable, though a touchdown would have been preferable.",
    "The field goal was made. His career accuracy is 87.3%. As expected.",
    "Points are points, I suppose. Though settling for field goals rarely wins championships.",
]


@app.post("/simulate_event")
async def simulate_event(request: Request):
    """
    SWARM INTELLIGENCE: Multi-Agent Debate
    Returns TWO agents with opposing viewpoints for each event:
    - Agent A: 'The Homer' (fanatic, high energy)
    - Agent B: 'The Skeptic' (analyst, dry wit)
    """
    body = await request.json()
    event_type = body.get("type", "PLAY").upper()
    
    # Select messages based on event type
    if event_type in ["TD", "TOUCHDOWN"]:
        homer_text = random.choice(HOMER_TD)
        skeptic_text = random.choice(SKEPTIC_TD)
        homer_score = random.randint(92, 100)
        skeptic_score = random.randint(55, 72)
    elif event_type in ["INT", "INTERCEPTION"]:
        homer_text = random.choice(HOMER_INT)
        skeptic_text = random.choice(SKEPTIC_INT)
        homer_score = random.randint(85, 98)
        skeptic_score = random.randint(50, 68)
    elif event_type == "PENALTY":
        homer_text = random.choice(HOMER_PENALTY)
        skeptic_text = random.choice(SKEPTIC_PENALTY)
        homer_score = random.randint(70, 85)
        skeptic_score = random.randint(30, 48)
    elif event_type == "FUMBLE":
        homer_text = random.choice(HOMER_FUMBLE)
        skeptic_text = random.choice(SKEPTIC_FUMBLE)
        homer_score = random.randint(88, 100)
        skeptic_score = random.randint(52, 70)
    elif event_type in ["FG", "FIELD_GOAL"]:
        homer_text = random.choice(HOMER_FG)
        skeptic_text = random.choice(SKEPTIC_FG)
        homer_score = random.randint(72, 88)
        skeptic_score = random.randint(40, 58)
    else:
        homer_text = "SOMETHING HAPPENED! LET'S GOOO!"
        skeptic_text = "A play occurred. Moving on."
        homer_score = random.randint(50, 70)
        skeptic_score = random.randint(30, 45)
    
    # Return BOTH agents as a list
    swarm_response = [
        {
            "agent_id": "agent-homer",
            "persona": "fanatic",
            "name": "The Homer 🔥",
            "content": homer_text,
            "excitement_score": homer_score,
            "event_type": event_type,
            "alignment": "left",
            "timestamp": "Live"
        },
        {
            "agent_id": "agent-skeptic",
            "persona": "analyst",
            "name": "The Skeptic 🧐",
            "content": skeptic_text,
            "excitement_score": skeptic_score,
            "event_type": event_type,
            "alignment": "right",
            "timestamp": "Live"
        }
    ]
    
    return swarm_response


@app.post("/speak")
async def speak(request: Request):
    """
    Generates MP3 audio from text using Google Cloud TTS.
    Now supports persona-based voice selection for the Swarm!
    """
    body = await request.json()
    text = body.get("text", "")
    persona = body.get("persona", "")
    voice_locale = body.get("voice_locale", "")  # Optional override
    
    if not text:
        return Response(status_code=400)

    try:
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        
        # Voice pools for variety
        HOMER_VOICES = [
            ("en-US", "en-US-Polyglot-1"),    # American high energy
            ("pt-BR", "pt-BR-Neural2-B"),      # Brazilian GOOOL energy
            ("es-MX", "es-MX-Neural2-A"),      # Mexican Andrés Cantor
            ("en-AU", "en-AU-Neural2-B"),      # Aussie excitement
        ]
        
        SKEPTIC_VOICES = [
            ("en-GB", "en-GB-Neural2-B"),      # British dry wit
            ("de-DE", "de-DE-Neural2-B"),      # German analytical
            ("ja-JP", "ja-JP-Neural2-C"),      # Japanese precise
        ]
        
        # Select voice based on persona and optional override
        if voice_locale:
            # Direct locale override
            language_code = voice_locale
            if voice_locale == "pt-BR":
                voice_name = "pt-BR-Neural2-B"
            elif voice_locale == "en-GB":
                voice_name = "en-GB-Neural2-B"
            elif voice_locale == "es-MX":
                voice_name = "es-MX-Neural2-A"
            elif voice_locale == "en-AU":
                voice_name = "en-AU-Neural2-B"
            elif voice_locale == "de-DE":
                voice_name = "de-DE-Neural2-B"
            elif voice_locale == "ja-JP":
                voice_name = "ja-JP-Neural2-C"
            else:
                voice_name = "en-US-Journey-D"
                language_code = "en-US"
        elif persona == "fanatic":
            # Random exciting voice for The Homer
            language_code, voice_name = random.choice(HOMER_VOICES)
        elif persona == "analyst":
            # Random calm/analytical voice for The Skeptic
            language_code, voice_name = random.choice(SKEPTIC_VOICES)
        else:
            # Default journey voice
            voice_name = "en-US-Journey-D"
            language_code = "en-US"
            
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        
        return Response(content=response.audio_content, media_type="audio/mpeg")
        
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return Response(status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
