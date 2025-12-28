"""
dashboard_api_cloud.py - Cloud Run version of the Dashboard API

This version is optimized for Cloud Run deployment with:
- aiokafka instead of confluent-kafka (pure Python, no C deps)
- Google Cloud TTS for voice synthesis
"""

import asyncio
import json
import os
import base64
import uuid
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

# --- Configuration ---
ORIGINS = ["*"]

# Kafka config
KAFKA_BOOTSTRAP = os.environ.get(
    "KAFKA_BOOTSTRAP", 
    "pkc-619z3.us-east1.gcp.confluent.cloud:9092"
)
KAFKA_USERNAME = os.environ.get("KAFKA_USERNAME", "UEAFJBH67LNNBKPC")
KAFKA_PASSWORD = os.environ.get(
    "KAFKA_PASSWORD", 
    "cfltGY0RWLd/2RRmmYZWM+5dNDexNRC733PEdub4iF7s60s0mTI9QgKv8y44VHNg"
)

app = FastAPI(title="Neuron Dashboard API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    type: str
    team: str
    description: str


class TTSRequest(BaseModel):
    text: str
    locale: str = "en-US"


# Voice mapping by locale (Google Cloud TTS Neural2 voices)
VOICE_MAP = {
    "en-US": {"languageCode": "en-US", "name": "en-US-Neural2-D", "ssmlGender": "MALE"},
    "pt-BR": {"languageCode": "pt-BR", "name": "pt-BR-Neural2-B", "ssmlGender": "MALE"},
    "en-GB": {"languageCode": "en-GB", "name": "en-GB-Neural2-B", "ssmlGender": "MALE"},
    "es-MX": {"languageCode": "es-US", "name": "es-US-Neural2-B", "ssmlGender": "MALE"},
    "en-AU": {"languageCode": "en-AU", "name": "en-AU-Neural2-B", "ssmlGender": "MALE"},
    "ja-JP": {"languageCode": "ja-JP", "name": "ja-JP-Neural2-C", "ssmlGender": "MALE"},
}

# Global producer (lazy init)
_producer = None


async def get_producer():
    """Lazy initialize aiokafka producer."""
    global _producer
    if _producer is None:
        from aiokafka import AIOKafkaProducer
        import ssl
        
        ssl_context = ssl.create_default_context()
        
        _producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_plain_username=KAFKA_USERNAME,
            sasl_plain_password=KAFKA_PASSWORD,
            ssl_context=ssl_context
        )
        await _producer.start()
    return _producer


async def kafka_stream() -> AsyncGenerator[dict, None]:
    """Generates SSE events from Kafka messages."""
    from aiokafka import AIOKafkaConsumer
    import ssl
    
    ssl_context = ssl.create_default_context()
    consumer_group = f'dashboard-{uuid.uuid4()}'
    
    consumer = AIOKafkaConsumer(
        'agent-debates',
        bootstrap_servers=KAFKA_BOOTSTRAP,
        security_protocol="SASL_SSL",
        sasl_mechanism="PLAIN",
        sasl_plain_username=KAFKA_USERNAME,
        sasl_plain_password=KAFKA_PASSWORD,
        ssl_context=ssl_context,
        group_id=consumer_group,
        auto_offset_reset='latest'
    )
    
    await consumer.start()
    
    try:
        while True:
            # Get batch of messages with timeout
            data = await consumer.getmany(timeout_ms=500)
            
            if not data:
                yield {"comment": "keep-alive"}
                await asyncio.sleep(0.1)
                continue
            
            for tp, messages in data.items():
                for msg in messages:
                    raw_val = msg.value.decode('utf-8')
                    print(f"📡 Sending to frontend: {raw_val[:100]}...")
                    yield {"event": "debate", "data": raw_val}
                    await asyncio.sleep(0.01)
                    
    except asyncio.CancelledError:
        print("Client disconnected")
    finally:
        await consumer.stop()


# --- Routes ---

@app.get("/api/stream")
async def stream():
    """Server-Sent Events endpoint for the frontend."""
    return EventSourceResponse(kafka_stream())


@app.post("/api/simulate")
async def simulate_event(req: SimulationRequest):
    """Injects a simulated event into the Kafka stream."""
    try:
        if req.type == "FIELD GOAL":
            event_text = f"{req.type}! {req.team} kicks a 45-yarder. It is GOOD."
        elif req.type == "PENALTY":
            event_text = f"{req.type} on {req.team}. 10 yards. Replay down."
        else:
            event_text = f"{req.type} {req.team}! {req.description}"

        print(f"🏈 Simulating: {event_text}")
        
        producer = await get_producer()
        await producer.send_and_wait('nfl-game-events', event_text.encode('utf-8'))
        
        return {"status": "success", "message": event_text}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "detail": str(e)}


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Generate speech audio from text using Google Cloud TTS."""
    try:
        from google.cloud import texttospeech
        
        client = texttospeech.TextToSpeechClient()
        
        voice_config = VOICE_MAP.get(req.locale, VOICE_MAP["en-US"])
        
        synthesis_input = texttospeech.SynthesisInput(text=req.text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config["languageCode"],
            name=voice_config["name"],
            ssml_gender=texttospeech.SsmlVoiceGender[voice_config["ssmlGender"]]
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.1,
            pitch=0.0
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        print(f"🎤 TTS generated: {len(response.audio_content)} bytes ({req.locale})")
        
        return {"audio": audio_base64, "locale": req.locale}
        
    except ImportError:
        print("⚠️ Google Cloud TTS not available")
        return {"audio": None, "error": "TTS not configured", "fallback": True}
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return {"audio": None, "error": str(e)}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("services.dashboard_api_cloud:app", host="0.0.0.0", port=port)
