#!/usr/bin/env python3
"""
dashboard_api.py - Real-Time SSE Backend for Neuron Dashboard

FastAPI server that streams Kafka debate messages to React frontend
using Server-Sent Events (SSE).
"""

import json
import logging
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from confluent_kafka import Consumer, KafkaError
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Neuron Dashboard API",
    description="Real-time SSE streaming from Kafka debates",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': 'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': 'UEAFJBH67LNNBKPC',
    'sasl.password': 'cfltGY0RWLd/2RRmmYZWM+5dNDexNRC733PEdub4iF7s60s0mTI9QgKv8y44VHNg',
    'group.id': 'react-dashboard',
    'auto.offset.reset': 'latest',
    'enable.auto.commit': True
}


async def kafka_stream() -> AsyncGenerator[dict, None]:
    """
    Async generator that streams Kafka messages as SSE events.
    
    Yields:
        SSE-formatted events with Kafka message data
    """
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe(['agent-debates'])
    
    logger.info("SSE stream started - consuming from agent-debates")
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                # Send keep-alive to maintain connection
                yield {"event": "keepalive", "data": "ping"}
                await asyncio.sleep(0.5)
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                continue
            
            try:
                value = msg.value().decode('utf-8') if msg.value() else ""
                data = json.loads(value)
                
                # Enrich with metadata
                event_data = {
                    "content": data.get("answer", str(data)),
                    "locale": data.get("locale", "en-US"),
                    "confidence": data.get("confidence", 0),
                    "version": data.get("version", "1.0.0"),
                    "agent_id": data.get("agent_id", "unknown"),
                    "is_safe": data.get("is_safe", True),
                    "safety_reason": data.get("safety_reason")
                }
                
                logger.info(f"Streaming event: {event_data['content'][:50]}...")
                
                yield {"event": "debate", "data": json.dumps(event_data)}
                
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Skipping malformed message: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        consumer.close()
        logger.info("SSE stream closed")


@app.get("/stream")
async def stream_debates():
    """
    SSE endpoint for real-time debate streaming.
    
    Connect with EventSource('http://localhost:8000/stream')
    """
    return EventSourceResponse(kafka_stream())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Neuron Dashboard API"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to API docs."""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/docs">
        </head>
        <body>
            <p>Redirecting to <a href="/docs">API Documentation</a>...</p>
        </body>
    </html>
    """


if __name__ == "__main__":
    print("🚀 Starting Neuron Dashboard API...")
    print("   SSE Stream: http://localhost:8000/stream")
    print("   API Docs:   http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
