#!/usr/bin/env python3
"""
simple_bridge.py - Simple Echo Bridge for Dashboard Testing

Listens to nfl-game-events and echoes them back to agent-debates
for immediate dashboard testing without Vertex AI dependency.
"""

import json
import logging
import time
from confluent_kafka import Producer, Consumer, KafkaError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Kafka config (same as dashboard_api.py)
KAFKA_CONFIG = {
    'bootstrap.servers': 'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': 'UEAFJBH67LNNBKPC',
    'sasl.password': 'cfltGY0RWLd/2RRmmYZWM+5dNDexNRC733PEdub4iF7s60s0mTI9QgKv8y44VHNg',
}

INPUT_TOPIC = 'nfl-game-events'
OUTPUT_TOPIC = 'agent-debates'


def calculate_excitement(event_text: str) -> int:
    """
    Calculate excitement score (0-100) based on event content.
    
    Scoring:
    - Touchdowns/Big Plays: 80-100
    - Interceptions/Fumbles: 60-80  
    - Penalties: 40-60
    - Standard plays: 0-40
    """
    text = event_text.upper()
    
    # Touchdowns & Big Plays (80-100)
    if 'TOUCHDOWN' in text:
        return 95
    if 'TD' in text:
        return 90
    if 'FIELD GOAL' in text and 'GOOD' in text:
        return 75
    
    # Turnovers (60-80)
    if 'INTERCEPTION' in text or 'INT' in text:
        return 78
    if 'FUMBLE' in text:
        return 72
    if 'SACK' in text:
        return 68
    
    # Penalties (40-60)
    if 'PENALTY' in text:
        return 50
    if 'FLAG' in text:
        return 45
    
    # Field Goals Missed
    if 'FIELD GOAL' in text and ('MISS' in text or 'NO GOOD' in text):
        return 55
    
    # Default - moderate excitement
    return 35


def main():
    print("=" * 60)
    print("🌉 Simple Echo Bridge Starting...")
    print(f"   Input:  {INPUT_TOPIC}")
    print(f"   Output: {OUTPUT_TOPIC}")
    print("=" * 60)
    
    # Producer for output
    producer = Producer(KAFKA_CONFIG)
    
    # Consumer for input
    consumer_config = {
        **KAFKA_CONFIG,
        'group.id': 'simple-bridge-group',
        'auto.offset.reset': 'latest',
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe([INPUT_TOPIC])
    
    print("✅ Waiting for events... (Ctrl+C to stop)")
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                    continue
            
            try:
                value = msg.value().decode('utf-8') if msg.value() else ""
            except UnicodeDecodeError:
                continue
            
            print(f"🏈 Received: {value}")
            
            # Calculate excitement score based on event type
            excitement_score = calculate_excitement(value)
            
            # Create a response with excitement analytics
            response = {
                "content": f"🎙️ BREAKING: {value}",
                "agent_id": "NeuronBot",
                "confidence": 95,
                "excitement_score": excitement_score,
                "timestamp": time.strftime("%H:%M:%S")
            }
            
            # Send to output topic
            producer.produce(
                OUTPUT_TOPIC,
                value=json.dumps(response).encode('utf-8')
            )
            producer.flush()
            
            print(f"✅ Sent to {OUTPUT_TOPIC}: {response['content']} (excitement: {excitement_score})")
            
    except KeyboardInterrupt:
        print("\n🛑 Bridge stopped.")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    main()
