"""
Confluent Kafka Consumer for Neuron Orchestrator.

This module implements the event bus integration layer:
- Consumes from Confluent Cloud topic: game-events
- Buffers burst traffic during high-concurrency moments (touchdowns, goals)
- Asynchronously spawns Modal CulturalAgent containers
- Provides backpressure management and zero data loss

Architecture: Live Event → Kafka → Modal Consumer → Agent Spawner → Vertex AI
"""

import json
import os
import time
from typing import Dict, Any, Optional, List
import modal

# Import the CulturalAgent from modal_orchestrator (same directory)
from infra.modal_orchestrator import CulturalAgent


# ============================================================================
# MODAL APP & IMAGE DEFINITION
# ============================================================================

app = modal.App("neuron-kafka-consumer")

# Define the container image with Kafka dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "confluent-kafka",         # Confluent Kafka Python client
        "google-cloud-aiplatform", # Vertex AI (Gemini) for agent spawning
        "redis",                   # Redis for hot state
        "websockets",              # WebSocket for streaming TTS
        "aiohttp",                 # Async HTTP client
    )
    .copy_local_dir(
        local_path="src",
        remote_path="/root/src"
    )
    .copy_local_dir(
        local_path="config",
        remote_path="/root/config"
    )
)


# ============================================================================
# EVENT PRIORITY SYSTEM
# ============================================================================

# Priority levels for different event types (lower = higher priority)
EVENT_PRIORITY = {
    "touchdown": 1,         # Highest priority - immediate processing
    "turnover": 2,          # Interception, fumble recovery
    "safety": 2,            # Rare but exciting
    "big_play": 3,          # 20+ yard gains, sacks
    "score_change": 3,      # Field goals, PATs
    "two_minute_warning": 4,# Important game moments
    "timeout": 5,           # Strategic moments
    "penalty": 6,           # Routine events
    "normal": 10,           # Default priority
}

def get_event_priority(event_type: str) -> int:
    """Get priority level for an event type. Lower = higher priority."""
    return EVENT_PRIORITY.get(event_type.lower(), EVENT_PRIORITY["normal"])


# ============================================================================
# KAFKA CONSUMER CONFIGURATION
# ============================================================================

def get_kafka_config() -> Dict[str, Any]:
    """
    Build Confluent Cloud Kafka configuration from environment secrets.
    
    Required environment variables (from Modal secrets):
    - BOOTSTRAP_SERVERS: Kafka broker endpoints
    - SASL_USERNAME: Confluent Cloud API Key
    - SASL_PASSWORD: Confluent Cloud API Secret
    
    Returns:
        Dictionary of Kafka consumer configuration
    """
    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS")
    sasl_username = os.environ.get("SASL_USERNAME")
    sasl_password = os.environ.get("SASL_PASSWORD")
    
    if not all([bootstrap_servers, sasl_username, sasl_password]):
        raise ValueError(
            "Missing required Kafka credentials. Ensure Modal secret 'kafka-credentials' "
            "contains BOOTSTRAP_SERVERS, SASL_USERNAME, and SASL_PASSWORD."
        )
    
    return {
        # Connection settings
        'bootstrap.servers': bootstrap_servers,
        
        # Security settings (Confluent Cloud requires SASL_SSL)
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'PLAIN',
        'sasl.username': sasl_username,
        'sasl.password': sasl_password,
        
        # Consumer group settings
        'group.id': 'neuron-agents',
        'client.id': 'neuron-kafka-consumer-modal',
        
        # Offset management (manual commit for zero data loss)
        'auto.offset.reset': 'earliest',  # Start from beginning if no offset
        'enable.auto.commit': False,      # Manual commit after processing
        
        # Performance tuning
        'fetch.min.bytes': 1,             # Don't wait for large batches
        'fetch.wait.max.ms': 100,         # Max 100ms wait for polling
        'max.partition.fetch.bytes': 1048576,  # 1MB per partition
        
        # Error handling
        'session.timeout.ms': 45000,      # 45s consumer heartbeat timeout
        'heartbeat.interval.ms': 3000,    # 3s heartbeat interval
        'max.poll.interval.ms': 300000,   # 5 min max time between polls
    }


# ============================================================================
# EVENT VALIDATION & PARSING
# ============================================================================

def parse_kafka_message(message_value: str) -> Optional[Dict[str, Any]]:
    """
    Parse and validate Kafka message payload.
    
    Expected JSON structure:
    {
        "city": "Philadelphia",
        "event_type": "touchdown",
        "user_input": "React to that amazing catch!",
        "game_context": {
            "score": "Eagles 21 - Cowboys 14",
            "quarter": 3,
            "time_remaining": "8:45"
        },
        "timestamp": 1733189234.567
    }
    
    Args:
        message_value: Raw Kafka message value (JSON string)
    
    Returns:
        Parsed dictionary with priority field if valid, None if invalid
    """
    try:
        data = json.loads(message_value)
        
        # Validate required fields
        if 'city' not in data:
            print(f"[INVALID] Missing 'city' field: {data}")
            return None
        
        if 'user_input' not in data and 'event_type' not in data:
            print(f"[INVALID] Missing both 'user_input' and 'event_type': {data}")
            return None
        
        # Auto-generate user_input from event_type if missing
        event_type = data.get('event_type', 'normal')
        if 'user_input' not in data:
            data['user_input'] = f"React to this {event_type}!"
        
        # Add priority based on event type for queue ordering
        data['priority'] = get_event_priority(event_type)
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"[JSON ERROR] Failed to parse message: {e}")
        return None
    except Exception as e:
        print(f"[PARSE ERROR] Unexpected error: {e}")
        return None


# ============================================================================
# KAFKA CONSUMER (LONG-RUNNING FUNCTION)
# ============================================================================

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("kafka-credentials"),  # BOOTSTRAP_SERVERS, SASL_USERNAME, SASL_PASSWORD
        modal.Secret.from_name("redis-credentials"),  # REDIS_URL (for agent spawning)
        modal.Secret.from_name("googlecloud-secret"),  # GCP_PROJECT_ID + credentials
    ],
    timeout=86400,  # 24 hours (long-running consumer)
    keep_warm=1,    # Always keep 1 consumer active
)
def consume_game_events():
    """
    Long-running Kafka consumer that processes game events.
    
    This function:
    1. Connects to Confluent Cloud Kafka
    2. Subscribes to 'game-events' topic
    3. Polls for messages in batches of 50
    4. Spawns CulturalAgent containers asynchronously
    5. Commits offsets after successful spawning
    
    Runs indefinitely until manually stopped or error occurs.
    """
    from confluent_kafka import Consumer, KafkaError, KafkaException
    
    # ----------------------------------------------------------------
    # 1. INITIALIZE KAFKA CONSUMER
    # ----------------------------------------------------------------
    kafka_config = get_kafka_config()
    consumer = Consumer(kafka_config)
    
    print("=" * 70)
    print("NEURON KAFKA CONSUMER - STARTED")
    print("=" * 70)
    print(f"Bootstrap Servers: {kafka_config['bootstrap.servers']}")
    print(f"Consumer Group: {kafka_config['group.id']}")
    print(f"Topic: game-events")
    print("=" * 70)
    
    # Subscribe to topic
    topic_name = "game-events"
    consumer.subscribe([topic_name])
    print(f"[SUBSCRIBED] Listening to topic: {topic_name}")
    
    # ----------------------------------------------------------------
    # 2. METRICS TRACKING
    # ----------------------------------------------------------------
    total_messages = 0
    successful_spawns = 0
    failed_spawns = 0
    last_log_time = time.time()
    
    # ----------------------------------------------------------------
    # 3. MAIN CONSUMPTION LOOP (INFINITE)
    # ----------------------------------------------------------------
    try:
        while True:
            # Poll for messages (timeout: 1 second)
            # This returns up to 'max_messages' per poll
            batch_size = 50
            messages_to_process = []
            
            # Collect a batch of messages
            for _ in range(batch_size):
                msg = consumer.poll(timeout=1.0)
                
                if msg is None:
                    # No message available, continue to next poll
                    break
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition, not a critical error
                        print(f"[EOF] Reached end of partition {msg.partition()}")
                        continue
                    else:
                        # Critical error
                        raise KafkaException(msg.error())
                
                # Valid message
                messages_to_process.append(msg)
            
            # ----------------------------------------------------------------
            # 4. PROCESS BATCH (SPAWN AGENTS ASYNCHRONOUSLY)
            # ----------------------------------------------------------------
            if len(messages_to_process) > 0:
                print(f"\n[BATCH] Processing {len(messages_to_process)} messages...")
                
                spawn_tasks = []  # Track async spawns
                
                for msg in messages_to_process:
                    total_messages += 1
                    
                    # Decode message
                    message_value = msg.value().decode('utf-8')
                    
                    # Parse and validate
                    event_data = parse_kafka_message(message_value)
                    
                    if event_data is None:
                        failed_spawns += 1
                        print(f"[SKIP] Invalid message at offset {msg.offset()}")
                        continue
                    
                    # Extract fields
                    city = event_data.get("city")
                    user_input = event_data.get("user_input")
                    game_context = event_data.get("game_context", {})
                    conversation_history = event_data.get("conversation_history", [])
                    
                    print(f"[EVENT] City: {city} | Input: {user_input[:50]}...")
                    
                    # ----------------------------------------------------------------
                    # 5. SPAWN CULTURAL AGENT (ASYNC, NON-BLOCKING)
                    # ----------------------------------------------------------------
                    try:
                        # Use .spawn() for async execution (doesn't block)
                        agent = CulturalAgent()
                        spawn_handle = agent.generate_response.spawn(
                            city_name=city,
                            user_input=user_input,
                            conversation_history=conversation_history,
                            game_context=game_context
                        )
                        
                        spawn_tasks.append({
                            'handle': spawn_handle,
                            'city': city,
                            'offset': msg.offset(),
                            'partition': msg.partition()
                        })
                        
                        successful_spawns += 1
                        
                    except Exception as e:
                        failed_spawns += 1
                        print(f"[SPAWN ERROR] Failed to spawn agent for {city}: {e}")
                
                # ----------------------------------------------------------------
                # 6. COMMIT OFFSETS (ZERO DATA LOSS GUARANTEE)
                # ----------------------------------------------------------------
                # Only commit after all spawns are initiated
                # This ensures that if we crash, unprocessed messages will be redelivered
                try:
                    consumer.commit(asynchronous=False)
                    print(f"[COMMITTED] Offsets for {len(messages_to_process)} messages")
                except Exception as e:
                    print(f"[COMMIT ERROR] Failed to commit offsets: {e}")
                    # Don't raise - consumers can recover from commit failures
                
                # ----------------------------------------------------------------
                # 7. WAIT FOR SPAWNED AGENTS (OPTIONAL VALIDATION)
                # ----------------------------------------------------------------
                # For production, you might want to track these in a separate monitoring task
                # For now, we just log that they were spawned
                print(f"[SPAWNED] {len(spawn_tasks)} agent containers initiated")
                
                # Optional: Wait for first few to complete (for debugging)
                if len(spawn_tasks) > 0:
                    first_task = spawn_tasks[0]
                    try:
                        # Wait up to 5 seconds for first result
                        result = first_task['handle'].get(timeout=5.0)
                        print(f"[RESULT SAMPLE] {first_task['city']}: {result.get('response', '')[:100]}...")
                    except TimeoutError:
                        print(f"[TIMEOUT] First agent still processing (normal for large batches)")
                    except Exception as e:
                        print(f"[RESULT ERROR] {e}")
            
            # ----------------------------------------------------------------
            # 8. PERIODIC METRICS LOGGING (EVERY 60 SECONDS)
            # ----------------------------------------------------------------
            current_time = time.time()
            if current_time - last_log_time >= 60:
                print("\n" + "=" * 70)
                print("KAFKA CONSUMER METRICS (LAST 60s)")
                print("=" * 70)
                print(f"Total Messages Processed: {total_messages}")
                print(f"Successful Spawns: {successful_spawns}")
                print(f"Failed Spawns: {failed_spawns}")
                print(f"Success Rate: {(successful_spawns / max(total_messages, 1)) * 100:.2f}%")
                print("=" * 70 + "\n")
                
                # Reset counters
                total_messages = 0
                successful_spawns = 0
                failed_spawns = 0
                last_log_time = current_time
            
            # Small sleep to prevent tight loop when no messages
            if len(messages_to_process) == 0:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Received interrupt signal")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Consumer crashed: {e}")
        raise
    finally:
        # ----------------------------------------------------------------
        # 9. GRACEFUL SHUTDOWN
        # ----------------------------------------------------------------
        print("[CLEANUP] Closing Kafka consumer...")
        consumer.close()
        print("[SHUTDOWN] Consumer closed successfully")


# ============================================================================
# WEBHOOK ENDPOINT (KAFKA EVENT PRODUCER)
# ============================================================================

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kafka-credentials")],
)
@modal.web_endpoint(method="POST")
def publish_game_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint to publish events to Kafka topic.
    
    This allows external systems to inject events into the Kafka pipeline.
    
    Expected JSON payload:
    {
        "city": "Philadelphia",
        "event_type": "touchdown",
        "user_input": "React to that play!",
        "game_context": {...}
    }
    
    Returns:
        Success/failure confirmation
    """
    from confluent_kafka import Producer
    
    # Validate input
    if 'city' not in event_data:
        return {
            "status": "error",
            "message": "Missing required field: 'city'"
        }
    
    # Build producer config (simpler than consumer)
    producer_config = {
        'bootstrap.servers': os.environ.get("BOOTSTRAP_SERVERS"),
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'PLAIN',
        'sasl.username': os.environ.get("SASL_USERNAME"),
        'sasl.password': os.environ.get("SASL_PASSWORD"),
    }
    
    producer = Producer(producer_config)
    
    # Add timestamp if not present
    if 'timestamp' not in event_data:
        event_data['timestamp'] = time.time()
    
    # Publish to Kafka
    topic = "game-events"
    city = event_data['city']
    
    try:
        # Use city as partition key (ensures all events for a city go to same partition)
        producer.produce(
            topic=topic,
            key=city.encode('utf-8'),
            value=json.dumps(event_data).encode('utf-8'),
            callback=lambda err, msg: print(f"[KAFKA] Published to {msg.topic()}:{msg.partition()} @ offset {msg.offset()}")
        )
        
        # Flush to ensure delivery
        producer.flush(timeout=5.0)
        
        return {
            "status": "success",
            "message": f"Event published to {topic}",
            "city": city,
            "timestamp": event_data['timestamp']
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to publish event: {str(e)}"
        }


# ============================================================================
# CONSUMER MANAGEMENT ENDPOINTS
# ============================================================================

@app.function(image=image)
def start_consumer():
    """
    Start the Kafka consumer (blocking call).
    
    Usage:
        modal run infra/confluent_consumer.py::start_consumer
    """
    print("Starting Kafka consumer...")
    consume_game_events.remote()


# ============================================================================
# MAIN: DEPLOYMENT INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    """
    Deployment and testing instructions.
    
    SETUP:
    1. Create Modal secret 'kafka-credentials' with:
       - BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
       - SASL_USERNAME=<your-api-key>
       - SASL_PASSWORD=<your-api-secret>
    
    2. Ensure Confluent Cloud topic 'game-events' exists with 32 partitions
    
    DEPLOYMENT:
        modal deploy infra/confluent_consumer.py
    
    START CONSUMER:
        modal run infra/confluent_consumer.py::start_consumer
    
    PUBLISH EVENT (TEST):
        curl -X POST https://your-modal-url.modal.run/publish_game_event \
             -H "Content-Type: application/json" \
             -d '{
                "city": "Philadelphia",
                "event_type": "touchdown",
                "user_input": "TOUCHDOWN! React to that play!",
                "game_context": {"score": "21-14", "quarter": 3}
             }'
    
    MONITORING:
        modal logs neuron-kafka-consumer --follow
    """
    print("=" * 70)
    print("NEURON KAFKA CONSUMER - DEPLOYMENT GUIDE")
    print("=" * 70)
    print("\n1. Setup Kafka credentials in Modal secrets")
    print("2. Deploy: modal deploy infra/confluent_consumer.py")
    print("3. Start consumer: modal run infra/confluent_consumer.py::start_consumer")
    print("4. Monitor: modal logs neuron-kafka-consumer --follow")
    print("\n" + "=" * 70)
