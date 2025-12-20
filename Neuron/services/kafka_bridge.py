#!/usr/bin/env python3
"""
kafka_bridge.py - The Bridge between Confluent Kafka and Vertex AI

Listens to NFL game events from Kafka, queries the Neuron agent,
validates responses through a safety layer, logs analytics to BigQuery,
and produces safe AI-generated commentary back to Kafka.

BLACK BOX RECORDER: Every decision is logged to BigQuery for post-game analysis.
"""

import json
import logging
import os
import sys
import time
import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confluent_kafka import Producer, Consumer, KafkaError
import vertexai
from vertexai.preview import reasoning_engines
from google.cloud import bigquery

from neuron_core.agents.validator_agent import ValidatorAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class NeuronBridge:
    """
    Bridge service connecting Kafka streams to Vertex AI Reasoning Engine.
    
    Features:
    - Vertex AI agent integration
    - ValidatorAgent safety layer (Circuit Breaker pattern)
    - Automatic blocking of unsafe content
    - BigQuery analytics logging (Black Box Recorder)
    """
    
    def __init__(
        self, 
        kafka_config: Dict[str, Any],
        agent_resource_id: str,
        input_topic: str = "nfl-game-events",
        output_topic: str = "agent-debates",
        consumer_group: str = "neuron-bridge-group"
    ):
        """
        Initialize the Neuron Bridge with safety layer and analytics.
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.agent_resource_id = agent_resource_id
        
        # Initialize Safety Layer (Circuit Breaker)
        self.validator = ValidatorAgent(name='SafetyNet')
        logger.info(f"🛡️ Safety Layer initialized: ValidatorAgent 'SafetyNet'")
        
        # Initialize BigQuery Client (Black Box Recorder)
        try:
            self.bq_client = bigquery.Client()
            self.table_id = 'leafy-sanctuary-476515-t2.nfl_analysis.agent_debates_log'
            logger.info(f"📊 BigQuery initialized: {self.table_id}")
        except Exception as e:
            logger.warning(f"⚠️ BigQuery init failed (analytics disabled): {e}")
            self.bq_client = None
            self.table_id = None
        
        # Initialize Kafka Producer
        producer_config = {
            **kafka_config,
            'client.id': 'neuron-bridge-producer'
        }
        self.producer = Producer(producer_config)
        logger.info(f"Producer initialized for topic: {output_topic}")
        
        # Initialize Kafka Consumer
        consumer_config = {
            **kafka_config,
            'group.id': consumer_group,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True
        }
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([input_topic])
        logger.info(f"Consumer subscribed to topic: {input_topic}")
        
        # Initialize Vertex AI agent
        self._init_agent()
        
        # Stats for monitoring
        self.stats = {
            'messages_processed': 0,
            'messages_passed': 0,
            'messages_blocked': 0,
            'bq_logged': 0,
            'bq_errors': 0
        }
    
    def _init_agent(self):
        """Initialize connection to Vertex AI Reasoning Engine."""
        try:
            self.remote_agent = reasoning_engines.ReasoningEngine(
                self.agent_resource_id
            )
            logger.info(f"Connected to Vertex AI agent: {self.agent_resource_id}")
        except Exception as e:
            logger.error(f"Failed to connect to agent: {e}")
            raise
    
    def _delivery_callback(self, err, msg):
        """Callback for Kafka producer delivery reports."""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def _log_to_bigquery(
        self,
        input_payload: str,
        agent_response: Dict[str, Any],
        is_safe: bool,
        safety_reason: Optional[str],
        latency_ms: float
    ):
        """
        Log analytics to BigQuery (Black Box Recorder).
        
        Errors are caught silently to not disrupt the stream.
        """
        if not self.bq_client:
            return
        
        try:
            row = {
                "event_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "input_payload": input_payload,
                "agent_response": json.dumps(agent_response) if agent_response else None,
                "is_safe": is_safe,
                "safety_reason": safety_reason,
                "latency_ms": latency_ms
            }
            
            errors = self.bq_client.insert_rows_json(self.table_id, [row])
            
            if errors:
                logger.warning(f"BQ insert error: {errors}")
                self.stats['bq_errors'] += 1
            else:
                logger.debug(f"📊 Logged to BigQuery (latency: {latency_ms:.1f}ms)")
                self.stats['bq_logged'] += 1
                
        except Exception as e:
            logger.warning(f"BQ logging failed (non-fatal): {e}")
            self.stats['bq_errors'] += 1
    
    def _validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response through the safety layer."""
        content = ""
        if isinstance(response, dict):
            content = response.get('answer', '') or response.get('response', '') or str(response)
        else:
            content = str(response)
        
        try:
            result = self.validator.validate(content)
            return result
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {'is_safe': False, 'reason': f'Validation error: {e}'}
    
    def process_message(self, message_value: str) -> Optional[Dict[str, Any]]:
        """Process a single message through the Neuron agent."""
        try:
            logger.info(f"🏈 Received Event: {message_value}")
            response = self.remote_agent.query(input_text=message_value)
            logger.info(f"🧠 Agent Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def produce_response(self, response: Dict[str, Any], key: Optional[str] = None):
        """Produce agent response to the output topic."""
        try:
            value = json.dumps(response)
            self.producer.produce(
                topic=self.output_topic,
                key=key,
                value=value,
                callback=self._delivery_callback
            )
            self.producer.poll(0)
        except Exception as e:
            logger.error(f"Error producing message: {e}")
    
    def run(self):
        """
        Start the bridge consumer loop with full analytics.
        """
        logger.info("=" * 60)
        logger.info("🌉 Neuron Bridge Starting...")
        logger.info(f"   Input Topic:  {self.input_topic}")
        logger.info(f"   Output Topic: {self.output_topic}")
        logger.info(f"   Agent:        {self.agent_resource_id}")
        logger.info(f"   🛡️ Safety:    ValidatorAgent 'SafetyNet' ACTIVE")
        logger.info(f"   📊 Analytics: BigQuery {'ACTIVE' if self.bq_client else 'DISABLED'}")
        logger.info("=" * 60)
        logger.info("Waiting for events... (Ctrl+C to stop)")
        
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        continue
                
                key = msg.key().decode('utf-8') if msg.key() else None
                try:
                    value = msg.value().decode('utf-8') if msg.value() else ""
                except UnicodeDecodeError:
                    logger.warning("Skipping binary message (schema registry metadata)")
                    continue
                
                # ========== START TIMING ==========
                start_time = time.time()
                
                # Process through Vertex AI agent
                response = self.process_message(value)
                self.stats['messages_processed'] += 1
                
                if response:
                    # Validate through safety layer
                    validation_result = self._validate_response(response)
                    
                    # ========== END TIMING ==========
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    is_safe = validation_result.get('is_safe', False)
                    safety_reason = validation_result.get('reason') if not is_safe else None
                    
                    # ========== LOG TO BIGQUERY (Black Box Recorder) ==========
                    self._log_to_bigquery(
                        input_payload=value,
                        agent_response=response,
                        is_safe=is_safe,
                        safety_reason=safety_reason,
                        latency_ms=latency_ms
                    )
                    
                    # ========== SAFETY DECISION ==========
                    if not is_safe:
                        logger.warning(f"{RED}🚫 [BLOCKED] Content unsafe: {safety_reason}{RESET}")
                        self.stats['messages_blocked'] += 1
                        continue
                    
                    logger.info(f"{GREEN}✅ [PASSED] Content validated (latency: {latency_ms:.0f}ms){RESET}")
                    self.stats['messages_passed'] += 1
                    self.produce_response(response, key=key)
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Bridge shutting down...")
            logger.info(f"📊 Final Stats:")
            logger.info(f"   Processed: {self.stats['messages_processed']}")
            logger.info(f"   Passed:    {self.stats['messages_passed']}")
            logger.info(f"   Blocked:   {self.stats['messages_blocked']}")
            logger.info(f"   BQ Logged: {self.stats['bq_logged']}")
            logger.info(f"   BQ Errors: {self.stats['bq_errors']}")
        finally:
            self.consumer.close()
            self.producer.flush()
            logger.info("Bridge stopped.")


def main():
    """Main entry point for the Kafka bridge."""
    
    kafka_config = {
        'bootstrap.servers': 'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': 'UEAFJBH67LNNBKPC',
        'sasl.password': 'cfltGY0RWLd/2RRmmYZWM+5dNDexNRC733PEdub4iF7s60s0mTI9QgKv8y44VHNg'
    }
    
    agent_resource_id = 'projects/488602940935/locations/us-central1/reasoningEngines/205135884394168320'
    
    print("🧠 Initializing Vertex AI...")
    vertexai.init(project='leafy-sanctuary-476515-t2', location='us-central1')
    
    bridge = NeuronBridge(
        kafka_config=kafka_config,
        agent_resource_id=agent_resource_id,
        input_topic="nfl-game-events",
        output_topic="agent-debates"
    )
    
    bridge.run()


if __name__ == "__main__":
    main()
