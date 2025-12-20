#!/usr/bin/env python3
"""
kafka_bridge.py - The Bridge between Confluent Kafka and Vertex AI

Listens to NFL game events from Kafka, queries the Neuron agent,
and produces AI-generated commentary back to Kafka.
"""

import json
import logging
from typing import Dict, Any, Optional

from confluent_kafka import Producer, Consumer, KafkaError
import vertexai
from vertexai.preview import reasoning_engines

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuronBridge:
    """
    Bridge service connecting Kafka streams to Vertex AI Reasoning Engine.
    
    Consumes events from 'nfl-game-events' topic, processes them through
    the Neuron agent, and produces responses to 'agent-debates' topic.
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
        Initialize the Neuron Bridge.
        
        Args:
            kafka_config: Confluent Kafka connection configuration
            agent_resource_id: Vertex AI Reasoning Engine resource ID
            input_topic: Topic to consume game events from
            output_topic: Topic to produce agent responses to
            consumer_group: Kafka consumer group ID
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.agent_resource_id = agent_resource_id
        
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
    
    def process_message(self, message_value: str) -> Optional[Dict[str, Any]]:
        """
        Process a single message through the Neuron agent.
        
        Args:
            message_value: The game event message content
            
        Returns:
            Agent response dictionary, or None if processing failed
        """
        try:
            logger.info(f"🏈 Received Event: {message_value}")
            
            # Query the Vertex AI agent
            response = self.remote_agent.query(input_text=message_value)
            
            logger.info(f"🧠 Agent Response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def produce_response(self, response: Dict[str, Any], key: Optional[str] = None):
        """
        Produce agent response to the output topic.
        
        Args:
            response: Agent response dictionary
            key: Optional message key
        """
        try:
            value = json.dumps(response)
            self.producer.produce(
                topic=self.output_topic,
                key=key,
                value=value,
                callback=self._delivery_callback
            )
            self.producer.poll(0)  # Trigger delivery callbacks
            
        except Exception as e:
            logger.error(f"Error producing message: {e}")
    
    def run(self):
        """
        Start the bridge consumer loop.
        
        Continuously consumes messages from the input topic,
        processes them through the agent, and produces responses.
        """
        logger.info("=" * 60)
        logger.info("🌉 Neuron Bridge Starting...")
        logger.info(f"   Input Topic:  {self.input_topic}")
        logger.info(f"   Output Topic: {self.output_topic}")
        logger.info(f"   Agent:        {self.agent_resource_id}")
        logger.info("=" * 60)
        logger.info("Waiting for events... (Ctrl+C to stop)")
        
        try:
            while True:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        continue
                
                # Extract message content
                key = msg.key().decode('utf-8') if msg.key() else None
                try:
                    value = msg.value().decode('utf-8') if msg.value() else ""
                except UnicodeDecodeError:
                    logger.warning("Skipping binary message (schema registry metadata)")
                    continue
                
                # Process through agent
                response = self.process_message(value)
                
                if response:
                    # Produce response to output topic
                    self.produce_response(response, key=key)
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Bridge shutting down...")
        finally:
            self.consumer.close()
            self.producer.flush()
            logger.info("Bridge stopped.")


def main():
    """Main entry point for the Kafka bridge."""
    
    # Confluent Cloud Configuration (Hackathon Demo)
    kafka_config = {
        'bootstrap.servers': 'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': 'UEAFJBH67LNNBKPC',
        'sasl.password': 'cfltGY0RWLd/2RRmmYZWM+5dNDexNRC733PEdub4iF7s60s0mTI9QgKv8y44VHNg'
    }
    
    # Vertex AI Agent Resource ID (v1.0.4)
    agent_resource_id = 'projects/488602940935/locations/us-central1/reasoningEngines/205135884394168320'
    
    # Initialize Vertex AI
    print("🧠 Initializing Vertex AI...")
    vertexai.init(project='leafy-sanctuary-476515-t2', location='us-central1')
    
    # Create and run the bridge
    bridge = NeuronBridge(
        kafka_config=kafka_config,
        agent_resource_id=agent_resource_id,
        input_topic="nfl-game-events",
        output_topic="agent-debates"
    )
    
    bridge.run()


if __name__ == "__main__":
    main()
