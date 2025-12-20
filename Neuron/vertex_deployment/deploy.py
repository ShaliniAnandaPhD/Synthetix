"""
deploy.py - Deploy Neuron Core to Vertex AI Reasoning Engine

v1.0.4 - Persistence + Observability (Fully Self-Contained)
"""

from vertexai.preview import reasoning_engines
import vertexai
import os
import uuid
from typing import Any, Dict

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "leafy-sanctuary-476515-t2")
STAGING_BUCKET = f"gs://neuron-staging-{PROJECT_ID}"
LOCATION = "us-central1"


# ============================================================================
# FULLY SELF-CONTAINED AGENT CLASS (all imports happen lazily in set_up)
# ============================================================================

class NeuronCloudAgent:
    """
    Self-contained agent for Vertex AI Reasoning Engine.
    
    All cloud resources (Firestore, Trace) are initialized in set_up()
    which runs on the cloud worker, avoiding pickle issues.
    """
    
    def __init__(self, agent_name: str = "VertexNeuronBot"):
        self.agent_name = agent_name
        self.agent_id = str(uuid.uuid4())
        # These will be initialized in set_up()
        self._firestore_client = None
        self._tracer = None
        self._is_setup = False
    
    def set_up(self):
        """Initialize resources on the cloud worker."""
        if self._is_setup:
            return
            
        import logging
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        
        # 1. Initialize Cloud Trace
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME
            
            resource = Resource.create({SERVICE_NAME: "neuron-cloud-agent"})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("neuron-cloud-agent")
            self._logger.info("Cloud Trace initialized")
        except Exception as e:
            self._logger.warning(f"Trace init failed: {e}")
            self._tracer = None
        
        # 2. Initialize Firestore
        try:
            from google.cloud import firestore
            self._firestore_client = firestore.Client()
            self._collection = self._firestore_client.collection('vertex_production_memory')
            self._logger.info("Firestore initialized with 'vertex_production_memory' collection")
        except Exception as e:
            self._logger.warning(f"Firestore init failed: {e}")
            self._firestore_client = None
            self._collection = None
        
        self._is_setup = True
        self._logger.info(f"NeuronCloudAgent '{self.agent_name}' setup complete")
    
    def _save_state(self, input_text: str, response: Dict[str, Any]):
        """Persist interaction state to Firestore."""
        if not self._collection:
            return
            
        try:
            import time
            doc_ref = self._collection.document(self.agent_id)
            doc_ref.set({
                "agent_name": self.agent_name,
                "agent_id": self.agent_id,
                "last_input": input_text,
                "last_response": str(response),
                "last_updated": time.time(),
                "interaction_count": firestore.Increment(1) if hasattr(firestore, 'Increment') else 1
            }, merge=True)
            self._logger.info(f"State saved to Firestore for agent {self.agent_id}")
        except Exception as e:
            self._logger.warning(f"Failed to save state: {e}")
    
    def query(self, input_text: str) -> Dict[str, Any]:
        """Main entry point for Vertex AI Reasoning Engine."""
        # Ensure setup runs on first query
        if not self._is_setup:
            self.set_up()
        
        # Create response
        response = {
            "answer": f"Processed by {self.agent_name}: {input_text}",
            "agent_id": self.agent_id,
            "confidence": 0.95,
            "source": "neuron_core",
            "status": "success",
            "persistence": "Firestore" if self._firestore_client else "Ephemeral"
        }
        
        # Trace the query
        if self._tracer:
            with self._tracer.start_as_current_span("vertex.query") as span:
                span.set_attribute("neuron.input", input_text)
                span.set_attribute("neuron.agent_name", self.agent_name)
                span.set_attribute("neuron.agent_id", self.agent_id)
                span.set_attribute("neuron.persistence", "Firestore" if self._firestore_client else "Ephemeral")
                
                # Save state to Firestore
                self._save_state(input_text, response)
                
                return response
        else:
            # No tracing, still save state
            self._save_state(input_text, response)
            return response
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": "NeuronCloudAgent",
            "state": "READY",
            "version": "1.0.4"
        }
    
    # Pickle support
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_firestore_client'] = None
        state['_tracer'] = None
        state['_collection'] = None
        state['_is_setup'] = False
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)


# ============================================================================
# DEPLOYMENT
# ============================================================================

if __name__ == "__main__":
    print(f"🚀 Initializing Vertex AI for project: {PROJECT_ID}")
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
    
    print(f"📦 Deploying Neuron Core v1.0.4 to Vertex AI Reasoning Engine...")
    print(f"   Bucket: {STAGING_BUCKET}")
    
    try:
        remote_agent = reasoning_engines.ReasoningEngine.create(
            NeuronCloudAgent(agent_name="VertexNeuronBot"),
            requirements=[
                "google-cloud-aiplatform",
                "google-cloud-firestore>=2.11.0",
                "pydantic",
                "cloudpickle>=2.2.0,<3.0.0",
                "opentelemetry-api",
                "opentelemetry-sdk",
                "opentelemetry-exporter-gcp-trace",
                "opentelemetry-instrumentation"
            ],
            display_name="Neuron-Core-v1.0.4",
            description="Neuron Core v1.0.4 (Firestore Persistence + Cloud Trace)"
        )
        
        print(f"✅ Deployment Complete!")
        print(f"Resource Name: {remote_agent.resource_name}")
        print(f"To query: remote_agent.query(input_text='Trace test')")
        
    except Exception as e:
        print(f"❌ Deployment Failed: {e}")
