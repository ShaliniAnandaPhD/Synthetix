"""
Fine-Tuning Pipeline for Sportscaster AI Personalities

This module provides functionality to:
1. Export training data from Supabase
2. Upload training data to GCS
3. Launch Vertex AI supervised fine-tuning jobs
4. Monitor training progress
5. Deploy fine-tuned models

Usage:
    python -m infra.fine_tuning_pipeline --personality <id> --min-examples 100
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf import json_format
import tempfile


class FineTuningPipeline:
    """Pipeline for fine-tuning Gemini models on custom sportscaster data."""
    
    def __init__(
        self,
        project_id: str = "leafy-sanctuary-476515-t2",
        location: str = "us-central1",
        bucket_name: str = "sportscaster-training-data"
    ):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            project_id: GCP project ID
            location: Vertex AI region
            bucket_name: GCS bucket for training data
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        
        # Initialize clients
        aiplatform.init(project=project_id, location=location)
        self.storage_client = storage.Client(project=project_id)
        
    def export_training_data(
        self,
        personality_id: str,
        supabase_url: str,
        supabase_key: str,
        min_examples: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Export training data from Supabase.
        
        Args:
            personality_id: UUID of the personality
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
            min_examples: Minimum examples required
            
        Returns:
            List of training examples in Vertex AI format
        """
        from supabase import create_client
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Fetch positive, edited, and saved examples
        response = supabase.table('training_examples') \
            .select('*') \
            .eq('personality_id', personality_id) \
            .in_('rating', ['positive', 'edited', 'saved']) \
            .execute()
        
        examples = response.data or []
        
        if len(examples) < min_examples:
            raise ValueError(
                f"Insufficient training data: {len(examples)} examples "
                f"(minimum: {min_examples})"
            )
        
        # Convert to Vertex AI format
        training_data = []
        for example in examples:
            # Use edited output if available, otherwise original
            output = example.get('edited_output') or example.get('original_output')
            city = example.get('city', 'Unknown')
            prompt = example.get('input_prompt', '')
            
            training_data.append({
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{city}: {prompt}"}]
                    },
                    {
                        "role": "model", 
                        "parts": [{"text": output}]
                    }
                ]
            })
        
        print(f"[FINE-TUNING] Exported {len(training_data)} training examples")
        return training_data
    
    def upload_to_gcs(
        self,
        training_data: List[Dict[str, Any]],
        personality_id: str
    ) -> str:
        """
        Upload training data to GCS in JSONL format.
        
        Args:
            training_data: List of training examples
            personality_id: UUID for naming
            
        Returns:
            GCS URI of uploaded file
        """
        # Ensure bucket exists
        try:
            bucket = self.storage_client.get_bucket(self.bucket_name)
        except Exception:
            bucket = self.storage_client.create_bucket(
                self.bucket_name,
                location=self.location
            )
            print(f"[GCS] Created bucket: {self.bucket_name}")
        
        # Create JSONL content
        jsonl_content = "\n".join([json.dumps(ex) for ex in training_data])
        
        # Upload to GCS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"training/{personality_id}/{timestamp}_training.jsonl"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(jsonl_content, content_type="application/jsonl")
        
        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
        print(f"[GCS] Uploaded training data to: {gcs_uri}")
        
        return gcs_uri
    
    def create_tuning_job(
        self,
        training_data_uri: str,
        personality_name: str,
        base_model: str = "gemini-1.5-pro-002",
        epochs: int = 3,
        learning_rate: float = 0.0002
    ) -> str:
        """
        Create a Vertex AI supervised fine-tuning job.
        
        Args:
            training_data_uri: GCS URI of training JSONL
            personality_name: Display name for tuned model
            base_model: Base model to fine-tune
            epochs: Training epochs
            learning_rate: Learning rate
            
        Returns:
            Tuning job resource name
        """
        from vertexai.tuning import sft
        
        # Sanitize model name
        safe_name = personality_name.replace(" ", "_").lower()[:30]
        model_display_name = f"sportscaster-{safe_name}-{datetime.now().strftime('%Y%m%d')}"
        
        print(f"[TUNING] Starting fine-tuning job: {model_display_name}")
        print(f"[TUNING] Base model: {base_model}")
        print(f"[TUNING] Training data: {training_data_uri}")
        
        # Create tuning job
        tuning_job = sft.train(
            source_model=base_model,
            train_dataset=training_data_uri,
            tuned_model_display_name=model_display_name,
            epochs=epochs,
            adapter_size=4,  # LoRA rank for efficient tuning
            learning_rate_multiplier=learning_rate
        )
        
        print(f"[TUNING] Job started: {tuning_job.name}")
        print(f"[TUNING] Monitor at: https://console.cloud.google.com/vertex-ai/tuning")
        
        return tuning_job.name
    
    def check_tuning_status(self, job_name: str) -> Dict[str, Any]:
        """
        Check the status of a tuning job.
        
        Args:
            job_name: Resource name of the tuning job
            
        Returns:
            Dict with status information
        """
        from vertexai.tuning import sft
        
        job = sft.SupervisedTuningJob(job_name)
        
        return {
            "name": job.name,
            "state": job.state.name if hasattr(job, 'state') else "UNKNOWN",
            "tuned_model": getattr(job, 'tuned_model_name', None),
            "error": getattr(job, 'error', None)
        }
    
    def list_tuned_models(self) -> List[Dict[str, Any]]:
        """
        List all tuned models in the project.
        
        Returns:
            List of tuned model info
        """
        models = aiplatform.Model.list(
            filter='labels.task_type=tuning'
        )
        
        return [
            {
                "name": m.display_name,
                "resource_name": m.resource_name,
                "created": str(m.create_time)
            }
            for m in models
        ]
    
    def get_tuned_model_endpoint(self, personality_id: str) -> Optional[str]:
        """
        Get the endpoint for a fine-tuned model.
        
        Args:
            personality_id: Personality UUID or name pattern
            
        Returns:
            Model resource name or None
        """
        models = self.list_tuned_models()
        
        for model in models:
            if personality_id in model["name"]:
                return model["resource_name"]
        
        return None


def run_full_pipeline(
    personality_id: str,
    personality_name: str,
    supabase_url: str,
    supabase_key: str,
    min_examples: int = 100,
    epochs: int = 3
) -> Dict[str, Any]:
    """
    Run the complete fine-tuning pipeline.
    
    Args:
        personality_id: UUID of the personality
        personality_name: Human-readable name
        supabase_url: Supabase project URL
        supabase_key: Supabase service key
        min_examples: Minimum training examples
        epochs: Training epochs
        
    Returns:
        Dict with pipeline results
    """
    pipeline = FineTuningPipeline()
    
    # Step 1: Export training data
    print("\n=== Step 1: Exporting Training Data ===")
    training_data = pipeline.export_training_data(
        personality_id=personality_id,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        min_examples=min_examples
    )
    
    # Step 2: Upload to GCS
    print("\n=== Step 2: Uploading to GCS ===")
    gcs_uri = pipeline.upload_to_gcs(training_data, personality_id)
    
    # Step 3: Create tuning job
    print("\n=== Step 3: Starting Tuning Job ===")
    job_name = pipeline.create_tuning_job(
        training_data_uri=gcs_uri,
        personality_name=personality_name,
        epochs=epochs
    )
    
    return {
        "status": "submitted",
        "job_name": job_name,
        "training_examples": len(training_data),
        "gcs_uri": gcs_uri,
        "monitor_url": "https://console.cloud.google.com/vertex-ai/tuning"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning pipeline")
    parser.add_argument("--personality-id", required=True, help="Personality UUID")
    parser.add_argument("--personality-name", required=True, help="Personality name")
    parser.add_argument("--min-examples", type=int, default=100, help="Minimum examples")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_SERVICE_KEY"))
    
    args = parser.parse_args()
    
    if not args.supabase_url or not args.supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY required")
        exit(1)
    
    result = run_full_pipeline(
        personality_id=args.personality_id,
        personality_name=args.personality_name,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        min_examples=args.min_examples,
        epochs=args.epochs
    )
    
    print("\n=== Pipeline Complete ===")
    print(json.dumps(result, indent=2))
