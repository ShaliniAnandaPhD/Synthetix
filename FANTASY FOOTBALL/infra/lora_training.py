"""
LoRA Training Pipeline for Sportscaster AI

Uses Modal GPU infrastructure to fine-tune open-source LLMs (Llama 3.1)
with LoRA adapters for efficient, customizable sportscaster personalities.

Features:
- A100 GPU acceleration via Modal
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- QLoRA (Quantized LoRA) for memory efficiency
- City-specific adapter training
- Inference endpoint with adapter switching

Usage:
    modal run infra/lora_training.py --data training_data.jsonl
"""

import modal
from typing import List, Dict, Any, Optional
import json
import os

# ============================================================================
# MODAL APP CONFIGURATION
# ============================================================================

app = modal.App("sportscaster-lora")

# GPU-enabled image with all dependencies
lora_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "datasets>=2.16.0",
        "trl>=0.7.0",
        "huggingface_hub>=0.20.0",
        "safetensors>=0.4.0",
        "scipy>=1.11.0",
        "fastapi>=0.100.0"
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

# Volume for caching models and adapters
model_cache = modal.Volume.from_name("sportscaster-model-cache", create_if_missing=True)
adapter_store = modal.Volume.from_name("sportscaster-adapters", create_if_missing=True)


# ============================================================================
# LORA CONFIGURATION
# ============================================================================

LORA_CONFIG = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # Scaling factor
    "target_modules": [         # Modules to adapt
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "optim": "paged_adamw_8bit"
}


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

@app.function(
    image=lora_image,
    gpu="A100",
    timeout=7200,  # 2 hours max
    volumes={
        "/cache": model_cache,
        "/adapters": adapter_store
    },
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_lora_adapter(
    training_data: List[Dict[str, Any]],
    adapter_name: str,
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_epochs: int = 3
) -> Dict[str, Any]:
    """
    Train a LoRA adapter on custom sportscaster data.
    
    Args:
        training_data: List of training examples
        adapter_name: Name for the trained adapter
        base_model: Base model to adapt
        num_epochs: Training epochs
        
    Returns:
        Dict with training metrics and adapter path
    """
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    
    print(f"[LORA] Starting training for adapter: {adapter_name}")
    print(f"[LORA] Base model: {base_model}")
    print(f"[LORA] Training examples: {len(training_data)}")
    
    # Prepare training data
    formatted_data = []
    for ex in training_data:
        city = ex.get("city", "Unknown")
        prompt = ex.get("input_prompt", "")
        output = ex.get("edited_output") or ex.get("original_output", "")
        
        # Chat format for Llama 3.1
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a {city} sportscaster with passionate local pride and deep knowledge of the team's history.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
        
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    print(f"[LORA] Dataset prepared: {len(dataset)} examples")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir="/cache",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model with quantization
    print("[LORA] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="/cache",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA config
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[LORA] Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Training arguments
    output_dir = f"/adapters/{adapter_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_train_epochs=num_epochs,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        fp16=TRAINING_CONFIG["fp16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        optim=TRAINING_CONFIG["optim"],
        report_to="none"
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048
    )
    
    # Train
    print("[LORA] Starting training...")
    train_result = trainer.train()
    
    # Save adapter
    print(f"[LORA] Saving adapter to: {output_dir}")
    trainer.save_model(output_dir)
    
    # Commit to volume
    adapter_store.commit()
    
    return {
        "status": "success",
        "adapter_name": adapter_name,
        "adapter_path": output_dir,
        "training_loss": train_result.training_loss,
        "epochs": num_epochs,
        "examples_trained": len(training_data)
    }


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

@app.function(
    image=lora_image,
    gpu="T4",
    timeout=120,
    volumes={
        "/cache": model_cache,
        "/adapters": adapter_store
    },
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def generate_with_lora(
    prompt: str,
    adapter_name: str,
    city: str = "Philadelphia",
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_tokens: int = 256
) -> Dict[str, Any]:
    """
    Generate text using a trained LoRA adapter.
    
    Args:
        prompt: User prompt
        adapter_name: Name of the adapter to load
        city: City for system prompt
        base_model: Base model
        max_tokens: Maximum generation length
        
    Returns:
        Generated response
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    print(f"[LORA INFERENCE] Loading adapter: {adapter_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir="/cache",
        trust_remote_code=True
    )
    
    # Quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="/cache",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    adapter_path = f"/adapters/{adapter_name}"
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Format prompt
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a {city} sportscaster with passionate local pride and deep knowledge of the team's history.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    
    return {
        "status": "success",
        "response": response,
        "adapter": adapter_name,
        "city": city
    }


# ============================================================================
# ADAPTER MANAGEMENT
# ============================================================================

@app.function(image=lora_image, volumes={"/adapters": adapter_store})
def list_adapters() -> List[str]:
    """List all available LoRA adapters."""
    import os
    
    adapters = []
    adapter_dir = "/adapters"
    
    if os.path.exists(adapter_dir):
        for name in os.listdir(adapter_dir):
            adapter_path = os.path.join(adapter_dir, name)
            if os.path.isdir(adapter_path):
                adapters.append(name)
    
    return adapters


@app.function(image=lora_image, volumes={"/adapters": adapter_store})
def delete_adapter(adapter_name: str) -> Dict[str, Any]:
    """Delete a LoRA adapter."""
    import shutil
    import os
    
    adapter_path = f"/adapters/{adapter_name}"
    
    if os.path.exists(adapter_path):
        shutil.rmtree(adapter_path)
        adapter_store.commit()
        return {"status": "success", "deleted": adapter_name}
    else:
        return {"status": "error", "message": f"Adapter not found: {adapter_name}"}


# ============================================================================
# CALLABLE FUNCTIONS (No web endpoints to stay under 8-endpoint limit)
# These can be called from the main orchestrator app
# ============================================================================

@app.function(image=lora_image, timeout=7200, gpu="A100", volumes={"/cache": model_cache, "/adapters": adapter_store}, secrets=[modal.Secret.from_name("huggingface-secret")])
def train_adapter_api(
    training_data: List[Dict[str, Any]],
    adapter_name: str = "custom-adapter",
    num_epochs: int = 3
) -> Dict[str, Any]:
    """
    API function to trigger LoRA training.
    Call from main orchestrator: sportscaster_lora.train_adapter_api.remote(...)
    """
    if len(training_data) < 50:
        return {
            "status": "error",
            "error": f"Insufficient training data: {len(training_data)} examples (minimum: 50)"
        }
    
    result = train_lora_adapter.remote(
        training_data=training_data,
        adapter_name=adapter_name,
        num_epochs=num_epochs
    )
    
    return result


@app.function(image=lora_image, timeout=120, gpu="T4", volumes={"/cache": model_cache, "/adapters": adapter_store}, secrets=[modal.Secret.from_name("huggingface-secret")])
def generate_api(
    prompt: str,
    adapter_name: str,
    city: str = "Philadelphia"
) -> Dict[str, Any]:
    """
    API function for LoRA inference.
    Call from main orchestrator: sportscaster_lora.generate_api.remote(...)
    """
    if not prompt:
        return {"status": "error", "error": "Prompt required"}
    
    if not adapter_name:
        return {"status": "error", "error": "Adapter name required"}
    
    result = generate_with_lora.remote(
        prompt=prompt,
        adapter_name=adapter_name,
        city=city
    )
    
    return result


@app.function(image=lora_image, volumes={"/adapters": adapter_store})
def list_adapters_api() -> Dict[str, Any]:
    """List all available LoRA adapters."""
    adapters = list_adapters.remote()
    return {
        "status": "success",
        "adapters": adapters,
        "count": len(adapters)
    }


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """
    Local testing entrypoint.
    
    Usage:
        modal run infra/lora_training.py
    """
    print("=" * 60)
    print("SPORTSCASTER LORA TRAINING PIPELINE")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /train_adapter_endpoint  - Train new LoRA adapter")
    print("  POST /generate_endpoint       - Generate with adapter")
    print("  GET  /list_adapters_endpoint  - List available adapters")
    print("\nTo train an adapter:")
    print("  1. Collect 50+ training examples")
    print("  2. POST to /train_adapter_endpoint")
    print("  3. Wait for A100 training (~1-2 hours)")
    print("  4. Use adapter via /generate_endpoint")
    print("=" * 60)
