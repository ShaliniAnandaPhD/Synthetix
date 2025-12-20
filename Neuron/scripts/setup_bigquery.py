#!/usr/bin/env python3
"""
setup_bigquery.py - Setup BigQuery infrastructure for NFL analysis

Creates the dataset and tables needed to store agent debate logs
for analytics and "Proof of Life" dashboards.
"""

import os
import sys
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "leafy-sanctuary-476515-t2")
DATASET_ID = "nfl_analysis"
TABLE_ID = "agent_debates_log"


def setup_infrastructure():
    """
    Set up BigQuery infrastructure for NFL analysis.
    
    Creates:
    - Dataset: nfl_analysis
    - Table: agent_debates_log with schema for logging agent responses
    """
    print(f"🏗️ Setting up BigQuery infrastructure in project: {PROJECT_ID}")
    
    # Initialize client
    client = bigquery.Client(project=PROJECT_ID)
    
    # Full references
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    # ========== CREATE DATASET ==========
    print(f"\n📁 Creating dataset: {DATASET_ID}")
    
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    dataset.description = "NFL Analysis and Agent Debate Logs"
    
    try:
        client.get_dataset(dataset_ref)
        print(f"   ✅ Dataset '{DATASET_ID}' already exists")
    except NotFound:
        dataset = client.create_dataset(dataset)
        print(f"   ✅ Dataset '{DATASET_ID}' created")
    
    # ========== CREATE TABLE ==========
    print(f"\n📊 Creating table: {TABLE_ID}")
    
    schema = [
        bigquery.SchemaField("event_timestamp", "TIMESTAMP", 
                           description="When the event was processed"),
        bigquery.SchemaField("input_payload", "STRING", 
                           description="Original input from Kafka"),
        bigquery.SchemaField("agent_response", "STRING", 
                           description="Full agent response JSON"),
        bigquery.SchemaField("is_safe", "BOOLEAN", 
                           description="Was the content validated as safe"),
        bigquery.SchemaField("safety_reason", "STRING", 
                           description="Reason if blocked, null if safe"),
        bigquery.SchemaField("latency_ms", "FLOAT", 
                           description="Processing time in milliseconds"),
    ]
    
    table = bigquery.Table(table_ref, schema=schema)
    table.description = "Log of all agent debates processed through the Kafka bridge"
    
    # Add partitioning by timestamp for efficient queries
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="event_timestamp"
    )
    
    try:
        client.get_table(table_ref)
        print(f"   ✅ Table '{TABLE_ID}' already exists")
    except NotFound:
        table = client.create_table(table)
        print(f"   ✅ Table '{TABLE_ID}' created with schema:")
        for field in schema:
            print(f"      - {field.name}: {field.field_type}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 50)
    print("✅ BigQuery infrastructure ready!")
    print(f"   Dataset: {dataset_ref}")
    print(f"   Table:   {table_ref}")
    print("\nTo query:")
    print(f"   SELECT * FROM `{table_ref}` LIMIT 10")
    print("=" * 50)
    
    return {
        "dataset": dataset_ref,
        "table": table_ref
    }


def insert_test_row():
    """Insert a test row to verify the setup."""
    from datetime import datetime
    import json
    
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    test_row = {
        "event_timestamp": datetime.utcnow().isoformat(),
        "input_payload": "Test play: Mahomes throws to Kelce",
        "agent_response": json.dumps({"answer": "Test response", "confidence": 0.95}),
        "is_safe": True,
        "safety_reason": None,
        "latency_ms": 125.5
    }
    
    errors = client.insert_rows_json(table_ref, [test_row])
    
    if errors:
        print(f"❌ Insert failed: {errors}")
        return False
    else:
        print(f"✅ Test row inserted successfully")
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup BigQuery infrastructure for NFL Analysis"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Insert a test row after setup"
    )
    
    args = parser.parse_args()
    
    # Run setup
    result = setup_infrastructure()
    
    if args.test:
        print("\n🧪 Inserting test row...")
        insert_test_row()


if __name__ == "__main__":
    main()
