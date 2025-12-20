#!/usr/bin/env python3
"""
manage_cache.py - CLI for Vertex AI Context Cache Management

Usage:
    python scripts/manage_cache.py --create path/to/document.pdf
    python scripts/manage_cache.py --list
    python scripts/manage_cache.py --delete <RESOURCE_NAME>
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vertexai
from neuron_core.memory.cache_manager import ContextCacheManager

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "leafy-sanctuary-476515-t2")
LOCATION = "us-central1"


def create_cache(file_path: str, display_name: str = None, ttl_hours: int = 1):
    """Create a cache from a file."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    # Generate display name from filename if not provided
    if not display_name:
        display_name = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"📦 Creating cache from: {file_path}")
    print(f"   Display Name: {display_name}")
    print(f"   TTL: {ttl_hours} hour(s)")
    
    manager = ContextCacheManager(project=PROJECT_ID, location=LOCATION)
    
    try:
        cache_name = manager.create_cache(
            display_name=display_name,
            content_path=file_path,
            ttl_hours=ttl_hours
        )
        print(f"\n✅ Cache created successfully!")
        print(f"📛 Resource Name: {cache_name}")
        print(f"\nUse this ID with GenerativeModel.from_cached_content()")
        return cache_name
    except Exception as e:
        print(f"❌ Failed to create cache: {e}")
        return None


def list_caches():
    """List all active caches."""
    print(f"📋 Listing caches for project: {PROJECT_ID}")
    
    manager = ContextCacheManager(project=PROJECT_ID, location=LOCATION)
    caches = manager.list_caches()
    
    if not caches:
        print("   No active caches found.")
        return
    
    print(f"\nFound {len(caches)} cache(s):\n")
    for cache in caches:
        print(f"  📦 {cache.display_name}")
        print(f"     Name: {cache.name}")
        print(f"     Model: {cache.model_name}")
        print(f"     Expires: {cache.expire_time}")
        print()


def delete_cache(resource_name: str):
    """Delete a specific cache."""
    print(f"🗑️ Deleting cache: {resource_name}")
    
    manager = ContextCacheManager(project=PROJECT_ID, location=LOCATION)
    
    if manager.delete_cache(resource_name):
        print(f"✅ Cache deleted successfully!")
    else:
        print(f"❌ Failed to delete cache.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Vertex AI Context Caches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_cache.py --create data/nfl_rulebook.pdf
  python scripts/manage_cache.py --create data/rules.txt --name "NFL-Rules-2024" --ttl 24
  python scripts/manage_cache.py --list
  python scripts/manage_cache.py --delete "projects/123/locations/us-central1/cachedContents/abc"
        """
    )
    
    parser.add_argument(
        "--create", 
        metavar="FILE_PATH",
        help="Create a cache from a file (.txt or .pdf)"
    )
    parser.add_argument(
        "--name",
        help="Display name for the cache (optional, defaults to filename)"
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=1,
        help="Time-to-live in hours (default: 1)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all active caches"
    )
    parser.add_argument(
        "--delete",
        metavar="RESOURCE_NAME",
        help="Delete a cache by resource name"
    )
    
    args = parser.parse_args()
    
    # Initialize Vertex AI
    print(f"🧠 Initializing Vertex AI (Project: {PROJECT_ID}, Location: {LOCATION})")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Execute command
    if args.create:
        create_cache(args.create, args.name, args.ttl)
    elif args.list:
        list_caches()
    elif args.delete:
        delete_cache(args.delete)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
