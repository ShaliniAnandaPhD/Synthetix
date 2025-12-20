"""
cache_manager.py - Vertex AI Context Caching for neuron_core

Provides context caching for Gemini models to reduce latency and costs
when repeatedly processing the same large documents or system instructions.
"""

import logging
from typing import Optional, Any
from datetime import timedelta

import vertexai
from vertexai.preview import caching

logger = logging.getLogger(__name__)


class ContextCacheManager:
    """
    Manager for Vertex AI Context Caching.
    
    Context caching allows you to cache large documents, system instructions,
    or other content that will be reused across multiple queries, reducing
    latency and token costs.
    
    Usage:
        manager = ContextCacheManager()
        cache_name = manager.create_cache("my-docs", "path/to/document.pdf")
        # Use cache_name when creating GenerativeModel
        manager.delete_cache(cache_name)  # Cleanup when done
    """
    
    def __init__(self, project: Optional[str] = None, location: str = "us-central1"):
        """
        Initialize the Context Cache Manager.
        
        Args:
            project: GCP project ID (uses default if not specified)
            location: GCP region for Vertex AI
        """
        self.project = project
        self.location = location
        
        # Initialize Vertex AI if not already done
        try:
            vertexai.init(project=project, location=location)
            logger.info(f"ContextCacheManager initialized for {location}")
        except Exception as e:
            logger.warning(f"Vertex AI init failed (may already be initialized): {e}")
    
    def create_cache(
        self, 
        display_name: str, 
        content_path: str, 
        ttl_hours: int = 1,
        model_name: str = "gemini-1.5-flash"
    ) -> str:
        """
        Create a context cache from a file.
        
        Args:
            display_name: Human-readable name for the cache
            content_path: Path to the content file (.txt or .pdf)
            ttl_hours: Time-to-live in hours (default 1)
            model_name: Gemini model to use (default gemini-1.5-pro-002)
            
        Returns:
            Cache resource name (ID) for use with GenerativeModel
        """
        # Read the file content
        content = self._read_file(content_path)
        
        if content is None:
            raise ValueError(f"Could not read file at {content_path}")
        
        # Create the cached content
        try:
            cached_content = caching.CachedContent.create(
                model_name=model_name,
                contents=[content],
                display_name=display_name,
                ttl=timedelta(hours=ttl_hours)
            )
            
            logger.info(f"Created cache '{display_name}' with TTL {ttl_hours}h")
            logger.info(f"Cache resource name: {cached_content.name}")
            
            return cached_content.name
            
        except Exception as e:
            logger.error(f"Failed to create cache: {e}")
            raise
    
    def _read_file(self, file_path: str) -> Optional[Any]:
        """
        Read content from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content suitable for caching, or None if read fails
        """
        import os
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif ext == '.pdf':
                # For PDF, we need to use Part.from_data
                from vertexai.generative_models import Part
                
                with open(file_path, 'rb') as f:
                    pdf_data = f.read()
                
                return Part.from_data(pdf_data, mime_type="application/pdf")
                
            else:
                # Try reading as text for other extensions
                logger.warning(f"Unknown extension {ext}, attempting text read")
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def get_cache(self, name: str) -> Optional[caching.CachedContent]:
        """
        Retrieve an existing cache by name.
        
        Args:
            name: Cache resource name (returned from create_cache)
            
        Returns:
            CachedContent object, or None if not found
        """
        try:
            cached_content = caching.CachedContent(cached_content_name=name)
            logger.info(f"Retrieved cache: {name}")
            return cached_content
        except Exception as e:
            logger.error(f"Failed to retrieve cache {name}: {e}")
            return None
    
    def delete_cache(self, name: str) -> bool:
        """
        Delete a cache to free up resources.
        
        Args:
            name: Cache resource name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cached_content = caching.CachedContent(cached_content_name=name)
            cached_content.delete()
            logger.info(f"Deleted cache: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache {name}: {e}")
            return False
    
    def list_caches(self) -> list:
        """
        List all available caches.
        
        Returns:
            List of CachedContent objects
        """
        try:
            caches = list(caching.CachedContent.list())
            logger.info(f"Found {len(caches)} cached contents")
            return caches
        except Exception as e:
            logger.error(f"Failed to list caches: {e}")
            return []
