"""
firestore_store.py - Firestore-based Memory Storage for neuron_core

Provides persistent cloud storage for agent memory using Google Cloud Firestore.
"""

import logging
from typing import Any, Dict, Optional

from google.cloud import firestore
from google.api_core import exceptions as gcp_exceptions

from ..core.instrumentation import get_tracer

logger = logging.getLogger(__name__)


class FirestoreMemoryStore:
    """
    Persistent memory store backed by Google Cloud Firestore.
    
    This class provides a durable, cloud-native storage backend for
    agent memory that persists across sessions and deployments.
    
    Usage:
        store = FirestoreMemoryStore(collection_name='my_agent_memory')
        store.store('session_123', {'context': 'user is asking about weather'})
        data = store.retrieve('session_123')
    """
    
    def __init__(self, collection_name: str = "neuron_memory", project: Optional[str] = None):
        """
        Initialize the Firestore memory store.
        
        Args:
            collection_name: Name of the Firestore collection to use
            project: Optional GCP project ID (uses default if not specified)
        """
        self.collection_name = collection_name
        self._client = None
        self._project = project
        
        try:
            self._client = firestore.Client(project=project)
            self._collection = self._client.collection(collection_name)
            logger.info(f"FirestoreMemoryStore initialized with collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            raise ConnectionError(f"Could not connect to Firestore: {e}")
    
    def store(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Store a dictionary in Firestore.
        
        Args:
            key: Document ID (unique identifier)
            value: Dictionary to store
            
        Returns:
            True if successful, False otherwise
        """
        tracer = get_tracer('firestore')
        with tracer.start_as_current_span('firestore.write') as span:
            span.set_attribute('firestore.collection', self.collection_name)
            span.set_attribute('firestore.document_id', key)
            
            try:
                doc_ref = self._collection.document(key)
                doc_ref.set(value)
                logger.debug(f"Stored document: {key}")
                span.set_attribute('firestore.success', True)
                return True
            except gcp_exceptions.GoogleAPICallError as e:
                logger.error(f"Firestore API error storing {key}: {e}")
                span.set_attribute('firestore.success', False)
                span.record_exception(e)
                return False
            except Exception as e:
                logger.error(f"Error storing document {key}: {e}")
                span.set_attribute('firestore.success', False)
                span.record_exception(e)
                return False
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from Firestore.
        
        Args:
            key: Document ID to retrieve
            
        Returns:
            Dictionary if found, None if document doesn't exist
        """
        tracer = get_tracer('firestore')
        with tracer.start_as_current_span('firestore.read') as span:
            span.set_attribute('firestore.collection', self.collection_name)
            span.set_attribute('firestore.document_id', key)
            
            try:
                doc_ref = self._collection.document(key)
                doc = doc_ref.get()
                
                if doc.exists:
                    logger.debug(f"Retrieved document: {key}")
                    span.set_attribute('firestore.found', True)
                    return doc.to_dict()
                else:
                    logger.debug(f"Document not found: {key}")
                    span.set_attribute('firestore.found', False)
                    return None
                    
            except gcp_exceptions.GoogleAPICallError as e:
                logger.error(f"Firestore API error retrieving {key}: {e}")
                span.record_exception(e)
                return None
            except Exception as e:
                logger.error(f"Error retrieving document {key}: {e}")
                span.record_exception(e)
                return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a document from Firestore.
        
        Args:
            key: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_ref = self._collection.document(key)
            doc_ref.delete()
            logger.debug(f"Deleted document: {key}")
            return True
        except gcp_exceptions.GoogleAPICallError as e:
            logger.error(f"Firestore API error deleting {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a document exists.
        
        Args:
            key: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            doc_ref = self._collection.document(key)
            return doc_ref.get().exists
        except Exception as e:
            logger.error(f"Error checking document existence {key}: {e}")
            return False
    
    def list_keys(self, limit: int = 100) -> list:
        """
        List document IDs in the collection.
        
        Args:
            limit: Maximum number of keys to return
            
        Returns:
            List of document IDs
        """
        try:
            docs = self._collection.limit(limit).stream()
            return [doc.id for doc in docs]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
