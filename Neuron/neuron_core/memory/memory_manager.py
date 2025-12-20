"""
memory_manager.py - Memory Management for neuron_core

Implements memory systems inspired by human memory architecture.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..types import MemoryEntry, MemoryID, MemoryType
from ..exceptions import MemoryAccessError, MemoryStorageError

logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        pass
    
    @abstractmethod
    def retrieve(self, memory_id: MemoryID) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: MemoryID) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    def query(self, query: Dict[str, Any], limit: Optional[int] = None) -> List[MemoryEntry]:
        """Query for memory entries matching criteria."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory entries."""
        pass


class InMemoryStore(MemoryStore):
    """In-memory implementation of MemoryStore."""
    
    def __init__(self):
        self._memories: Dict[MemoryID, MemoryEntry] = {}
        self._lock = threading.Lock()
    
    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        with self._lock:
            self._memories[entry.id] = entry
    
    def retrieve(self, memory_id: MemoryID) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        with self._lock:
            entry = self._memories.get(memory_id)
            if entry:
                entry.access()
            return entry
    
    def delete(self, memory_id: MemoryID) -> bool:
        """Delete a memory entry."""
        with self._lock:
            if memory_id in self._memories:
                del self._memories[memory_id]
                return True
            return False
    
    def query(self, query: Dict[str, Any], limit: Optional[int] = None) -> List[MemoryEntry]:
        """Query for memory entries matching criteria."""
        results = []
        
        with self._lock:
            for entry in self._memories.values():
                matches = True
                
                for key, value in query.items():
                    if key == "memory_type":
                        if entry.memory_type != value:
                            matches = False
                            break
                    elif key == "importance_gte":
                        if entry.importance < value:
                            matches = False
                            break
                    elif key in entry.metadata:
                        if entry.metadata[key] != value:
                            matches = False
                            break
                
                if matches:
                    results.append(entry)
                    if limit and len(results) >= limit:
                        break
        
        return results
    
    def clear(self) -> None:
        """Clear all memory entries."""
        with self._lock:
            self._memories.clear()


class MemoryManager:
    """
    Central manager for agent memory systems.
    
    Provides unified access to different memory types and
    implements memory lifecycle management.
    """
    
    def __init__(self, store: Optional[MemoryStore] = None):
        self._store = store or InMemoryStore()
        self._lock = threading.Lock()
    
    def store(self, content: Any, memory_type: MemoryType,
              importance: float = 0.5,
              metadata: Optional[Dict[str, Any]] = None) -> MemoryID:
        """
        Store information in memory.
        
        Args:
            content: The information to store
            memory_type: Type of memory (working, episodic, etc.)
            importance: Importance score (0.0 to 1.0)
            metadata: Additional context
            
        Returns:
            ID of the stored memory
        """
        entry = MemoryEntry.create(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata
        )
        
        self._store.store(entry)
        logger.debug(f"Stored memory {entry.id} of type {memory_type.name}")
        return entry.id
    
    def retrieve(self, memory_id: MemoryID) -> Optional[Any]:
        """
        Retrieve memory content by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory content, or None if not found
        """
        entry = self._store.retrieve(memory_id)
        if entry:
            return entry.content
        return None
    
    def retrieve_entry(self, memory_id: MemoryID) -> Optional[MemoryEntry]:
        """Retrieve full memory entry by ID."""
        return self._store.retrieve(memory_id)
    
    def delete(self, memory_id: MemoryID) -> bool:
        """Delete a memory entry."""
        return self._store.delete(memory_id)
    
    def query(self, memory_type: Optional[MemoryType] = None,
              min_importance: Optional[float] = None,
              metadata_filter: Optional[Dict[str, Any]] = None,
              limit: Optional[int] = None) -> List[MemoryEntry]:
        """
        Query for memories matching criteria.
        
        Args:
            memory_type: Filter by memory type
            min_importance: Minimum importance score
            metadata_filter: Match against metadata fields
            limit: Maximum number of results
            
        Returns:
            List of matching memory entries
        """
        query_dict = {}
        
        if memory_type:
            query_dict["memory_type"] = memory_type
        if min_importance is not None:
            query_dict["importance_gte"] = min_importance
        if metadata_filter:
            query_dict.update(metadata_filter)
        
        return self._store.query(query_dict, limit)
    
    def get_working_memory(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent working memory entries."""
        entries = self.query(memory_type=MemoryType.WORKING)
        entries.sort(key=lambda e: e.last_accessed, reverse=True)
        return entries[:limit]
    
    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear memories, optionally filtered by type."""
        if memory_type is None:
            self._store.clear()
        else:
            entries = self.query(memory_type=memory_type)
            for entry in entries:
                self._store.delete(entry.id)
