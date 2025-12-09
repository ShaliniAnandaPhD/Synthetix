"""
Live Commentary Module

Core infrastructure for real-time NFL game commentary.
"""

from .event_classifier import (
    EventClassifier,
    ClassifiedEvent,
    EventType,
    EventUrgency,
    classify_event
)

from .live_dispatcher import (
    LiveAgentDispatcher,
    AgentResponse,
    create_dispatcher
)

from .live_tempo_coordinator import (
    LiveTempoCoordinator,
    CommentarySegment,
    get_tempo_coordinator
)

from .live_voice_stream import (
    LiveVoiceStream,
    AudioChunk,
    create_voice_stream
)

from .live_phrase_cache import (
    LivePhraseCache,
    CachedReaction,
    get_phrase_cache
)

from .live_fallback_manager import (
    LiveFallbackManager,
    FallbackResult,
    FallbackLevel,
    create_fallback_manager
)

from .live_session_manager import (
    LiveSessionManager,
    LiveSession,
    SessionState,
    get_session_manager
)

from .live_audio_queue import (
    LiveAudioQueue,
    AudioQueueItem,
    AudioPriority
)

from .live_latency_dashboard import (
    LiveLatencyDashboard,
    DashboardStats,
    MetricType,
    get_dashboard
)

from .live_cost_cap import (
    LiveCostCap,
    CostCategory,
    CostConfig,
    create_cost_cap
)

__all__ = [
    # Event Classification
    'EventClassifier', 'ClassifiedEvent', 'EventType', 'EventUrgency', 'classify_event',
    # Agent Dispatch
    'LiveAgentDispatcher', 'AgentResponse', 'create_dispatcher',
    # Tempo Coordination
    'LiveTempoCoordinator', 'CommentarySegment', 'get_tempo_coordinator',
    # Voice Streaming
    'LiveVoiceStream', 'AudioChunk', 'create_voice_stream',
    # Phrase Cache
    'LivePhraseCache', 'CachedReaction', 'get_phrase_cache',
    # Fallback Manager
    'LiveFallbackManager', 'FallbackResult', 'FallbackLevel', 'create_fallback_manager',
    # Session Manager
    'LiveSessionManager', 'LiveSession', 'SessionState', 'get_session_manager',
    # Audio Queue
    'LiveAudioQueue', 'AudioQueueItem', 'AudioPriority',
    # Latency Dashboard
    'LiveLatencyDashboard', 'DashboardStats', 'MetricType', 'get_dashboard',
    # Cost Cap
    'LiveCostCap', 'CostCategory', 'CostConfig', 'create_cost_cap',
]
